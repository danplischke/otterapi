"""Layout-agnostic core that emits every function family for an endpoint.

For each endpoint OtterAPI emits sync **and** async variants of several
*function families*: the core request function, pagination
(``paginate_*`` + ``_iter``), DataFrame converters (``_df`` / ``_pl``, paginated
and non-paginated), and streaming export (``_export``, paginated and
non-paginated).

Historically this per-endpoint orchestration existed twice -- once in
:class:`otterapi.codegen.codegen.Codegen` (the single-file, non-split output) and
once in :class:`otterapi.codegen.splitting.SplitModuleEmitter` (one file per
module) -- along with a copied block of type-introspection helpers. The two
copies drifted and grew bugs (split silently ignored ``generate_sync`` /
``generate_async``; non-split paginated export crashed). This module is the
single source of truth both writers now share.

Design mirrors :mod:`otterapi.codegen._features` (``FeatureModule`` /
``all_features()``), which registers the static *runtime modules*
(``_pagination.py`` etc.). Here :class:`EndpointFeature` / :func:`endpoint_features`
is the symmetric registry for per-endpoint *function families*.

Extending with a new "way" to generate an endpoint:

* A **new orthogonal family** (e.g. a ``_stream`` variant that does not
  participate in the pagination/dataframe/export coupling) is a one-file change:
  subclass :class:`EndpointFeature`, implement :meth:`~EndpointFeature.emit` (and
  optionally :meth:`~EndpointFeature.file_imports`), and append an instance to
  :data:`ADDITIONAL_FEATURES`. :func:`emit_endpoint` runs it for every endpoint
  and :func:`finalize_file_imports` picks up its file-level imports -- in both
  split and non-split output automatically.
* The **core / dataframe / export triad** is intrinsically coupled by data
  dependencies (pagination *replaces* the core pair; the paginated DataFrame
  *replaces* the non-paginated one; export wraps ``_iter`` when paginated else the
  base function; and the DataFrame-vs-export emission order flips on whether the
  endpoint is paginated). That coupling is encoded once, explicitly, in
  :func:`emit_endpoint` -- it deliberately cannot be a flat feature loop.
"""

from __future__ import annotations

import ast
import fnmatch
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, cast

from otterapi.codegen.ast_utils import (
    MODELS_MODULE,
    ImportCollector,
    _all,
    _name,
)
from otterapi.codegen.dataframes import get_dataframe_config_for_endpoint
from otterapi.codegen.endpoints import (
    build_default_client_code,
    build_standalone_dataframe_fn,
    build_standalone_endpoint_fn,
    build_standalone_paginated_dataframe_fn,
    build_standalone_paginated_fn,
    build_standalone_paginated_iter_fn,
)
from otterapi.codegen.export import (
    build_standalone_export_fn,
    build_standalone_paginated_export_fn,
)
from otterapi.codegen.pagination import get_pagination_config_for_endpoint
from otterapi.codegen.types import collect_used_model_names

if TYPE_CHECKING:
    from otterapi.codegen.pagination import PaginationMethodConfig
    from otterapi.codegen.types import Endpoint, Type
    from otterapi.config import (
        DataFrameConfig,
        DocumentConfig,
        ExportConfig,
        PaginationConfig,
        ResponseUnwrapConfig,
    )


# =============================================================================
# Config + supporting data structures
# =============================================================================


@dataclass(frozen=True)
class EmitConfig:
    """Normalized view of the feature config the emitter needs.

    ``Codegen`` reads ``self.config.<section>`` (always present);
    ``SplitModuleEmitter`` historically received each section as a separate,
    optionally-``None`` constructor argument. This dataclass reconciles both so
    the emission core has a single, uniform accessor, and carries
    ``generate_sync`` / ``generate_async`` -- previously honored only by the
    non-split path.
    """

    pagination: PaginationConfig | None
    dataframe: DataFrameConfig | None
    response_unwrap: ResponseUnwrapConfig | None
    export: ExportConfig | None
    generate_sync: bool = True
    generate_async: bool = True

    @classmethod
    def from_document(cls, config: DocumentConfig) -> EmitConfig:
        """Build from a full :class:`~otterapi.config.DocumentConfig`."""
        return cls(
            pagination=config.pagination,
            dataframe=config.dataframe,
            response_unwrap=config.response_unwrap,
            export=config.export,
            generate_sync=config.generate_sync,
            generate_async=config.generate_async,
        )

    @property
    def pagination_enabled(self) -> bool:
        return bool(self.pagination and self.pagination.enabled)

    @property
    def dataframe_enabled(self) -> bool:
        return bool(self.dataframe and self.dataframe.enabled)

    @property
    def export_enabled(self) -> bool:
        return bool(self.export and self.export.enabled)


class TypeResolver:
    """Owns the type-introspection helpers, seeded once with the generated types.

    These previously existed as identical private methods on both ``Codegen`` and
    ``SplitModuleEmitter``. Behaviour is unchanged; the split path's cleaner
    ``_lookup_response_field_annotation`` factoring is adopted here.
    """

    def __init__(self, typegen_types: dict[str, Type]):
        self._types = typegen_types

    # -- list / item extraction ------------------------------------------------

    @staticmethod
    def _list_item_type(type_ast: ast.expr | None) -> ast.expr | None:
        """Return the item-type AST if ``type_ast`` is ``list[ItemType]``, else None."""
        if (
            isinstance(type_ast, ast.Subscript)
            and isinstance(type_ast.value, ast.Name)
            and type_ast.value.id == 'list'
        ):
            return type_ast.slice
        return None

    def item_type_ast(
        self, endpoint: Endpoint, data_path: str | None = None
    ) -> tuple[ast.expr | None, dict[str, set[str]]]:
        """Extract the item type AST (and its imports) from a list response.

        Handles both a direct ``list[X]`` response type and an envelope response
        whose ``data_path`` field is the list.
        """
        if not endpoint.response_type or not endpoint.response_type.annotation_ast:
            return None, {}

        ann = endpoint.response_type.annotation_ast
        item = self._list_item_type(ann)
        if item is not None:
            return item, self.collect_model_imports_from_ast(item)

        if data_path:
            field_ast = self._lookup_response_field_annotation(endpoint, data_path)
            item = self._list_item_type(field_ast)
            if item is not None:
                return item, self.collect_model_imports_from_ast(item)

        return None, {}

    # -- response-model field lookup ------------------------------------------

    @staticmethod
    def _resolve_response_type_name(endpoint: Endpoint) -> str | None:
        """Resolve the model name backing an endpoint's response type.

        Union response types (e.g. ``SuccessResponse | ErrorResponse``) have no
        name on ``response_type`` directly, so fall back to the 2xx success
        response in ``response_infos``.
        """
        type_name = endpoint.response_type.name if endpoint.response_type else None
        if type_name or not endpoint.response_infos:
            return type_name

        for response_info in endpoint.response_infos:
            if 200 <= response_info.status_code < 300 and response_info.type:
                return response_info.type.name

        return None

    @staticmethod
    def _find_field_annotation(
        class_def: ast.ClassDef, field_name: str
    ) -> ast.expr | None:
        """Find the annotation AST for a field declared in a class body."""
        for stmt in class_def.body:
            if (
                isinstance(stmt, ast.AnnAssign)
                and isinstance(stmt.target, ast.Name)
                and stmt.target.id == field_name
            ):
                return stmt.annotation
        return None

    def _lookup_response_field_annotation(
        self, endpoint: Endpoint, field_path: str
    ) -> ast.expr | None:
        """Return the annotation AST for the first segment of ``field_path``."""
        if not endpoint.response_type:
            return None

        field_name = field_path.split('.')[0]
        type_name = self._resolve_response_type_name(endpoint)
        if not type_name:
            return None

        type_def = self._types.get(type_name)
        if not type_def or not type_def.implementation_ast:
            return None

        impl = type_def.implementation_ast
        if not isinstance(impl, ast.ClassDef):
            return None

        return self._find_field_annotation(impl, field_name)

    def unwrapped_type_ast(
        self, endpoint: Endpoint, data_path: str
    ) -> tuple[ast.expr | None, dict[str, set[str]]]:
        """Extract the type AST (and imports) for the unwrapped data field."""
        annotation = self._lookup_response_field_annotation(endpoint, data_path)
        if annotation is None:
            return None, {}
        return annotation, self.collect_model_imports_from_ast(annotation)

    # -- imports ---------------------------------------------------------------

    def collect_model_imports_from_ast(
        self, annotation_ast: ast.expr
    ) -> dict[str, set[str]]:
        """Collect ``.models`` imports for every model name referenced in an AST."""
        imports: dict[str, set[str]] = {}
        available_models = {
            name
            for name, type_ in self._types.items()
            if type_.implementation_ast is not None
        }
        for node in ast.walk(annotation_ast):
            if isinstance(node, ast.Name) and node.id in available_models:
                imports.setdefault(MODELS_MODULE, set()).add(node.id)
        return imports

    def used_model_names(self, endpoints: list[Endpoint]) -> set[str]:
        """Model names actually referenced by the given endpoints' signatures."""
        return collect_used_model_names(endpoints, self._types)


@dataclass
class EndpointContext:
    """Everything :func:`emit_endpoint` needs for a single endpoint, resolved once.

    ``name_pairs`` is the single choke point for sync/async selection: it honours
    ``generate_sync`` / ``generate_async`` for both output layouts (the fix for
    split mode, which previously always emitted both).
    """

    endpoint: Endpoint
    config: EmitConfig
    resolver: TypeResolver
    pag_config: PaginationMethodConfig | None
    should_unwrap: bool
    unwrap_path: str | None
    name_pairs: list[tuple[bool, str]]

    @classmethod
    def build(
        cls, endpoint: Endpoint, config: EmitConfig, resolver: TypeResolver
    ) -> EndpointContext:
        pag_config = None
        if config.pagination_enabled:
            pag_config = get_pagination_config_for_endpoint(
                endpoint.sync_fn_name, config.pagination, endpoint.parameters
            )

        should_unwrap, unwrap_path = False, None
        if config.response_unwrap:
            should_unwrap, unwrap_path = (
                config.response_unwrap.get_unwrap_config_for_endpoint(
                    endpoint.sync_fn_name
                )
            )

        name_pairs: list[tuple[bool, str]] = []
        if config.generate_sync:
            name_pairs.append((False, endpoint.sync_fn_name))
        if config.generate_async:
            name_pairs.append((True, endpoint.async_fn_name))

        return cls(
            endpoint=endpoint,
            config=config,
            resolver=resolver,
            pag_config=pag_config,
            should_unwrap=should_unwrap,
            unwrap_path=unwrap_path,
            name_pairs=name_pairs,
        )


@dataclass
class EmitSink:
    """Accumulator for one output file: statements, imports, names, feature flags."""

    body: list[ast.stmt] = field(default_factory=list)
    imports: ImportCollector = field(default_factory=ImportCollector)
    names: list[str] = field(default_factory=list)
    has_dataframe: bool = False
    has_pagination: bool = False
    has_export: bool = False
    #: Maps each emitted function name to the endpoint it was generated for.
    #: Populated only while :func:`emit_endpoint` is running (the client
    #: preamble functions have no owner). Used by layout writers that group
    #: functions by endpoint / resource.
    owners: dict[str, Endpoint] = field(default_factory=dict)
    _current_endpoint: Endpoint | None = None

    def add(self, fn: ast.stmt, name: str, imports: dict[str, set[str]]) -> None:
        """Append a generated function, record its name, and merge its imports."""
        self.body.append(fn)
        self.names.append(name)
        self.imports.add_imports(imports)
        if self._current_endpoint is not None:
            self.owners[name] = self._current_endpoint


def build_pagination_config_dict(pag_config: PaginationMethodConfig) -> dict:
    """Build the plain-dict pagination config passed to the AST builders."""
    return {
        'offset_param': pag_config.offset_param,
        'limit_param': pag_config.limit_param,
        'cursor_param': pag_config.cursor_param,
        'page_param': pag_config.page_param,
        'per_page_param': pag_config.per_page_param,
        'data_path': pag_config.data_path,
        'total_path': pag_config.total_path,
        'next_cursor_path': pag_config.next_cursor_path,
        'total_pages_path': pag_config.total_pages_path,
        'default_page_size': pag_config.default_page_size,
        'send_page_size': pag_config.send_page_size,
    }


# =============================================================================
# Endpoint features (the registry, symmetric to _features.all_features())
# =============================================================================


class EndpointFeature(ABC):
    """A family of functions emitted per endpoint (core, dataframe, export, ...).

    Subclasses implement :meth:`emit` (append their variants to the sink) and may
    override :meth:`file_imports` to register file-level imports once per file.
    """

    name: str

    @abstractmethod
    def emit(self, ctx: EndpointContext, sink: EmitSink) -> None:
        """Emit this family's variants for one endpoint into ``sink``."""

    def file_imports(self, sink: EmitSink) -> list[ast.stmt]:
        """Register file-level imports on ``sink.imports`` if this family was used.

        Called once per file, after every endpoint is emitted, so implementations
        can consult ``sink.has_*`` flags. Returns any extra top-level statements to
        splice into the module (e.g. a ``TYPE_CHECKING`` block); default: none.
        """
        return []


class CoreEndpointFeature(EndpointFeature):
    """The primary request function -- plain, or replaced by pagination fns."""

    name = 'core'

    def emit(self, ctx: EndpointContext, sink: EmitSink) -> None:
        if ctx.pag_config is not None:
            self._emit_paginated(ctx, sink)
        else:
            self._emit_standalone(ctx, sink)

    def _emit_standalone(self, ctx: EndpointContext, sink: EmitSink) -> None:
        ep = ctx.endpoint
        unwrap_type_ast = None
        unwrap_type_imports = None
        if ctx.should_unwrap and ctx.unwrap_path:
            unwrap_type_ast, unwrap_type_imports = ctx.resolver.unwrapped_type_ast(
                ep, ctx.unwrap_path
            )

        response_type_imports = None
        if ep.response_type and ep.response_type.annotation_ast:
            response_type_imports = ctx.resolver.collect_model_imports_from_ast(
                ep.response_type.annotation_ast
            )

        for is_async, fn_name in ctx.name_pairs:
            fn, imports = build_standalone_endpoint_fn(
                fn_name=fn_name,
                method=ep.method,
                path=ep.path,
                parameters=ep.parameters,
                request_body_info=ep.request_body,
                response_type=ep.response_type,
                response_infos=ep.response_infos,
                docs=ep.description,
                is_async=is_async,
                unwrap_data_path=ctx.unwrap_path if ctx.should_unwrap else None,
                unwrap_type_ast=unwrap_type_ast,
                unwrap_type_imports=unwrap_type_imports,
                response_type_imports=response_type_imports,
            )
            sink.add(fn, fn_name, imports)

    def _emit_paginated(self, ctx: EndpointContext, sink: EmitSink) -> None:
        ep = ctx.endpoint
        assert ctx.pag_config is not None
        item_type_ast, item_type_imports = ctx.resolver.item_type_ast(
            ep, ctx.pag_config.data_path
        )
        pag_dict = build_pagination_config_dict(ctx.pag_config)

        for is_async, fn_name in ctx.name_pairs:
            fn, imports = build_standalone_paginated_fn(
                fn_name=fn_name,
                method=ep.method,
                path=ep.path,
                parameters=ep.parameters,
                request_body_info=ep.request_body,
                response_type=ep.response_type,
                pagination_style=ctx.pag_config.style,
                pagination_config=pag_dict,
                item_type_ast=item_type_ast,
                item_type_imports=item_type_imports,
                docs=ep.description,
                is_async=is_async,
            )
            sink.add(fn, fn_name, imports)

        for is_async, base_fn_name in ctx.name_pairs:
            iter_fn_name = f'{base_fn_name}_iter'
            fn, imports = build_standalone_paginated_iter_fn(
                fn_name=iter_fn_name,
                method=ep.method,
                path=ep.path,
                parameters=ep.parameters,
                request_body_info=ep.request_body,
                response_type=ep.response_type,
                pagination_style=ctx.pag_config.style,
                pagination_config=pag_dict,
                item_type_ast=item_type_ast,
                item_type_imports=item_type_imports,
                docs=ep.description,
                is_async=is_async,
            )
            sink.add(fn, iter_fn_name, imports)

    def file_imports(self, sink: EmitSink) -> list[ast.stmt]:
        if not sink.has_pagination:
            return []
        sink.imports.add_imports({'collections.abc': {'Iterator', 'AsyncIterator'}})
        sink.imports.add_imports(
            {
                '._pagination': {
                    'paginate_offset',
                    'paginate_offset_async',
                    'paginate_cursor',
                    'paginate_cursor_async',
                    'paginate_page',
                    'paginate_page_async',
                    'iterate_offset',
                    'iterate_offset_async',
                    'iterate_cursor',
                    'iterate_cursor_async',
                    'iterate_page',
                    'iterate_page_async',
                    'extract_path',
                }
            }
        )
        return []


class DataFrameFeature(EndpointFeature):
    """pandas ``_df`` / polars ``_pl`` converters (paginated or non-paginated)."""

    name = 'dataframe'

    def emit(self, ctx: EndpointContext, sink: EmitSink) -> None:
        # emit_endpoint drives the paginated / standalone variants directly to
        # preserve the triad's emission order; this generic entry point is for a
        # hypothetical standalone (registry-only) invocation.
        if ctx.pag_config is not None:
            if self.emit_paginated(ctx, sink):
                return
        self.emit_standalone(ctx, sink)

    def emit_paginated(self, ctx: EndpointContext, sink: EmitSink) -> bool:
        """Emit paginated DataFrame methods; return whether any were emitted.

        When any is emitted the caller skips the non-paginated DataFrame block for
        this endpoint. Mirrors the non-split path: an endpoint whose per-endpoint
        DataFrame config is disabled emits neither library.
        """
        cfg = ctx.config
        if not cfg.dataframe_enabled:
            return False
        assert cfg.dataframe is not None and ctx.pag_config is not None

        endpoint_df_config = cfg.dataframe.endpoints.get(ctx.endpoint.sync_fn_name)
        if endpoint_df_config and endpoint_df_config.enabled is False:
            return False

        item_type_ast, item_type_imports = ctx.resolver.item_type_ast(
            ctx.endpoint, ctx.pag_config.data_path
        )
        pag_dict = build_pagination_config_dict(ctx.pag_config)

        libraries: list[Literal['pandas', 'polars']] = []
        if cfg.dataframe.pandas:
            libraries.append('pandas')
        if cfg.dataframe.polars:
            libraries.append('polars')

        ep = ctx.endpoint
        emitted = False
        for library in libraries:
            suffix = '_df' if library == 'pandas' else '_pl'
            for is_async, base_name in ctx.name_pairs:
                fn_name = f'{base_name}{suffix}'
                fn, imports = build_standalone_paginated_dataframe_fn(
                    fn_name=fn_name,
                    method=ep.method,
                    path=ep.path,
                    parameters=ep.parameters,
                    request_body_info=ep.request_body,
                    response_type=ep.response_type,
                    pagination_style=ctx.pag_config.style,
                    pagination_config=pag_dict,
                    library=library,
                    item_type_ast=item_type_ast,
                    item_type_imports=item_type_imports,
                    docs=ep.description,
                    is_async=is_async,
                )
                sink.add(fn, fn_name, imports)
                emitted = True
        return emitted

    def emit_standalone(self, ctx: EndpointContext, sink: EmitSink) -> bool:
        """Emit non-paginated DataFrame methods; return whether any were emitted."""
        cfg = ctx.config
        if not cfg.dataframe_enabled:
            return False
        assert cfg.dataframe is not None

        # When response unwrapping is active the unwrapped data type is the
        # endpoint's real return type, so feed it to list detection -- otherwise
        # non-paginated envelope list endpoints are misclassified and lose their
        # DataFrame variants.
        unwrap_type_ast = None
        if ctx.should_unwrap and ctx.unwrap_path:
            unwrap_type_ast, _ = ctx.resolver.unwrapped_type_ast(
                ctx.endpoint, ctx.unwrap_path
            )

        df_config = get_dataframe_config_for_endpoint(
            ctx.endpoint, cfg.dataframe, unwrap_type_ast=unwrap_type_ast
        )

        ep = ctx.endpoint
        emitted = False
        for library, generate, suffix in (
            ('pandas', df_config.generate_pandas, '_df'),
            ('polars', df_config.generate_polars, '_pl'),
        ):
            if not generate:
                continue
            for is_async, base_name in ctx.name_pairs:
                fn_name = f'{base_name}{suffix}'
                fn, imports = build_standalone_dataframe_fn(
                    fn_name=fn_name,
                    method=ep.method,
                    path=ep.path,
                    parameters=ep.parameters,
                    request_body_info=ep.request_body,
                    library=cast("Literal['pandas', 'polars']", library),
                    default_path=df_config.path,
                    docs=ep.description,
                    is_async=is_async,
                )
                sink.add(fn, fn_name, imports)
                emitted = True
        return emitted

    def file_imports(self, sink: EmitSink) -> list[ast.stmt]:
        if not sink.has_dataframe:
            return []
        sink.imports.add_imports({'typing': {'TYPE_CHECKING'}})
        sink.imports.add_imports({'._dataframe': {'to_pandas', 'to_polars'}})
        return [
            ast.If(
                test=_name('TYPE_CHECKING'),
                body=[
                    ast.Import(names=[ast.alias(name='pandas', asname='pd')]),
                    ast.Import(names=[ast.alias(name='polars', asname='pl')]),
                ],
                orelse=[],
            )
        ]


class ExportFeature(EndpointFeature):
    """Streaming ``_export`` helpers (wrap ``_iter`` when paginated, else base fn)."""

    name = 'export'

    def emit(self, ctx: EndpointContext, sink: EmitSink) -> None:
        if ctx.pag_config is not None:
            self.emit_paginated(ctx, sink)
        else:
            self.emit_standalone(ctx, sink)

    def emit_paginated(self, ctx: EndpointContext, sink: EmitSink) -> bool:
        """Emit sync+async export wrappers around the paginated ``_iter`` fns."""
        cfg = ctx.config
        if not cfg.export_enabled:
            return False
        assert cfg.export is not None and ctx.pag_config is not None

        item_type_ast, item_type_imports = ctx.resolver.item_type_ast(
            ctx.endpoint, ctx.pag_config.data_path
        )
        if item_type_ast is None:
            return False

        should_generate, formats, _path = cfg.export.should_generate_for_endpoint(
            endpoint_name=ctx.endpoint.sync_fn_name,
            returns_list=True,
        )
        if not should_generate:
            return False

        ep = ctx.endpoint
        default_format = formats[0] if formats else 'csv'
        for is_async, base_name in ctx.name_pairs:
            fn_name = f'{base_name}_export'
            fn, imports = build_standalone_paginated_export_fn(
                fn_name=fn_name,
                target_iter_fn_name=f'{base_name}_iter',
                parameters=ep.parameters,
                request_body_info=ep.request_body,
                item_type_ast=item_type_ast,
                item_type_imports=item_type_imports,
                docs=ep.description,
                is_async=is_async,
                default_format=default_format,
                default_batch_size=cfg.export.batch_size,
                pagination_limit_param=ctx.pag_config.limit_param,
            )
            sink.add(fn, fn_name, imports)
        return True

    def emit_standalone(self, ctx: EndpointContext, sink: EmitSink) -> bool:
        """Emit sync+async export wrappers for a non-paginated list endpoint."""
        cfg = ctx.config
        if not cfg.export_enabled:
            return False
        assert cfg.export is not None

        # When response unwrapping is active the list lives behind the data path
        # (e.g. envelope.data) rather than directly on the response type, so
        # resolve the item type through that path.
        data_path = ctx.unwrap_path if ctx.should_unwrap else None
        item_type_ast, item_type_imports = ctx.resolver.item_type_ast(
            ctx.endpoint, data_path
        )
        if item_type_ast is None:
            return False

        should_generate, formats, _path = cfg.export.should_generate_for_endpoint(
            endpoint_name=ctx.endpoint.sync_fn_name,
            returns_list=True,
        )
        if not should_generate:
            return False

        ep = ctx.endpoint
        default_format = formats[0] if formats else 'csv'
        for is_async, base_name in ctx.name_pairs:
            fn_name = f'{base_name}_export'
            fn, imports = build_standalone_export_fn(
                fn_name=fn_name,
                target_fn_name=base_name,
                parameters=ep.parameters,
                request_body_info=ep.request_body,
                item_type_ast=item_type_ast,
                item_type_imports=item_type_imports,
                docs=ep.description,
                is_async=is_async,
                default_format=default_format,
                default_batch_size=cfg.export.batch_size,
            )
            sink.add(fn, fn_name, imports)
        return True


# The coupled triad, shared as stateless singletons.
_CORE = CoreEndpointFeature()
_DATAFRAME = DataFrameFeature()
_EXPORT = ExportFeature()

#: Orthogonal per-endpoint families that are NOT part of the core/dataframe/export
#: coupling. Register a new "way" here: :func:`emit_endpoint` runs it generically
#: after the triad, and :func:`finalize_file_imports` collects its file imports.
ADDITIONAL_FEATURES: list[EndpointFeature] = []


def endpoint_features() -> list[EndpointFeature]:
    """All per-endpoint families, symmetric to ``_features.all_features()``."""
    return [_CORE, _DATAFRAME, _EXPORT, *ADDITIONAL_FEATURES]


# =============================================================================
# Orchestration
# =============================================================================


def emit_endpoint(ctx: EndpointContext, sink: EmitSink) -> None:
    """Emit every function for one endpoint, encoding the triad's data coupling.

    Emission order (preserved from the canonical non-split path):

    * paginated endpoint -> ``paginate`` pair, ``_iter`` pair, paginated
      DataFrame, paginated export, then a non-paginated DataFrame *fall-through*
      (only if no paginated DataFrame was emitted -- e.g. a per-endpoint override
      re-enables DataFrames the globals turned off);
    * non-paginated endpoint -> core pair, DataFrame, export.

    The DataFrame-before-export vs export-in-between ordering is exactly why this
    is a hand-written orchestrator rather than a flat feature loop.
    """
    sink._current_endpoint = ctx.endpoint
    generated_paginated_df = False
    if ctx.pag_config is not None:
        sink.has_pagination = True
        _CORE.emit(ctx, sink)
        generated_paginated_df = _DATAFRAME.emit_paginated(ctx, sink)
        sink.has_dataframe = sink.has_dataframe or generated_paginated_df
        sink.has_export = sink.has_export or _EXPORT.emit_paginated(ctx, sink)
    else:
        _CORE.emit(ctx, sink)

    if not generated_paginated_df:
        sink.has_dataframe = sink.has_dataframe or _DATAFRAME.emit_standalone(ctx, sink)

    if ctx.pag_config is None:
        sink.has_export = sink.has_export or _EXPORT.emit_standalone(ctx, sink)

    # Orthogonal families run generically after the coupled triad.
    for feature in ADDITIONAL_FEATURES:
        feature.emit(ctx, sink)


def finalize_file_imports(
    sink: EmitSink, resolver: TypeResolver, endpoints: list[Endpoint]
) -> list[ast.stmt]:
    """Add file-level imports (Client, models, feature imports) after the loop.

    Returns extra top-level statements (e.g. the DataFrame ``TYPE_CHECKING``
    block) to splice into the module body.
    """
    sink.imports.add_imports({'.client': {'Client'}})
    for name in resolver.used_model_names(endpoints):
        sink.imports.add_imports({MODELS_MODULE: {name}})

    extra_stmts: list[ast.stmt] = []
    for feature in endpoint_features():
        extra_stmts.extend(feature.file_imports(sink))
    return extra_stmts


def assemble_module_body(
    sink: EmitSink,
    extra_stmts: list[ast.stmt],
    *,
    reexport_models: bool,
    reexport_model_exclude_patterns: list[str] | None,
) -> list[ast.stmt]:
    """Assemble the final module body: future import, imports, __all__, body."""
    final_body: list[ast.stmt] = []
    final_body.extend(sink.imports.to_ast())
    final_body.extend(extra_stmts)

    all_names = set(sink.names)
    if reexport_models:
        model_names = sink.imports._imports.get(MODELS_MODULE, set())
        if reexport_model_exclude_patterns:
            model_names = {
                n
                for n in model_names
                if not any(
                    fnmatch.fnmatch(n, pat) for pat in reexport_model_exclude_patterns
                )
            }
        all_names |= model_names
    final_body.append(_all(sorted(all_names)))

    final_body.extend(sink.body)

    # ``from __future__ import annotations`` must be the first statement so
    # forward references in annotations (model names imported from .models)
    # resolve lazily under cyclic models.
    final_body.insert(
        0,
        ast.ImportFrom(
            module='__future__',
            names=[ast.alias(name='annotations')],
            level=0,
        ),
    )
    return final_body


def build_endpoint_sink(
    endpoints: list[Endpoint],
    config: EmitConfig,
    resolver: TypeResolver,
    description: str | None = None,
) -> EmitSink:
    """Emit the client preamble + every endpoint's functions into a fresh sink.

    Stops short of import finalization and module assembly so callers can either
    finish the module (:func:`build_endpoints_module_body`) or consume the raw
    generated function defs (e.g. the ``client``-style writer, which wraps them
    in delegating methods).
    """
    sink = EmitSink()
    if description:
        sink.body.append(ast.Expr(value=ast.Constant(value=description)))

    client_stmts, client_imports = build_default_client_code()
    sink.body.extend(client_stmts)
    sink.imports.add_imports(client_imports)

    for endpoint in endpoints:
        ctx = EndpointContext.build(endpoint, config, resolver)
        emit_endpoint(ctx, sink)

    return sink


def endpoint_function_defs(
    sink: EmitSink,
) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    """Return only the emitted endpoint function defs from a sink.

    Filters out the client preamble (``_get_client`` etc.) by keeping just the
    defs whose name was recorded in ``sink.names``.
    """
    wanted = set(sink.names)
    return [
        stmt
        for stmt in sink.body
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef))
        and stmt.name in wanted
    ]


def build_endpoints_module_body(
    endpoints: list[Endpoint],
    config: EmitConfig,
    resolver: TypeResolver,
    *,
    reexport_models: bool,
    reexport_model_exclude_patterns: list[str] | None = None,
    description: str | None = None,
) -> tuple[list[ast.stmt], list[str]]:
    """Build a complete endpoint-module body and the emitted function names.

    This is the single entry point shared by both the non-split writer
    (:meth:`Codegen._generate_endpoint_file`) and the split writer
    (:meth:`SplitModuleEmitter._emit_module_file`).
    """
    sink = build_endpoint_sink(endpoints, config, resolver, description)

    extra_stmts = finalize_file_imports(sink, resolver, endpoints)
    body = assemble_module_body(
        sink,
        extra_stmts,
        reexport_models=reexport_models,
        reexport_model_exclude_patterns=reexport_model_exclude_patterns,
    )
    return body, sink.names
