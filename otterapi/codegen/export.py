"""File export utilities for OtterAPI code generation.

This module provides utilities for:
- Generating the ``_export.py`` runtime module in the output directory.
- Resolving per-endpoint export configuration.
- Checking whether an endpoint is eligible for export helper generation.

The generated ``_export.py`` exposes streaming writers for CSV, TSV, JSONL,
and Parquet. Schemas are derived from the endpoint's Pydantic item model
(never inferred from data) so types and nullability stay correct even for
empty results.

CSV / TSV / JSONL use only the Python standard library. Parquet support is
an optional extra (``pip install otterapi[parquet]``) that imports
``pyarrow`` lazily at call time, mirroring the pandas/polars pattern in
:mod:`otterapi.codegen.dataframes`.
"""

import ast
import copy
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING

from upath import UPath

from otterapi.codegen.ast_utils import (
    ImportDict,
    _argument,
    _assign,
    _async_func,
    _call,
    _func,
    _name,
    _subscript,
    _union_expr,
)

if TYPE_CHECKING:
    from otterapi.codegen.types import Endpoint, Parameter, RequestBodyInfo, Type
    from otterapi.config import ExportConfig, ExportFormat

__all__ = [
    'ExportMethodConfig',
    'generate_export_module',
    'get_export_config_for_endpoint',
    'get_export_config_from_parts',
    'endpoint_is_exportable',
    'build_standalone_export_fn',
    'build_standalone_paginated_export_fn',
]


# =============================================================================
# Runtime module content (written verbatim as ``_export.py``)
# =============================================================================


# =============================================================================
# Module generation
# =============================================================================


def generate_export_module(output_dir: Path | UPath) -> Path | UPath:
    """Generate the ``_export.py`` utility module in the output directory."""
    output_dir = UPath(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / '_export.py'
    file_path.write_text(
        files('otterapi.codegen.runtime').joinpath('_export.py').read_text('utf-8')
    )
    return file_path


# =============================================================================
# Configuration resolution
# =============================================================================


@dataclass
class ExportMethodConfig:
    """Resolved per-endpoint export configuration.

    Attributes:
        enabled: Whether export helpers should be generated.
        formats: Formats the helpers should support.
        path: JSON path for extracting list data from the response.
        batch_size: Streaming batch size.
    """

    enabled: bool = False
    formats: 'list[ExportFormat]' = None  # type: ignore[assignment]
    path: str | None = None
    batch_size: int = 1000

    def __post_init__(self) -> None:
        if self.formats is None:
            self.formats = []


def _returns_list(response_type: 'Type | None') -> bool:
    # Local import avoids a circular dependency at module load.
    from otterapi.codegen.dataframes import response_type_returns_list

    return response_type_returns_list(response_type)


def get_export_config_for_endpoint(
    endpoint: 'Endpoint',
    export_config: 'ExportConfig',
) -> ExportMethodConfig:
    """Resolve export configuration for a fully-constructed endpoint."""
    if not export_config.enabled:
        return ExportMethodConfig()

    returns_list = _returns_list(endpoint.response_type)
    should_generate, formats, path = export_config.should_generate_for_endpoint(
        endpoint_name=endpoint.sync_fn_name,
        returns_list=returns_list,
    )
    return ExportMethodConfig(
        enabled=should_generate,
        formats=formats,
        path=path,
        batch_size=export_config.batch_size,
    )


def get_export_config_from_parts(
    endpoint_name: str,
    response_type: 'Type | None',
    export_config: 'ExportConfig',
) -> ExportMethodConfig:
    """Resolve export configuration from individual endpoint parts."""
    if not export_config.enabled:
        return ExportMethodConfig()

    returns_list = _returns_list(response_type)
    should_generate, formats, path = export_config.should_generate_for_endpoint(
        endpoint_name=endpoint_name,
        returns_list=returns_list,
    )
    return ExportMethodConfig(
        enabled=should_generate,
        formats=formats,
        path=path,
        batch_size=export_config.batch_size,
    )


def endpoint_is_exportable(
    endpoint_name: str,
    response_type: 'Type | None',
    export_config: 'ExportConfig',
) -> bool:
    """Return whether export helpers would be generated for this endpoint."""
    resolved = get_export_config_from_parts(endpoint_name, response_type, export_config)
    return resolved.enabled


# =============================================================================
# Per-endpoint AST builders
# =============================================================================
#
# These wrap an existing endpoint (or its ``_iter`` paginated variant) in a
# small forwarder that pipes results into the runtime ``export()``. The
# wrapper signature mirrors the underlying function's parameters exactly,
# adds export-specific kw-only options (``output_path``, ``format``,
# ``batch_size``), and returns the row count from the writer.
#
# The body is intentionally thin -- pagination, request handling, response
# parsing all stay in the underlying function we delegate to. Keeps the
# generated diff small and avoids re-implementing pagination AST.

_EXPORT_FORMAT_LITERAL_VALUES = ('csv', 'tsv', 'jsonl', 'parquet')


def _format_literal_annotation() -> ast.expr:
    """Build the AST for ``Literal['csv', 'tsv', 'jsonl', 'parquet']``."""
    return _subscript(
        'Literal',
        ast.Tuple(
            elts=[ast.Constant(value=v) for v in _EXPORT_FORMAT_LITERAL_VALUES],
            ctx=ast.Load(),
        ),
    )


def _output_path_annotation() -> ast.expr:
    """Build the AST for ``str | Path`` (the wrapper's path argument type)."""
    return _union_expr([_name('str'), _name('Path')])


def _add_export_kwonly_args(
    kwonlyargs: list[ast.arg],
    kw_defaults: list[ast.expr],
    default_format: str,
    default_batch_size: int,
) -> None:
    """Append the export-specific keyword-only args + defaults in place."""
    kwonlyargs.append(_argument('output_path', _output_path_annotation()))
    kw_defaults.append(ast.Constant(value=None))  # required at call time

    kwonlyargs.append(_argument('format', _format_literal_annotation()))
    kw_defaults.append(ast.Constant(value=default_format))

    kwonlyargs.append(
        _argument('batch_size', _name('int')),
    )
    kw_defaults.append(ast.Constant(value=default_batch_size))


def _forward_call_keywords(
    parameters: 'list[Parameter] | None',
    request_body_info: 'RequestBodyInfo | None',
    extra_kw_names: list[str],
) -> list[ast.keyword]:
    """Build keyword arguments for forwarding to the underlying function.

    Forwards every cloned parameter and any extra kwargs (``client``, the
    pagination knobs, etc.) by name, leaving the values bound to the
    same-named locals in the wrapper's scope.
    """
    keywords: list[ast.keyword] = []
    for param in parameters or []:
        name = param.name_sanitized
        keywords.append(ast.keyword(arg=name, value=_name(name)))
    if request_body_info is not None:
        keywords.append(ast.keyword(arg='body', value=_name('body')))
    for kw_name in extra_kw_names:
        keywords.append(ast.keyword(arg=kw_name, value=_name(kw_name)))
    return keywords


def _build_export_dispatch(
    *,
    target_fn_name: str,
    target_call_keywords: list[ast.keyword],
    item_type_ast: ast.expr,
    is_async: bool,
    consumes_async_iter: bool,
) -> list[ast.stmt]:
    """Build the function body for an export wrapper.

    Body shape:
        rows = <target>(...)            # or `await ...`
        return export(rows, output_path, model=<Item>, format=format,
                      batch_size=batch_size)

    For async + paginated case, ``rows`` is an AsyncIterator and we route
    through ``export_async`` (and ``await`` it).
    """
    target_call: ast.expr = _call(
        func=_name(target_fn_name),
        args=[],
        keywords=target_call_keywords,
    )
    if is_async and not consumes_async_iter:
        # ``await get_users_async(...)`` returns the materialized list.
        target_call = ast.Await(value=target_call)

    rows_assign = _assign(_name('rows'), target_call)

    writer_name = 'export_async' if consumes_async_iter else 'export'
    writer_call: ast.expr = _call(
        func=_name(writer_name),
        args=[
            _name('rows'),
            _name('output_path'),
        ],
        keywords=[
            ast.keyword(arg='model', value=copy.deepcopy(item_type_ast)),
            ast.keyword(arg='format', value=_name('format')),
            ast.keyword(arg='batch_size', value=_name('batch_size')),
        ],
    )
    if consumes_async_iter:
        writer_call = ast.Await(value=writer_call)

    return [rows_assign, ast.Return(value=writer_call)]


def _build_export_function(
    *,
    fn_name: str,
    target_fn_name: str,
    parameters: 'list[Parameter] | None',
    request_body_info: 'RequestBodyInfo | None',
    item_type_ast: ast.expr,
    item_type_imports: ImportDict,
    docs: str | None,
    is_async: bool,
    is_paginated: bool,
    default_format: str,
    default_batch_size: int,
) -> tuple[ast.FunctionDef | ast.AsyncFunctionDef, ImportDict]:
    """Shared builder body for both standalone + paginated export wrappers."""
    # Local import avoids a circular dependency at module load.
    from otterapi.codegen.endpoints import FunctionSignatureBuilder

    builder = FunctionSignatureBuilder()
    builder.add_parameters(parameters)
    builder.add_request_body(request_body_info)
    builder.add_client_parameter()
    signature = builder.build()

    extra_kw_forwards: list[str] = ['client']

    # Mirror pagination's wrapper params so they end up forwarded to the
    # underlying ``_iter`` function. We add them here on the wrapper so that
    # they're locally bound names; the actual pagination logic stays in
    # ``_iter`` itself.
    if is_paginated:
        for kw_name, annotation, default in (
            (
                'page_size',
                _name('int'),
                ast.Constant(value=100),
            ),
            (
                'max_items',
                _union_expr([_name('int'), ast.Constant(value=None)]),
                ast.Constant(value=None),
            ),
        ):
            signature.kwonlyargs.append(_argument(kw_name, annotation))
            signature.kw_defaults.append(default)
            extra_kw_forwards.append(kw_name)

    _add_export_kwonly_args(
        signature.kwonlyargs,
        signature.kw_defaults,
        default_format=default_format,
        default_batch_size=default_batch_size,
    )

    target_call_keywords = _forward_call_keywords(
        parameters=parameters,
        request_body_info=request_body_info,
        extra_kw_names=extra_kw_forwards,
    )

    body = _build_export_dispatch(
        target_fn_name=target_fn_name,
        target_call_keywords=target_call_keywords,
        item_type_ast=item_type_ast,
        is_async=is_async,
        consumes_async_iter=is_async and is_paginated,
    )

    docstring = (
        f'Export results of ``{target_fn_name}`` to a file. '
        f'Returns the number of rows written.'
    )
    if docs:
        docstring = f'{docstring}\n\n{docs.strip()}'
    body = [ast.Expr(value=ast.Constant(value=docstring))] + body

    builder_fn = _async_func if is_async else _func
    func_ast = builder_fn(
        name=fn_name,
        args=signature.args,
        body=body,
        kwargs=_argument('format_kwargs', _name('Any')),
        kwonlyargs=signature.kwonlyargs,
        kw_defaults=signature.kw_defaults,
        returns=_name('int'),
    )

    imports: ImportDict = {
        'pathlib': {'Path'},
        'typing': {'Any', 'Literal'},
        '._export': {'export_async' if (is_async and is_paginated) else 'export'},
    }
    for module, names in (signature.imports or {}).items():
        imports.setdefault(module, set()).update(names)
    for module, names in (item_type_imports or {}).items():
        imports.setdefault(module, set()).update(names)

    return func_ast, imports


def build_standalone_export_fn(
    *,
    fn_name: str,
    target_fn_name: str,
    parameters: 'list[Parameter] | None',
    request_body_info: 'RequestBodyInfo | None',
    item_type_ast: ast.expr,
    item_type_imports: ImportDict | None,
    docs: str | None,
    is_async: bool,
    default_format: str = 'csv',
    default_batch_size: int = 1000,
) -> tuple[ast.FunctionDef | ast.AsyncFunctionDef, ImportDict]:
    """Build an export wrapper around a non-paginated list endpoint.

    The wrapper mirrors ``target_fn_name``'s signature, adds keyword-only
    ``output_path`` / ``format`` / ``batch_size`` arguments, calls the
    underlying endpoint, and pipes the resulting list into the runtime
    ``export(...)`` writer. Returns the row count.
    """
    return _build_export_function(
        fn_name=fn_name,
        target_fn_name=target_fn_name,
        parameters=parameters,
        request_body_info=request_body_info,
        item_type_ast=item_type_ast,
        item_type_imports=item_type_imports or {},
        docs=docs,
        is_async=is_async,
        is_paginated=False,
        default_format=default_format,
        default_batch_size=default_batch_size,
    )


def build_standalone_paginated_export_fn(
    *,
    fn_name: str,
    target_iter_fn_name: str,
    parameters: 'list[Parameter] | None',
    request_body_info: 'RequestBodyInfo | None',
    item_type_ast: ast.expr,
    item_type_imports: ImportDict | None,
    docs: str | None,
    is_async: bool,
    default_format: str = 'csv',
    default_batch_size: int = 1000,
) -> tuple[ast.FunctionDef | ast.AsyncFunctionDef, ImportDict]:
    """Build an export wrapper around a paginated endpoint's ``_iter`` variant.

    The wrapper mirrors the paginated endpoint's signature plus
    ``page_size``/``max_items`` (forwarded to the iter function), drives the
    iterator, and pipes items into ``export(...)`` (sync) or
    ``export_async(...)`` (async). Memory stays bounded by ``batch_size``.
    """
    return _build_export_function(
        fn_name=fn_name,
        target_fn_name=target_iter_fn_name,
        parameters=parameters,
        request_body_info=request_body_info,
        item_type_ast=item_type_ast,
        item_type_imports=item_type_imports or {},
        docs=docs,
        is_async=is_async,
        is_paginated=True,
        default_format=default_format,
        default_batch_size=default_batch_size,
    )
