"""Endpoint generation module for creating sync and async HTTP request functions.

This module provides utilities for generating Python AST nodes that represent
HTTP endpoint functions. It supports both synchronous and asynchronous request
patterns with automatic parameter handling, response validation, and type hints.

Key Features:
    - Generates sync and async base request functions with response validation
    - Creates endpoint-specific functions with proper parameter handling
    - Supports query, path, header, and body parameters
    - Automatic TypeAdapter validation and RootModel unwrapping
    - Configurable BASE_URL handling
    - File upload support (multipart/form-data)
    - Form data support (application/x-www-form-urlencoded)
    - Binary response handling with streaming
    - Request timeout configuration
    - Unified EndpointFunctionFactory for consistent endpoint generation
"""

from __future__ import annotations

import ast
import copy
import re
import textwrap
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Literal, Self, cast

from otterapi.codegen.ast_utils import (
    HTTP_MODULE,
    ImportDict,
    _argument,
    _assign,
    _async_func,
    _attr,
    _call,
    _func,
    _function_def,
    _name,
    _subscript,
    _union_expr,
    annotation_includes_none,
)
from otterapi.exceptions import ConfigurationError
from otterapi.openapi.constants import MediaType

if TYPE_CHECKING:
    from otterapi.codegen.types import Parameter, RequestBodyInfo, ResponseInfo, Type

# String annotations for the optional dataframe-conversion return types
_PANDAS_DATAFRAME_ANNOTATION = 'pd.DataFrame'
_POLARS_DATAFRAME_ANNOTATION = 'pl.DataFrame'

# Module providing the Iterator/AsyncIterator types used for streamed responses
_COLLECTIONS_ABC_MODULE = 'collections.abc'

__all__ = [
    # Enums and dataclasses
    'EndpointMode',
    'DataFrameLibrary',
    'PaginationStyle',
    'EndpointFunctionConfig',
    'FunctionSignature',
    # Factory class
    'EndpointFunctionFactory',
    # Builder classes
    'ParameterASTBuilder',
    'FunctionSignatureBuilder',
    # Base request functions
    'base_request_fn',
    'base_async_request_fn',
    'request_fn',
    'async_request_fn',
    # Helper functions
    'clean_docstring',
    'get_parameters',
    'get_base_call_keywords',
    'build_header_params',
    'build_query_params',
    'build_path_params',
    'build_param_styles',
    'flattened_body_args',
    'build_body_params',
    'prepare_call_from_parameters',
    'build_default_client_code',
    'pagination_owned_param_names',
    # Convenience functions
    'build_standalone_endpoint_fn',
    'build_delegating_endpoint_fn',
    'build_standalone_dataframe_fn',
    'build_delegating_dataframe_fn',
    'build_standalone_paginated_fn',
    'build_standalone_paginated_iter_fn',
    'build_standalone_paginated_dataframe_fn',
]


# =============================================================================
# Enums and Configuration
# =============================================================================


class EndpointMode(Enum):
    """Mode of endpoint function generation."""

    STANDALONE = 'standalone'
    """Standalone function that uses a Client instance internally."""

    DELEGATING = 'delegating'
    """Delegating function that calls _get_client().method_name()."""


class DataFrameLibrary(Enum):
    """DataFrame library for DataFrame endpoint functions."""

    PANDAS = 'pandas'
    POLARS = 'polars'


class PaginationStyle(Enum):
    """Pagination style for paginated endpoint functions."""

    OFFSET = 'offset'
    CURSOR = 'cursor'
    PAGE = 'page'


@dataclass(frozen=True)
class _StyleSpec:
    """How one pagination style maps onto signatures, wire names and runtime.

    Attributes:
        knob: The generated function's start-position argument.
        knob_type: Its type (``int`` / ``str``), always ``| None``.
        knob_default: Passed for an unset knob (``offset or 0``); None passes
            the knob as is.
        start_key / size_key: The pagination config keys naming the spec's
            position and page-size parameters, with their conventional names.
        runtime: The item-level runtime helpers, ``paginate_<runtime>`` /
            ``iterate_<runtime>``, which take the start as ``start_<runtime>``.
        fetcher: The page-level runtime helper, which takes it as ``knob``.
    """

    knob: str
    knob_type: str
    knob_default: int | None
    start_key: tuple[str, str]
    size_key: tuple[str, str]
    runtime: str
    fetcher: str


_STYLES: dict[PaginationStyle, _StyleSpec] = {
    PaginationStyle.OFFSET: _StyleSpec(
        'offset', 'int', 0, ('offset_param', 'offset'), ('limit_param', 'limit'),
        'offset', 'fetch_offset_page',
    ),
    PaginationStyle.CURSOR: _StyleSpec(
        'cursor', 'str', None, ('cursor_param', 'cursor'), ('limit_param', 'limit'),
        'cursor', 'fetch_cursor_page',
    ),
    PaginationStyle.PAGE: _StyleSpec(
        'page', 'int', 1, ('page_param', 'page'), ('per_page_param', 'per_page'),
        'page', 'fetch_numbered_page',
    ),
}  # fmt: skip


def _style_spec(pagination_style: PaginationStyle | str | None) -> _StyleSpec | None:
    if pagination_style is None:
        return None
    return _STYLES[PaginationStyle(pagination_style)]


def pagination_start_knob(
    pagination_style: PaginationStyle | str | None,
) -> tuple[str, str] | None:
    """``(name, type)`` of a paginated function's starting-position knob.

    Every wrapper that forwards to ``_iter`` mirrors it through this helper.
    """
    spec = _style_spec(pagination_style)
    return (spec.knob, spec.knob_type) if spec else None


def pagination_omits_page_size(
    pagination_style: PaginationStyle | str | None,
    pagination_config: dict | None,
) -> bool:
    """Whether a paginated function drops its ``page_size`` knob.

    Only cursor style honours ``send_page_size: false`` (forward-token feeds
    with no page-size parameter).
    """
    if _style_spec(pagination_style) is not _STYLES[PaginationStyle.CURSOR]:
        return False
    return not (pagination_config or {}).get('send_page_size', True)


def pagination_fields(
    pagination_style: PaginationStyle | str | None,
    pagination_config: dict | None,
) -> list[tuple[str, str, str]]:
    """``(wire name, fetch_page argument, location)`` of each paging field.

    The location (``query`` / ``body``) comes from ``cursor_location`` and
    ``limit_location``; the page size is absent when it is not sent.
    """
    spec = _style_spec(pagination_style)
    if spec is None:
        return []
    config = pagination_config or {}
    fields = [
        (
            config.get(spec.start_key[0]) or spec.start_key[1],
            '_start',
            config.get('cursor_location', 'query'),
        )
    ]
    if not pagination_omits_page_size(pagination_style, config):
        fields.append(
            (
                config.get(spec.size_key[0]) or spec.size_key[1],
                '_size',
                config.get('limit_location', 'query'),
            )
        )
    return fields


def pagination_owned_param_names(
    pagination_style: PaginationStyle | str | None,
    pagination_config: dict | None,
) -> set[str]:
    """The spec parameter names a paginated function drives itself.

    They are dropped from every paginated signature (``_iter``, ``_df``,
    ``_export`` ...), replaced by its own start / ``page_size`` knobs.
    """
    spec = _style_spec(pagination_style)
    if spec is None:
        return set()
    config = pagination_config or {}
    return {
        config.get(key) or fallback for key, fallback in (spec.start_key, spec.size_key)
    }


def strip_pagination_body_fields(
    request_body_info: RequestBodyInfo | None,
    pagination_style: PaginationStyle | str | None,
    pagination_config: dict | None,
) -> RequestBodyInfo | None:
    """``request_body_info`` without the flattened fields the paginator owns."""
    if request_body_info is None or request_body_info.flattened_fields is None:
        return request_body_info
    owned = {
        wire
        for wire, _, location in pagination_fields(pagination_style, pagination_config)
        if location == 'body'
    }
    if not owned:
        return request_body_info
    stripped = copy.copy(request_body_info)
    stripped.flattened_fields = [
        f for f in request_body_info.flattened_fields if f.wire_name not in owned
    ]
    return stripped


def _flattened_values(fields: list, merge_fields: ast.expr | None) -> ast.expr:
    """``{'wire': local, ...}``, overlaid with ``merge_fields`` when given."""
    values: ast.expr = ast.Dict(
        keys=[ast.Constant(value=item.wire_name) for item in fields],
        values=[_name(item.name) for item in fields],
    )
    if merge_fields is None:
        return values
    return _call(func=_name('merge_body_fields'), args=[values, merge_fields])


def clean_docstring(docstring: str) -> str:
    """Clean and normalize a docstring by removing excess indentation.

    Adds leading and trailing newlines to create blank lines before and
    after the docstring content when rendered.

    Args:
        docstring: The raw docstring to clean.

    Returns:
        A cleaned docstring with normalized indentation and surrounding newlines.
    """
    return '\n' + textwrap.dedent(f'\n{docstring}\n').strip() + '\n'


# =============================================================================
# Function Signature Building
# =============================================================================


@dataclass
class FunctionSignature:
    """Represents a function signature with args, kwargs, and annotations.

    This dataclass holds all the components needed to build the arguments
    portion of a function definition AST node.

    Attributes:
        args: Required positional arguments.
        kwonlyargs: Optional keyword-only arguments.
        kw_defaults: Default values for keyword-only arguments.
        imports: Required imports for type annotations.
    """

    args: list[ast.arg] = field(default_factory=list)
    kwonlyargs: list[ast.arg] = field(default_factory=list)
    kw_defaults: list[ast.expr] = field(default_factory=list)
    imports: ImportDict = field(default_factory=dict)


class FunctionSignatureBuilder:
    """Builder for function signatures from endpoint parameters.

    This builder provides a fluent interface for constructing function
    signatures, handling the common patterns of:
    - Converting parameters to function arguments
    - Separating required and optional parameters
    - Adding body parameters
    - Adding optional client/path parameters
    - Collecting type annotation imports

    Example:
        >>> builder = FunctionSignatureBuilder()
        >>> signature = (
        ...     builder
        ...     .add_parameters(endpoint.parameters)
        ...     .add_request_body(endpoint.request_body)
        ...     .add_client_parameter()
        ...     .build()
        ... )
    """

    def __init__(self):
        """Initialize an empty signature builder."""
        self._args: list[ast.arg] = []
        self._kwonlyargs: list[ast.arg] = []
        self._kw_defaults: list[ast.expr] = []
        self._imports: ImportDict = {}

    def _add_import(self, module: str, name: str) -> None:
        """Add a single import to the collection."""
        if module not in self._imports:
            self._imports[module] = set()
        self._imports[module].add(name)

    def _merge_imports(self, imports: ImportDict) -> None:
        """Merge imports from another ImportDict."""
        for module, names in imports.items():
            if module not in self._imports:
                self._imports[module] = set()
            self._imports[module].update(names)

    def add_parameters(self, parameters: list[Parameter] | None) -> Self:
        """Add endpoint parameters to the signature.

        Separates required parameters into positional args and optional
        parameters into keyword-only args with None defaults.

        Args:
            parameters: List of Parameter objects, or None.

        Returns:
            Self for method chaining.
        """
        if not parameters:
            return self

        for param in parameters:
            if param.type:
                annotation = param.type.annotation_ast or _name('Any')
                self._merge_imports(param.type.annotation_imports)
            else:
                annotation = _name('Any')
                self._add_import('typing', 'Any')

            arg = _argument(param.name_sanitized, annotation)

            if param.required:
                self._args.append(arg)
            else:
                # Wrap annotation in `| None` only if not already nullable.
                # The OpenAPI schema may already encode nullability (nullable:true
                # or type:[..., null]), in which case schema_to_type emits
                # `T | None` and we must not add a second `| None`.
                from otterapi.codegen.types import _binop_includes_none

                if _binop_includes_none(annotation):
                    optional_annotation = annotation
                else:
                    optional_annotation = _union_expr(
                        [annotation, ast.Constant(value=None)]
                    )
                self._kwonlyargs.append(
                    _argument(param.name_sanitized, optional_annotation)
                )
                self._kw_defaults.append(ast.Constant(value=None))

        return self

    def add_request_body(
        self,
        body: RequestBodyInfo | None,
    ) -> Self:
        """Add request body parameter to the signature.

        Adds a 'body' parameter with the appropriate type annotation.
        Required bodies become positional args, optional become keyword-only.

        Args:
            body: The RequestBodyInfo object, or None.

        Returns:
            Self for method chaining.
        """
        if not body:
            return self

        if body.flattened_fields is not None:
            return self._add_flattened_body(body)

        if body.type:
            body_annotation = body.type.annotation_ast or _name('Any')
            self._merge_imports(body.type.annotation_imports)
        else:
            body_annotation = _name('Any')
            self._add_import('typing', 'Any')

        body_arg = _argument('body', body_annotation)

        if body.required:
            self._args.append(body_arg)
        else:
            optional_body_annotation = _union_expr(
                [body_annotation, ast.Constant(value=None)]
            )
            self._kwonlyargs.append(_argument('body', optional_body_annotation))
            self._kw_defaults.append(ast.Constant(value=None))

        return self

    def _add_flattened_body(self, body: RequestBodyInfo) -> Self:
        """Add the body model's fields as parameters instead of one ``body=``."""
        imports: ImportDict = {}
        args, kwonlyargs, kw_defaults = flattened_body_args(body, imports)
        self._args.extend(args)
        self._kwonlyargs.extend(kwonlyargs)
        self._kw_defaults.extend(kw_defaults)
        self._merge_imports(imports)
        return self

    def add_client_parameter(self, client_type: str = 'Client') -> Self:
        """Add an optional client parameter to the signature.

        Adds a keyword-only 'client' parameter with type `Client | None`
        and default value None.

        Args:
            client_type: The name of the client class (default: 'Client').

        Returns:
            Self for method chaining.
        """
        self._kwonlyargs.append(
            _argument(
                'client',
                _union_expr([_name(client_type), ast.Constant(value=None)]),
            )
        )
        self._kw_defaults.append(ast.Constant(value=None))

        return self

    def add_path_parameter(self, default: str | None = None) -> Self:
        """Add an optional path parameter for DataFrame methods.

        Adds a keyword-only 'path' parameter with type `str | None`
        and the specified default value.

        Args:
            default: Default value for the path parameter.

        Returns:
            Self for method chaining.
        """
        self._kwonlyargs.append(
            _argument(
                'path',
                _union_expr([_name('str'), ast.Constant(value=None)]),
            )
        )
        self._kw_defaults.append(ast.Constant(value=default))

        return self

    def add_custom_kwarg(
        self,
        name: str,
        annotation: ast.expr,
        default: ast.expr,
        imports: ImportDict | None = None,
    ) -> Self:
        """Add a custom keyword-only argument.

        Args:
            name: The argument name.
            annotation: The type annotation AST node.
            default: The default value AST node.
            imports: Optional imports required for the annotation.

        Returns:
            Self for method chaining.
        """
        self._kwonlyargs.append(_argument(name, annotation))
        self._kw_defaults.append(default)

        if imports:
            self._merge_imports(imports)

        return self

    def build(self) -> FunctionSignature:
        """Build and return the function signature.

        Returns:
            A FunctionSignature containing all accumulated arguments and imports.
        """
        return FunctionSignature(
            args=self._args.copy(),
            kwonlyargs=self._kwonlyargs.copy(),
            kw_defaults=self._kw_defaults.copy(),
            imports=self._imports.copy(),
        )


# =============================================================================
# Parameter AST Building
# =============================================================================


class ParameterASTBuilder:
    """Unified builder for parameter-related AST nodes.

    This class provides static methods for building AST nodes that represent
    various types of request parameters.

    Example:
        >>> query_dict = ParameterASTBuilder.build_query_params(parameters)
        >>> path_expr = ParameterASTBuilder.build_path_expr('/pet/{petId}', parameters)
    """

    @staticmethod
    def build_query_params(parameters: list[Parameter]) -> ast.Dict | None:
        """Build a dictionary AST node for query parameters.

        Creates an AST Dict node mapping query parameter names to their values.
        Only includes parameters where location == 'query'.

        Args:
            parameters: List of Parameter objects to filter for query params.

        Returns:
            An AST Dict node for query parameters, or None if no query params.
        """
        query_params = [p for p in parameters if p.location == 'query']
        if not query_params:
            return None

        return ast.Dict(
            keys=[ast.Constant(value=param.name) for param in query_params],
            values=[_name(param.name_sanitized) for param in query_params],
        )

    @staticmethod
    def build_header_params(parameters: list[Parameter]) -> ast.Dict | None:
        """Build a dictionary AST node for header parameters.

        Creates an AST Dict node mapping header parameter names to their values.
        Only includes parameters where location == 'header'.

        Args:
            parameters: List of Parameter objects to filter for headers.

        Returns:
            An AST Dict node for header parameters, or None if no header params.
        """
        header_params = [p for p in parameters if p.location == 'header']
        if not header_params:
            return None

        return ast.Dict(
            keys=[ast.Constant(value=param.name) for param in header_params],
            values=[_name(param.name_sanitized) for param in header_params],
        )

    @staticmethod
    def build_path_expr(path: str, parameters: list[Parameter]) -> ast.expr:
        """Build an f-string or constant for the request path.

        Replaces OpenAPI path parameters like {petId} with Python f-string
        expressions using the sanitized parameter names.

        Args:
            path: The OpenAPI path string with placeholders (e.g., '/pet/{petId}').
            parameters: List of Parameter objects to map placeholders to variables.

        Returns:
            An AST expression:
            - ast.Constant for static paths without parameters
            - ast.JoinedStr (f-string) for paths with parameter substitution
        """
        path_params = {p.name: p for p in parameters if p.location == 'path'}

        if not path_params:
            return ast.Constant(value=path)

        pattern = r'\{([^}]+)\}'
        parts = re.split(pattern, path)
        values: list[ast.expr] = []

        for i, part in enumerate(parts):
            if i % 2 == 0:
                if part:
                    values.append(ast.Constant(value=part))
            else:
                param = path_params.get(part)
                sanitized = param.name_sanitized if param else part
                # Interpolating the raw value lets a path parameter containing
                # '/' or '?' escape its URL segment, and renders enums and
                # datetimes with str(). format_path_param percent-encodes and
                # applies the parameter's OpenAPI style.
                keywords: list[ast.keyword] = []
                if param is not None and not param.uses_default_style:
                    keywords.append(
                        ast.keyword(arg='style', value=ast.Constant(value=param.style))
                    )
                    keywords.append(
                        ast.keyword(
                            arg='explode', value=ast.Constant(value=param.explode)
                        )
                    )
                    if param.style == 'matrix':
                        keywords.append(
                            ast.keyword(
                                arg='name', value=ast.Constant(value=param.name)
                            )
                        )
                values.append(
                    ast.FormattedValue(
                        value=_call(
                            _name('format_path_param'),
                            args=[_name(sanitized)],
                            keywords=keywords,
                        ),
                        conversion=-1,
                    )
                )

        if len(values) == 1 and isinstance(values[0], ast.Constant):
            return values[0]

        return ast.JoinedStr(values=values)

    @staticmethod
    def build_param_styles(parameters: list[Parameter]) -> ast.Dict | None:
        """Build the ``param_styles`` mapping for non-default query parameters.

        Returns ``None`` when every query parameter uses the OpenAPI default
        (``form``/``explode=true``), so the common case emits no extra keyword.
        """
        deviating = [
            p for p in parameters if p.location == 'query' and not p.uses_default_style
        ]
        if not deviating:
            return None

        return ast.Dict(
            keys=[ast.Constant(value=p.name) for p in deviating],
            values=[
                ast.Tuple(
                    elts=[
                        ast.Constant(value=p.style),
                        ast.Constant(value=p.explode),
                    ],
                    ctx=ast.Load(),
                )
                for p in deviating
            ],
        )

    @staticmethod
    def build_body_expr(
        body: RequestBodyInfo | None,
        merge_fields: ast.expr | None = None,
    ) -> tuple[ast.expr | None, str | None]:
        """Build an AST expression for the request body parameter.

        Handles different content types and determines the appropriate
        httpx2 parameter name to use.

        Args:
            body: The RequestBodyInfo object, or None if no body.
            merge_fields: For a flattened body, a dict expression overlaid on
                the field values (``merge_body_fields``) before validation.

        Returns:
            A tuple of (body_expr, httpx_param_name).
        """
        if not body:
            return None, None

        body_name = 'body'

        def _dump(source: ast.expr) -> ast.expr:
            """``<source>.model_dump(mode='json', by_alias=True, exclude_unset=True)``.

            The default ``model_dump()`` is Python-mode, so a ``datetime``,
            ``UUID`` or ``Decimal`` field reaches ``json=`` as a native object
            and raises ``TypeError``.  ``by_alias`` restores wire names for
            fields whose spec name is not a Python identifier, and
            ``exclude_unset`` stops every untouched optional field being sent
            as an explicit null -- omitted entirely when the document turns
            that off, so the generated call says what it does.
            """
            keywords = [
                ast.keyword(arg='mode', value=ast.Constant(value='json')),
                ast.keyword(arg='by_alias', value=ast.Constant(value=True)),
            ]
            if body.exclude_unset:
                keywords.append(
                    ast.keyword(arg='exclude_unset', value=ast.Constant(value=True))
                )
            return _call(func=_attr(source, 'model_dump'), args=[], keywords=keywords)

        def _flattened_call() -> ast.expr:
            """``build_request_body(Model, {'id': id, ...}, optional=...)``.

            The filtering, the empty-payload case and the dump all live in the
            runtime helper: inlining them here produced a call expression no
            one could read.
            """
            fields = body.flattened_fields or []
            model = (
                body.type.annotation_ast
                if body.type and body.type.annotation_ast
                else _name('Any')
            )
            keywords: list[ast.keyword] = []
            if not body.exclude_unset:
                keywords.append(
                    ast.keyword(arg='exclude_unset', value=ast.Constant(value=False))
                )
            if not body.required:
                keywords.append(
                    ast.keyword(arg='optional', value=ast.Constant(value=True))
                )
            return _call(
                func=_name('build_request_body'),
                args=[model, _flattened_values(fields, merge_fields)],
                keywords=keywords,
            )

        def _body_value() -> ast.expr:
            """The ``json=``/``data=`` value for a model-backed body."""
            if body.flattened_fields is not None:
                return _flattened_call()
            may_be_none = not body.required or (
                body.type is not None
                and body.type.annotation_ast is not None
                and annotation_includes_none(body.type.annotation_ast)
            )
            if not may_be_none:
                return _dump(_name(body_name))
            # An optional or nullable body may be None, and None has no
            # model_dump; guard the call and pass None straight through.
            return ast.IfExp(
                test=ast.Compare(
                    left=_name(body_name),
                    ops=[ast.IsNot()],
                    comparators=[ast.Constant(value=None)],
                ),
                body=_dump(_name(body_name)),
                orelse=ast.Constant(value=None),
            )

        body_expr: ast.expr
        if body.is_json and body.type and body.type.type in ('model', 'root'):
            body_expr = _body_value()
        elif body.is_multipart:
            keywords: list[ast.keyword] = []
            if not body.exclude_unset:
                keywords.append(
                    ast.keyword(arg='exclude_unset', value=ast.Constant(value=False))
                )
            body_expr = _call(
                func=_name('build_multipart_body'),
                args=[_name(body_name)],
                keywords=keywords,
            )
        elif body.is_form:
            if body.type and body.type.type in ('model', 'root'):
                body_expr = _body_value()
            else:
                body_expr = _name(body_name)
        else:
            body_expr = _name(body_name)

        return body_expr, body.httpx_param_name

    @staticmethod
    def prepare_all_params(
        parameters: list[Parameter] | None,
        path: str,
        request_body_info: RequestBodyInfo | None = None,
    ) -> tuple[ast.expr | None, ast.expr | None, ast.expr | None, str | None, ast.expr]:
        """Prepare all parameter AST nodes for a request function call.

        Args:
            parameters: List of Parameter objects, or None.
            path: The API path with optional placeholders.
            request_body_info: Request body info, or None.

        Returns:
            A tuple of (query_params, header_params, body_expr, body_param_name, path_expr).
        """
        if not parameters:
            parameters = []

        query_params = ParameterASTBuilder.build_query_params(parameters)
        header_params = ParameterASTBuilder.build_header_params(parameters)
        body_expr, body_param_name = ParameterASTBuilder.build_body_expr(
            request_body_info
        )
        path_expr = ParameterASTBuilder.build_path_expr(path, parameters)

        return query_params, header_params, body_expr, body_param_name, path_expr


# =============================================================================
# Endpoint Function Configuration
# =============================================================================


@dataclass
class EndpointFunctionConfig:
    """Configuration for endpoint function generation.

    This dataclass holds all the parameters needed to generate an endpoint
    function. By using a configuration object, we can easily create different
    types of endpoint functions through a single factory.

    Attributes:
        fn_name: The name of the function to generate.
        method: HTTP method (GET, POST, etc.).
        path: The API path with optional placeholders.
        parameters: List of Parameter objects for the endpoint.
        request_body_info: RequestBodyInfo object for the request body, or None.
        response_type: The response Type for the return annotation.
        response_infos: List of ResponseInfo objects for status code handling.
        docs: Optional docstring for the generated function.
        is_async: Whether to generate an async function.
        mode: The generation mode (standalone or delegating).
        client_method_name: For delegating mode, the client method to call.
        dataframe_library: For DataFrame functions, which library to use.
        dataframe_path: For DataFrame functions, the default path parameter.
    """

    fn_name: str
    method: str
    path: str
    parameters: list[Parameter] | None = None
    request_body_info: RequestBodyInfo | None = None
    response_type: Type | None = None
    response_infos: list[ResponseInfo] | None = None
    docs: str | None = None
    is_async: bool = False
    mode: EndpointMode = EndpointMode.STANDALONE
    client_method_name: str | None = None
    dataframe_library: DataFrameLibrary | None = None
    dataframe_path: str | None = None
    # Pagination config
    pagination_style: PaginationStyle | None = None
    pagination_config: dict | None = None  # Contains offset_param, limit_param, etc.
    is_iterator: bool = False  # If True, generates _iter function
    # 'page' -> one Page (the ``_page`` variant); 'pages' -> an iterator of
    # pages (``_pages``); None -> the item-level functions.
    page_mode: Literal['page', 'pages'] | None = None
    is_paginated_dataframe: bool = (
        False  # If True, generates DataFrame from paginated results
    )
    item_type_ast: ast.expr | None = None  # The type of items in the list
    item_type_imports: dict[str, set[str]] | None = None  # Imports for item type
    # Response unwrap config
    unwrap_data_path: str | None = None  # If set, extract response.{path}
    unwrap_type_ast: ast.expr | None = None  # The type of the unwrapped data
    unwrap_type_imports: dict[str, set[str]] | None = None  # Imports for unwrapped type
    # Response type imports (for model names in response_type.annotation_ast)
    response_type_imports: dict[str, set[str]] | None = None
    # The ``_with_response`` variant: return a RawResponse wrapping the HTTP
    # response, with ``parse()`` yielding what the plain function returns.
    with_response: bool = False


# =============================================================================
# Endpoint Function Factory
# =============================================================================


class EndpointFunctionFactory:
    """Factory for generating endpoint functions.

    This factory consolidates the logic for generating various types of
    endpoint functions, reducing code duplication and improving maintainability.

    Example:
        >>> config = EndpointFunctionConfig(
        ...     fn_name='get_pet_by_id',
        ...     method='get',
        ...     path='/pet/{petId}',
        ...     parameters=[...],
        ...     mode=EndpointMode.STANDALONE,
        ... )
        >>> factory = EndpointFunctionFactory(config)
        >>> func_ast, imports = factory.build()
    """

    def __init__(self, config: EndpointFunctionConfig):
        """Initialize the factory with configuration.

        Args:
            config: The endpoint function configuration.
        """
        self.config = config
        self._imports: ImportDict = {}

    def _add_import(self, module: str, name: str) -> None:
        """Add a single import to the collection."""
        if module not in self._imports:
            self._imports[module] = set()
        self._imports[module].add(name)

    def _merge_imports(self, imports: ImportDict) -> None:
        """Merge imports from another ImportDict."""
        for module, names in imports.items():
            if module not in self._imports:
                self._imports[module] = set()
            self._imports[module].update(names)

    def build(self) -> tuple[ast.FunctionDef | ast.AsyncFunctionDef, ImportDict]:
        """Build the endpoint function based on configuration.

        Returns:
            A tuple of (function_ast, imports).
        """
        self._imports = {}

        signature = self._build_signature()

        if self.config.pagination_style:
            if self.config.page_mode is not None:
                body = self._build_page_body()
            elif self.config.is_iterator:
                body = self._build_paginated_iter_body()
            elif self.config.is_paginated_dataframe:
                body = self._build_paginated_dataframe_body()
            else:
                body = self._build_paginated_body()
        elif self.config.dataframe_library:
            body = self._build_dataframe_body()
        elif self.config.mode == EndpointMode.STANDALONE:
            body = self._build_standalone_body()
        else:
            body = self._build_delegating_body()

        returns = self._build_return_type()

        func_builder = _async_func if self.config.is_async else _func

        # Add Any import for **kwargs type annotation
        self._add_import('typing', 'Any')

        func_ast = func_builder(
            name=self.config.fn_name,
            args=signature.args,
            body=body,
            kwargs=_argument('kwargs', _name('Any')),
            kwonlyargs=signature.kwonlyargs,
            kw_defaults=signature.kw_defaults,
            returns=returns,
        )

        self._merge_imports(signature.imports)

        return func_ast, self._imports

    def _build_signature(self) -> FunctionSignature:
        """Build the function signature using FunctionSignatureBuilder."""
        builder = FunctionSignatureBuilder()

        if self.config.pagination_style and self.config.pagination_config:
            skip_params = pagination_owned_param_names(
                self.config.pagination_style, self.config.pagination_config
            )
            filtered_params = [
                p for p in (self.config.parameters or []) if p.name not in skip_params
            ]
            builder.add_parameters(filtered_params)
            self._add_pagination_parameters(builder)
            builder.add_request_body(self._request_body_for_signature())
        else:
            builder.add_parameters(self.config.parameters)
            builder.add_request_body(self.config.request_body_info)

        if self.config.dataframe_library and not self.config.is_paginated_dataframe:
            # Only add path parameter for non-paginated DataFrame functions
            # Paginated DataFrame functions don't need it - data is already extracted
            builder.add_path_parameter(default=self.config.dataframe_path)

        if self.config.mode == EndpointMode.STANDALONE:
            builder.add_client_parameter()

        return builder.build()

    def _add_pagination_parameters(self, builder: FunctionSignatureBuilder) -> None:
        """Add pagination-specific parameters to the signature builder."""
        pag_config = self.config.pagination_config or {}
        default_page_size = pag_config.get('default_page_size', 100)

        # offset: int | None = None  /  cursor: str | None  /  page: int | None
        start_knob = pagination_start_knob(self.config.pagination_style)
        if start_knob is not None:
            knob_name, knob_type = start_knob
            builder.add_custom_kwarg(
                name=knob_name,
                annotation=_union_expr([_name(knob_type), ast.Constant(value=None)]),
                default=ast.Constant(value=None),
            )

        # page_size: int = default_page_size
        if not self._omit_page_size():
            builder.add_custom_kwarg(
                name='page_size',
                annotation=_name('int'),
                default=ast.Constant(value=default_page_size),
            )

        # max_items: int | None = None  (max_pages for the ``_pages`` variant;
        # a single ``_page`` has neither)
        limit_knob = {None: 'max_items', 'pages': 'max_pages'}.get(
            self.config.page_mode
        )
        if limit_knob is not None:
            builder.add_custom_kwarg(
                name=limit_knob,
                annotation=_union_expr([_name('int'), ast.Constant(value=None)]),
                default=ast.Constant(value=None),
            )

    def _build_return_type(self) -> ast.expr:
        """Build the return type annotation."""
        if self.config.dataframe_library:
            return self._dataframe_return_type()

        if self.config.pagination_style:
            return self._pagination_return_type()

        if self.config.response_type and self.config.with_response:
            raw_cls = 'AsyncRawResponse' if self.config.is_async else 'RawResponse'
            self._add_import('._raw_response', raw_cls)
            return _subscript(raw_cls, self._response_return_type())

        if self.config.response_type:
            return self._response_return_type()

        self._add_import(HTTP_MODULE, 'Response')
        return _name('Response')

    def _dataframe_return_type(self) -> ast.expr:
        """Build the return type annotation for DataFrame-returning methods.

        Emitted unquoted (``pd.DataFrame`` / ``pl.DataFrame``): every generated
        module carries ``from __future__ import annotations``, so the annotation
        is a deferred string at runtime and ``pandas`` / ``polars`` only need to
        resolve under ``TYPE_CHECKING``. A quoted string constant would be
        redundant there (ruff ``UP037``).
        """
        annotation = (
            _PANDAS_DATAFRAME_ANNOTATION
            if self.config.dataframe_library == DataFrameLibrary.PANDAS
            else _POLARS_DATAFRAME_ANNOTATION
        )
        module, _, attr = annotation.partition('.')
        return _attr(module, attr)

    def _iterator_return_type(self, item_type_ast: ast.expr) -> ast.expr:
        """Build an ``Iterator[X]`` / ``AsyncIterator[X]`` return type annotation."""
        self._add_import(_COLLECTIONS_ABC_MODULE, 'Iterator')
        self._add_import(_COLLECTIONS_ABC_MODULE, 'AsyncIterator')
        iter_type = 'AsyncIterator' if self.config.is_async else 'Iterator'
        return _subscript(iter_type, item_type_ast)

    def _page_return_type(self) -> ast.expr:
        """``Page[Item, Envelope]``, or an iterator of them for ``_pages``."""
        if self.config.item_type_imports:
            self._merge_imports(self.config.item_type_imports)
        item = self.config.item_type_ast
        if item is None:
            self._add_import('typing', 'Any')
            item = _name('Any')
        response_type = self.config.response_type
        envelope: ast.expr
        if response_type is not None and response_type.annotation_ast is not None:
            self._merge_imports(response_type.annotation_imports)
            envelope = response_type.annotation_ast
        else:
            self._add_import('typing', 'Any')
            envelope = _name('Any')
        page_cls = 'AsyncPage' if self.config.is_async else 'Page'
        page_type = ast.Subscript(
            value=_name(page_cls),
            slice=ast.Tuple(elts=[item, envelope], ctx=ast.Load()),
            ctx=ast.Load(),
        )
        if self.config.page_mode == 'pages':
            return self._iterator_return_type(page_type)
        return page_type

    def _pagination_return_type(self) -> ast.expr:
        """Build the return type annotation for paginated methods."""
        if self.config.page_mode is not None:
            return self._page_return_type()
        if self.config.item_type_ast:
            if self.config.item_type_imports:
                self._merge_imports(self.config.item_type_imports)
            if self.config.is_iterator:
                return self._iterator_return_type(self.config.item_type_ast)
            return _subscript('list', self.config.item_type_ast)

        # Fallback to list[Any] / Iterator[Any] when the item type is unknown
        self._add_import('typing', 'Any')
        if self.config.is_iterator:
            return self._iterator_return_type(_name('Any'))
        return _subscript('list', _name('Any'))

    def _response_return_type(self) -> ast.expr:
        """Build the return type annotation for a regular JSON response."""
        if self.config.unwrap_data_path:
            if self.config.unwrap_type_ast:
                if self.config.unwrap_type_imports:
                    self._merge_imports(self.config.unwrap_type_imports)
                return self.config.unwrap_type_ast
            # Fallback to Any if we can't determine the unwrapped type
            self._add_import('typing', 'Any')
            return _name('Any')

        response_type = self.config.response_type
        if response_type is None or response_type.annotation_ast is None:
            self._add_import('typing', 'Any')
            return _name('Any')
        self._merge_imports(response_type.annotation_imports)
        return response_type.annotation_ast

    def _build_standalone_body(self) -> list[ast.stmt]:
        """Build the body for a standalone endpoint function.

        Shape:
            [docstring]
            c = client or Client()
            [raw_response]  return [await] c._request(...)
            [json path]     response = [await] c._request(...)
                            [return / unwrap] [await] c._parse_response(...)
        """
        from otterapi.codegen._body_builder import BodyStatementBuilder

        builder = BodyStatementBuilder()
        if self.config.docs:
            builder.add_docstring(self.config.docs)
        builder.add_client_init()

        request_keywords = self._build_request_keywords()
        request_method = '_request_async' if self.config.is_async else '_request'

        # The request call + its await wrapping gets reused in three places
        # below; build it once and re-wrap when needed.
        def _request_call() -> ast.expr:
            call: ast.expr = _call(
                func=_attr('c', request_method), keywords=request_keywords
            )
            return ast.Await(value=call) if self.config.is_async else call

        if not self.config.response_type:
            return builder.add_return(_request_call()).build()

        self._merge_imports(self.config.response_type.annotation_imports)
        if self.config.response_type_imports:
            self._merge_imports(self.config.response_type_imports)

        if self._is_raw_response_type(self.config.response_type):
            # Raw types skip Pydantic parsing, but a function annotated
            # ``-> str`` returning the httpx2 Response is a lie: the caller gets
            # an object, not the text it asked for. Pull the body out for the
            # scalar cases; a genuine ``-> Response`` is returned as-is.
            attribute = self._raw_response_attribute(self.config.response_type)
            if attribute is None:
                return builder.add_return(_request_call()).build()
            builder.add_method_call_assignment(
                target_var='response',
                receiver='c',
                method=request_method,
                keywords=request_keywords,
                is_async=self.config.is_async,
            )
            return builder.add_return(_attr('response', attribute)).build()

        # JSON response: request, parse, optionally unwrap.
        builder.add_method_call_assignment(
            target_var='response',
            receiver='c',
            method=request_method,
            keywords=request_keywords,
            is_async=self.config.is_async,
        )

        parse_method = (
            '_parse_response_async' if self.config.is_async else '_parse_response'
        )
        parse_call: ast.expr = _call(
            func=_attr('c', parse_method),
            args=[
                _name('response'),
                self.config.response_type.annotation_ast or _name('Any'),
            ],
        )
        if self.config.is_async:
            parse_call = ast.Await(value=parse_call)

        parse_body: list[ast.stmt]
        if self.config.unwrap_data_path:
            unwrap_expr = self._build_unwrap_expression(
                'result', self.config.unwrap_data_path
            )
            parse_body = [
                _assign(_name('result'), parse_call),
                ast.Return(value=unwrap_expr),
            ]
        else:
            parse_body = [ast.Return(value=parse_call)]

        if self.config.with_response:
            # def _parse() -> T: <parse + unwrap>
            # return RawResponse(response, _parse)
            raw_cls = 'AsyncRawResponse' if self.config.is_async else 'RawResponse'
            self._add_import('._raw_response', raw_cls)
            builder.add_statement(
                _function_def(
                    name='_parse',
                    args=ast.arguments(
                        posonlyargs=[],
                        args=[],
                        kwonlyargs=[],
                        kw_defaults=[],
                        kwarg=None,
                        defaults=[],
                    ),
                    body=parse_body,
                    is_async=self.config.is_async,
                    returns=self._response_return_type(),
                )
            )
            return builder.add_return(
                _call(_name(raw_cls), args=[_name('response'), _name('_parse')])
            ).build()

        for stmt in parse_body:
            builder.add_statement(stmt)
        return builder.build()

    @staticmethod
    def _is_raw_union_element(elt: ast.expr) -> bool:
        """Check if a Union member is a raw type (Response/bytes/str/None)."""
        if isinstance(elt, ast.Name):
            return elt.id in ('Response', 'bytes', 'str', 'None')
        return isinstance(elt, ast.Constant) and elt.value is None

    def _is_raw_response_type(self, response_type: Type) -> bool:
        """Check if the response type is a raw type that doesn't need JSON parsing.

        Raw types include: Response, bytes, str (for non-JSON content types).
        A Union is considered raw only if every member is itself a raw type.
        """
        if response_type is None:
            return True

        ann = response_type.annotation_ast
        if isinstance(ann, ast.Name):
            return ann.id in ('Response', 'bytes', 'str')

        if (
            isinstance(ann, ast.Subscript)
            and isinstance(ann.value, ast.Name)
            and ann.value.id == 'Union'
            and isinstance(ann.slice, ast.Tuple)
        ):
            return all(self._is_raw_union_element(elt) for elt in ann.slice.elts)

        return False

    @staticmethod
    def _raw_response_attribute(response_type: Type) -> str | None:
        """Return the ``Response`` attribute that yields *response_type*.

        ``str`` comes from ``.text`` and ``bytes`` from ``.content``.  Returns
        None for anything else -- including a real ``Response`` return type and
        unions -- meaning the response object itself is what to hand back.
        """
        ann = response_type.annotation_ast
        if not isinstance(ann, ast.Name):
            return None
        return {'str': 'text', 'bytes': 'content'}.get(ann.id)

    def _build_unwrap_expression(self, var_name: str, data_path: str) -> ast.expr:
        """Build an expression to extract data from a response.

        For data_path="data", generates: result.data
        For data_path="data.items", generates: result.data.items

        Args:
            var_name: The variable name holding the response.
            data_path: Dotted path to the data field.

        Returns:
            AST expression for accessing the data.
        """
        parts = data_path.split('.')
        expr: ast.expr = _name(var_name)
        for part in parts:
            expr = _attr(expr, part)
        return expr

    def _build_delegating_body(self) -> list[ast.stmt]:
        """Build the body for a delegating endpoint function.

        Pure forwarder: create a fresh ``Client()`` and call the same-named
        method on it, optionally awaited. Composed via
        :class:`BodyStatementBuilder` so the flow reads top-to-bottom.
        """
        from otterapi.codegen._body_builder import BodyStatementBuilder

        builder = BodyStatementBuilder()
        if self.config.docs:
            builder.add_docstring(self.config.docs)

        call_args, call_keywords = self._build_delegating_call_args()
        method_name = self.config.client_method_name or self.config.fn_name
        client_call: ast.expr = _call(
            func=_attr(_call(_name('Client')), method_name),
            args=call_args,
            keywords=call_keywords,
        )
        if self.config.is_async:
            client_call = ast.Await(value=client_call)

        return builder.add_return(client_call).build()

    # ------------------------------------------------------------------
    # Shared pagination call/keyword construction.
    #
    # ``paginate_*`` and ``iterate_*`` runtime helpers take an identical set
    # of keyword arguments per style (offset/cursor/page); only the function
    # name prefix and how the caller consumes the result differ. Factoring
    # this out collapses three near-identical style dispatches into one.
    # ------------------------------------------------------------------

    def _spec(self) -> _StyleSpec:
        spec = _style_spec(self.config.pagination_style)
        if spec is None:
            raise ValueError('not a paginated endpoint')
        return spec

    def _next_cursor_getter(
        self, pag_config: dict, extract_items: ast.Lambda
    ) -> ast.expr:
        """``get_next_cursor``: from the response, from the last item, or None."""
        if pag_config.get('next_cursor_from') == 'last_item':
            # lambda page: last_item_value(<items of page>, 'cursor')
            return ast.Lambda(
                args=copy.deepcopy(extract_items.args),
                body=_call(
                    func=_name('last_item_value'),
                    args=[
                        copy.deepcopy(extract_items.body),
                        ast.Constant(
                            value=pag_config.get('next_cursor_item_path') or 'cursor'
                        ),
                    ],
                ),
            )
        next_cursor_path = pag_config.get('next_cursor_path')
        if next_cursor_path:
            return self._build_extract_lambda(next_cursor_path)
        return ast.Lambda(
            args=copy.deepcopy(extract_items.args), body=ast.Constant(value=None)
        )

    def _build_pagination_call(
        self, prefix: Literal['paginate', 'iterate'] | None
    ) -> ast.expr:
        """A runtime call over ``fetch_page`` (not awaited).

        ``prefix`` picks the item-level helper (``paginate_*`` / ``iterate_*``,
        which take ``start_*`` and ``max_items``); None picks the page fetcher
        (``fetch_*_page``, which takes the start under the knob's own name).
        """
        spec = self._spec()
        style = self.config.pagination_style
        pag_config = self.config.pagination_config or {}
        extract_items = self._build_extract_lambda(pag_config.get('data_path'))
        keywords = [
            ast.keyword(arg='fetch_page', value=_name('fetch_page')),
            ast.keyword(arg='extract_items', value=extract_items),
        ]
        if style == PaginationStyle.CURSOR:
            keywords.append(
                ast.keyword(
                    arg='get_next_cursor',
                    value=self._next_cursor_getter(pag_config, extract_items),
                )
            )
        getters = (
            [('get_total_pages', 'total_pages_path')]
            if style == PaginationStyle.PAGE
            else []
        )
        # paginate_page / iterate_page take no get_total.
        if prefix is None or style != PaginationStyle.PAGE:
            getters.append(('get_total', 'total_path'))
        for keyword, path_key in getters:
            if pag_config.get(path_key):
                keywords.append(
                    ast.keyword(
                        arg=keyword,
                        value=self._build_extract_lambda(pag_config[path_key]),
                    )
                )
        start: ast.expr = _name(spec.knob)
        if spec.knob_default is not None:
            start = ast.BoolOp(
                op=ast.Or(), values=[start, ast.Constant(value=spec.knob_default)]
            )
        keywords.append(
            ast.keyword(
                arg=f'start_{spec.runtime}' if prefix else spec.knob, value=start
            )
        )
        if not self._omit_page_size():
            keywords.append(ast.keyword(arg='page_size', value=_name('page_size')))
        if prefix:
            keywords.append(ast.keyword(arg='max_items', value=_name('max_items')))
        fn = f'{prefix}_{spec.runtime}' if prefix else spec.fetcher
        if self.config.is_async:
            fn = f'{fn}_async'
        return _call(func=_name(fn), keywords=keywords)

    def _build_paginated_dataframe_body(self) -> list[ast.stmt]:
        """Build the body for a paginated DataFrame endpoint function.

        This combines pagination (to fetch all items) with DataFrame conversion.
        """
        from otterapi.codegen._body_builder import BodyStatementBuilder

        library = self.config.dataframe_library
        return_type_str = (
            _PANDAS_DATAFRAME_ANNOTATION
            if library == DataFrameLibrary.PANDAS
            else _POLARS_DATAFRAME_ANNOTATION
        )
        docstring = (self.config.docs or '') + f'\n\nReturns:\n    {return_type_str}'

        body = (
            BodyStatementBuilder()
            .add_docstring(docstring)
            .add_client_init()
            .add_statement(self._build_fetch_page_function())
            .build()
        )

        paginate_call = self._build_pagination_call('paginate')
        if self.config.is_async:
            paginate_call = ast.Await(value=paginate_call)

        # items = paginate_offset(...)
        body.append(_assign(_name('items'), paginate_call))

        # return to_pandas(items) or to_polars(items)
        helper_fn = 'to_pandas' if library == DataFrameLibrary.PANDAS else 'to_polars'
        body.append(
            ast.Return(value=_call(func=_name(helper_fn), args=[_name('items')]))
        )

        return body

    def _build_dataframe_body(self) -> list[ast.stmt]:
        """Build the body for a DataFrame endpoint function.

        Composed via :class:`BodyStatementBuilder` so the two modes
        (standalone vs delegating) read top-to-bottom instead of weaving
        AST construction in line with control flow.
        """
        from otterapi.codegen._body_builder import BodyStatementBuilder

        library = self.config.dataframe_library
        return_type_str = (
            _PANDAS_DATAFRAME_ANNOTATION
            if library == DataFrameLibrary.PANDAS
            else _POLARS_DATAFRAME_ANNOTATION
        )
        helper_fn = 'to_pandas' if library == DataFrameLibrary.PANDAS else 'to_polars'
        docstring = (self.config.docs or '') + f'\n\nReturns:\n    {return_type_str}'

        builder = BodyStatementBuilder().add_docstring(docstring)

        if self.config.mode != EndpointMode.STANDALONE:
            # Delegating mode: defer to the corresponding method on a fresh
            # ``Client()`` instance.
            call_args, call_keywords = self._build_delegating_call_args()
            method_name = self.config.client_method_name or self.config.fn_name
            client_call: ast.expr = _call(
                func=_attr(_call(_name('Client')), method_name),
                args=call_args,
                keywords=call_keywords,
            )
            if self.config.is_async:
                client_call = ast.Await(value=client_call)
            return builder.add_return(client_call).build()

        request_method = (
            '_request_json_async' if self.config.is_async else '_request_json'
        )

        return (
            builder.add_client_init()
            .add_method_call_assignment(
                target_var='data',
                receiver='c',
                method=request_method,
                keywords=self._build_request_keywords(),
                is_async=self.config.is_async,
            )
            .add_return_call(
                helper_fn,
                args=[_name('data')],
                keywords=[ast.keyword(arg='path', value=_name('path'))],
            )
            .build()
        )

    def _build_paginated_body(self) -> list[ast.stmt]:
        """Build the body for a paginated endpoint function.

        Docstring + client init + fetch_page helper get composed via
        :class:`BodyStatementBuilder`; the ``paginate_{offset,cursor,page}``
        call is built by the shared :meth:`_build_pagination_call` helper.
        """
        from otterapi.codegen._body_builder import BodyStatementBuilder

        builder = BodyStatementBuilder()
        if self.config.docs:
            builder.add_docstring(self.config.docs)
        builder.add_client_init()
        builder.add_statement(self._build_fetch_page_function())
        body = builder.build()

        paginate_call = self._build_pagination_call('paginate')
        if self.config.is_async:
            paginate_call = ast.Await(value=paginate_call)

        body.append(ast.Return(value=paginate_call))

        return body

    def _build_paginated_iter_body(self) -> list[ast.stmt]:
        """Build the body for a paginated iterator endpoint function.

        Same scaffolding as :meth:`_build_paginated_body`; the call is built
        via :meth:`_build_pagination_call` with the ``iterate`` prefix.
        Unlike ``paginate_*``, the ``iterate_*_async`` result is an async
        iterable and must NOT be awaited — it is consumed with
        ``async for``/``yield from`` instead.
        """
        from otterapi.codegen._body_builder import BodyStatementBuilder

        builder = BodyStatementBuilder()
        if self.config.docs:
            builder.add_docstring(self.config.docs)
        builder.add_client_init()
        builder.add_statement(self._build_fetch_page_function())
        body = builder.build()

        iterate_call = self._build_pagination_call('iterate')

        if self.config.is_async:
            # async for item in iterate_..._async(...): yield item
            body.append(
                ast.AsyncFor(
                    target=ast.Name(id='item', ctx=ast.Store()),
                    iter=iterate_call,
                    body=[ast.Expr(value=ast.Yield(value=_name('item')))],
                    orelse=[],
                )
            )
        else:
            # yield from iterate_...()
            body.append(ast.Expr(value=ast.YieldFrom(value=iterate_call)))

        return body

    def _build_page_body(self) -> list[ast.stmt]:
        """Body of a ``_page`` (one page) or ``_pages`` (page iterator) function."""
        from otterapi.codegen._body_builder import BodyStatementBuilder

        builder = BodyStatementBuilder()
        if self.config.docs:
            builder.add_docstring(self.config.docs)
        builder.add_client_init()
        builder.add_statement(self._build_fetch_page_function())
        body = builder.build()

        first_page: ast.expr = self._build_pagination_call(None)
        if self.config.is_async:
            first_page = ast.Await(value=first_page)

        if self.config.page_mode == 'page':
            body.append(ast.Return(value=first_page))
            return body

        # _pages: lazily fetch the first page, then follow it.
        iter_pages = _call(
            func=_attr(first_page, 'iter_pages'),
            keywords=[ast.keyword(arg='max_pages', value=_name('max_pages'))],
        )
        if self.config.is_async:
            body.append(
                ast.AsyncFor(
                    target=ast.Name(id='page', ctx=ast.Store()),
                    iter=iter_pages,
                    body=[ast.Expr(value=ast.Yield(value=_name('page')))],
                    orelse=[],
                )
            )
        else:
            body.append(ast.Expr(value=ast.YieldFrom(value=iter_pages)))
        return body

    def _omit_page_size(self) -> bool:
        """Whether to drop the page-size (limit) query param.

        Only honored for cursor style: some cursor APIs (e.g. forward-token
        feeds) have no page-size parameter, so sending ``limit`` is noise.
        """
        return pagination_omits_page_size(
            self.config.pagination_style, self.config.pagination_config
        )

    def _request_body_for_signature(self) -> RequestBodyInfo | None:
        """The request body as a paginated function exposes it."""
        return strip_pagination_body_fields(
            self.config.request_body_info,
            self.config.pagination_style,
            self.config.pagination_config,
        )

    def _build_fetch_page_params_dict(self) -> ast.Dict:
        """The ``params`` dict: query-located paging fields plus passthrough params."""
        pag_config = self.config.pagination_config or {}
        fields = pagination_fields(self.config.pagination_style, pag_config)
        param_keys: list[ast.expr | None] = [
            ast.Constant(value=wire) for wire, _, loc in fields if loc == 'query'
        ]
        param_values: list[ast.expr] = [
            _name(arg) for _, arg, loc in fields if loc == 'query'
        ]
        skip_params = {
            pag_config.get('offset_param'),
            pag_config.get('limit_param'),
            pag_config.get('cursor_param'),
            pag_config.get('page_param'),
            pag_config.get('per_page_param'),
        }
        for param in self.config.parameters or []:
            if param.location == 'query' and param.name not in skip_params:
                param_keys.append(ast.Constant(value=param.name))
                param_values.append(_name(param.name_sanitized))

        return ast.Dict(keys=param_keys, values=param_values)

    def _build_fetch_page_body(self, params_dict: ast.Dict) -> list[ast.stmt]:
        """Build the fetch_page body: request call, plus optional response parsing."""
        request_method = '_request_async' if self.config.is_async else '_request'
        path_expr = ParameterASTBuilder.build_path_expr(
            self.config.path, self.config.parameters or []
        )

        page_keywords = [
            ast.keyword(
                arg='method', value=ast.Constant(value=self.config.method.lower())
            ),
            ast.keyword(arg='path', value=path_expr),
        ]
        if params_dict.keys:
            page_keywords.append(ast.keyword(arg='params', value=params_dict))
        param_styles = ParameterASTBuilder.build_param_styles(
            self.config.parameters or []
        )
        if param_styles:
            page_keywords.append(ast.keyword(arg='param_styles', value=param_styles))
        # Header parameters and the request body travel with every page; they
        # were previously accepted by the signature but never sent.
        header_params = ParameterASTBuilder.build_header_params(
            self.config.parameters or []
        )
        if header_params:
            page_keywords.append(ast.keyword(arg='headers', value=header_params))
        body_keyword = self._build_fetch_page_body_keyword()
        if body_keyword is not None:
            page_keywords.append(body_keyword)
        page_keywords.append(ast.keyword(arg=None, value=_name('kwargs')))

        request_call = _call(func=_attr('c', request_method), keywords=page_keywords)
        request_expr: ast.expr = request_call
        if self.config.is_async:
            request_expr = ast.Await(value=request_call)

        if not self.config.response_type:
            return [ast.Return(value=request_expr)]

        parse_method = (
            '_parse_response_async' if self.config.is_async else '_parse_response'
        )
        parse_call: ast.expr = _call(
            func=_attr('c', parse_method),
            args=[
                _name('response'),
                self.config.response_type.annotation_ast or _name('Any'),
            ],
        )
        if self.config.is_async:
            parse_call = ast.Await(value=parse_call)

        return [
            _assign(_name('response'), request_expr),
            ast.Return(value=parse_call),
        ]

    def _build_fetch_page_body_keyword(self) -> ast.keyword | None:
        """The ``json=`` / ``data=`` keyword for each page request, if any.

        Body-located paging fields are overlaid with ``merge_body_fields`` on
        their wire names: onto the dumped body, or -- for a flattened body --
        onto the field dict before ``build_request_body`` validates it, so a
        required paging field is present on the first page.
        """
        body_info = self._request_body_for_signature()
        fields = [
            (wire, arg)
            for wire, arg, loc in pagination_fields(
                self.config.pagination_style, self.config.pagination_config
            )
            if loc == 'body'
        ]
        if not fields:
            body_expr, keyword = ParameterASTBuilder.build_body_expr(body_info)
            if body_expr is None or not keyword:
                return None
            return ast.keyword(arg=keyword, value=body_expr)
        if (
            body_info is None
            or not body_info.is_json
            or body_info.type is None
            or body_info.type.type != 'model'
        ):
            raise ConfigurationError(
                f'{self.config.fn_name}: cursor_location/limit_location: body '
                'needs a JSON object request body',
                field='pagination.endpoints.cursor_location',
            )
        paging = ast.Dict(
            keys=[ast.Constant(value=wire) for wire, _ in fields],
            values=[_name(arg) for _, arg in fields],
        )
        if body_info.flattened_fields is not None:
            body_expr, keyword = ParameterASTBuilder.build_body_expr(
                body_info, merge_fields=paging
            )
        else:
            dumped, keyword = ParameterASTBuilder.build_body_expr(body_info)
            assert dumped is not None  # a JSON model body always dumps
            body_expr = _call(func=_name('merge_body_fields'), args=[dumped, paging])
        assert body_expr is not None
        return ast.keyword(arg=keyword or 'json', value=body_expr)

    def _build_fetch_page_function(self) -> ast.FunctionDef | ast.AsyncFunctionDef:
        """``def fetch_page(_start, _size)``: one page request.

        The arguments are underscore-prefixed so they can never shadow a spec
        parameter or flattened body field the request forwards.
        """
        fetch_body = self._build_fetch_page_body(self._build_fetch_page_params_dict())
        start_annotation = (
            _union_expr([_name('str'), ast.Constant(value=None)])
            if self._spec().knob_type == 'str'
            else _name('int')
        )
        fn_args = ast.arguments(
            posonlyargs=[],
            args=[
                _argument('_start', start_annotation),
                _argument('_size', _name('int')),
            ],
            kwonlyargs=[],
            kw_defaults=[],
            kwarg=None,
            defaults=[],
        )
        return _function_def(
            name='fetch_page',
            args=fn_args,
            body=fetch_body,
            is_async=self.config.is_async,
        )

    def _build_extract_lambda(self, path: str | None) -> ast.Lambda:
        """Build a lambda to extract data from a response using a path."""
        page_arg = ast.arg(arg='page', annotation=None)

        body: ast.expr
        if path:
            # Check if path looks like a simple attribute access (no dots)
            if '.' not in path:
                # lambda page: page.attr
                body = ast.Attribute(
                    value=ast.Name(id='page', ctx=ast.Load()),
                    attr=path,
                    ctx=ast.Load(),
                )
            else:
                # lambda page: extract_path(page, "path")
                body = ast.Call(
                    func=_name('extract_path'),
                    args=[
                        ast.Name(id='page', ctx=ast.Load()),
                        ast.Constant(value=path),
                    ],
                    keywords=[],
                )
        else:
            # lambda page: page
            body = ast.Name(id='page', ctx=ast.Load())

        return ast.Lambda(
            args=ast.arguments(
                posonlyargs=[],
                args=[page_arg],
                kwonlyargs=[],
                kw_defaults=[],
                defaults=[],
            ),
            body=body,
        )

    def _build_request_keywords(self) -> list[ast.keyword]:
        """Build the keywords for a request call."""
        parameters = self.config.parameters or []

        path_expr = ParameterASTBuilder.build_path_expr(self.config.path, parameters)
        query_params = ParameterASTBuilder.build_query_params(parameters)
        header_params = ParameterASTBuilder.build_header_params(parameters)
        body_expr, body_param_name = ParameterASTBuilder.build_body_expr(
            self.config.request_body_info
        )

        request_keywords: list[ast.keyword] = [
            ast.keyword(
                arg='method', value=ast.Constant(value=self.config.method.lower())
            ),
            ast.keyword(arg='path', value=path_expr),
        ]

        if query_params:
            request_keywords.append(ast.keyword(arg='params', value=query_params))

        param_styles = ParameterASTBuilder.build_param_styles(parameters)
        if param_styles:
            request_keywords.append(ast.keyword(arg='param_styles', value=param_styles))

        if header_params:
            request_keywords.append(ast.keyword(arg='headers', value=header_params))

        if body_expr and body_param_name:
            request_keywords.append(ast.keyword(arg=body_param_name, value=body_expr))

        request_keywords.append(ast.keyword(arg=None, value=_name('kwargs')))

        return request_keywords

    def _build_delegating_call_args(
        self,
    ) -> tuple[list[ast.expr], list[ast.keyword]]:
        """Build the arguments for a delegating client method call."""
        call_args: list[ast.expr] = []
        call_keywords: list[ast.keyword] = []

        builder = FunctionSignatureBuilder()
        builder.add_parameters(self.config.parameters)
        builder.add_request_body(self.config.request_body_info)

        if self.config.dataframe_library:
            builder.add_path_parameter(default=self.config.dataframe_path)

        signature = builder.build()

        for arg in signature.args:
            call_args.append(_name(arg.arg))

        for kwarg in signature.kwonlyargs:
            call_keywords.append(ast.keyword(arg=kwarg.arg, value=_name(kwarg.arg)))

        call_keywords.append(ast.keyword(arg=None, value=_name('kwargs')))

        return call_args, call_keywords


# =============================================================================
# Base Request Function Generation
# =============================================================================


def _get_base_request_arguments() -> tuple[list[ast.arg], list[ast.arg], ast.arg]:
    """Build the argument signature for base request functions.

    Returns:
        A tuple of (args, kwonlyargs, kwargs).
    """
    args = [
        _argument('method', _name('str')),
        _argument('path', _name('str')),
    ]
    kwonlyargs = [
        _argument(
            'response_model',
            _union_expr([_subscript('Type', _name('T')), ast.Constant(value=None)]),
        ),
        _argument('supported_status_codes', _subscript('list', _name('int'))),
        _argument('timeout', _union_expr([_name('float'), ast.Constant(value=None)])),
        _argument('stream', _name('bool')),
    ]
    kwargs = _argument('kwargs', _name('Any'))

    return args, kwonlyargs, kwargs


def _build_url_fstring(base_url_var: str = 'BASE_URL') -> ast.JoinedStr:
    """Build an f-string AST node for URL construction.

    Args:
        base_url_var: The variable name containing the base URL.

    Returns:
        An AST JoinedStr node representing f"{base_url_var}{path}".
    """
    return ast.JoinedStr(
        values=[
            ast.FormattedValue(value=_name(base_url_var), conversion=-1),
            ast.FormattedValue(value=_name('path'), conversion=-1),
        ]
    )


def _build_response_validation_body() -> list[ast.stmt]:
    """Build the shared response validation AST statements.

    Returns:
        List of AST statements for response validation.
    """
    stream_check = ast.If(
        test=_name('stream'),
        body=[ast.Return(value=_name('response'))],
        orelse=[],
    )

    return [
        ast.If(
            test=ast.BoolOp(
                op=ast.Or(),
                values=[
                    ast.UnaryOp(op=ast.Not(), operand=_name('supported_status_codes')),
                    ast.Compare(
                        left=_attr('response', 'status_code'),
                        ops=[ast.NotIn()],
                        comparators=[_name('supported_status_codes')],
                    ),
                ],
            ),
            body=[ast.Expr(value=_call(func=_attr('response', 'raise_for_status')))],
            orelse=[],
        ),
        stream_check,
        _assign(
            target=_name('content_type'),
            value=_call(
                func=_attr(_attr('response', 'headers'), 'get'),
                args=[ast.Constant(value='content-type'), ast.Constant(value='')],
            ),
        ),
        ast.If(
            test=ast.BoolOp(
                op=ast.Or(),
                values=[
                    ast.Compare(
                        left=ast.Constant(value=MediaType.JSON),
                        ops=[ast.In()],
                        comparators=[_name('content_type')],
                    ),
                    _call(
                        func=_attr(_name('content_type'), 'endswith'),
                        args=[ast.Constant(value='+json')],
                    ),
                ],
            ),
            body=[
                _assign(
                    target=_name('data'),
                    value=_call(func=_attr('response', 'json')),
                ),
                ast.If(
                    test=ast.UnaryOp(op=ast.Not(), operand=_name('response_model')),
                    body=[ast.Return(value=_name('data'))],
                    orelse=[],
                ),
                _assign(
                    target=_name('validated_data'),
                    value=_call(
                        func=_attr(
                            _call(
                                func=_name('TypeAdapter'),
                                args=[_name('response_model')],
                            ),
                            'validate_python',
                        ),
                        args=[_name('data')],
                    ),
                ),
                ast.If(
                    test=_call(
                        func=_name('isinstance'),
                        args=[_name('validated_data'), _name('RootModel')],
                    ),
                    body=[ast.Return(value=_attr('validated_data', 'root'))],
                    orelse=[],
                ),
                ast.Return(value=_name('validated_data')),
            ],
            orelse=[
                ast.If(
                    test=_call(
                        func=_attr(_name('content_type'), 'startswith'),
                        args=[ast.Constant(value='text/')],
                    ),
                    body=[
                        ast.Return(value=_attr('response', 'text')),
                    ],
                    orelse=[
                        ast.If(
                            test=ast.BoolOp(
                                op=ast.Or(),
                                values=[
                                    ast.Compare(
                                        left=ast.Constant(value=MediaType.OCTET_STREAM),
                                        ops=[ast.In()],
                                        comparators=[_name('content_type')],
                                    ),
                                    _call(
                                        func=_attr(_name('content_type'), 'startswith'),
                                        args=[ast.Constant(value='image/')],
                                    ),
                                    _call(
                                        func=_attr(_name('content_type'), 'startswith'),
                                        args=[ast.Constant(value='audio/')],
                                    ),
                                    _call(
                                        func=_attr(_name('content_type'), 'startswith'),
                                        args=[ast.Constant(value='video/')],
                                    ),
                                    _call(
                                        func=_attr(_name('content_type'), 'startswith'),
                                        args=[ast.Constant(value='application/pdf')],
                                    ),
                                ],
                            ),
                            body=[
                                ast.Return(value=_attr('response', 'content')),
                            ],
                            orelse=[
                                ast.Return(value=_name('response')),
                            ],
                        ),
                    ],
                ),
            ],
        ),
    ]


def _build_base_request_fn(
    is_async: bool,
    base_url_var: str = 'BASE_URL',
) -> tuple[ast.FunctionDef | ast.AsyncFunctionDef, ImportDict]:
    """Build a base request function (sync or async).

    Args:
        is_async: If True, generates an async function; otherwise sync.
        base_url_var: The variable name containing the base URL configuration.

    Returns:
        A tuple of (function_ast, imports).
    """
    args, kwonlyargs, kwargs = _get_base_request_arguments()
    url_fstring = _build_url_fstring(base_url_var)
    validation_body = _build_response_validation_body()

    request_keywords = [
        ast.keyword(arg=None, value=_name('kwargs')),
        ast.keyword(
            arg='timeout',
            value=_name('timeout'),
        ),
    ]

    if is_async:
        request_call: ast.expr = ast.Await(
            value=_call(
                func=_attr('client', 'request'),
                args=[_name('method'), url_fstring],
                keywords=request_keywords,
            )
        )
        inner_body = [
            _assign(target=_name('response'), value=request_call),
            *validation_body,
        ]
        body: list[ast.stmt] = [
            ast.AsyncWith(
                items=[
                    ast.withitem(
                        context_expr=_call(func=_name('AsyncClient')),
                        optional_vars=_name('client'),
                    )
                ],
                body=inner_body,
            )
        ]
        func_builder: Callable[..., ast.FunctionDef | ast.AsyncFunctionDef] = (
            _async_func
        )
        func_name = 'request_async'
        http_imports = {HTTP_MODULE: {'AsyncClient'}}
    else:
        request_call = _call(
            func=_name('request'),
            args=[_name('method'), url_fstring],
            keywords=request_keywords,
        )
        body = [
            _assign(target=_name('response'), value=request_call),
            *validation_body,
        ]
        func_builder = _func
        func_name = 'request_sync'
        http_imports = {HTTP_MODULE: {'request'}}

    func_ast = func_builder(
        name=func_name,
        args=args,
        body=body,
        kwargs=kwargs,
        kwonlyargs=kwonlyargs,
        kw_defaults=[
            _name('Json'),
            ast.Constant(value=None),
            ast.Constant(value=None),
            ast.Constant(value=False),
        ],
        returns=_name('T'),
    )

    imports: ImportDict = {
        **http_imports,
        'pydantic': {'TypeAdapter', 'Json', 'RootModel'},
        'typing': {'Any', 'Type', 'TypeVar'},
    }

    return func_ast, imports


def base_request_fn(
    base_url_var: str = 'BASE_URL',
) -> tuple[ast.FunctionDef, ImportDict]:
    """Generate a synchronous base request function.

    Creates a `request_sync` function that handles HTTP requests with response
    validation and model deserialization.

    Args:
        base_url_var: The variable name containing the base URL configuration.

    Returns:
        A tuple of (function_ast, imports) for the sync request function.
    """
    fn, imports = _build_base_request_fn(is_async=False, base_url_var=base_url_var)
    return cast(ast.FunctionDef, fn), imports


def base_async_request_fn(
    base_url_var: str = 'BASE_URL',
) -> tuple[ast.AsyncFunctionDef, ImportDict]:
    """Generate an asynchronous base request function.

    Creates a `request_async` function that handles HTTP requests with response
    validation and model deserialization using httpx2.AsyncClient.

    Args:
        base_url_var: The variable name containing the base URL configuration.

    Returns:
        A tuple of (function_ast, imports) for the async request function.
    """
    fn, imports = _build_base_request_fn(is_async=True, base_url_var=base_url_var)
    return cast(ast.AsyncFunctionDef, fn), imports


# =============================================================================
# Parameter and Call Building Helpers
# =============================================================================


def get_parameters(
    parameters: list[Parameter],
) -> tuple[list[ast.arg], list[ast.arg], list[ast.expr], ImportDict]:
    """Extract function arguments from OpenAPI parameters.

    Separates required and optional parameters and generates appropriate
    AST argument nodes for each.

    Args:
        parameters: List of Parameter objects from the OpenAPI spec.

    Returns:
        A tuple of (args, kwonlyargs, kw_defaults, imports).
    """
    args: list[ast.arg] = []
    kwonlyargs: list[ast.arg] = []
    kw_defaults: list[ast.expr] = []
    imports: ImportDict = {}

    for param in parameters:
        if param.type:
            annotation = param.type.annotation_ast
            imports.update(param.type.annotation_imports)
        else:
            annotation = _name('Any')
            imports.setdefault('typing', set()).add('Any')

        arg = _argument(param.name_sanitized, annotation)

        if param.required:
            args.append(arg)
        else:
            kwonlyargs.append(arg)
            kw_defaults.append(ast.Constant(value=None))

    return args, kwonlyargs, kw_defaults, imports


def get_base_call_keywords(
    query_params: ast.expr | None,
    header_params: ast.expr | None,
    response_model_ast: ast.expr | None,
    supported_status_codes: list[int] | None,
    body_expr: ast.expr | None = None,
    body_param_name: str | None = None,
    timeout: float | None = None,
    stream: bool = False,
    param_styles: ast.expr | None = None,
) -> list[ast.keyword]:
    """Build the keyword arguments for a request function call.

    Args:
        query_params: AST expression for query parameters dictionary.
        header_params: AST expression for header parameters dictionary.
        response_model_ast: AST expression for the response model type.
        supported_status_codes: List of valid HTTP status codes.
        body_expr: AST expression for the request body.
        body_param_name: The httpx2 parameter name for the body.
        timeout: Optional request timeout in seconds.
        stream: Whether to stream the response.
        param_styles: AST expression for the non-default ``style``/``explode``
            mapping, or None when every parameter uses the OpenAPI default.

    Returns:
        List of AST keyword nodes for the function call.
    """
    keywords: list[ast.keyword] = []

    keywords.append(
        ast.keyword(
            arg='response_model', value=response_model_ast or ast.Constant(value=None)
        )
    )

    if supported_status_codes:
        keywords.append(
            ast.keyword(
                arg='supported_status_codes',
                value=ast.List(
                    elts=[ast.Constant(value=code) for code in supported_status_codes]
                ),
            )
        )
    else:
        keywords.append(
            ast.keyword(arg='supported_status_codes', value=ast.Constant(value=None))
        )

    if timeout is not None:
        keywords.append(ast.keyword(arg='timeout', value=ast.Constant(value=timeout)))

    if stream:
        keywords.append(ast.keyword(arg='stream', value=ast.Constant(value=True)))

    if query_params:
        keywords.append(ast.keyword(arg='params', value=query_params))

    if param_styles:
        keywords.append(ast.keyword(arg='param_styles', value=param_styles))

    if header_params:
        keywords.append(ast.keyword(arg='headers', value=header_params))

    if body_expr and body_param_name:
        keywords.append(ast.keyword(arg=body_param_name, value=body_expr))

    return keywords


def build_header_params(
    parameters: list[Parameter],
) -> ast.Dict | None:
    """Build a dictionary AST node for header parameters.

    Args:
        parameters: List of Parameter objects to filter for headers.

    Returns:
        An AST Dict node for header parameters, or None if no header params.
    """
    return ParameterASTBuilder.build_header_params(parameters)


def build_query_params(
    parameters: list[Parameter],
) -> ast.Dict | None:
    """Build a dictionary AST node for query parameters.

    Args:
        parameters: List of Parameter objects to filter for query params.

    Returns:
        An AST Dict node for query parameters, or None if no query params.
    """
    return ParameterASTBuilder.build_query_params(parameters)


def build_param_styles(
    parameters: list[Parameter],
) -> ast.Dict | None:
    """Build the ``param_styles`` mapping for non-default query parameters.

    Args:
        parameters: List of Parameter objects to inspect.

    Returns:
        An AST Dict node, or None when every query parameter uses the
        OpenAPI default serialization.
    """
    return ParameterASTBuilder.build_param_styles(parameters)


def build_path_params(
    path: str,
    parameters: list[Parameter],
) -> ast.expr:
    """Build an f-string or constant for the request path.

    Args:
        path: The OpenAPI path string with placeholders.
        parameters: List of Parameter objects to map placeholders to variables.

    Returns:
        An AST expression (JoinedStr for f-strings, Constant for static paths).
    """
    return ParameterASTBuilder.build_path_expr(path, parameters)


def build_body_params(
    body: RequestBodyInfo | None,
) -> tuple[ast.expr | None, str | None]:
    """Build an AST expression for the request body parameter.

    Args:
        body: The RequestBodyInfo object, or None if no body.

    Returns:
        A tuple of (body_expr, httpx_param_name).
    """
    return ParameterASTBuilder.build_body_expr(body)


def prepare_call_from_parameters(
    parameters: list[Parameter] | None,
    path: str,
    request_body_info: RequestBodyInfo | None = None,
) -> tuple[ast.expr | None, ast.expr | None, ast.expr | None, str | None, ast.expr]:
    """Prepare all parameter AST nodes for a request function call.

    Args:
        parameters: List of Parameter objects, or None.
        path: The API path with optional placeholders.
        request_body_info: Request body info, or None.

    Returns:
        A tuple of (query_params, header_params, body_expr, body_param_name, path_expr).
    """
    return ParameterASTBuilder.prepare_all_params(parameters, path, request_body_info)


# =============================================================================
# Endpoint Function Building
# =============================================================================


def flattened_body_args(
    body: RequestBodyInfo,
    imports: ImportDict,
) -> tuple[list[ast.arg], list[ast.arg], list[ast.expr]]:
    """Build the parameters that replace ``body=`` for a flattened body.

    Required fields stay positional so a caller cannot forget them; everything
    else is keyword-only with a ``None`` default, matching how optional
    endpoint parameters already read.

    Registers the imports the field annotations need into *imports* in place.

    Args:
        body: The request body, whose ``flattened_fields`` drive the result.
        imports: Import dict to extend.

    Returns:
        A tuple of (args, kwonlyargs, kw_defaults).
    """
    # The model's own imports cover the field annotations (datetime, UUID,
    # referenced enums). Registering the whole set is deliberately optimistic
    # -- write_mod prunes whatever the module ends up not using.
    if body.type:
        for source in (body.type.annotation_imports, body.type.implementation_imports):
            for module, names in source.items():
                imports.setdefault(module, set()).update(names)

    args: list[ast.arg] = []
    kwonlyargs: list[ast.arg] = []
    kw_defaults: list[ast.expr] = []

    optional_fields = [f for f in (body.flattened_fields or []) if not f.required]
    if optional_fields:
        imports.setdefault('._serialization', set()).update({'NOTSET', 'NotSet'})

    for body_field in body.flattened_fields or []:
        if body_field.required:
            args.append(_argument(body_field.name, body_field.annotation_ast))
            continue
        # NOTSET rather than None: None already means "send a JSON null" for a
        # nullable field, and one default cannot carry both meanings. The
        # annotation is left otherwise untouched, so a field the model will not
        # accept as null does not advertise that it would.
        annotation = _union_expr([body_field.annotation_ast, _name('NotSet')])
        kwonlyargs.append(_argument(body_field.name, annotation))
        kw_defaults.append(_name('NOTSET'))

    return args, kwonlyargs, kw_defaults


def _build_endpoint_fn_signature(
    parameters: list[Parameter] | None,
    request_body_info: RequestBodyInfo | None,
    imports: ImportDict,
) -> tuple[list[ast.arg], list[ast.arg], list[ast.expr]]:
    """Build (args, kwonlyargs, kw_defaults) for an endpoint fn, registering imports in place."""
    if parameters:
        args, kwonlyargs, kw_defaults, param_imports = get_parameters(parameters)
        imports.update(param_imports)
    else:
        args, kwonlyargs, kw_defaults = [], [], []

    if request_body_info and request_body_info.flattened_fields is not None:
        body_args, body_kwonly, body_defaults = flattened_body_args(
            request_body_info, imports
        )
        args.extend(body_args)
        kwonlyargs.extend(body_kwonly)
        kw_defaults.extend(body_defaults)
    elif request_body_info:
        body_annotation = (
            request_body_info.type.annotation_ast
            if request_body_info.type and request_body_info.type.annotation_ast
            else _name('Any')
        )
        if request_body_info.type:
            imports.update(request_body_info.type.annotation_imports)
        else:
            imports.setdefault('typing', set()).add('Any')

        body_arg = _argument('body', body_annotation)
        if request_body_info.required:
            args.append(body_arg)
        else:
            kwonlyargs.append(body_arg)
            kw_defaults.append(ast.Constant(value=None))

    return args, kwonlyargs, kw_defaults


def _build_endpoint_request_call(
    *,
    method: str,
    path: str,
    parameters: list[Parameter] | None,
    request_body_info: RequestBodyInfo | None,
    response_model: Type | None,
    response_infos: list[ResponseInfo] | None,
    base_fn_name: str,
    is_async: bool,
    imports: ImportDict,
) -> ast.expr:
    """Build the (possibly awaited) request_sync/request_async call expr.

    Registers any response-model imports into ``imports`` in place.
    """
    query_params, header_params, body_expr, body_param_name, path_expr = (
        prepare_call_from_parameters(parameters, path, request_body_info)
    )

    if response_model and response_model.annotation_ast is not None:
        response_model_ast: ast.expr = response_model.annotation_ast
        imports.update(response_model.annotation_imports)
    else:
        response_model_ast = ast.Constant(value=None)

    supported_status_codes = None
    if response_infos:
        supported_status_codes = [
            r.status_code for r in response_infos if r.status_code
        ]

    call_keywords = get_base_call_keywords(
        query_params=query_params,
        header_params=header_params,
        response_model_ast=response_model_ast,
        supported_status_codes=supported_status_codes,
        body_expr=body_expr,
        body_param_name=body_param_name,
        stream=False,
        param_styles=ParameterASTBuilder.build_param_styles(parameters or []),
    )

    call_args = [
        ast.keyword(arg='method', value=ast.Constant(value=method.lower())),
        ast.keyword(arg='path', value=path_expr),
        *call_keywords,
        ast.keyword(arg=None, value=_name('kwargs')),
    ]

    request_call: ast.expr = _call(func=_name(base_fn_name), keywords=call_args)
    if is_async:
        request_call = ast.Await(value=request_call)
    return request_call


def _resolve_endpoint_return_type(
    response_model: Type | None, imports: ImportDict
) -> ast.expr:
    """Resolve the endpoint's return annotation, registering imports in place."""
    if response_model and response_model.annotation_ast is not None:
        return response_model.annotation_ast
    imports.setdefault(HTTP_MODULE, set()).add('Response')
    return _name('Response')


def _build_endpoint_fn(
    name: str,
    method: str,
    path: str,
    response_model: Type | None,
    is_async: bool,
    docs: str | None = None,
    parameters: list[Parameter] | None = None,
    response_infos: list[ResponseInfo] | None = None,
    request_body_info: RequestBodyInfo | None = None,
) -> tuple[ast.FunctionDef | ast.AsyncFunctionDef, ImportDict]:
    """Build an endpoint function (sync or async).

    Args:
        name: The function name.
        method: HTTP method (GET, POST, etc.).
        path: The API path with optional placeholders.
        response_model: The response Type for validation, or None.
        is_async: Whether to generate an async function.
        docs: Optional docstring.
        parameters: List of Parameter objects.
        response_infos: List of ResponseInfo objects.
        request_body_info: RequestBodyInfo object for the body, or None.

    Returns:
        A tuple of (function_ast, imports).
    """
    imports: ImportDict = {}
    func_builder = _async_func if is_async else _func
    base_fn_name = 'request_async' if is_async else 'request_sync'

    args, kwonlyargs, kw_defaults = _build_endpoint_fn_signature(
        parameters, request_body_info, imports
    )

    request_call = _build_endpoint_request_call(
        method=method,
        path=path,
        parameters=parameters,
        request_body_info=request_body_info,
        response_model=response_model,
        response_infos=response_infos,
        base_fn_name=base_fn_name,
        is_async=is_async,
        imports=imports,
    )

    body: list[ast.stmt] = []
    if docs:
        body.append(ast.Expr(value=ast.Constant(value=clean_docstring(docs))))
    body.append(ast.Return(value=request_call))

    returns = _resolve_endpoint_return_type(response_model, imports)

    # Add Any import for **kwargs type annotation
    imports.setdefault('typing', set()).add('Any')

    func_ast = func_builder(
        name=name,
        args=args,
        body=body,
        kwargs=_argument('kwargs', _name('Any')),
        kwonlyargs=kwonlyargs,
        kw_defaults=kw_defaults,
        returns=returns,
    )

    return func_ast, imports


def request_fn(
    name: str,
    method: str,
    path: str,
    response_model: Type | None,
    docs: str | None = None,
    parameters: list[Parameter] | None = None,
    response_infos: list[ResponseInfo] | None = None,
    request_body_info: RequestBodyInfo | None = None,
) -> tuple[ast.FunctionDef, ImportDict]:
    """Generate a synchronous endpoint function.

    Args:
        name: The function name to generate.
        method: HTTP method (GET, POST, etc.).
        path: The API path with optional placeholders.
        response_model: The response type for validation, or None.
        docs: Optional docstring for the generated function.
        parameters: List of Parameter objects for the endpoint.
        response_infos: List of ResponseInfo objects describing response content types.
        request_body_info: RequestBodyInfo object for the request body, or None.

    Returns:
        A tuple of (function_ast, imports) for the sync endpoint function.
    """
    fn, imports = _build_endpoint_fn(
        name=name,
        method=method,
        path=path,
        response_model=response_model,
        is_async=False,
        docs=docs,
        parameters=parameters,
        response_infos=response_infos,
        request_body_info=request_body_info,
    )
    return cast(ast.FunctionDef, fn), imports


def async_request_fn(
    name: str,
    method: str,
    path: str,
    response_model: Type | None,
    docs: str | None = None,
    parameters: list[Parameter] | None = None,
    response_infos: list[ResponseInfo] | None = None,
    request_body_info: RequestBodyInfo | None = None,
) -> tuple[ast.AsyncFunctionDef, ImportDict]:
    """Generate an asynchronous endpoint function.

    Args:
        name: The function name to generate.
        method: HTTP method (GET, POST, etc.).
        path: The API path with optional placeholders.
        response_model: The response type for validation, or None.
        docs: Optional docstring for the generated function.
        parameters: List of Parameter objects for the endpoint.
        response_infos: List of ResponseInfo objects describing response content types.
        request_body_info: RequestBodyInfo object for the request body, or None.

    Returns:
        A tuple of (function_ast, imports) for the async endpoint function.
    """
    fn, imports = _build_endpoint_fn(
        name=name,
        method=method,
        path=path,
        response_model=response_model,
        is_async=True,
        docs=docs,
        parameters=parameters,
        response_infos=response_infos,
        request_body_info=request_body_info,
    )
    return cast(ast.AsyncFunctionDef, fn), imports


# =============================================================================
# Default Client Code Generation
# =============================================================================


def build_default_client_code() -> tuple[list[ast.stmt], ImportDict]:
    """Build any module-level client code needed.

    Previously this created a global singleton pattern. Now it returns
    an empty list since each endpoint creates its own Client instance
    when none is provided.

    Returns:
        A tuple of (statements, imports) - both empty.
    """
    imports: ImportDict = {}

    # No global state needed - endpoints use `client or Client()`
    return [], imports


# =============================================================================
# Convenience Functions (Factory Wrappers)
# =============================================================================


def build_standalone_endpoint_fn(
    fn_name: str,
    method: str,
    path: str,
    parameters: list[Parameter] | None,
    request_body_info: RequestBodyInfo | None,
    response_type: Type | None,
    response_infos: list[ResponseInfo] | None = None,
    docs: str | None = None,
    is_async: bool = False,
    unwrap_data_path: str | None = None,
    unwrap_type_ast: ast.expr | None = None,
    unwrap_type_imports: dict[str, set[str]] | None = None,
    response_type_imports: dict[str, set[str]] | None = None,
    with_response: bool = False,
) -> tuple[ast.FunctionDef | ast.AsyncFunctionDef, ImportDict]:
    """Build a standalone endpoint function with full implementation.

    This is a convenience wrapper around EndpointFunctionFactory.

    Args:
        fn_name: The function name to generate.
        method: HTTP method (GET, POST, etc.).
        path: The API path with optional placeholders.
        parameters: List of Parameter objects for the endpoint.
        request_body_info: RequestBodyInfo object for the request body, or None.
        response_type: The response Type for the return annotation.
        response_infos: List of ResponseInfo objects for status code handling.
        docs: Optional docstring for the generated function.
        is_async: Whether to generate an async function.
        unwrap_data_path: If set, extract response.{path} and return it.
        unwrap_type_ast: The AST for the unwrapped return type.
        unwrap_type_imports: Imports needed for the unwrapped return type.
        response_type_imports: Imports needed for model names in response type.
        with_response: Build the ``_with_response`` variant, returning a
            ``RawResponse`` whose ``parse()`` gives the usual result.

    Returns:
        A tuple of (function_ast, imports).
    """
    config = EndpointFunctionConfig(
        fn_name=fn_name,
        method=method,
        path=path,
        parameters=parameters,
        request_body_info=request_body_info,
        response_type=response_type,
        response_infos=response_infos,
        docs=docs,
        is_async=is_async,
        mode=EndpointMode.STANDALONE,
        unwrap_data_path=unwrap_data_path,
        unwrap_type_ast=unwrap_type_ast,
        unwrap_type_imports=unwrap_type_imports,
        response_type_imports=response_type_imports,
        with_response=with_response,
    )
    return EndpointFunctionFactory(config).build()


def endpoint_supports_with_response(response_type: Type | None) -> bool:
    """Whether an endpoint gets a ``_with_response`` variant.

    Only endpoints whose body is parsed: a raw ``Response`` / ``bytes`` /
    ``str`` endpoint (or one with no response body) already hands back the
    response, or has nothing to parse.
    """
    if response_type is None or response_type.annotation_ast is None:
        return False
    factory = EndpointFunctionFactory(
        EndpointFunctionConfig(fn_name='_', method='get', path='/')
    )
    return not factory._is_raw_response_type(response_type)


def build_delegating_endpoint_fn(
    fn_name: str,
    client_method_name: str,
    parameters: list[Parameter] | None,
    request_body_info: RequestBodyInfo | None,
    response_type: Type | None,
    docs: str | None = None,
    is_async: bool = False,
) -> tuple[ast.FunctionDef | ast.AsyncFunctionDef, ImportDict]:
    """Build an endpoint function that delegates to a client method.

    This is a convenience wrapper around EndpointFunctionFactory.

    Args:
        fn_name: The function name to generate.
        client_method_name: The client method to delegate to.
        parameters: List of Parameter objects for the endpoint.
        request_body_info: RequestBodyInfo object for the request body, or None.
        response_type: The response Type for the return annotation.
        docs: Optional docstring for the generated function.
        is_async: Whether to generate an async function.

    Returns:
        A tuple of (function_ast, imports).
    """
    config = EndpointFunctionConfig(
        fn_name=fn_name,
        method='',
        path='',
        parameters=parameters,
        request_body_info=request_body_info,
        response_type=response_type,
        docs=docs,
        is_async=is_async,
        mode=EndpointMode.DELEGATING,
        client_method_name=client_method_name,
    )
    return EndpointFunctionFactory(config).build()


def build_standalone_dataframe_fn(
    fn_name: str,
    method: str,
    path: str,
    parameters: list[Parameter] | None,
    request_body_info: RequestBodyInfo | None,
    library: Literal['pandas', 'polars'],
    default_path: str | None = None,
    docs: str | None = None,
    is_async: bool = False,
) -> tuple[ast.FunctionDef | ast.AsyncFunctionDef, ImportDict]:
    """Build a standalone DataFrame endpoint function.

    This is a convenience wrapper around EndpointFunctionFactory.

    Args:
        fn_name: The function name to generate.
        method: HTTP method (GET, POST, etc.).
        path: The API path with optional placeholders.
        parameters: List of Parameter objects for the endpoint.
        request_body_info: RequestBodyInfo object for the request body, or None.
        library: The DataFrame library ('pandas' or 'polars').
        default_path: Default path value for the path parameter.
        docs: Optional docstring for the generated function.
        is_async: Whether to generate an async function.

    Returns:
        A tuple of (function_ast, imports).
    """
    df_library = (
        DataFrameLibrary.PANDAS if library == 'pandas' else DataFrameLibrary.POLARS
    )
    config = EndpointFunctionConfig(
        fn_name=fn_name,
        method=method,
        path=path,
        parameters=parameters,
        request_body_info=request_body_info,
        response_type=None,
        docs=docs,
        is_async=is_async,
        mode=EndpointMode.STANDALONE,
        dataframe_library=df_library,
        dataframe_path=default_path,
    )
    return EndpointFunctionFactory(config).build()


def build_standalone_paginated_dataframe_fn(
    fn_name: str,
    method: str,
    path: str,
    parameters: list[Parameter] | None,
    request_body_info: RequestBodyInfo | None,
    response_type: Type | None,
    pagination_style: str,
    pagination_config: dict,
    library: Literal['pandas', 'polars'],
    item_type_ast: ast.expr | None = None,
    item_type_imports: dict[str, set[str]] | None = None,
    docs: str | None = None,
    is_async: bool = False,
) -> tuple[ast.FunctionDef | ast.AsyncFunctionDef, ImportDict]:
    """Build a standalone paginated DataFrame endpoint function.

    This function fetches all paginated items and returns them as a DataFrame.

    Args:
        fn_name: The function name to generate.
        method: HTTP method (GET, POST, etc.).
        path: The API path with optional placeholders.
        parameters: List of Parameter objects for the endpoint.
        request_body_info: RequestBodyInfo object for the request body, or None.
        response_type: The response Type for parsing.
        pagination_style: The pagination style ('offset', 'cursor', 'page').
        pagination_config: Dict with pagination parameter names and paths.
        library: The DataFrame library ('pandas' or 'polars').
        item_type_ast: AST for the item type in the list.
        item_type_imports: Imports needed for the item type.
        docs: Optional docstring for the generated function.
        is_async: Whether to generate an async function.

    Returns:
        A tuple of (function_ast, imports).
    """
    pag_style = PaginationStyle(pagination_style)
    df_library = (
        DataFrameLibrary.PANDAS if library == 'pandas' else DataFrameLibrary.POLARS
    )
    config = EndpointFunctionConfig(
        fn_name=fn_name,
        method=method,
        path=path,
        parameters=parameters,
        request_body_info=request_body_info,
        response_type=response_type,
        docs=docs,
        is_async=is_async,
        mode=EndpointMode.STANDALONE,
        pagination_style=pag_style,
        pagination_config=pagination_config,
        is_paginated_dataframe=True,
        dataframe_library=df_library,
        item_type_ast=item_type_ast,
        item_type_imports=item_type_imports,
    )
    return EndpointFunctionFactory(config).build()


def build_delegating_dataframe_fn(
    fn_name: str,
    client_method_name: str,
    parameters: list[Parameter] | None,
    request_body_info: RequestBodyInfo | None,
    library: Literal['pandas', 'polars'],
    default_path: str | None = None,
    docs: str | None = None,
    is_async: bool = False,
) -> tuple[ast.FunctionDef | ast.AsyncFunctionDef, ImportDict]:
    """Build a DataFrame endpoint function that delegates to a client method.

    This is a convenience wrapper around EndpointFunctionFactory.

    Args:
        fn_name: The function name to generate.
        client_method_name: The client method to delegate to.
        parameters: List of Parameter objects for the endpoint.
        request_body_info: RequestBodyInfo object for the request body, or None.
        library: The DataFrame library ('pandas' or 'polars').
        default_path: Default path value for the path parameter.
        docs: Optional docstring for the generated function.
        is_async: Whether to generate an async function.

    Returns:
        A tuple of (function_ast, imports).
    """
    df_library = (
        DataFrameLibrary.PANDAS if library == 'pandas' else DataFrameLibrary.POLARS
    )
    config = EndpointFunctionConfig(
        fn_name=fn_name,
        method='',
        path='',
        parameters=parameters,
        request_body_info=request_body_info,
        response_type=None,
        docs=docs,
        is_async=is_async,
        mode=EndpointMode.DELEGATING,
        client_method_name=client_method_name,
        dataframe_library=df_library,
        dataframe_path=default_path,
    )
    return EndpointFunctionFactory(config).build()


def build_standalone_paginated_fn(
    fn_name: str,
    method: str,
    path: str,
    parameters: list[Parameter] | None,
    request_body_info: RequestBodyInfo | None,
    response_type: Type | None,
    pagination_style: str,
    pagination_config: dict,
    item_type_ast: ast.expr | None = None,
    item_type_imports: dict[str, set[str]] | None = None,
    docs: str | None = None,
    is_async: bool = False,
) -> tuple[ast.FunctionDef | ast.AsyncFunctionDef, ImportDict]:
    """Build a standalone paginated endpoint function.

    This function returns all items by automatically handling pagination.

    Args:
        fn_name: The function name to generate.
        method: HTTP method (GET, POST, etc.).
        path: The API path with optional placeholders.
        parameters: List of Parameter objects for the endpoint.
        request_body_info: RequestBodyInfo object for the request body, or None.
        response_type: The response Type for parsing.
        pagination_style: The pagination style ('offset', 'cursor', 'page').
        pagination_config: Dict with pagination parameter names and paths.
        item_type_ast: AST for the item type in the list.
        item_type_imports: Imports needed for the item type.
        docs: Optional docstring for the generated function.
        is_async: Whether to generate an async function.

    Returns:
        A tuple of (function_ast, imports).
    """
    pag_style = PaginationStyle(pagination_style)
    config = EndpointFunctionConfig(
        fn_name=fn_name,
        method=method,
        path=path,
        parameters=parameters,
        request_body_info=request_body_info,
        response_type=response_type,
        docs=docs,
        is_async=is_async,
        mode=EndpointMode.STANDALONE,
        pagination_style=pag_style,
        pagination_config=pagination_config,
        item_type_ast=item_type_ast,
        item_type_imports=item_type_imports,
    )
    return EndpointFunctionFactory(config).build()


def build_standalone_page_fn(
    fn_name: str,
    method: str,
    path: str,
    parameters: list[Parameter] | None,
    request_body_info: RequestBodyInfo | None,
    response_type: Type | None,
    pagination_style: str,
    pagination_config: dict,
    page_mode: Literal['page', 'pages'],
    item_type_ast: ast.expr | None = None,
    item_type_imports: dict[str, set[str]] | None = None,
    docs: str | None = None,
    is_async: bool = False,
) -> tuple[ast.FunctionDef | ast.AsyncFunctionDef, ImportDict]:
    """Build a ``_page`` (one ``Page``) or ``_pages`` (page iterator) function.

    Page objects keep what the item-level variants discard: the total, where
    the next page starts and the validated response envelope.
    """
    config = EndpointFunctionConfig(
        fn_name=fn_name,
        method=method,
        path=path,
        parameters=parameters,
        request_body_info=request_body_info,
        response_type=response_type,
        docs=docs,
        is_async=is_async,
        mode=EndpointMode.STANDALONE,
        pagination_style=PaginationStyle(pagination_style),
        pagination_config=pagination_config,
        page_mode=page_mode,
        item_type_ast=item_type_ast,
        item_type_imports=item_type_imports,
    )
    return EndpointFunctionFactory(config).build()


def build_standalone_paginated_iter_fn(
    fn_name: str,
    method: str,
    path: str,
    parameters: list[Parameter] | None,
    request_body_info: RequestBodyInfo | None,
    response_type: Type | None,
    pagination_style: str,
    pagination_config: dict,
    item_type_ast: ast.expr | None = None,
    item_type_imports: dict[str, set[str]] | None = None,
    docs: str | None = None,
    is_async: bool = False,
) -> tuple[ast.FunctionDef | ast.AsyncFunctionDef, ImportDict]:
    """Build a standalone paginated iterator endpoint function.

    This function yields items one at a time for memory-efficient streaming.

    Args:
        fn_name: The function name to generate.
        method: HTTP method (GET, POST, etc.).
        path: The API path with optional placeholders.
        parameters: List of Parameter objects for the endpoint.
        request_body_info: RequestBodyInfo object for the request body, or None.
        response_type: The response Type for parsing.
        pagination_style: The pagination style ('offset', 'cursor', 'page').
        pagination_config: Dict with pagination parameter names and paths.
        item_type_ast: AST for the item type in the list.
        item_type_imports: Imports needed for the item type.
        docs: Optional docstring for the generated function.
        is_async: Whether to generate an async function.

    Returns:
        A tuple of (function_ast, imports).
    """
    pag_style = PaginationStyle(pagination_style)
    config = EndpointFunctionConfig(
        fn_name=fn_name,
        method=method,
        path=path,
        parameters=parameters,
        request_body_info=request_body_info,
        response_type=response_type,
        docs=docs,
        is_async=is_async,
        mode=EndpointMode.STANDALONE,
        pagination_style=pag_style,
        pagination_config=pagination_config,
        is_iterator=True,
        item_type_ast=item_type_ast,
        item_type_imports=item_type_imports,
    )
    return EndpointFunctionFactory(config).build()
