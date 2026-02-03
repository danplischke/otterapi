"""Client class generation module for OtterAPI.

This module provides utilities for generating a client class that wraps
all API endpoints with configurable base URL, timeout, headers, and
HTTP client injection support.
"""

import ast
from dataclasses import dataclass
from typing import TYPE_CHECKING

from otterapi.codegen.ast_utils import (
    _argument,
    _assign,
    _attr,
    _call,
    _name,
    _subscript,
    _union_expr,
)

if TYPE_CHECKING:
    from otterapi.codegen.types import Parameter, RequestBodyInfo, ResponseInfo, Type

# Type alias for import dictionaries
ImportDict = dict[str, set[str]]


@dataclass
class EndpointInfo:
    """Information about an endpoint for client method generation."""

    name: str
    async_name: str
    method: str
    path: str
    parameters: list['Parameter'] | None
    request_body: 'RequestBodyInfo | None'
    response_type: 'Type | None'
    response_infos: list['ResponseInfo'] | None
    description: str | None


def generate_base_client_class(
    class_name: str,
    endpoints: list[EndpointInfo],
    default_base_url: str,
    default_timeout: float = 30.0,
) -> tuple[ast.ClassDef, ImportDict]:
    """Generate a BaseClient class with all endpoints as methods.

    Args:
        class_name: Name for the generated class (e.g., 'BasePetStoreClient').
        endpoints: List of EndpointInfo objects to include as methods.
        default_base_url: Default base URL from the OpenAPI spec.
        default_timeout: Default request timeout in seconds.

    Returns:
        Tuple of (class AST node, required imports).
    """
    imports: ImportDict = {
        'httpx': {'Client', 'AsyncClient', 'Response'},
        'typing': {'Any', 'Union'},
        'pydantic': {'TypeAdapter', 'RootModel'},
    }

    # Build __init__ method
    init_method = _build_init_method(default_base_url, default_timeout)

    # Build _request method (sync)
    request_method = _build_request_method(is_async=False)

    # Build _request_async method (async)
    async_request_method = _build_request_method(is_async=True)

    # Build endpoint methods
    endpoint_methods: list[ast.stmt] = []
    for endpoint in endpoints:
        # Generate sync method
        sync_method, sync_imports = _build_endpoint_method(
            endpoint=endpoint,
            is_async=False,
        )
        endpoint_methods.append(sync_method)
        _merge_imports(imports, sync_imports)

        # Generate async method
        async_method, async_imports = _build_endpoint_method(
            endpoint=endpoint,
            is_async=True,
        )
        endpoint_methods.append(async_method)
        _merge_imports(imports, async_imports)

    # Build class body
    class_body: list[ast.stmt] = [
        ast.Expr(
            value=ast.Constant(
                value=f"""Base client for API requests.

This class is regenerated on each code generation run.
To customize, subclass this in client.py.

Args:
    base_url: Base URL for API requests. Default: {default_base_url}
    timeout: Request timeout in seconds. Default: {default_timeout}
    headers: Default headers to include in all requests.
    http_client: Custom httpx.Client for sync requests.
    async_http_client: Custom httpx.AsyncClient for async requests.
"""
            )
        ),
        init_method,
        request_method,
        async_request_method,
        *endpoint_methods,
    ]

    class_def = ast.ClassDef(
        name=class_name,
        bases=[],
        keywords=[],
        body=class_body,
        decorator_list=[],
    )

    return class_def, imports


def _build_init_method(
    default_base_url: str, default_timeout: float
) -> ast.FunctionDef:
    """Build the __init__ method for the client class."""
    init_body: list[ast.stmt] = [
        # self.base_url = base_url.rstrip('/')
        _assign(
            _attr('self', 'base_url'),
            _call(_attr(_name('base_url'), 'rstrip'), [ast.Constant(value='/')]),
        ),
        # self.timeout = timeout
        _assign(
            _attr('self', 'timeout'),
            _name('timeout'),
        ),
        # self.headers = headers or {}
        _assign(
            _attr('self', 'headers'),
            ast.BoolOp(
                op=ast.Or(),
                values=[_name('headers'), ast.Dict(keys=[], values=[])],
            ),
        ),
        # self._client = http_client
        _assign(
            _attr('self', '_client'),
            _name('http_client'),
        ),
        # self._async_client = async_http_client
        _assign(
            _attr('self', '_async_client'),
            _name('async_http_client'),
        ),
    ]

    init_method = ast.FunctionDef(
        name='__init__',
        args=ast.arguments(
            posonlyargs=[],
            args=[
                _argument('self'),
                _argument('base_url', _name('str')),
                _argument('timeout', _name('float')),
                _argument(
                    'headers',
                    _union_expr(
                        [
                            _subscript(
                                'dict', ast.Tuple(elts=[_name('str'), _name('str')])
                            ),
                            ast.Constant(value=None),
                        ]
                    ),
                ),
                _argument(
                    'http_client',
                    _union_expr([_name('Client'), ast.Constant(value=None)]),
                ),
                _argument(
                    'async_http_client',
                    _union_expr([_name('AsyncClient'), ast.Constant(value=None)]),
                ),
            ],
            kwonlyargs=[],
            kw_defaults=[],
            kwarg=None,
            defaults=[
                ast.Constant(value=default_base_url),
                ast.Constant(value=default_timeout),
                ast.Constant(value=None),
                ast.Constant(value=None),
                ast.Constant(value=None),
            ],
        ),
        body=init_body,
        decorator_list=[],
        returns=ast.Constant(value=None),
    )

    return init_method


def _build_request_method(is_async: bool) -> ast.FunctionDef | ast.AsyncFunctionDef:
    """Build the internal _request or _request_async method."""
    method_name = '_request_async' if is_async else '_request'

    args = ast.arguments(
        posonlyargs=[],
        args=[
            _argument('self'),
            _argument('method', _name('str')),
            _argument('path', _name('str')),
        ],
        kwonlyargs=[
            _argument('params', _union_expr([_name('dict'), ast.Constant(value=None)])),
            _argument(
                'headers', _union_expr([_name('dict'), ast.Constant(value=None)])
            ),
            _argument('json', _union_expr([_name('Any'), ast.Constant(value=None)])),
            _argument('data', _union_expr([_name('Any'), ast.Constant(value=None)])),
            _argument('files', _union_expr([_name('Any'), ast.Constant(value=None)])),
            _argument('content', _union_expr([_name('Any'), ast.Constant(value=None)])),
            _argument(
                'timeout', _union_expr([_name('float'), ast.Constant(value=None)])
            ),
        ],
        kw_defaults=[
            ast.Constant(value=None),  # params
            ast.Constant(value=None),  # headers
            ast.Constant(value=None),  # json
            ast.Constant(value=None),  # data
            ast.Constant(value=None),  # files
            ast.Constant(value=None),  # content
            ast.Constant(value=None),  # timeout
        ],
        kwarg=None,
        defaults=[],
    )

    # Build URL: f"{self.base_url}{path}"
    url_expr = ast.JoinedStr(
        values=[
            ast.FormattedValue(value=_attr('self', 'base_url'), conversion=-1),
            ast.FormattedValue(value=_name('path'), conversion=-1),
        ]
    )

    # merged_headers = {**self.headers, **(headers or {})}
    merged_headers = ast.Dict(
        keys=[None, None],
        values=[
            _attr('self', 'headers'),
            ast.BoolOp(
                op=ast.Or(),
                values=[_name('headers'), ast.Dict(keys=[], values=[])],
            ),
        ],
    )

    # actual_timeout = timeout if timeout is not None else self.timeout
    timeout_expr = ast.IfExp(
        test=ast.Compare(
            left=_name('timeout'),
            ops=[ast.IsNot()],
            comparators=[ast.Constant(value=None)],
        ),
        body=_name('timeout'),
        orelse=_attr('self', 'timeout'),
    )

    if is_async:
        body = _build_async_request_body(url_expr, merged_headers, timeout_expr)
        return ast.AsyncFunctionDef(
            name=method_name,
            args=args,
            body=body,
            decorator_list=[],
            returns=_name('Response'),
        )
    else:
        body = _build_sync_request_body(url_expr, merged_headers, timeout_expr)
        return ast.FunctionDef(
            name=method_name,
            args=args,
            body=body,
            decorator_list=[],
            returns=_name('Response'),
        )


def _build_sync_request_body(
    url_expr: ast.expr, merged_headers: ast.expr, timeout_expr: ast.expr
) -> list[ast.stmt]:
    """Build the body for sync _request method."""
    request_call = _call(
        _attr('client', 'request'),
        args=[_name('method'), url_expr],
        keywords=[
            ast.keyword(arg='params', value=_name('params')),
            ast.keyword(arg='headers', value=merged_headers),
            ast.keyword(arg='json', value=_name('json')),
            ast.keyword(arg='data', value=_name('data')),
            ast.keyword(arg='files', value=_name('files')),
            ast.keyword(arg='content', value=_name('content')),
            ast.keyword(arg='timeout', value=timeout_expr),
        ],
    )

    return [
        ast.If(
            test=_attr('self', '_client'),
            body=[
                _assign(
                    _name('response'),
                    _call(
                        _attr(_attr('self', '_client'), 'request'),
                        args=[_name('method'), url_expr],
                        keywords=[
                            ast.keyword(arg='params', value=_name('params')),
                            ast.keyword(arg='headers', value=merged_headers),
                            ast.keyword(arg='json', value=_name('json')),
                            ast.keyword(arg='data', value=_name('data')),
                            ast.keyword(arg='files', value=_name('files')),
                            ast.keyword(arg='content', value=_name('content')),
                            ast.keyword(arg='timeout', value=timeout_expr),
                        ],
                    ),
                ),
            ],
            orelse=[
                ast.With(
                    items=[
                        ast.withitem(
                            context_expr=_call(_name('Client')),
                            optional_vars=_name('client'),
                        )
                    ],
                    body=[
                        _assign(_name('response'), request_call),
                    ],
                ),
            ],
        ),
        ast.Expr(value=_call(_attr('response', 'raise_for_status'))),
        ast.Return(value=_name('response')),
    ]


def _build_async_request_body(
    url_expr: ast.expr, merged_headers: ast.expr, timeout_expr: ast.expr
) -> list[ast.stmt]:
    """Build the body for async _request_async method."""
    request_call = ast.Await(
        value=_call(
            _attr('client', 'request'),
            args=[_name('method'), url_expr],
            keywords=[
                ast.keyword(arg='params', value=_name('params')),
                ast.keyword(arg='headers', value=merged_headers),
                ast.keyword(arg='json', value=_name('json')),
                ast.keyword(arg='data', value=_name('data')),
                ast.keyword(arg='files', value=_name('files')),
                ast.keyword(arg='content', value=_name('content')),
                ast.keyword(arg='timeout', value=timeout_expr),
            ],
        )
    )

    return [
        ast.If(
            test=_attr('self', '_async_client'),
            body=[
                _assign(
                    _name('response'),
                    ast.Await(
                        value=_call(
                            _attr(_attr('self', '_async_client'), 'request'),
                            args=[_name('method'), url_expr],
                            keywords=[
                                ast.keyword(arg='params', value=_name('params')),
                                ast.keyword(arg='headers', value=merged_headers),
                                ast.keyword(arg='json', value=_name('json')),
                                ast.keyword(arg='data', value=_name('data')),
                                ast.keyword(arg='files', value=_name('files')),
                                ast.keyword(arg='content', value=_name('content')),
                                ast.keyword(arg='timeout', value=timeout_expr),
                            ],
                        )
                    ),
                ),
            ],
            orelse=[
                ast.AsyncWith(
                    items=[
                        ast.withitem(
                            context_expr=_call(_name('AsyncClient')),
                            optional_vars=_name('client'),
                        )
                    ],
                    body=[
                        _assign(_name('response'), request_call),
                    ],
                ),
            ],
        ),
        ast.Expr(value=_call(_attr('response', 'raise_for_status'))),
        ast.Return(value=_name('response')),
    ]


def _build_endpoint_method(
    endpoint: EndpointInfo,
    is_async: bool,
) -> tuple[ast.FunctionDef | ast.AsyncFunctionDef, ImportDict]:
    """Build an endpoint method for the client class."""
    imports: ImportDict = {}
    fn_name = endpoint.async_name if is_async else endpoint.name
    request_method = '_request_async' if is_async else '_request'

    # Build function arguments
    args_list = [_argument('self')]
    kwonlyargs = []
    kw_defaults = []

    # Add parameters as arguments
    if endpoint.parameters:
        for param in endpoint.parameters:
            if param.type:
                annotation = param.type.annotation_ast
                _merge_imports(imports, param.type.annotation_imports)
            else:
                annotation = _name('Any')
                imports.setdefault('typing', set()).add('Any')

            arg = _argument(param.name_sanitized, annotation)

            if param.required:
                args_list.append(arg)
            else:
                kwonlyargs.append(arg)
                kw_defaults.append(ast.Constant(value=None))

    # Add body parameter if present
    if endpoint.request_body:
        if endpoint.request_body.type:
            body_annotation = endpoint.request_body.type.annotation_ast
            _merge_imports(imports, endpoint.request_body.type.annotation_imports)
        else:
            body_annotation = _name('Any')
            imports.setdefault('typing', set()).add('Any')

        body_arg = _argument('body', body_annotation)
        if endpoint.request_body.required:
            args_list.append(body_arg)
        else:
            kwonlyargs.append(body_arg)
            kw_defaults.append(ast.Constant(value=None))

    # Build arguments for _request call
    request_keywords = [
        ast.keyword(arg='method', value=ast.Constant(value=endpoint.method.lower())),
        ast.keyword(
            arg='path', value=_build_path_expr(endpoint.path, endpoint.parameters)
        ),
    ]

    # Add query params
    query_params = _build_query_params(endpoint.parameters)
    if query_params:
        request_keywords.append(ast.keyword(arg='params', value=query_params))

    # Add header params
    header_params = _build_header_params(endpoint.parameters)
    if header_params:
        request_keywords.append(ast.keyword(arg='headers', value=header_params))

    # Add body
    if endpoint.request_body:
        body_expr, param_name = _build_body_expr(endpoint.request_body)
        if body_expr and param_name:
            request_keywords.append(ast.keyword(arg=param_name, value=body_expr))

    # Build the method body
    body: list[ast.stmt] = []

    # Add docstring
    if endpoint.description:
        body.append(ast.Expr(value=ast.Constant(value=endpoint.description)))

    # Call _request
    request_call = _call(
        _attr('self', request_method),
        keywords=request_keywords,
    )

    if is_async:
        request_call = ast.Await(value=request_call)

    # Handle response
    if endpoint.response_type:
        _merge_imports(imports, endpoint.response_type.annotation_imports)
        # response = self._request(...)
        body.append(_assign(_name('response'), request_call))
        # Process JSON response
        body.extend(
            _build_response_processing(endpoint.response_type, endpoint.response_infos)
        )
    else:
        # return self._request(...)
        body.append(ast.Return(value=request_call))

    # Build return type
    if endpoint.response_type:
        returns = endpoint.response_type.annotation_ast
    else:
        returns = _name('Response')
        imports.setdefault('httpx', set()).add('Response')

    args = ast.arguments(
        posonlyargs=[],
        args=args_list,
        kwonlyargs=kwonlyargs,
        kw_defaults=kw_defaults,
        kwarg=None,
        defaults=[],
    )

    if is_async:
        method = ast.AsyncFunctionDef(
            name=fn_name,
            args=args,
            body=body,
            decorator_list=[],
            returns=returns,
        )
    else:
        method = ast.FunctionDef(
            name=fn_name,
            args=args,
            body=body,
            decorator_list=[],
            returns=returns,
        )

    return method, imports


def _build_path_expr(path: str, parameters: list['Parameter'] | None) -> ast.expr:
    """Build the path expression with parameter substitution."""
    if not parameters:
        return ast.Constant(value=path)

    path_params = {p.name: p.name_sanitized for p in parameters if p.location == 'path'}
    if not path_params:
        return ast.Constant(value=path)

    # Build f-string
    import re

    pattern = r'\{([^}]+)\}'
    parts = re.split(pattern, path)
    values: list[ast.expr] = []

    for i, part in enumerate(parts):
        if i % 2 == 0:
            if part:
                values.append(ast.Constant(value=part))
        else:
            sanitized = path_params.get(part, part)
            values.append(ast.FormattedValue(value=_name(sanitized), conversion=-1))

    if len(values) == 1 and isinstance(values[0], ast.Constant):
        return values[0]

    return ast.JoinedStr(values=values)


def _build_query_params(parameters: list['Parameter'] | None) -> ast.Dict | None:
    """Build query parameters dict."""
    if not parameters:
        return None

    query_params = [p for p in parameters if p.location == 'query']
    if not query_params:
        return None

    return ast.Dict(
        keys=[ast.Constant(value=p.name) for p in query_params],
        values=[_name(p.name_sanitized) for p in query_params],
    )


def _build_header_params(parameters: list['Parameter'] | None) -> ast.Dict | None:
    """Build header parameters dict."""
    if not parameters:
        return None

    header_params = [p for p in parameters if p.location == 'header']
    if not header_params:
        return None

    return ast.Dict(
        keys=[ast.Constant(value=p.name) for p in header_params],
        values=[_name(p.name_sanitized) for p in header_params],
    )


def _build_body_expr(
    request_body: 'RequestBodyInfo',
) -> tuple[ast.expr | None, str | None]:
    """Build the body expression for the request."""
    if not request_body:
        return None, None

    body_name = 'body'

    if (
        request_body.is_json
        and request_body.type
        and request_body.type.type in ('model', 'root')
    ):
        body_expr = _call(_attr(_name(body_name), 'model_dump'))
    else:
        body_expr = _name(body_name)

    return body_expr, request_body.httpx_param_name


def _build_response_processing(
    response_type: 'Type', response_infos: list['ResponseInfo'] | None
) -> list[ast.stmt]:
    """Build statements for processing the response."""
    stmts = []

    # data = response.json()
    stmts.append(_assign(_name('data'), _call(_attr('response', 'json'))))

    # validated = TypeAdapter(response_type).validate_python(data)
    stmts.append(
        _assign(
            _name('validated'),
            _call(
                _attr(
                    _call(_name('TypeAdapter'), [response_type.annotation_ast]),
                    'validate_python',
                ),
                [_name('data')],
            ),
        )
    )

    # if isinstance(validated, RootModel): return validated.root
    stmts.append(
        ast.If(
            test=_call(_name('isinstance'), [_name('validated'), _name('RootModel')]),
            body=[ast.Return(value=_attr('validated', 'root'))],
            orelse=[],
        )
    )

    # return validated
    stmts.append(ast.Return(value=_name('validated')))

    return stmts


def _merge_imports(target: ImportDict, source: ImportDict) -> None:
    """Merge source imports into target imports."""
    for module, names in source.items():
        if module not in target:
            target[module] = set()
        target[module].update(names)


def generate_client_stub(
    class_name: str,
    base_class_name: str,
    module_name: str = '_client',
) -> str:
    """Generate the user-customizable client.py stub content.

    Args:
        class_name: Name for the client class (e.g., 'PetStoreClient').
        base_class_name: Name of the base class to inherit from.
        module_name: Module name where base class is defined.

    Returns:
        String content for client.py file.
    """
    return f'''"""API Client.

This file is generated once and will NOT be overwritten on regeneration.
You can safely customize this file to add authentication, logging,
error handling, or other client-specific functionality.
"""

from .{module_name} import {base_class_name}


class {class_name}({base_class_name}):
    """API client with customizable configuration.

    This class inherits from the generated {base_class_name} and can be
    customized without being overwritten on code regeneration.

    Example:
        >>> client = {class_name}()
        >>> # Use default base URL from OpenAPI spec

        >>> client = {class_name}(base_url="https://staging.api.example.com")
        >>> # Override base URL

        >>> client = {class_name}(timeout=60.0, headers={{"Authorization": "Bearer token"}})
        >>> # Custom timeout and headers

        >>> import httpx
        >>> with httpx.Client() as http_client:
        ...     client = {class_name}(http_client=http_client)
        ...     # Use custom HTTP client (useful for testing/mocking)
    """

    pass

    # Add custom methods or override base class methods below.
    #
    # Example - adding authentication:
    #
    # def __init__(self, api_key: str | None = None, **kwargs):
    #     super().__init__(**kwargs)
    #     if api_key:
    #         self.headers["Authorization"] = f"Bearer {{api_key}}"
    #
    # Example - overriding a method:
    #
    # def get_pet_by_id(self, pet_id: int, **kwargs):
    #     \"\"\"Get pet with custom error handling.\"\"\"
    #     try:
    #         return super().get_pet_by_id(pet_id, **kwargs)
    #     except httpx.HTTPStatusError as e:
    #         if e.response.status_code == 404:
    #             return None
    #         raise


# Convenience alias for shorter imports
Client = {class_name}

__all__ = ["{class_name}", "Client"]
'''
