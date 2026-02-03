"""
Endpoint generation module for creating sync and async HTTP request functions.

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
"""

import ast
import textwrap

from otterapi.codegen.ast_utils import (
    _argument,
    _assign,
    _async_func,
    _attr,
    _call,
    _func,
    _name,
    _subscript,
    _union_expr,
)
from otterapi.codegen.types import Parameter, RequestBodyInfo, ResponseInfo, Type

# Type alias for import dictionaries used throughout this module
ImportDict = dict[str, set[str]]


def clean_docstring(docstring: str) -> str:
    """Clean and normalize a docstring by removing excess indentation.

    Args:
        docstring: The raw docstring to clean.

    Returns:
        A cleaned docstring with normalized indentation.
    """
    return textwrap.dedent(f'\n{docstring}\n').strip()


def _get_base_request_arguments() -> tuple[list[ast.arg], list[ast.arg], ast.arg]:
    """Build the argument signature for base request functions.

    Returns:
        A tuple of (args, kwonlyargs, kwargs) representing:
        - args: Positional arguments (method, path)
        - kwonlyargs: Keyword-only arguments (response_model, supported_status_codes, timeout, stream)
        - kwargs: The **kwargs argument
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
    kwargs = _argument('kwargs', _name('dict'))

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


def _build_response_validation_body(is_async: bool = False) -> list[ast.stmt]:
    """Build the shared response validation AST statements.

    This creates the common logic for:
    - Status code checking and raise_for_status
    - Streaming response handling
    - Content-type based response handling (JSON, text, binary, raw)
    - Response model validation with TypeAdapter
    - RootModel unwrapping

    Args:
        is_async: Whether this is for async functions (affects streaming).

    Returns:
        List of AST statements for response validation.
    """
    # Build the stream handling first
    # if stream:
    #     return response
    stream_check = ast.If(
        test=_name('stream'),
        body=[ast.Return(value=_name('response'))],
        orelse=[],
    )

    return [
        # if not supported_status_codes or response.status_code not in supported_status_codes:
        #     response.raise_for_status()
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
        # if stream: return response
        stream_check,
        # content_type = response.headers.get('content-type', '')
        _assign(
            target=_name('content_type'),
            value=_call(
                func=_attr(_attr('response', 'headers'), 'get'),
                args=[ast.Constant(value='content-type'), ast.Constant(value='')],
            ),
        ),
        # Handle different content types
        # if 'application/json' in content_type or content_type.endswith('+json'):
        ast.If(
            test=ast.BoolOp(
                op=ast.Or(),
                values=[
                    ast.Compare(
                        left=ast.Constant(value='application/json'),
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
                # data = response.json()
                _assign(
                    target=_name('data'),
                    value=_call(func=_attr('response', 'json')),
                ),
                # if not response_model: return data
                ast.If(
                    test=ast.UnaryOp(op=ast.Not(), operand=_name('response_model')),
                    body=[ast.Return(value=_name('data'))],
                    orelse=[],
                ),
                # validated_data = TypeAdapter(response_model).validate_python(data)
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
                # if isinstance(validated_data, RootModel): return validated_data.root
                ast.If(
                    test=_call(
                        func=_name('isinstance'),
                        args=[_name('validated_data'), _name('RootModel')],
                    ),
                    body=[ast.Return(value=_attr('validated_data', 'root'))],
                    orelse=[],
                ),
                # return validated_data
                ast.Return(value=_name('validated_data')),
            ],
            orelse=[
                # elif content_type.startswith('text/'):
                ast.If(
                    test=_call(
                        func=_attr(_name('content_type'), 'startswith'),
                        args=[ast.Constant(value='text/')],
                    ),
                    body=[
                        # return response.text
                        ast.Return(value=_attr('response', 'text')),
                    ],
                    orelse=[
                        # elif binary content types
                        ast.If(
                            test=ast.BoolOp(
                                op=ast.Or(),
                                values=[
                                    ast.Compare(
                                        left=ast.Constant(
                                            value='application/octet-stream'
                                        ),
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
                                # return response.content
                                ast.Return(value=_attr('response', 'content')),
                            ],
                            orelse=[
                                # else: return response (raw httpx.Response)
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

    This generates either `request_sync` or `request_async` function that handles
    the core HTTP request logic including URL construction, response validation,
    and model deserialization.

    Args:
        is_async: If True, generates an async function; otherwise sync.
        base_url_var: The variable name containing the base URL configuration.

    Returns:
        A tuple of (function_ast, imports) where:
        - function_ast: The generated FunctionDef or AsyncFunctionDef
        - imports: Dictionary of required imports
    """
    args, kwonlyargs, kwargs = _get_base_request_arguments()
    url_fstring = _build_url_fstring(base_url_var)
    validation_body = _build_response_validation_body(is_async)

    # Build request keywords including timeout
    request_keywords = [
        ast.keyword(arg=None, value=_name('kwargs')),
        # timeout=timeout if timeout else None
        ast.keyword(
            arg='timeout',
            value=_name('timeout'),
        ),
    ]

    if is_async:
        # Async version: uses AsyncClient context manager
        request_call = ast.Await(
            value=_call(
                func=_attr('client', 'request'),
                args=[_name('method'), ast.Expr(value=url_fstring)],
                keywords=request_keywords,
            )
        )
        inner_body = [
            _assign(target=_name('response'), value=request_call),
            *validation_body,
        ]
        body = [
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
        func_builder = _async_func
        func_name = 'request_async'
        http_imports = {'httpx': {'AsyncClient'}}
    else:
        # Sync version: direct request call
        request_call = _call(
            func=_name('request'),
            args=[_name('method'), ast.Expr(value=url_fstring)],
            keywords=request_keywords,
        )
        body = [
            _assign(target=_name('response'), value=request_call),
            *validation_body,
        ]
        func_builder = _func
        func_name = 'request_sync'
        http_imports = {'httpx': {'request'}}

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
        'typing': {'Type', 'TypeVar', 'Union'},
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
    return _build_base_request_fn(is_async=False, base_url_var=base_url_var)


def base_async_request_fn(
    base_url_var: str = 'BASE_URL',
) -> tuple[ast.AsyncFunctionDef, ImportDict]:
    """Generate an asynchronous base request function.

    Creates a `request_async` function that handles HTTP requests with response
    validation and model deserialization using httpx.AsyncClient.

    Args:
        base_url_var: The variable name containing the base URL configuration.

    Returns:
        A tuple of (function_ast, imports) for the async request function.
    """
    return _build_base_request_fn(is_async=True, base_url_var=base_url_var)


def get_parameters(
    parameters: list[Parameter],
) -> tuple[list[ast.arg], list[ast.arg], list[ast.expr], ImportDict]:
    """Extract function arguments from OpenAPI parameters.

    Separates required and optional parameters and generates appropriate
    AST argument nodes for each.

    Args:
        parameters: List of Parameter objects from the OpenAPI spec.

    Returns:
        A tuple of (args, kwonlyargs, kw_defaults, imports) where:
        - args: Required positional arguments
        - kwonlyargs: Optional keyword-only arguments
        - kw_defaults: Default values for optional arguments
        - imports: Required imports for type annotations
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
) -> list[ast.keyword]:
    """Build the keyword arguments for a request function call.

    Args:
        query_params: AST expression for query parameters dictionary.
        header_params: AST expression for header parameters dictionary.
        response_model_ast: AST expression for the response model type.
        supported_status_codes: List of valid HTTP status codes.
        body_expr: AST expression for the request body.
        body_param_name: The httpx parameter name for the body ('json', 'data', 'files', 'content').
        timeout: Optional request timeout in seconds.
        stream: Whether to stream the response.

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

    # Add timeout parameter
    if timeout is not None:
        keywords.append(ast.keyword(arg='timeout', value=ast.Constant(value=timeout)))

    # Add stream parameter
    if stream:
        keywords.append(ast.keyword(arg='stream', value=ast.Constant(value=True)))

    if query_params:
        keywords.append(ast.keyword(arg='params', value=query_params))

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
    header_params = [p for p in parameters if p.location == 'header']
    if not header_params:
        return None

    return ast.Dict(
        keys=[ast.Constant(value=param.name) for param in header_params],
        values=[_name(param.name_sanitized) for param in header_params],
    )


def build_query_params(
    parameters: list[Parameter],
) -> ast.Dict | None:
    """Build a dictionary AST node for query parameters.

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


def build_path_params(
    path: str,
    parameters: list[Parameter],
) -> ast.expr:
    """Build an f-string or constant for the request path.

    Replaces OpenAPI path parameters like {petId} with Python f-string
    expressions using the sanitized parameter names.

    Args:
        path: The OpenAPI path string with placeholders.
        parameters: List of Parameter objects to map placeholders to variables.

    Returns:
        An AST expression (JoinedStr for f-strings, Constant for static paths).
    """
    path_params = {p.name: p.name_sanitized for p in parameters if p.location == 'path'}

    # Check if there are any path parameters
    if not path_params:
        return ast.Constant(value=path)

    # Build an f-string with interpolated path parameters
    import re

    pattern = r'\{([^}]+)\}'
    parts = re.split(pattern, path)
    values: list[ast.expr] = []
    current_pos = 0

    for i, part in enumerate(parts):
        if i % 2 == 0:
            # Static part
            if part:
                values.append(ast.Constant(value=part))
        else:
            # Parameter placeholder
            sanitized = path_params.get(part, part)
            values.append(ast.FormattedValue(value=_name(sanitized), conversion=-1))

    if len(values) == 1 and isinstance(values[0], ast.Constant):
        return values[0]

    return ast.JoinedStr(values=values)


def build_body_params(
    body: RequestBodyInfo | None,
) -> tuple[ast.expr | None, str | None]:
    """Build an AST expression for the request body parameter.

    Handles different content types:
    - JSON: Uses model_dump() for Pydantic models
    - Form data: Returns data dict directly
    - Multipart: Returns files dict
    - Binary: Returns content directly

    Args:
        body: The RequestBodyInfo object, or None if no body.

    Returns:
        A tuple of (body_expr, httpx_param_name) where:
        - body_expr: AST expression for the body value, or None
        - httpx_param_name: The httpx keyword to use ('json', 'data', 'files', 'content')
    """
    if not body:
        return None, None

    body_name = 'body'

    # For JSON content with model types, use model_dump()
    if body.is_json and body.type and body.type.type in ('model', 'root'):
        body_expr = _call(
            func=_attr(_name(body_name), 'model_dump'),
            args=[],
        )
    elif body.is_multipart:
        # For multipart, the body should be a dict of file tuples
        # The user passes files as {'field': (filename, content, content_type)}
        body_expr = _name(body_name)
    elif body.is_form:
        # For form data, pass as dict
        if body.type and body.type.type in ('model', 'root'):
            body_expr = _call(
                func=_attr(_name(body_name), 'model_dump'),
                args=[],
            )
        else:
            body_expr = _name(body_name)
    else:
        # For other content types or primitive types, use the value directly
        body_expr = _name(body_name)

    return body_expr, body.httpx_param_name


def prepare_call_from_parameters(
    parameters: list[Parameter] | None,
    path: str,
    request_body_info: RequestBodyInfo | None = None,
) -> tuple[ast.expr | None, ast.expr | None, ast.expr | None, str | None, ast.expr]:
    """Prepare all parameter AST nodes for a request function call.

    Separates parameters by location (query, path, header) and builds
    the appropriate AST nodes for each.

    Args:
        parameters: List of Parameter objects, or None.
        path: The API path with optional placeholders.
        request_body_info: Request body info, or None.

    Returns:
        A tuple of (query_params, header_params, body_expr, body_param_name, path_expr).
    """
    if not parameters:
        parameters = []

    query_params = build_query_params(parameters)
    header_params = build_header_params(parameters)
    body_expr, body_param_name = build_body_params(request_body_info)
    path_expr = build_path_params(path, parameters)

    return query_params, header_params, body_expr, body_param_name, path_expr


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

    Creates a function that makes an HTTP request to the specified endpoint
    with proper parameter handling and response validation.

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

    # Build function arguments from parameters
    if parameters:
        args, kwonlyargs, kw_defaults, param_imports = get_parameters(parameters)
        imports.update(param_imports)
    else:
        args, kwonlyargs, kw_defaults = [], [], []

    # Add body parameter if present
    if request_body_info:
        body_annotation = (
            request_body_info.type.annotation_ast
            if request_body_info.type
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

    # Prepare parameters for the call
    query_params, header_params, body_expr, body_param_name, path_expr = (
        prepare_call_from_parameters(parameters, path, request_body_info)
    )

    # Build response model AST
    if response_model:
        response_model_ast = response_model.annotation_ast
        imports.update(response_model.annotation_imports)
    else:
        response_model_ast = ast.Constant(value=None)

    # Get supported status codes
    supported_status_codes = None
    if response_infos:
        supported_status_codes = [
            r.status_code for r in response_infos if r.status_code
        ]

    # Check if this is a binary response (might want to stream)
    is_binary_response = response_infos and any(r.is_binary for r in response_infos)

    # Build the request call keywords
    call_keywords = get_base_call_keywords(
        query_params=query_params,
        header_params=header_params,
        response_model_ast=response_model_ast,
        supported_status_codes=supported_status_codes,
        body_expr=body_expr,
        body_param_name=body_param_name,
        stream=False,  # Default, users can override via **kwargs
    )

    # Build the call to base request function
    call_args = [
        ast.keyword(arg='method', value=ast.Constant(value=method.lower())),
        ast.keyword(arg='path', value=path_expr),
        *call_keywords,
        # Pass through **kwargs for timeout, stream, and other options
        ast.keyword(arg=None, value=_name('kwargs')),
    ]

    request_call = _call(
        func=_name(base_fn_name),
        keywords=call_args,
    )

    if is_async:
        request_call = ast.Await(value=request_call)

    # Build function body
    body: list[ast.stmt] = []

    if docs:
        body.append(ast.Expr(value=ast.Constant(value=clean_docstring(docs))))

    body.append(ast.Return(value=request_call))

    # Build return type annotation
    if response_model:
        returns = response_model.annotation_ast
    else:
        returns = _name('Any')
        imports.setdefault('typing', set()).add('Any')

    func_ast = func_builder(
        name=name,
        args=args,
        body=body,
        kwargs=_argument('kwargs', _name('dict')),
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

    Creates a function that calls `request_sync` with the appropriate parameters
    for the specified endpoint.

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
    return _build_endpoint_fn(
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

    Creates an async function that calls `request_async` with the appropriate
    parameters for the specified endpoint.

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
    return _build_endpoint_fn(
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
