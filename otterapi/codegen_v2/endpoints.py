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
"""

import ast
import textwrap
from typing import Literal

from otterapi.codegen_v2.ast_utils import (
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
from otterapi.codegen_v2.types import Parameter, Type, ResponseInfo, RequestBodyInfo

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
        - kwonlyargs: Keyword-only arguments (response_model, supported_status_codes)
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


def _build_response_validation_body() -> list[ast.stmt]:
    """Build the shared response validation AST statements.

    This creates the common logic for:
    - Status code checking and raise_for_status
    - JSON parsing
    - Response model validation with TypeAdapter
    - RootModel unwrapping

    Returns:
        List of AST statements for response validation.
    """
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
            body=[
                ast.Expr(
                    value=_call(func=_attr('response', 'raise_for_status'))
                )
            ],
            orelse=[],
        ),
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
                    _call(func=_name('TypeAdapter'), args=[_name('response_model')]),
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
    validation_body = _build_response_validation_body()

    if is_async:
        # Async version: uses AsyncClient context manager
        request_call = ast.Await(
            value=_call(
                func=_attr('client', 'request'),
                args=[_name('method'), ast.Expr(value=url_fstring)],
                keywords=[ast.keyword(arg=None, value=_name('kwargs'))],
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
            keywords=[ast.keyword(arg=None, value=_name('kwargs'))],
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
        kw_defaults=[_name('Json'), ast.Constant(value=None)],
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

    Separates required and optional parameters, generating appropriate AST argument
    nodes with type annotations.

    Args:
        parameters: List of Parameter objects from the OpenAPI spec.

    Returns:
        A tuple of (args, kwonlyargs, kw_defaults, imports) where:
        - args: Required positional arguments
        - kwonlyargs: Optional keyword-only arguments
        - kw_defaults: Default values for keyword-only arguments (all None)
        - imports: Required imports for type annotations
    """
    args = []
    kwonlyargs = []
    kw_defaults = []
    imports: ImportDict = {}

    for param in parameters:
        param_name = param.name_sanitized
        param_type = param.type.annotation_ast if param.type else None
        param_required = param.required

        if param_type is None:
            param_type = _name('Any')
            imports.setdefault('typing', set()).add('Any')

        if param_required:
            # Required parameters go in regular args
            args.append(_argument(param_name, param_type))
        else:
            # Optional parameters go in kwonlyargs with None default
            kwonlyargs.append(_argument(param_name, param_type))
            kw_defaults.append(ast.Constant(None))

    return args, kwonlyargs, kw_defaults, imports


def get_base_call_keywords(
    method: str,
    path: ast.expr,
    response_model: Type | None,
    supported_status_codes: list[int] | None = None,
) -> list[ast.keyword]:
    """Build the base keyword arguments for request function calls.

    Args:
        method: HTTP method (GET, POST, etc.).
        path: AST expression for the request path.
        response_model: The response type for validation, or None.
        supported_status_codes: List of expected status codes, or None.

    Returns:
        List of AST keyword nodes for the function call.
    """
    return [
        ast.keyword(arg='method', value=ast.Constant(value=method)),
        ast.keyword(arg='path', value=path),
        ast.keyword(
            arg='response_model',
            value=response_model.annotation_ast
            if response_model
            else ast.Constant(None),
        ),
        ast.keyword(
            arg='supported_status_codes', value=ast.Constant(supported_status_codes)
        ),
    ]


def build_header_params(headers: list[Parameter]) -> ast.Dict | None:
    """Build a dictionary AST for header parameters.

    Args:
        headers: List of header Parameter objects.

    Returns:
        An AST Dict node mapping header names to values, or None if empty.
    """
    if not headers:
        return None

    return ast.Dict(
        keys=[ast.Constant(value=header.name) for header in headers],
        values=[_name(header.name_sanitized) for header in headers],
    )


def build_query_params(queries: list[Parameter]) -> ast.Dict | None:
    """Build a dictionary AST for query parameters.

    Args:
        queries: List of query Parameter objects.

    Returns:
        An AST Dict node mapping query param names to values, or None if empty.
    """
    if not queries:
        return None

    return ast.Dict(
        keys=[ast.Constant(value=query.name) for query in queries],
        values=[_name(query.name_sanitized) for query in queries],
    )


def build_path_params(
    paths: list[Parameter], path: str
) -> ast.JoinedStr | ast.Constant:
    """Build an f-string AST that interpolates path parameters.

    Args:
        paths: List of path Parameter objects.
        path: The original path string with placeholders like {id}.

    Returns:
        An AST JoinedStr (f-string) with interpolated parameters,
        or a Constant if no path parameters exist.
    """
    if not paths:
        return ast.Constant(value=path)

    # Split the path into parts and build the f-string
    values = []
    current_pos = 0

    for path_param in paths:
        param_placeholder = f'{{{path_param.name}}}'
        param_pos = path.find(param_placeholder, current_pos)

        if param_pos != -1:
            # Add any literal text before the parameter
            if param_pos > current_pos:
                literal_text = path[current_pos:param_pos]
                values.append(ast.Constant(value=literal_text))

            # Add the formatted value for the parameter
            values.append(
                ast.FormattedValue(
                    value=_name(path_param.name_sanitized),
                    conversion=-1,  # No conversion (default)
                )
            )

            current_pos = param_pos + len(param_placeholder)

    # Add any remaining literal text after the last parameter
    if current_pos < len(path):
        remaining_text = path[current_pos:]
        values.append(ast.Constant(value=remaining_text))

    return ast.JoinedStr(values=values)


def build_body_params(body: RequestBodyInfo | None) -> tuple[ast.expr | None, str | None]:
    """Build an AST expression for the request body parameter.

    For JSON bodies with Pydantic models, generates a `.model_dump()` call.
    For other content types, returns the parameter name directly.

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
        parameters: List of all Parameter objects for the endpoint.
        path: The original path string with placeholders.
        request_body_info: RequestBodyInfo object for the request body, or None.

    Returns:
        A tuple of (query_params, header_params, body_params, body_param_name, processed_path).
    """
    if not parameters:
        parameters = []

    query_params = [p for p in parameters if p.location == 'query']
    path_params = [p for p in parameters if p.location == 'path']
    header_params = [p for p in parameters if p.location == 'header']

    body_expr, body_param_name = build_body_params(request_body_info)

    return (
        build_query_params(query_params),
        build_header_params(header_params),
        body_expr,
        body_param_name,
        build_path_params(path_params, path),
    )


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

    This is the core function that generates endpoint-specific request functions.
    It handles parameter processing, docstring generation, and proper return types.

    Args:
        name: The function name to generate.
        method: HTTP method (GET, POST, etc.).
        path: The API path with optional placeholders.
        response_model: The response type for validation, or None.
        is_async: If True, generates an async function; otherwise sync.
        docs: Optional docstring for the generated function.
        parameters: List of Parameter objects for the endpoint.
        response_infos: List of ResponseInfo objects describing response content types.
        request_body_info: RequestBodyInfo object for the request body, or None.

    Returns:
        A tuple of (function_ast, imports) for the endpoint function.
    """
    # Get parameters and add body parameter if present
    all_params = list(parameters or [])
    if request_body_info:
        # Add body as a parameter
        body_param = Parameter(
            name='body',
            name_sanitized='body',
            location='body',
            required=request_body_info.required,
            type=request_body_info.type,
            description=request_body_info.description,
        )
        all_params.append(body_param)

    args, kwonlyargs, kw_defaults, imports = get_parameters(all_params)

    query_params, header_params, body_expr, body_param_name, processed_path = (
        prepare_call_from_parameters(parameters, path, request_body_info)
    )

    # Extract supported status codes from response_infos
    supported_status_codes = [r.status_code for r in (response_infos or [])] or None

    call_keywords = get_base_call_keywords(
        method, processed_path, response_model, supported_status_codes
    )

    if query_params:
        call_keywords.append(ast.keyword(arg='params', value=query_params))
    if header_params:
        call_keywords.append(ast.keyword(arg='headers', value=header_params))
    if body_expr and body_param_name:
        call_keywords.append(ast.keyword(arg=body_param_name, value=body_expr))

    # Add **kwargs to the call
    call_keywords.append(ast.keyword(arg=None, value=_name('kwargs')))

    # Build the return statement (with or without await)
    if is_async:
        base_func_name = 'request_async'
        return_value = ast.Await(
            value=_call(
                func=_name(base_func_name),
                args=[],
                keywords=call_keywords,
            )
        )
        func_builder = _async_func
        http_imports: ImportDict = {'httpx': {'AsyncClient'}}
    else:
        base_func_name = 'request_sync'
        return_value = _call(
            func=_name(base_func_name),
            args=[],
            keywords=call_keywords,
        )
        func_builder = _func
        http_imports = {'httpx': {'request'}}

    body: list[ast.stmt] = [ast.Return(value=return_value)]

    # Add docstring if provided
    if docs:
        cleaned_docs = textwrap.dedent(f'\n{docs}\n').strip()
        body.insert(0, ast.Expr(value=ast.Constant(value=cleaned_docs)))

    # Determine return type annotation
    response_model_ast = response_model.annotation_ast if response_model else None
    if not response_model_ast:
        response_model_ast = _name('Any')
        imports.setdefault('typing', set()).add('Any')

    func_ast = func_builder(
        name=name,
        args=args,
        body=body,
        kwargs=_argument('kwargs', _name('dict')),
        kwonlyargs=kwonlyargs,
        kw_defaults=kw_defaults,
        returns=response_model_ast,
    )

    # Merge imports
    merged_imports: ImportDict = {**http_imports}
    for module, names in imports.items():
        if module in merged_imports:
            merged_imports[module].update(names)
        else:
            merged_imports[module] = names

    return func_ast, merged_imports


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
