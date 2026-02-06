"""Client class generation module for OtterAPI.

This module provides utilities for generating a client class that wraps
all API endpoints with configurable base URL, timeout, headers, and
HTTP client injection support.
"""

import ast
from dataclasses import dataclass, field
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

# Re-export DataFrameMethodConfig from dataframes for backward compatibility
from otterapi.codegen.dataframes import DataFrameMethodConfig
from otterapi.codegen.endpoints import ParameterASTBuilder

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
    dataframe_config: DataFrameMethodConfig = field(
        default_factory=DataFrameMethodConfig
    )


def generate_api_error_class() -> ast.ClassDef:
    """Generate the APIError exception class for detailed error handling.

    Returns:
        AST ClassDef for the APIError class.
    """
    # Build __init__ method
    init_body = [
        # self.status_code = status_code
        _assign(_attr('self', 'status_code'), _name('status_code')),
        # self.response = response
        _assign(_attr('self', 'response'), _name('response')),
        # self.detail = detail
        _assign(_attr('self', 'detail'), _name('detail')),
        # self.body = body
        _assign(_attr('self', 'body'), _name('body')),
        # super().__init__(message)
        ast.Expr(
            value=_call(
                _attr(_call(_name('super')), '__init__'),
                args=[_name('message')],
            )
        ),
    ]

    init_method = ast.FunctionDef(
        name='__init__',
        args=ast.arguments(
            posonlyargs=[],
            args=[
                _argument('self'),
                _argument('message', _name('str')),
            ],
            kwonlyargs=[
                _argument('status_code', _name('int')),
                _argument('response', _name('Response')),
                _argument(
                    'detail', _union_expr([_name('Any'), ast.Constant(value=None)])
                ),
                _argument('body', _name('str')),
            ],
            kw_defaults=[
                None,  # status_code - required, no default
                None,  # response - required, no default
                ast.Constant(value=None),  # detail - default None
                ast.Constant(value=''),  # body - default ''
            ],
            kwarg=None,
            defaults=[],
        ),
        body=init_body,
        decorator_list=[],
        returns=ast.Constant(value=None),
    )

    # Build from_response classmethod
    from_response_body = [
        # status_code = response.status_code
        _assign(_name('status_code'), _attr('response', 'status_code')),
        # body = response.text
        _assign(_name('body'), _attr('response', 'text')),
        # detail = None
        _assign(_name('detail'), ast.Constant(value=None)),
        # try: ... except: ...
        ast.Try(
            body=[
                # json_body = response.json()
                _assign(_name('json_body'), _call(_attr('response', 'json'))),
                # if isinstance(json_body, dict):
                ast.If(
                    test=_call(
                        _name('isinstance'), args=[_name('json_body'), _name('dict')]
                    ),
                    body=[
                        # detail = json_body.get('detail', json_body)
                        _assign(
                            _name('detail'),
                            _call(
                                _attr('json_body', 'get'),
                                args=[ast.Constant(value='detail'), _name('json_body')],
                            ),
                        ),
                    ],
                    orelse=[
                        # detail = json_body
                        _assign(_name('detail'), _name('json_body')),
                    ],
                ),
            ],
            handlers=[
                ast.ExceptHandler(
                    type=_name('Exception'),
                    name=None,
                    body=[
                        # detail = body if body else None
                        _assign(
                            _name('detail'),
                            ast.IfExp(
                                test=_name('body'),
                                body=_name('body'),
                                orelse=ast.Constant(value=None),
                            ),
                        ),
                    ],
                ),
            ],
            orelse=[],
            finalbody=[],
        ),
        # message = f'HTTP {status_code} Error'
        _assign(
            _name('message'),
            ast.JoinedStr(
                values=[
                    ast.Constant(value='HTTP '),
                    ast.FormattedValue(value=_name('status_code'), conversion=-1),
                    ast.Constant(value=' Error'),
                ]
            ),
        ),
        # if detail:
        ast.If(
            test=_name('detail'),
            body=[
                # if isinstance(detail, list):
                ast.If(
                    test=_call(
                        _name('isinstance'), args=[_name('detail'), _name('list')]
                    ),
                    body=[
                        # error_msgs = []
                        _assign(_name('error_msgs'), ast.List(elts=[], ctx=ast.Load())),
                        # for err in detail:
                        ast.For(
                            target=ast.Name(id='err', ctx=ast.Store()),
                            iter=_name('detail'),
                            body=[
                                ast.If(
                                    test=_call(
                                        _name('isinstance'),
                                        args=[_name('err'), _name('dict')],
                                    ),
                                    body=[
                                        _assign(
                                            _name('loc'),
                                            _call(
                                                _attr('err', 'get'),
                                                args=[
                                                    ast.Constant(value='loc'),
                                                    ast.List(elts=[], ctx=ast.Load()),
                                                ],
                                            ),
                                        ),
                                        _assign(
                                            _name('msg'),
                                            _call(
                                                _attr('err', 'get'),
                                                args=[
                                                    ast.Constant(value='msg'),
                                                    _call(
                                                        _name('str'),
                                                        args=[_name('err')],
                                                    ),
                                                ],
                                            ),
                                        ),
                                        _assign(
                                            _name('loc_str'),
                                            ast.IfExp(
                                                test=_name('loc'),
                                                body=_call(
                                                    _attr(
                                                        ast.Constant(value=' -> '),
                                                        'join',
                                                    ),
                                                    args=[
                                                        ast.GeneratorExp(
                                                            elt=_call(
                                                                _name('str'),
                                                                args=[_name('x')],
                                                            ),
                                                            generators=[
                                                                ast.comprehension(
                                                                    target=ast.Name(
                                                                        id='x',
                                                                        ctx=ast.Store(),
                                                                    ),
                                                                    iter=_name('loc'),
                                                                    ifs=[],
                                                                    is_async=0,
                                                                )
                                                            ],
                                                        )
                                                    ],
                                                ),
                                                orelse=ast.Constant(value='unknown'),
                                            ),
                                        ),
                                        ast.Expr(
                                            value=_call(
                                                _attr('error_msgs', 'append'),
                                                args=[
                                                    ast.JoinedStr(
                                                        values=[
                                                            ast.Constant(value='  - '),
                                                            ast.FormattedValue(
                                                                value=_name('loc_str'),
                                                                conversion=-1,
                                                            ),
                                                            ast.Constant(value=': '),
                                                            ast.FormattedValue(
                                                                value=_name('msg'),
                                                                conversion=-1,
                                                            ),
                                                        ]
                                                    )
                                                ],
                                            )
                                        ),
                                    ],
                                    orelse=[
                                        ast.Expr(
                                            value=_call(
                                                _attr('error_msgs', 'append'),
                                                args=[
                                                    ast.JoinedStr(
                                                        values=[
                                                            ast.Constant(value='  - '),
                                                            ast.FormattedValue(
                                                                value=_name('err'),
                                                                conversion=-1,
                                                            ),
                                                        ]
                                                    )
                                                ],
                                            )
                                        ),
                                    ],
                                ),
                            ],
                            orelse=[],
                        ),
                        # if error_msgs:
                        ast.If(
                            test=_name('error_msgs'),
                            body=[
                                _assign(
                                    _name('message'),
                                    ast.BinOp(
                                        left=ast.JoinedStr(
                                            values=[
                                                ast.Constant(value='HTTP '),
                                                ast.FormattedValue(
                                                    value=_name('status_code'),
                                                    conversion=-1,
                                                ),
                                                ast.Constant(
                                                    value=' Validation Error:\n'
                                                ),
                                            ]
                                        ),
                                        op=ast.Add(),
                                        right=_call(
                                            _attr(ast.Constant(value='\n'), 'join'),
                                            args=[_name('error_msgs')],
                                        ),
                                    ),
                                ),
                            ],
                            orelse=[],
                        ),
                    ],
                    orelse=[
                        # elif isinstance(detail, str):
                        ast.If(
                            test=_call(
                                _name('isinstance'),
                                args=[_name('detail'), _name('str')],
                            ),
                            body=[
                                _assign(
                                    _name('message'),
                                    ast.JoinedStr(
                                        values=[
                                            ast.Constant(value='HTTP '),
                                            ast.FormattedValue(
                                                value=_name('status_code'),
                                                conversion=-1,
                                            ),
                                            ast.Constant(value=' Error: '),
                                            ast.FormattedValue(
                                                value=_name('detail'), conversion=-1
                                            ),
                                        ]
                                    ),
                                ),
                            ],
                            orelse=[
                                _assign(
                                    _name('message'),
                                    ast.JoinedStr(
                                        values=[
                                            ast.Constant(value='HTTP '),
                                            ast.FormattedValue(
                                                value=_name('status_code'),
                                                conversion=-1,
                                            ),
                                            ast.Constant(value=' Error: '),
                                            ast.FormattedValue(
                                                value=_name('detail'), conversion=-1
                                            ),
                                        ]
                                    ),
                                ),
                            ],
                        ),
                    ],
                ),
            ],
            orelse=[],
        ),
        # return cls(message, status_code=status_code, response=response, detail=detail, body=body)
        ast.Return(
            value=_call(
                _name('cls'),
                args=[_name('message')],
                keywords=[
                    ast.keyword(arg='status_code', value=_name('status_code')),
                    ast.keyword(arg='response', value=_name('response')),
                    ast.keyword(arg='detail', value=_name('detail')),
                    ast.keyword(arg='body', value=_name('body')),
                ],
            )
        ),
    ]

    from_response_method = ast.FunctionDef(
        name='from_response',
        args=ast.arguments(
            posonlyargs=[],
            args=[
                _argument('cls'),
                _argument('response', _name('Response')),
            ],
            kwonlyargs=[],
            kw_defaults=[],
            kwarg=None,
            defaults=[],
        ),
        body=from_response_body,
        decorator_list=[_name('classmethod')],
        returns=ast.Constant(value='APIError'),
    )

    # Build __str__ method
    str_method = ast.FunctionDef(
        name='__str__',
        args=ast.arguments(
            posonlyargs=[],
            args=[_argument('self')],
            kwonlyargs=[],
            kw_defaults=[],
            kwarg=None,
            defaults=[],
        ),
        body=[
            ast.Return(
                value=ast.Subscript(
                    value=_attr('self', 'args'),
                    slice=ast.Constant(value=0),
                    ctx=ast.Load(),
                )
            ),
        ],
        decorator_list=[],
        returns=_name('str'),
    )

    # Build __repr__ method
    repr_method = ast.FunctionDef(
        name='__repr__',
        args=ast.arguments(
            posonlyargs=[],
            args=[_argument('self')],
            kwonlyargs=[],
            kw_defaults=[],
            kwarg=None,
            defaults=[],
        ),
        body=[
            ast.Return(
                value=ast.JoinedStr(
                    values=[
                        ast.Constant(value='APIError(status_code='),
                        ast.FormattedValue(
                            value=_attr('self', 'status_code'), conversion=-1
                        ),
                        ast.Constant(value=', detail='),
                        ast.FormattedValue(
                            value=_attr('self', 'detail'), conversion=114
                        ),  # 114 = 'r' for repr
                        ast.Constant(value=')'),
                    ]
                )
            ),
        ],
        decorator_list=[],
        returns=_name('str'),
    )

    # Build class docstring
    docstring = ast.Expr(
        value=ast.Constant(
            value="""Exception raised when an API request fails with an error response.

This exception provides detailed error information from the API response,
including the HTTP status code, error message, and full response body.

Attributes:
    status_code: The HTTP status code of the response.
    response: The httpx Response object.
    detail: Parsed error detail from the response body (if available).
    body: Raw response body text.
"""
        )
    )

    class_def = ast.ClassDef(
        name='APIError',
        bases=[_name('Exception')],
        keywords=[],
        body=[docstring, init_method, from_response_method, str_method, repr_method],
        decorator_list=[],
    )

    return class_def


def generate_base_client_class(
    class_name: str,
    default_base_url: str,
    default_timeout: float = 30.0,
) -> tuple[ast.ClassDef, ImportDict]:
    """Generate a BaseClient class with only request infrastructure.

    This class contains only the HTTP request plumbing (__init__, _request,
    _request_async). Endpoint implementations live in the module files.

    Args:
        class_name: Name for the generated class (e.g., 'BasePetStoreClient').
        default_base_url: Default base URL from the OpenAPI spec.
        default_timeout: Default request timeout in seconds.

    Returns:
        Tuple of (class AST node, required imports).
    """
    imports: ImportDict = {
        'httpx': {'Client', 'AsyncClient', 'Response'},
        'typing': {'Any', 'Union', 'Type', 'TypeVar'},
        'pydantic': {'TypeAdapter', 'RootModel'},
    }

    # Build __init__ method
    init_method = _build_init_method(default_base_url, default_timeout)

    # Build _request method (sync)
    request_method = _build_request_method(is_async=False)

    # Build _request_async method (async)
    async_request_method = _build_request_method(is_async=True)

    # Build _request_json method (sync) - request + json parsing
    request_json_method = _build_request_json_method(is_async=False)

    # Build _request_json_async method (async) - request + json parsing
    async_request_json_method = _build_request_json_method(is_async=True)

    # Build _parse_response method (sync)
    parse_response_method = _build_parse_response_method(is_async=False)

    # Build _parse_response_async method (async)
    async_parse_response_method = _build_parse_response_method(is_async=True)

    # Build class body
    class_body: list[ast.stmt] = [
        ast.Expr(
            value=ast.Constant(
                value=f"""Base HTTP client with request infrastructure.

This class is regenerated on each code generation run.
To customize, subclass this in client.py.

Endpoint implementations are in the module files (e.g., pet.py, store.py).

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
        request_json_method,
        async_request_json_method,
        parse_response_method,
        async_parse_response_method,
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


def _build_parse_response_method(
    is_async: bool,
) -> ast.FunctionDef | ast.AsyncFunctionDef:
    """Build the _parse_response or _parse_response_async method.

    This method handles JSON parsing and Pydantic validation of responses.
    """
    method_name = '_parse_response_async' if is_async else '_parse_response'

    # T = TypeVar('T') is module-level, we reference it here
    args = ast.arguments(
        posonlyargs=[],
        args=[
            _argument('self'),
            _argument('response', _name('Response')),
            _argument('response_type', _subscript('Type', _name('T'))),
        ],
        kwonlyargs=[],
        kw_defaults=[],
        kwarg=None,
        defaults=[],
    )

    # Build the method body:
    # data = response.json()
    # validated = TypeAdapter(response_type).validate_python(data)
    # if isinstance(validated, RootModel):
    #     return validated.root
    # return validated
    body: list[ast.stmt] = [
        # data = response.json()
        _assign(
            _name('data'),
            _call(func=_attr('response', 'json')),
        ),
        # validated = TypeAdapter(response_type).validate_python(data)
        _assign(
            _name('validated'),
            _call(
                func=_attr(
                    _call(
                        func=_name('TypeAdapter'),
                        args=[_name('response_type')],
                    ),
                    'validate_python',
                ),
                args=[_name('data')],
            ),
        ),
        # if isinstance(validated, RootModel): return validated.root
        ast.If(
            test=_call(
                func=_name('isinstance'),
                args=[_name('validated'), _name('RootModel')],
            ),
            body=[ast.Return(value=_attr('validated', 'root'))],
            orelse=[],
        ),
        # return validated
        ast.Return(value=_name('validated')),
    ]

    if is_async:
        return ast.AsyncFunctionDef(
            name=method_name,
            args=args,
            body=body,
            decorator_list=[],
            returns=_name('T'),
        )
    else:
        return ast.FunctionDef(
            name=method_name,
            args=args,
            body=body,
            decorator_list=[],
            returns=_name('T'),
        )


def _build_request_json_method(
    is_async: bool,
) -> ast.FunctionDef | ast.AsyncFunctionDef:
    """Build the _request_json or _request_json_async method.

    This method combines _request and .json() parsing for convenience.
    """
    method_name = '_request_json_async' if is_async else '_request_json'
    request_method = '_request_async' if is_async else '_request'

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

    # Build the request call with all parameters
    request_call = _call(
        func=_attr('self', request_method),
        args=[_name('method'), _name('path')],
        keywords=[
            ast.keyword(arg='params', value=_name('params')),
            ast.keyword(arg='headers', value=_name('headers')),
            ast.keyword(arg='json', value=_name('json')),
            ast.keyword(arg='data', value=_name('data')),
            ast.keyword(arg='files', value=_name('files')),
            ast.keyword(arg='content', value=_name('content')),
            ast.keyword(arg='timeout', value=_name('timeout')),
        ],
    )

    if is_async:
        request_call = ast.Await(value=request_call)

    # response = self._request(...) or await self._request_async(...)
    # return response.json()
    body: list[ast.stmt] = [
        _assign(_name('response'), request_call),
        ast.Return(value=_call(func=_attr('response', 'json'))),
    ]

    if is_async:
        return ast.AsyncFunctionDef(
            name=method_name,
            args=args,
            body=body,
            decorator_list=[],
            returns=_name('Any'),
        )
    else:
        return ast.FunctionDef(
            name=method_name,
            args=args,
            body=body,
            decorator_list=[],
            returns=_name('Any'),
        )


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


def _build_filtered_params_expr() -> ast.expr:
    """Build expression to filter None values from params dict.

    Generates: {k: v for k, v in params.items() if v is not None} if params else None
    """
    # Build the dict comprehension: {k: v for k, v in params.items() if v is not None}
    dict_comp = ast.DictComp(
        key=_name('k'),
        value=_name('v'),
        generators=[
            ast.comprehension(
                target=ast.Tuple(
                    elts=[
                        ast.Name(id='k', ctx=ast.Store()),
                        ast.Name(id='v', ctx=ast.Store()),
                    ],
                    ctx=ast.Store(),
                ),
                iter=_call(_attr('params', 'items')),
                ifs=[
                    ast.Compare(
                        left=_name('v'),
                        ops=[ast.IsNot()],
                        comparators=[ast.Constant(value=None)],
                    )
                ],
                is_async=0,
            )
        ],
    )

    # Build the conditional: dict_comp if params else None
    return ast.IfExp(
        test=_name('params'),
        body=dict_comp,
        orelse=ast.Constant(value=None),
    )


def _build_sync_request_body(
    url_expr: ast.expr, merged_headers: ast.expr, timeout_expr: ast.expr
) -> list[ast.stmt]:
    """Build the body for sync _request method."""
    # Build filtered_params assignment
    filtered_params_stmt = _assign(
        _name('filtered_params'),
        _build_filtered_params_expr(),
    )

    request_call = _call(
        _attr('client', 'request'),
        args=[_name('method'), url_expr],
        keywords=[
            ast.keyword(arg='params', value=_name('filtered_params')),
            ast.keyword(arg='headers', value=merged_headers),
            ast.keyword(arg='json', value=_name('json')),
            ast.keyword(arg='data', value=_name('data')),
            ast.keyword(arg='files', value=_name('files')),
            ast.keyword(arg='content', value=_name('content')),
            ast.keyword(arg='timeout', value=timeout_expr),
        ],
    )

    return [
        filtered_params_stmt,
        ast.If(
            test=_attr('self', '_client'),
            body=[
                _assign(
                    _name('response'),
                    _call(
                        _attr(_attr('self', '_client'), 'request'),
                        args=[_name('method'), url_expr],
                        keywords=[
                            ast.keyword(arg='params', value=_name('filtered_params')),
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
        # if response.is_error: raise APIError.from_response(response)
        ast.If(
            test=_attr('response', 'is_error'),
            body=[
                ast.Raise(
                    exc=_call(
                        _attr(_name('APIError'), 'from_response'),
                        args=[_name('response')],
                    )
                ),
            ],
            orelse=[],
        ),
        ast.Return(value=_name('response')),
    ]


def _build_async_request_body(
    url_expr: ast.expr, merged_headers: ast.expr, timeout_expr: ast.expr
) -> list[ast.stmt]:
    """Build the body for async _request_async method."""
    # Build filtered_params assignment
    filtered_params_stmt = _assign(
        _name('filtered_params'),
        _build_filtered_params_expr(),
    )

    request_call = ast.Await(
        value=_call(
            _attr('client', 'request'),
            args=[_name('method'), url_expr],
            keywords=[
                ast.keyword(arg='params', value=_name('filtered_params')),
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
        filtered_params_stmt,
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
                                ast.keyword(
                                    arg='params', value=_name('filtered_params')
                                ),
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
        # if response.is_error: raise APIError.from_response(response)
        ast.If(
            test=_attr('response', 'is_error'),
            body=[
                ast.Raise(
                    exc=_call(
                        _attr(_name('APIError'), 'from_response'),
                        args=[_name('response')],
                    )
                ),
            ],
            orelse=[],
        ),
        ast.Return(value=_name('response')),
    ]


def _build_path_expr(path: str, parameters: list['Parameter'] | None) -> ast.expr:
    """Build the path expression with parameter substitution.

    Note:
        This function delegates to ParameterASTBuilder.build_path_expr().
    """
    return ParameterASTBuilder.build_path_expr(path, parameters or [])


def _build_query_params(parameters: list['Parameter'] | None) -> ast.Dict | None:
    """Build query parameters dict.

    Note:
        This function delegates to ParameterASTBuilder.build_query_params().
    """
    if not parameters:
        return None
    return ParameterASTBuilder.build_query_params(parameters)


def _build_header_params(parameters: list['Parameter'] | None) -> ast.Dict | None:
    """Build header parameters dict.

    Note:
        This function delegates to ParameterASTBuilder.build_header_params().
    """
    if not parameters:
        return None
    return ParameterASTBuilder.build_header_params(parameters)


def _build_body_expr(
    request_body: 'RequestBodyInfo',
) -> tuple[ast.expr | None, str | None]:
    """Build the body expression for the request.

    Note:
        This function delegates to ParameterASTBuilder.build_body_expr().
    """
    return ParameterASTBuilder.build_body_expr(request_body)


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
