"""Client class generation module for OtterAPI.

This module provides utilities for generating a client class that wraps
all API endpoints with configurable base URL, timeout, headers, and
HTTP client injection support.
"""

import ast
from dataclasses import dataclass, field
from importlib.resources import files
from typing import TYPE_CHECKING

from otterapi.codegen.ast_utils import (
    ImportDict,
    _argument,
    _assign,
    _attr,
    _call,
    _name,
    _subscript,
    _union_expr,
)
from otterapi.codegen.dataframes import DataFrameMethodConfig

if TYPE_CHECKING:
    from otterapi.codegen.types import Parameter, RequestBodyInfo, ResponseInfo, Type

# Prefix used when building `f'HTTP {status_code} ...'` error messages in generated code
_HTTP_ERROR_PREFIX = 'HTTP '


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
    """Generate the BaseAPIError exception class for detailed error handling.

    This generates a base class that can be subclassed by users in client.py
    to customize error handling behavior (e.g., different detail keys).

    Returns:
        AST ClassDef for the BaseAPIError class.
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
                    ast.Constant(value=_HTTP_ERROR_PREFIX),
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
                                                ast.Constant(value=_HTTP_ERROR_PREFIX),
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
                                            ast.Constant(value=_HTTP_ERROR_PREFIX),
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
                                            ast.Constant(value=_HTTP_ERROR_PREFIX),
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
        # cls = _resolve_error_class(status_code, cls)  -- pick the most
        # specific exception subclass registered for this status code,
        # falling through to ClientError / ServerError tier and finally
        # to ``cls`` itself when the status is unrecognised.
        _assign(
            _name('cls'),
            _call(
                _name('_resolve_error_class'),
                args=[_name('status_code'), _name('cls')],
            ),
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
        returns=ast.Constant(value='BaseAPIError'),
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
                        ast.Constant(value='BaseAPIError(status_code='),
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
        name='BaseAPIError',
        bases=[_name('Exception')],
        keywords=[],
        body=[docstring, init_method, from_response_method, str_method, repr_method],
        decorator_list=[],
    )

    return class_def


# Per-status-code exception hierarchy emitted alongside ``BaseAPIError`` so
# users can ``except NotFoundError`` instead of always inspecting
# ``e.status_code``. Two tier parents (``ClientError`` / ``ServerError``)
# let users catch every 4xx or every 5xx in one ``except``.
_API_ERROR_HIERARCHY_SOURCE = '''\
class ClientError(BaseAPIError):
    """Base class for 4xx HTTP errors."""

    pass


class ServerError(BaseAPIError):
    """Base class for 5xx HTTP errors."""

    pass


class BadRequestError(ClientError):
    """Raised on HTTP 400."""

    pass


class UnauthorizedError(ClientError):
    """Raised on HTTP 401."""

    pass


class ForbiddenError(ClientError):
    """Raised on HTTP 403."""

    pass


class NotFoundError(ClientError):
    """Raised on HTTP 404."""

    pass


class ConflictError(ClientError):
    """Raised on HTTP 409."""

    pass


class UnprocessableEntityError(ClientError):
    """Raised on HTTP 422."""

    pass


class RateLimitError(ClientError):
    """Raised on HTTP 429."""

    pass


class InternalServerError(ServerError):
    """Raised on HTTP 500."""

    pass


class BadGatewayError(ServerError):
    """Raised on HTTP 502."""

    pass


class ServiceUnavailableError(ServerError):
    """Raised on HTTP 503."""

    pass


class GatewayTimeoutError(ServerError):
    """Raised on HTTP 504."""

    pass


_STATUS_ERROR_MAP: dict[int, type[BaseAPIError]] = {
    400: BadRequestError,
    401: UnauthorizedError,
    403: ForbiddenError,
    404: NotFoundError,
    409: ConflictError,
    422: UnprocessableEntityError,
    429: RateLimitError,
    500: InternalServerError,
    502: BadGatewayError,
    503: ServiceUnavailableError,
    504: GatewayTimeoutError,
}


def _resolve_error_class(status_code: int, default: type[BaseAPIError]) -> type[BaseAPIError]:
    """Pick the most specific ``BaseAPIError`` subclass for a status code.

    Falls through ``_STATUS_ERROR_MAP`` -> ``ClientError`` (4xx) ->
    ``ServerError`` (5xx) -> ``default`` (typically ``BaseAPIError``).
    """
    explicit = _STATUS_ERROR_MAP.get(status_code)
    if explicit is not None:
        return explicit
    if 400 <= status_code < 500:
        return ClientError
    if 500 <= status_code < 600:
        return ServerError
    return default
'''


def _exported_error_names() -> list[str]:
    """Names of every emitted error class for ``__all__`` re-export."""
    return [
        'BaseAPIError',
        'ClientError',
        'ServerError',
        'BadRequestError',
        'UnauthorizedError',
        'ForbiddenError',
        'NotFoundError',
        'ConflictError',
        'UnprocessableEntityError',
        'RateLimitError',
        'InternalServerError',
        'BadGatewayError',
        'ServiceUnavailableError',
        'GatewayTimeoutError',
    ]


def generate_api_error_hierarchy() -> list[ast.stmt]:
    """Emit ``BaseAPIError`` plus the per-status-code subclass hierarchy.

    Returns the full list of statements ready to splice into the generated
    ``_client.py``: BaseAPIError ClassDef, tier parents, specific
    subclasses, the status -> class registry, and the resolver helper used
    by ``BaseAPIError.from_response``.
    """
    return [
        generate_api_error_class(),
        *ast.parse(_API_ERROR_HIERARCHY_SOURCE).body,
    ]


def generate_base_client_class(
    class_name: str,
    default_base_url: str,
    default_timeout: float = 30.0,
    pydantic_version: int = 2,
) -> tuple[ast.ClassDef, ImportDict]:
    """Generate a BaseClient class with only request infrastructure.

    This class contains only the HTTP request plumbing (__init__, _request,
    _request_async). Endpoint implementations live in the module files.

    Args:
        class_name: Name for the generated class (e.g., 'BasePetStoreClient').
        default_base_url: Default base URL from the OpenAPI spec.
        default_timeout: Default request timeout in seconds.
        pydantic_version: Target Pydantic version (1 or 2).  Affects which
            response-unwrapping pattern is emitted in ``_parse_response``.

    Returns:
        Tuple of (class AST node, required imports).
    """
    if pydantic_version == 1:
        imports: ImportDict = {
            'httpx': {'Client', 'AsyncClient', 'Response', 'TransportError'},
            'typing': {'Any'},
            'types': {'UnionType'},
            'pydantic': {'TypeAdapter'},
            '._retry': {'_backoff_sleep', '_backoff_sleep_async'},
        }
    else:
        imports = {
            'httpx': {'Client', 'AsyncClient', 'Response', 'TransportError'},
            'typing': {'Any'},
            'types': {'UnionType'},
            'pydantic': {'TypeAdapter', 'RootModel'},
            '._retry': {'_backoff_sleep', '_backoff_sleep_async'},
        }

    # Build __init__ method
    init_method = _build_init_method(default_base_url, default_timeout)

    # Build lifecycle methods
    lifecycle_methods = _build_lifecycle_methods()

    # Build _request method (sync)
    request_method = _build_request_method(is_async=False)

    # Build _request_async method (async)
    async_request_method = _build_request_method(is_async=True)

    # Build _request_json method (sync) - request + json parsing
    request_json_method = _build_request_json_method(is_async=False)

    # Build _request_json_async method (async) - request + json parsing
    async_request_json_method = _build_request_json_method(is_async=True)

    # Build _validate_response hook method
    validate_response_method = _build_validate_response_method()

    # Build _parse_response method (sync)
    parse_response_method = _build_parse_response_method(is_async=False, pydantic_version=pydantic_version)

    # Build _parse_response_async method (async)
    async_parse_response_method = _build_parse_response_method(is_async=True, pydantic_version=pydantic_version)

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
    http_client: Custom httpx.Client for sync requests (persisted and reused).
    async_http_client: Custom httpx.AsyncClient for async requests.
    max_retries: Retry attempts for transient failures (429/5xx/network errors).
        Set to 0 to disable retries. Default: 3.
    retry_statuses: HTTP status codes that trigger a retry.
    backoff_factor: Multiplier for exponential backoff between retries. Default: 0.5.
"""
            )
        ),
        init_method,
        *lifecycle_methods,
        request_method,
        async_request_method,
        request_json_method,
        async_request_json_method,
        validate_response_method,
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
    # frozenset({429, 500, 502, 503, 504}) literal
    default_retry_statuses = ast.Call(
        func=_name('frozenset'),
        args=[
            ast.Set(
                elts=[
                    ast.Constant(value=429),
                    ast.Constant(value=500),
                    ast.Constant(value=502),
                    ast.Constant(value=503),
                    ast.Constant(value=504),
                ]
            )
        ],
        keywords=[],
    )

    init_body: list[ast.stmt] = [
        # self.base_url = base_url.rstrip('/')
        _assign(
            _attr('self', 'base_url'),
            _call(_attr(_name('base_url'), 'rstrip'), [ast.Constant(value='/')]),
        ),
        # self.timeout = timeout
        _assign(_attr('self', 'timeout'), _name('timeout')),
        # self.headers = headers or {}
        _assign(
            _attr('self', 'headers'),
            ast.BoolOp(
                op=ast.Or(), values=[_name('headers'), ast.Dict(keys=[], values=[])]
            ),
        ),
        # self.max_retries = max_retries
        _assign(_attr('self', 'max_retries'), _name('max_retries')),
        # self.retry_statuses = retry_statuses
        _assign(_attr('self', 'retry_statuses'), _name('retry_statuses')),
        # self.backoff_factor = backoff_factor
        _assign(_attr('self', 'backoff_factor'), _name('backoff_factor')),
        # self._owns_sync_client = http_client is None
        _assign(
            _attr('self', '_owns_sync_client'),
            ast.Compare(
                left=_name('http_client'),
                ops=[ast.Is()],
                comparators=[ast.Constant(value=None)],
            ),
        ),
        # self._sync_client = http_client or Client()
        _assign(
            _attr('self', '_sync_client'),
            ast.BoolOp(
                op=ast.Or(), values=[_name('http_client'), _call(_name('Client'))]
            ),
        ),
        # self._async_client = async_http_client
        _assign(_attr('self', '_async_client'), _name('async_http_client')),
    ]

    return ast.FunctionDef(
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
                _argument('max_retries', _name('int')),
                _argument(
                    'retry_statuses',
                    _subscript('frozenset', _name('int')),
                ),
                _argument('backoff_factor', _name('float')),
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
                ast.Constant(value=3),
                default_retry_statuses,
                ast.Constant(value=0.5),
            ],
        ),
        body=init_body,
        decorator_list=[],
        returns=ast.Constant(value=None),
    )


def _build_lifecycle_methods() -> list[ast.stmt]:
    """Build close/aclose/__enter__/__exit__/__aenter__/__aexit__ methods."""

    def _simple_method(
        name: str, body: list[ast.stmt], is_async: bool = False
    ) -> ast.stmt:
        cls = ast.AsyncFunctionDef if is_async else ast.FunctionDef
        return cls(
            name=name,
            args=ast.arguments(
                posonlyargs=[],
                args=[_argument('self')],
                kwonlyargs=[],
                kw_defaults=[],
                kwarg=None,
                defaults=[],
            ),
            body=body,
            decorator_list=[],
            returns=ast.Constant(value=None),
        )

    def _ctx_method(
        name: str,
        body: list[ast.stmt],
        extra_args: list | None = None,
        is_async: bool = False,
    ) -> ast.stmt:
        cls = ast.AsyncFunctionDef if is_async else ast.FunctionDef
        args_list = [_argument('self')] + (extra_args or [])
        return cls(
            name=name,
            args=ast.arguments(
                posonlyargs=[],
                args=args_list,
                kwonlyargs=[],
                kw_defaults=[],
                kwarg=None,
                defaults=[],
            ),
            body=body,
            decorator_list=[],
            returns=ast.Constant(value=None),
        )

    # close(self): if self._owns_sync_client: self._sync_client.close()
    close_method = _simple_method(
        'close',
        [
            ast.If(
                test=_attr('self', '_owns_sync_client'),
                body=[ast.Expr(_call(_attr(_attr('self', '_sync_client'), 'close')))],
                orelse=[],
            )
        ],
    )

    # aclose(self): if self._async_client is not None: await self._async_client.aclose()
    aclose_method = _simple_method(
        'aclose',
        [
            ast.If(
                test=ast.Compare(
                    left=_attr('self', '_async_client'),
                    ops=[ast.IsNot()],
                    comparators=[ast.Constant(value=None)],
                ),
                body=[
                    ast.Expr(
                        ast.Await(
                            _call(_attr(_attr('self', '_async_client'), 'aclose'))
                        )
                    )
                ],
                orelse=[],
            )
        ],
        is_async=True,
    )

    # __enter__(self): return self  (no return annotation — inferred from body)
    enter_method = _ctx_method('__enter__', [ast.Return(_name('self'))])
    enter_method.returns = None  # type: ignore[assignment]

    # __exit__(self, *args): self.close()
    exit_method = _ctx_method(
        '__exit__',
        [ast.Expr(_call(_attr('self', 'close')))],
        extra_args=[
            _argument('exc_type', _name('Any')),
            _argument('exc_val', _name('Any')),
            _argument('exc_tb', _name('Any')),
        ],
    )

    # __aenter__(self): return self  (no return annotation — inferred from body)
    aenter_method = _ctx_method(
        '__aenter__', [ast.Return(_name('self'))], is_async=True
    )
    aenter_method.returns = None  # type: ignore[assignment]

    # __aexit__(self, *args): await self.aclose()
    aexit_method = _ctx_method(
        '__aexit__',
        [ast.Expr(ast.Await(_call(_attr('self', 'aclose'))))],
        extra_args=[
            _argument('exc_type', _name('Any')),
            _argument('exc_val', _name('Any')),
            _argument('exc_tb', _name('Any')),
        ],
        is_async=True,
    )

    return [
        close_method,
        aclose_method,
        enter_method,
        exit_method,
        aenter_method,
        aexit_method,
    ]


def _build_validate_response_method() -> ast.FunctionDef:
    """Build the _validate_response hook method.

    This method is called after parsing but before returning the response.
    Users can override this in their client.py to add custom validation,
    such as checking for API-level errors in wrapper objects.
    """
    args = ast.arguments(
        posonlyargs=[],
        args=[
            _argument('self'),
            _argument('response', _name('Response')),
            _argument('validated', _name('Any')),
        ],
        kwonlyargs=[],
        kw_defaults=[],
        kwarg=None,
        defaults=[],
    )

    # Build docstring
    docstring = """Validate the parsed response before returning.

        Override this method to add custom validation logic, such as
        checking for API-level errors in wrapper objects.

        This hook is called after Pydantic validation but before unwrapping
        RootModel responses.

        Args:
            response: The raw httpx Response object.
            validated: The parsed and validated Pydantic model.

        Raises:
            APIError: If validation fails (or any other exception).

        Example:
            def _validate_response(self, response, validated):
                # Check for API-level errors in wrapper objects
                if hasattr(validated, 'error') and validated.error is not None:
                    raise APIError(
                        status_code=response.status_code,
                        response=response,
                        detail=validated.error,
                    )
        """

    body: list[ast.stmt] = [
        ast.Expr(value=ast.Constant(value=docstring)),
        ast.Pass(),
    ]

    return ast.FunctionDef(
        name='_validate_response',
        args=args,
        body=body,
        decorator_list=[],
        returns=ast.Constant(value=None),
    )


def _build_parse_response_method(
    is_async: bool,
    pydantic_version: int = 2,
) -> ast.FunctionDef | ast.AsyncFunctionDef:
    """Build the _parse_response or _parse_response_async method.

    This method handles JSON parsing and Pydantic validation of responses.

    Args:
        is_async: Whether to generate the async variant.
        pydantic_version: Target Pydantic version (1 or 2).  For v2, the method
            emits an ``isinstance(validated, RootModel)`` check and returns
            ``validated.root``.  For v1, RootModel does not exist; instead the
            method checks ``hasattr(validated, '__root__')`` and returns
            ``validated.__root__``.
    """
    method_name = '_parse_response_async' if is_async else '_parse_response'

    # T = TypeVar('T') is module-level, we reference it here
    args = ast.arguments(
        posonlyargs=[],
        args=[
            _argument('self'),
            _argument('response', _name('Response')),
            _argument(
                'response_type',
                _union_expr([_name('type'), _name('UnionType')]),
            ),
        ],
        kwonlyargs=[],
        kw_defaults=[],
        kwarg=None,
        defaults=[],
    )

    # Common statements: parse JSON, validate with TypeAdapter, call hook
    common: list[ast.stmt] = [
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
        # self._validate_response(response, validated)
        ast.Expr(
            value=_call(
                func=_attr('self', '_validate_response'),
                args=[_name('response'), _name('validated')],
            )
        ),
    ]

    if pydantic_version == 1:
        # Pydantic v1: RootModel does not exist; use hasattr(validated, '__root__')
        # if hasattr(validated, '__root__'): return validated.__root__
        unwrap_stmt = ast.If(
            test=_call(
                func=_name('hasattr'),
                args=[_name('validated'), ast.Constant(value='__root__')],
            ),
            body=[ast.Return(value=_attr('validated', '__root__'))],
            orelse=[],
        )
    else:
        # Pydantic v2: if isinstance(validated, RootModel): return validated.root
        unwrap_stmt = ast.If(
            test=_call(
                func=_name('isinstance'),
                args=[_name('validated'), _name('RootModel')],
            ),
            body=[ast.Return(value=_attr('validated', 'root'))],
            orelse=[],
        )

    body: list[ast.stmt] = [
        *common,
        unwrap_stmt,
        # return validated
        ast.Return(value=_name('validated')),
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


def _build_request_keywords(
    merged_headers: ast.expr, timeout_expr: ast.expr
) -> list[ast.keyword]:
    """Shared keyword args for httpx request calls."""
    return [
        ast.keyword(arg='params', value=_name('filtered_params')),
        ast.keyword(arg='headers', value=merged_headers),
        ast.keyword(arg='json', value=_name('json')),
        ast.keyword(arg='data', value=_name('data')),
        ast.keyword(arg='files', value=_name('files')),
        ast.keyword(arg='content', value=_name('content')),
        ast.keyword(arg='timeout', value=timeout_expr),
    ]


def _build_retry_check(backoff_fn: str, is_async: bool) -> ast.If:
    """Build: if response.is_error: <retry-or-raise>."""
    backoff_call = _call(
        _name(backoff_fn),
        args=[
            _name('attempt'),
            _attr('self', 'backoff_factor'),
            _name('response'),
        ],
    )
    if is_async:
        backoff_stmt = ast.Expr(ast.Await(backoff_call))
    else:
        backoff_stmt = ast.Expr(backoff_call)

    # attempt < self.max_retries and response.status_code in self.retry_statuses
    should_retry = ast.BoolOp(
        op=ast.And(),
        values=[
            ast.Compare(
                left=_name('attempt'),
                ops=[ast.Lt()],
                comparators=[_attr('self', 'max_retries')],
            ),
            ast.Compare(
                left=_attr('response', 'status_code'),
                ops=[ast.In()],
                comparators=[_attr('self', 'retry_statuses')],
            ),
        ],
    )

    return ast.If(
        test=_attr('response', 'is_error'),
        body=[
            ast.If(
                test=should_retry,
                body=[backoff_stmt, ast.Continue()],
                orelse=[
                    ast.Raise(
                        exc=_call(
                            _attr(_name('APIError'), 'from_response'),
                            args=[_name('response')],
                        )
                    )
                ],
            )
        ],
        orelse=[],
    )


def _build_transport_handler(backoff_fn: str, is_async: bool) -> ast.ExceptHandler:
    """Build: except TransportError: retry or re-raise."""
    backoff_call = _call(
        _name(backoff_fn),
        args=[
            _name('attempt'),
            _attr('self', 'backoff_factor'),
            ast.Constant(value=None),
        ],
    )
    if is_async:
        backoff_stmt = ast.Expr(ast.Await(backoff_call))
    else:
        backoff_stmt = ast.Expr(backoff_call)

    return ast.ExceptHandler(
        type=_name('TransportError'),
        name=None,
        body=[
            ast.If(
                test=ast.Compare(
                    left=_name('attempt'),
                    ops=[ast.Lt()],
                    comparators=[_attr('self', 'max_retries')],
                ),
                body=[backoff_stmt, ast.Continue()],
                orelse=[ast.Raise()],
            )
        ],
    )


def _build_sync_request_body(
    url_expr: ast.expr, merged_headers: ast.expr, timeout_expr: ast.expr
) -> list[ast.stmt]:
    """Build the body for sync _request method with retry loop."""
    filtered_params_stmt = _assign(
        _name('filtered_params'), _build_filtered_params_expr()
    )

    request_stmt = _assign(
        _name('response'),
        _call(
            _attr(_attr('self', '_sync_client'), 'request'),
            args=[_name('method'), url_expr],
            keywords=_build_request_keywords(merged_headers, timeout_expr),
        ),
    )

    try_body = [
        request_stmt,
        _build_retry_check('_backoff_sleep', is_async=False),
        ast.Return(value=_name('response')),
    ]

    for_loop = ast.For(
        target=ast.Name(id='attempt', ctx=ast.Store()),
        iter=_call(
            _name('range'),
            [
                ast.BinOp(
                    left=_attr('self', 'max_retries'),
                    op=ast.Add(),
                    right=ast.Constant(value=1),
                )
            ],
        ),
        body=[
            ast.Try(
                body=try_body,
                handlers=[_build_transport_handler('_backoff_sleep', is_async=False)],
                orelse=[],
                finalbody=[],
            )
        ],
        orelse=[],
    )

    return [filtered_params_stmt, for_loop]


def _build_async_request_body(
    url_expr: ast.expr, merged_headers: ast.expr, timeout_expr: ast.expr
) -> list[ast.stmt]:
    """Build the body for async _request_async method with retry loop.

    Uses self._async_client when the caller provided one (e.g. a MockTransport in
    tests), otherwise creates a per-call AsyncClient to avoid event-loop binding
    issues when coroutines are dispatched to a thread via run_sync / run_concurrently.
    """
    filtered_params_stmt = _assign(
        _name('filtered_params'), _build_filtered_params_expr()
    )

    def _request_call(client_expr: ast.expr) -> ast.stmt:
        return _assign(
            _name('response'),
            ast.Await(
                value=_call(
                    _attr(client_expr, 'request'),
                    args=[_name('method'), url_expr],
                    keywords=_build_request_keywords(merged_headers, timeout_expr),
                )
            ),
        )

    # if self._async_client is not None:
    #     response = await self._async_client.request(...)
    # else:
    #     async with AsyncClient() as _owned_ac:
    #         response = await _owned_ac.request(...)
    async_with_stmt = ast.AsyncWith(
        items=[
            ast.withitem(
                context_expr=_call(_name('AsyncClient')),
                optional_vars=ast.Name(id='_owned_ac', ctx=ast.Store()),
            )
        ],
        body=[_request_call(_name('_owned_ac'))],
    )

    client_if = ast.If(
        test=ast.Compare(
            left=_attr('self', '_async_client'),
            ops=[ast.IsNot()],
            comparators=[ast.Constant(value=None)],
        ),
        body=[_request_call(_attr('self', '_async_client'))],
        orelse=[async_with_stmt],
    )

    try_body = [
        client_if,
        _build_retry_check('_backoff_sleep_async', is_async=True),
        ast.Return(value=_name('response')),
    ]

    for_loop = ast.For(
        target=ast.Name(id='attempt', ctx=ast.Store()),
        iter=_call(
            _name('range'),
            [
                ast.BinOp(
                    left=_attr('self', 'max_retries'),
                    op=ast.Add(),
                    right=ast.Constant(value=1),
                )
            ],
        ),
        body=[
            ast.Try(
                body=try_body,
                handlers=[
                    _build_transport_handler('_backoff_sleep_async', is_async=True)
                ],
                orelse=[],
                finalbody=[],
            )
        ],
        orelse=[],
    )

    return [filtered_params_stmt, for_loop]


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
    """Generate the user-customizable client.py stub content."""
    template = (
        files('otterapi.codegen.runtime')
        .joinpath('_client_stub.py.tpl')
        .read_text('utf-8')
    )
    all_names = sorted([class_name, 'Client', 'APIError', 'Error'])
    all_repr = '[' + ', '.join(f'"{n}"' for n in all_names) + ']'
    return (
        template.replace('__CLASS_NAME__', class_name)
        .replace('__BASE_CLASS_NAME__', base_class_name)
        .replace('__MODULE_NAME__', module_name)
        .replace('__ALL__', all_repr)
    )
