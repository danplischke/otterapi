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
from otterapi.codegen.type_generator import Parameter, Type


def clean_docstring(docstring: str) -> str:
    return textwrap.dedent(f'\n{docstring}\n').strip()


def get_base_request_arguments():
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


def base_request_fn():
    args, kwonlyargs, kwargs = get_base_request_arguments()

    body = [
        _assign(
            target=_name('response'),
            value=_call(
                func=_name('request'),
                args=[
                    _name('method'),
                    ast.Expr(
                        value=ast.JoinedStr(
                            values=[
                                ast.FormattedValue(
                                    value=_name('BASE_URL'), conversion=-1
                                ),
                                ast.FormattedValue(value=_name('path'), conversion=-1),
                            ]
                        )
                    ),
                ],
                keywords=[ast.keyword(arg=None, value=_name('kwargs'))],
            ),
        ),
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
                    value=_call(
                        func=_attr('response', 'raise_for_status'),
                    )
                )
            ],
            orelse=[],
        ),
        _assign(
            target=_name('data'),
            value=_call(
                func=_attr('response', 'json'),
            ),
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
    ]

    return _func(
        name='request_sync',
        args=args,
        body=body,
        kwargs=kwargs,
        kwonlyargs=kwonlyargs,
        kw_defaults=[_name('Json'), ast.Constant(value=None)],
        returns=_name('T'),
    ), {
        'httpx': ['request'],
        'pydantic': ['TypeAdapter', 'Json', 'RootModel'],
        'typing': ['Type', 'TypeVar', 'Union'],
    }


def base_async_request_fn():
    args, kwonlyargs, kwargs = get_base_request_arguments()

    body = [
        ast.AsyncWith(
            items=[
                ast.withitem(
                    context_expr=_call(func=_name('AsyncClient')),
                    optional_vars=_name('client'),
                )
            ],
            body=[
                _assign(
                    target=_name('response'),
                    value=ast.Await(
                        value=_call(
                            func=_attr('client', 'request'),
                            args=[
                                _name('method'),
                                ast.Expr(
                                    value=ast.JoinedStr(
                                        values=[
                                            ast.FormattedValue(
                                                value=_name('BASE_URL'), conversion=-1
                                            ),
                                            ast.FormattedValue(
                                                value=_name('path'), conversion=-1
                                            ),
                                        ]
                                    )
                                ),
                            ],
                            keywords=[ast.keyword(arg=None, value=_name('kwargs'))],
                        )
                    ),
                ),
                ast.If(
                    test=ast.BoolOp(
                        op=ast.Or(),
                        values=[
                            ast.UnaryOp(
                                op=ast.Not(), operand=_name('supported_status_codes')
                            ),
                            ast.Compare(
                                left=_attr('response', 'status_code'),
                                ops=[ast.NotIn()],
                                comparators=[_name('supported_status_codes')],
                            ),
                        ],
                    ),
                    body=[
                        ast.Expr(
                            value=_call(
                                func=_attr('response', 'raise_for_status'),
                            )
                        )
                    ],
                    orelse=[],
                ),
                _assign(
                    target=_name('data'),
                    value=_call(
                        func=_attr('response', 'json'),
                    ),
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
        )
    ]

    return _async_func(
        name='request_async',
        args=args,
        body=body,
        kwargs=kwargs,
        kwonlyargs=kwonlyargs,
        kw_defaults=[_name('Json'), ast.Constant(value=None)],
        returns=_name('T'),
    ), {
        'httpx': ['AsyncClient'],
        'pydantic': ['TypeAdapter', 'Json', 'RootModel'],
        'typing': ['Type', 'TypeVar', 'Union'],
    }


def get_parameters(
    parameters: list[Parameter],
) -> tuple[list[ast.arg], list[ast.arg], list[ast.expr], dict[str, set[str]]]:
    args = []
    kwonlyargs = []
    kw_defaults = []
    imports = {}

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
    method, path, response_model: Type, supported_status_codes: list | None = None
) -> list[ast.keyword]:
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
    if not headers:
        return None

    return ast.Dict(
        keys=[ast.Constant(value=header.name) for header in headers],
        values=[_name(header.name_sanitized) for header in headers],
    )


def build_query_params(queries: list[Parameter]) -> ast.Dict | None:
    if not queries:
        return None

    return ast.Dict(
        keys=[ast.Constant(value=query.name) for query in queries],
        values=[_name(query.name_sanitized) for query in queries],
    )


def build_path_params(
    paths: list[Parameter], path: str
) -> ast.JoinedStr | ast.Constant:
    """Build an f-string AST that interpolates path parameters."""
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


def build_body_params(body: Parameter | None) -> ast.expr | None:
    if not body:
        return None

    if body.type.type == 'model' or body.type.type == 'root_model':
        return _call(
            func=_attr(_name(body.name_sanitized), 'model_dump'),
            args=[],
        )

    return _name(body.name)


def prepare_call_from_parameters(
    parameters: list[Parameter] | None, path: str
) -> tuple[ast.expr, ast.expr, ast.expr, ast.expr]:
    if not parameters:
        parameters = []

    query_params = [p for p in parameters if p.location == 'query']
    path_params = [p for p in parameters if p.location == 'path']
    header_params = [p for p in parameters if p.location == 'header']
    body_params = [p for p in parameters if p.location == 'body']

    if len(body_params) > 1:
        raise ValueError('Multiple body parameters are not supported.')

    return (
        build_query_params(query_params),
        build_header_params(header_params),
        build_body_params(body_params[0] if len(body_params) == 1 else None),
        build_path_params(path_params, path),
    )


def request_fn(
    name: str,
    method: str,
    path: str,
    response_model: Type,
    docs: str | None = None,
    parameters: list | None = None,
    supported_status_codes: list | None = None,
):
    args, kwonlyargs, kw_defaults, imports = get_parameters(parameters)

    query_params, header_params, body_params, processed_path = (
        prepare_call_from_parameters(parameters, path)
    )
    call_keywords = get_base_call_keywords(
        method, processed_path, response_model, supported_status_codes
    )

    if query_params:
        call_keywords.append(ast.keyword(arg='params', value=query_params))
    if header_params:
        call_keywords.append(ast.keyword(arg='headers', value=header_params))
    if body_params:
        call_keywords.append(ast.keyword(arg='json', value=body_params))

    call_keywords.append(ast.keyword(arg=None, value=_name('kwargs')))

    body = [
        ast.Return(
            value=_call(
                func=_name('request_sync'),
                args=[],
                keywords=call_keywords,
            )
        )
    ]

    if docs:
        docs = textwrap.dedent(f'\n{docs}\n').strip()
        body.insert(0, ast.Expr(value=ast.Constant(value=docs)))

    response_model_ast = response_model.annotation_ast if response_model else None
    if not response_model_ast:
        response_model_ast = _name('Any')
        imports.setdefault('typing', set()).add('Any')

    return _func(
        name=name,
        args=args,
        body=body,
        kwargs=_argument('kwargs', _name('dict')),
        kwonlyargs=kwonlyargs,
        kw_defaults=kw_defaults,
        returns=response_model_ast,
    ), ({'httpx': ['request'], **imports})


def async_request_fn(
    name: str,
    method: str,
    path: str,
    response_model: Type,
    docs: str | None = None,
    parameters: list | None = None,
    supported_status_codes: list | None = None,
):
    args, kwonlyargs, kw_defaults, imports = get_parameters(parameters)

    query_params, header_params, body_params, processed_path = (
        prepare_call_from_parameters(parameters, path)
    )
    call_keywords = get_base_call_keywords(
        method, processed_path, response_model, supported_status_codes
    )

    if query_params:
        call_keywords.append(ast.keyword(arg='params', value=query_params))
    if header_params:
        call_keywords.append(ast.keyword(arg='headers', value=header_params))
    if body_params:
        call_keywords.append(ast.keyword(arg='json', value=body_params))

    # Add **kwargs to the call
    call_keywords.append(ast.keyword(arg=None, value=_name('kwargs')))

    body = [
        ast.Return(
            value=ast.Await(
                value=_call(
                    func=_name('request_async'),
                    args=[],
                    keywords=call_keywords,
                )
            )
        )
    ]

    if docs:
        docs = textwrap.dedent(f'\n{docs}\n').strip()
        body.insert(0, ast.Expr(value=ast.Constant(value=docs)))

    response_model_ast = response_model.annotation_ast if response_model else None
    if not response_model_ast:
        response_model_ast = _name('Any')
        imports.setdefault('typing', set()).add('Any')

    return _async_func(
        name=name,
        args=args,
        body=body,
        kwargs=_argument('kwargs', _name('dict')),
        kwonlyargs=kwonlyargs,
        kw_defaults=kw_defaults,
        returns=response_model_ast,
    ), {'httpx': ['AsyncClient'], **imports}
