"""Unified endpoint function factory.

This module provides a factory class for generating endpoint functions
that consolidates the logic previously duplicated across multiple generators:
- build_standalone_endpoint_fn
- build_delegating_endpoint_fn
- build_standalone_dataframe_fn
- build_delegating_dataframe_fn

The factory uses a configuration-based approach to generate the appropriate
function type, reducing duplication and improving maintainability.
"""

import ast
import textwrap
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Literal

from otterapi.codegen.ast_utils import (
    _argument,
    _assign,
    _async_func,
    _attr,
    _call,
    _func,
    _name,
)
from otterapi.codegen.builders.parameter_builder import ParameterASTBuilder
from otterapi.codegen.builders.signature_builder import FunctionSignatureBuilder

if TYPE_CHECKING:
    from otterapi.codegen.types import Parameter, RequestBodyInfo, ResponseInfo, Type

# Type alias for import dictionaries
ImportDict = dict[str, set[str]]

__all__ = [
    'EndpointFunctionConfig',
    'EndpointFunctionFactory',
    'EndpointMode',
    'DataFrameLibrary',
]


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


def _clean_docstring(docstring: str) -> str:
    """Clean and normalize a docstring by removing excess indentation."""
    return textwrap.dedent(f'\n{docstring}\n').strip()


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
    parameters: list['Parameter'] | None = None
    request_body_info: 'RequestBodyInfo | None' = None
    response_type: 'Type | None' = None
    response_infos: list['ResponseInfo'] | None = None
    docs: str | None = None
    is_async: bool = False
    mode: EndpointMode = EndpointMode.STANDALONE
    client_method_name: str | None = None
    dataframe_library: DataFrameLibrary | None = None
    dataframe_path: str | None = None


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

        # Build function signature
        signature = self._build_signature()

        # Build function body based on mode and type
        if self.config.dataframe_library:
            body = self._build_dataframe_body()
        elif self.config.mode == EndpointMode.STANDALONE:
            body = self._build_standalone_body()
        else:
            body = self._build_delegating_body()

        # Build return type annotation
        returns = self._build_return_type()

        # Assemble the function
        func_builder = _async_func if self.config.is_async else _func

        func_ast = func_builder(
            name=self.config.fn_name,
            args=signature.args,
            body=body,
            kwargs=_argument('kwargs', _name('dict')),
            kwonlyargs=signature.kwonlyargs,
            kw_defaults=signature.kw_defaults,
            returns=returns,
        )

        # Merge signature imports
        self._merge_imports(signature.imports)

        return func_ast, self._imports

    def _build_signature(self):
        """Build the function signature using FunctionSignatureBuilder."""
        builder = FunctionSignatureBuilder()

        # Add endpoint parameters
        builder.add_parameters(self.config.parameters)

        # Add request body parameter
        builder.add_request_body(self.config.request_body_info)

        # Add DataFrame path parameter if needed
        if self.config.dataframe_library:
            builder.add_path_parameter(default=self.config.dataframe_path)

        # Add client parameter for standalone mode
        if self.config.mode == EndpointMode.STANDALONE:
            builder.add_client_parameter()

        return builder.build()

    def _build_return_type(self) -> ast.expr:
        """Build the return type annotation."""
        if self.config.dataframe_library:
            # DataFrame methods return string annotation for TYPE_CHECKING
            if self.config.dataframe_library == DataFrameLibrary.PANDAS:
                return ast.Constant(value='pd.DataFrame')
            else:
                return ast.Constant(value='pl.DataFrame')

        if self.config.response_type:
            self._merge_imports(self.config.response_type.annotation_imports)
            return self.config.response_type.annotation_ast

        self._add_import('typing', 'Any')
        return _name('Any')

    def _build_standalone_body(self) -> list[ast.stmt]:
        """Build the body for a standalone endpoint function.

        Generates code like:
            '''Docstring'''
            c = client or _get_client()
            response = c._request(method='get', path=f'/pet/{petId}', **kwargs)
            return c._parse_response(response, Pet)
        """
        body: list[ast.stmt] = []

        # Add docstring
        if self.config.docs:
            body.append(
                ast.Expr(value=ast.Constant(value=_clean_docstring(self.config.docs)))
            )

        # c = client or _get_client()
        body.append(
            _assign(
                _name('c'),
                ast.BoolOp(
                    op=ast.Or(),
                    values=[_name('client'), _call(_name('_get_client'))],
                ),
            )
        )

        # Build request keywords
        request_keywords = self._build_request_keywords()

        # Build the request call
        request_method = '_request_async' if self.config.is_async else '_request'
        request_call = _call(
            func=_attr('c', request_method),
            keywords=request_keywords,
        )

        if self.config.is_async:
            request_call = ast.Await(value=request_call)

        # Handle response
        if self.config.response_type:
            self._merge_imports(self.config.response_type.annotation_imports)

            # response = c._request(...)
            body.append(_assign(_name('response'), request_call))

            # return c._parse_response(response, ResponseType)
            parse_method = (
                '_parse_response_async' if self.config.is_async else '_parse_response'
            )
            parse_call = _call(
                func=_attr('c', parse_method),
                args=[_name('response'), self.config.response_type.annotation_ast],
            )
            if self.config.is_async:
                parse_call = ast.Await(value=parse_call)
            body.append(ast.Return(value=parse_call))
        else:
            # No response type - just return the response
            body.append(ast.Return(value=request_call))

        return body

    def _build_delegating_body(self) -> list[ast.stmt]:
        """Build the body for a delegating endpoint function.

        Generates code like:
            '''Docstring'''
            return _get_client().method_name(args, **kwargs)
        """
        body: list[ast.stmt] = []

        # Add docstring
        if self.config.docs:
            body.append(
                ast.Expr(value=ast.Constant(value=_clean_docstring(self.config.docs)))
            )

        # Build call arguments
        call_args, call_keywords = self._build_delegating_call_args()

        # Build the client method call
        method_name = self.config.client_method_name or self.config.fn_name
        client_call = _call(
            func=_attr(_call(_name('_get_client')), method_name),
            args=call_args,
            keywords=call_keywords,
        )

        if self.config.is_async:
            client_call = ast.Await(value=client_call)

        body.append(ast.Return(value=client_call))

        return body

    def _build_dataframe_body(self) -> list[ast.stmt]:
        """Build the body for a DataFrame endpoint function.

        Generates code like:
            '''Docstring

            Returns:
                pd.DataFrame
            '''
            c = client or _get_client()
            data = c._request_json(method='get', path=f'/pet/findByStatus', **kwargs)
            return to_pandas(data, path=path)
        """
        body: list[ast.stmt] = []

        # Build docstring with return type info
        library = self.config.dataframe_library
        return_type_str = (
            'pd.DataFrame' if library == DataFrameLibrary.PANDAS else 'pl.DataFrame'
        )

        doc_content = self.config.docs or ''
        doc_suffix = f'\n\nReturns:\n    {return_type_str}'
        body.append(
            ast.Expr(
                value=ast.Constant(value=_clean_docstring(doc_content + doc_suffix))
            )
        )

        if self.config.mode == EndpointMode.STANDALONE:
            # c = client or _get_client()
            body.append(
                _assign(
                    _name('c'),
                    ast.BoolOp(
                        op=ast.Or(),
                        values=[_name('client'), _call(_name('_get_client'))],
                    ),
                )
            )

            # Build request keywords
            request_keywords = self._build_request_keywords()

            # Build the request call: c._request_json(...) or c._request_json_async(...)
            request_json_method = (
                '_request_json_async' if self.config.is_async else '_request_json'
            )
            request_call = _call(
                func=_attr('c', request_json_method),
                keywords=request_keywords,
            )

            if self.config.is_async:
                request_call = ast.Await(value=request_call)

            # data = c._request_json(...)
            body.append(_assign(_name('data'), request_call))
        else:
            # Delegating mode: call client method
            call_args, call_keywords = self._build_delegating_call_args()
            method_name = self.config.client_method_name or self.config.fn_name
            client_call = _call(
                func=_attr(_call(_name('_get_client')), method_name),
                args=call_args,
                keywords=call_keywords,
            )

            if self.config.is_async:
                client_call = ast.Await(value=client_call)

            body.append(ast.Return(value=client_call))
            return body

        # Determine conversion function
        helper_fn = 'to_pandas' if library == DataFrameLibrary.PANDAS else 'to_polars'

        # return to_pandas(data, path=path) or to_polars(data, path=path)
        body.append(
            ast.Return(
                value=_call(
                    func=_name(helper_fn),
                    args=[_name('data')],
                    keywords=[ast.keyword(arg='path', value=_name('path'))],
                )
            )
        )

        return body

    def _build_request_keywords(self) -> list[ast.keyword]:
        """Build the keywords for a request call."""
        parameters = self.config.parameters or []

        # Build parameter expressions
        path_expr = ParameterASTBuilder.build_path_expr(self.config.path, parameters)
        query_params = ParameterASTBuilder.build_query_params(parameters)
        header_params = ParameterASTBuilder.build_header_params(parameters)
        body_expr, body_param_name = ParameterASTBuilder.build_body_expr(
            self.config.request_body_info
        )

        # Build request call keywords
        request_keywords: list[ast.keyword] = [
            ast.keyword(
                arg='method', value=ast.Constant(value=self.config.method.lower())
            ),
            ast.keyword(arg='path', value=path_expr),
        ]

        if query_params:
            request_keywords.append(ast.keyword(arg='params', value=query_params))

        if header_params:
            request_keywords.append(ast.keyword(arg='headers', value=header_params))

        if body_expr and body_param_name:
            request_keywords.append(ast.keyword(arg=body_param_name, value=body_expr))

        # Add **kwargs pass-through
        request_keywords.append(ast.keyword(arg=None, value=_name('kwargs')))

        return request_keywords

    def _build_delegating_call_args(
        self,
    ) -> tuple[list[ast.expr], list[ast.keyword]]:
        """Build the arguments for a delegating client method call."""
        call_args: list[ast.expr] = []
        call_keywords: list[ast.keyword] = []

        # Build signature to get arg names
        builder = FunctionSignatureBuilder()
        builder.add_parameters(self.config.parameters)
        builder.add_request_body(self.config.request_body_info)

        if self.config.dataframe_library:
            builder.add_path_parameter(default=self.config.dataframe_path)

        signature = builder.build()

        # Add positional args (required parameters)
        for arg in signature.args:
            call_args.append(_name(arg.arg))

        # Add keyword args (optional parameters)
        for kwarg in signature.kwonlyargs:
            call_keywords.append(ast.keyword(arg=kwarg.arg, value=_name(kwarg.arg)))

        # Add **kwargs pass-through
        call_keywords.append(ast.keyword(arg=None, value=_name('kwargs')))

        return call_args, call_keywords


# Convenience functions that wrap the factory for backward compatibility


def build_standalone_endpoint_fn(
    fn_name: str,
    method: str,
    path: str,
    parameters: list['Parameter'] | None,
    request_body_info: 'RequestBodyInfo | None',
    response_type: 'Type | None',
    response_infos: list['ResponseInfo'] | None = None,
    docs: str | None = None,
    is_async: bool = False,
) -> tuple[ast.FunctionDef | ast.AsyncFunctionDef, ImportDict]:
    """Build a standalone endpoint function with full implementation.

    This is a convenience wrapper around EndpointFunctionFactory.
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
    )
    return EndpointFunctionFactory(config).build()


def build_delegating_endpoint_fn(
    fn_name: str,
    client_method_name: str,
    parameters: list['Parameter'] | None,
    request_body_info: 'RequestBodyInfo | None',
    response_type: 'Type | None',
    docs: str | None = None,
    is_async: bool = False,
) -> tuple[ast.FunctionDef | ast.AsyncFunctionDef, ImportDict]:
    """Build an endpoint function that delegates to a client method.

    This is a convenience wrapper around EndpointFunctionFactory.
    """
    config = EndpointFunctionConfig(
        fn_name=fn_name,
        method='',  # Not used in delegating mode
        path='',  # Not used in delegating mode
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
    parameters: list['Parameter'] | None,
    request_body_info: 'RequestBodyInfo | None',
    library: Literal['pandas', 'polars'],
    default_path: str | None = None,
    docs: str | None = None,
    is_async: bool = False,
) -> tuple[ast.FunctionDef | ast.AsyncFunctionDef, ImportDict]:
    """Build a standalone DataFrame endpoint function.

    This is a convenience wrapper around EndpointFunctionFactory.
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


def build_delegating_dataframe_fn(
    fn_name: str,
    client_method_name: str,
    parameters: list['Parameter'] | None,
    request_body_info: 'RequestBodyInfo | None',
    library: Literal['pandas', 'polars'],
    default_path: str | None = None,
    docs: str | None = None,
    is_async: bool = False,
) -> tuple[ast.FunctionDef | ast.AsyncFunctionDef, ImportDict]:
    """Build a DataFrame endpoint function that delegates to a client method.

    This is a convenience wrapper around EndpointFunctionFactory.
    """
    df_library = (
        DataFrameLibrary.PANDAS if library == 'pandas' else DataFrameLibrary.POLARS
    )
    config = EndpointFunctionConfig(
        fn_name=fn_name,
        method='',  # Not used in delegating mode
        path='',  # Not used in delegating mode
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
