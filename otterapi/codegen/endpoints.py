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
    - Unified EndpointFunctionFactory for consistent endpoint generation
"""

import ast
import re
import textwrap
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Literal, Self

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

if TYPE_CHECKING:
    from otterapi.codegen.types import Parameter, RequestBodyInfo, ResponseInfo, Type

# Type alias for import dictionaries used throughout this module
ImportDict = dict[str, set[str]]

__all__ = [
    # Enums and dataclasses
    'EndpointMode',
    'DataFrameLibrary',
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
    'build_body_params',
    'prepare_call_from_parameters',
    'build_default_client_code',
    # Convenience functions
    'build_standalone_endpoint_fn',
    'build_delegating_endpoint_fn',
    'build_standalone_dataframe_fn',
    'build_delegating_dataframe_fn',
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


def clean_docstring(docstring: str) -> str:
    """Clean and normalize a docstring by removing excess indentation.

    Args:
        docstring: The raw docstring to clean.

    Returns:
        A cleaned docstring with normalized indentation.
    """
    return textwrap.dedent(f'\n{docstring}\n').strip()


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

    def to_ast_arguments(
        self,
        kwargs: ast.arg | None = None,
    ) -> ast.arguments:
        """Convert to an ast.arguments node.

        Args:
            kwargs: Optional **kwargs argument to include.

        Returns:
            An ast.arguments node ready for use in a FunctionDef.
        """
        return ast.arguments(
            posonlyargs=[],
            args=self.args,
            kwonlyargs=self.kwonlyargs,
            kw_defaults=self.kw_defaults,
            kwarg=kwargs,
            defaults=[],
        )


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

    def add_parameters(self, parameters: list['Parameter'] | None) -> Self:
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
                annotation = param.type.annotation_ast
                self._merge_imports(param.type.annotation_imports)
            else:
                annotation = _name('Any')
                self._add_import('typing', 'Any')

            arg = _argument(param.name_sanitized, annotation)

            if param.required:
                self._args.append(arg)
            else:
                self._kwonlyargs.append(arg)
                self._kw_defaults.append(ast.Constant(value=None))

        return self

    def add_request_body(
        self,
        body: 'RequestBodyInfo | None',
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

        if body.type:
            body_annotation = body.type.annotation_ast
            self._merge_imports(body.type.annotation_imports)
        else:
            body_annotation = _name('Any')
            self._add_import('typing', 'Any')

        body_arg = _argument('body', body_annotation)

        if body.required:
            self._args.append(body_arg)
        else:
            self._kwonlyargs.append(body_arg)
            self._kw_defaults.append(ast.Constant(value=None))

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

    def add_self_parameter(self) -> Self:
        """Add 'self' as the first positional argument.

        Used when building method signatures for classes.

        Returns:
            Self for method chaining.
        """
        self._args.insert(0, _argument('self'))
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

    def reset(self) -> Self:
        """Reset the builder to empty state.

        Returns:
            Self for method chaining.
        """
        self._args = []
        self._kwonlyargs = []
        self._kw_defaults = []
        self._imports = {}
        return self


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
    def build_query_params(parameters: list['Parameter']) -> ast.Dict | None:
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
    def build_header_params(parameters: list['Parameter']) -> ast.Dict | None:
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
    def build_path_expr(path: str, parameters: list['Parameter']) -> ast.expr:
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
        path_params = {
            p.name: p.name_sanitized for p in parameters if p.location == 'path'
        }

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
                sanitized = path_params.get(part, part)
                values.append(ast.FormattedValue(value=_name(sanitized), conversion=-1))

        if len(values) == 1 and isinstance(values[0], ast.Constant):
            return values[0]

        return ast.JoinedStr(values=values)

    @staticmethod
    def build_body_expr(
        body: 'RequestBodyInfo | None',
    ) -> tuple[ast.expr | None, str | None]:
        """Build an AST expression for the request body parameter.

        Handles different content types and determines the appropriate
        httpx parameter name to use.

        Args:
            body: The RequestBodyInfo object, or None if no body.

        Returns:
            A tuple of (body_expr, httpx_param_name).
        """
        if not body:
            return None, None

        body_name = 'body'

        if body.is_json and body.type and body.type.type in ('model', 'root'):
            body_expr = _call(
                func=_attr(_name(body_name), 'model_dump'),
                args=[],
            )
        elif body.is_multipart:
            body_expr = _name(body_name)
        elif body.is_form:
            if body.type and body.type.type in ('model', 'root'):
                body_expr = _call(
                    func=_attr(_name(body_name), 'model_dump'),
                    args=[],
                )
            else:
                body_expr = _name(body_name)
        else:
            body_expr = _name(body_name)

        return body_expr, body.httpx_param_name

    @staticmethod
    def prepare_all_params(
        parameters: list['Parameter'] | None,
        path: str,
        request_body_info: 'RequestBodyInfo | None' = None,
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

        if self.config.dataframe_library:
            body = self._build_dataframe_body()
        elif self.config.mode == EndpointMode.STANDALONE:
            body = self._build_standalone_body()
        else:
            body = self._build_delegating_body()

        returns = self._build_return_type()

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

        self._merge_imports(signature.imports)

        return func_ast, self._imports

    def _build_signature(self) -> FunctionSignature:
        """Build the function signature using FunctionSignatureBuilder."""
        builder = FunctionSignatureBuilder()

        builder.add_parameters(self.config.parameters)
        builder.add_request_body(self.config.request_body_info)

        if self.config.dataframe_library:
            builder.add_path_parameter(default=self.config.dataframe_path)

        if self.config.mode == EndpointMode.STANDALONE:
            builder.add_client_parameter()

        return builder.build()

    def _build_return_type(self) -> ast.expr:
        """Build the return type annotation."""
        if self.config.dataframe_library:
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
        """Build the body for a standalone endpoint function."""
        body: list[ast.stmt] = []

        if self.config.docs:
            body.append(
                ast.Expr(value=ast.Constant(value=clean_docstring(self.config.docs)))
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

        request_keywords = self._build_request_keywords()

        request_method = '_request_async' if self.config.is_async else '_request'
        request_call = _call(
            func=_attr('c', request_method),
            keywords=request_keywords,
        )

        if self.config.is_async:
            request_call = ast.Await(value=request_call)

        if self.config.response_type:
            self._merge_imports(self.config.response_type.annotation_imports)

            body.append(_assign(_name('response'), request_call))

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
            body.append(ast.Return(value=request_call))

        return body

    def _build_delegating_body(self) -> list[ast.stmt]:
        """Build the body for a delegating endpoint function."""
        body: list[ast.stmt] = []

        if self.config.docs:
            body.append(
                ast.Expr(value=ast.Constant(value=clean_docstring(self.config.docs)))
            )

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

    def _build_dataframe_body(self) -> list[ast.stmt]:
        """Build the body for a DataFrame endpoint function."""
        body: list[ast.stmt] = []

        library = self.config.dataframe_library
        return_type_str = (
            'pd.DataFrame' if library == DataFrameLibrary.PANDAS else 'pl.DataFrame'
        )

        doc_content = self.config.docs or ''
        doc_suffix = f'\n\nReturns:\n    {return_type_str}'
        body.append(
            ast.Expr(
                value=ast.Constant(value=clean_docstring(doc_content + doc_suffix))
            )
        )

        if self.config.mode == EndpointMode.STANDALONE:
            body.append(
                _assign(
                    _name('c'),
                    ast.BoolOp(
                        op=ast.Or(),
                        values=[_name('client'), _call(_name('_get_client'))],
                    ),
                )
            )

            request_keywords = self._build_request_keywords()

            request_json_method = (
                '_request_json_async' if self.config.is_async else '_request_json'
            )
            request_call = _call(
                func=_attr('c', request_json_method),
                keywords=request_keywords,
            )

            if self.config.is_async:
                request_call = ast.Await(value=request_call)

            body.append(_assign(_name('data'), request_call))
        else:
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

        helper_fn = 'to_pandas' if library == DataFrameLibrary.PANDAS else 'to_polars'

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

    Args:
        is_async: Whether this is for async functions.

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
    validation_body = _build_response_validation_body(is_async)

    request_keywords = [
        ast.keyword(arg=None, value=_name('kwargs')),
        ast.keyword(
            arg='timeout',
            value=_name('timeout'),
        ),
    ]

    if is_async:
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


# =============================================================================
# Parameter and Call Building Helpers
# =============================================================================


def get_parameters(
    parameters: list['Parameter'],
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
) -> list[ast.keyword]:
    """Build the keyword arguments for a request function call.

    Args:
        query_params: AST expression for query parameters dictionary.
        header_params: AST expression for header parameters dictionary.
        response_model_ast: AST expression for the response model type.
        supported_status_codes: List of valid HTTP status codes.
        body_expr: AST expression for the request body.
        body_param_name: The httpx parameter name for the body.
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

    if timeout is not None:
        keywords.append(ast.keyword(arg='timeout', value=ast.Constant(value=timeout)))

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
    parameters: list['Parameter'],
) -> ast.Dict | None:
    """Build a dictionary AST node for header parameters.

    Args:
        parameters: List of Parameter objects to filter for headers.

    Returns:
        An AST Dict node for header parameters, or None if no header params.
    """
    return ParameterASTBuilder.build_header_params(parameters)


def build_query_params(
    parameters: list['Parameter'],
) -> ast.Dict | None:
    """Build a dictionary AST node for query parameters.

    Args:
        parameters: List of Parameter objects to filter for query params.

    Returns:
        An AST Dict node for query parameters, or None if no query params.
    """
    return ParameterASTBuilder.build_query_params(parameters)


def build_path_params(
    path: str,
    parameters: list['Parameter'],
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
    body: 'RequestBodyInfo | None',
) -> tuple[ast.expr | None, str | None]:
    """Build an AST expression for the request body parameter.

    Args:
        body: The RequestBodyInfo object, or None if no body.

    Returns:
        A tuple of (body_expr, httpx_param_name).
    """
    return ParameterASTBuilder.build_body_expr(body)


def prepare_call_from_parameters(
    parameters: list['Parameter'] | None,
    path: str,
    request_body_info: 'RequestBodyInfo | None' = None,
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


def _build_endpoint_fn(
    name: str,
    method: str,
    path: str,
    response_model: 'Type | None',
    is_async: bool,
    docs: str | None = None,
    parameters: list['Parameter'] | None = None,
    response_infos: list['ResponseInfo'] | None = None,
    request_body_info: 'RequestBodyInfo | None' = None,
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

    if parameters:
        args, kwonlyargs, kw_defaults, param_imports = get_parameters(parameters)
        imports.update(param_imports)
    else:
        args, kwonlyargs, kw_defaults = [], [], []

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

    query_params, header_params, body_expr, body_param_name, path_expr = (
        prepare_call_from_parameters(parameters, path, request_body_info)
    )

    if response_model:
        response_model_ast = response_model.annotation_ast
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
    )

    call_args = [
        ast.keyword(arg='method', value=ast.Constant(value=method.lower())),
        ast.keyword(arg='path', value=path_expr),
        *call_keywords,
        ast.keyword(arg=None, value=_name('kwargs')),
    ]

    request_call = _call(
        func=_name(base_fn_name),
        keywords=call_args,
    )

    if is_async:
        request_call = ast.Await(value=request_call)

    body: list[ast.stmt] = []

    if docs:
        body.append(ast.Expr(value=ast.Constant(value=clean_docstring(docs))))

    body.append(ast.Return(value=request_call))

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
    response_model: 'Type | None',
    docs: str | None = None,
    parameters: list['Parameter'] | None = None,
    response_infos: list['ResponseInfo'] | None = None,
    request_body_info: 'RequestBodyInfo | None' = None,
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
    response_model: 'Type | None',
    docs: str | None = None,
    parameters: list['Parameter'] | None = None,
    response_infos: list['ResponseInfo'] | None = None,
    request_body_info: 'RequestBodyInfo | None' = None,
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


# =============================================================================
# Default Client Code Generation
# =============================================================================


def build_default_client_code() -> tuple[list[ast.stmt], ImportDict]:
    """Build the default client variable and _get_client() function.

    Generates:
        _default_client: Client | None = None

        def _get_client() -> Client:
            '''Get or lazily initialize the default client.'''
            global _default_client
            if _default_client is None:
                _default_client = Client()
            return _default_client

    Returns:
        A tuple of (statements, imports).
    """
    imports: ImportDict = {}

    default_client_var = ast.AnnAssign(
        target=ast.Name(id='_default_client', ctx=ast.Store()),
        annotation=_union_expr([_name('Client'), ast.Constant(value=None)]),
        value=ast.Constant(value=None),
        simple=1,
    )

    get_client_body: list[ast.stmt] = [
        ast.Expr(
            value=ast.Constant(value='Get or lazily initialize the default client.')
        ),
        ast.Global(names=['_default_client']),
        ast.If(
            test=ast.Compare(
                left=_name('_default_client'),
                ops=[ast.Is()],
                comparators=[ast.Constant(value=None)],
            ),
            body=[
                _assign(_name('_default_client'), _call(_name('Client'))),
            ],
            orelse=[],
        ),
        ast.Return(value=_name('_default_client')),
    ]

    get_client_fn = _func(
        name='_get_client',
        args=[],
        body=get_client_body,
        returns=_name('Client'),
    )

    imports.setdefault('typing', set()).add('Union')

    return [default_client_var, get_client_fn], imports


# =============================================================================
# Convenience Functions (Factory Wrappers)
# =============================================================================


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
    parameters: list['Parameter'] | None,
    request_body_info: 'RequestBodyInfo | None',
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
