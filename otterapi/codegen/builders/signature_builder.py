"""Unified function signature building utilities.

This module provides a builder class for constructing function signatures
from endpoint parameters. It consolidates the repeated pattern of iterating
over parameters, creating annotations, and separating required/optional args.
"""

import ast
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Self

from otterapi.codegen.ast_utils import _argument, _name, _union_expr

if TYPE_CHECKING:
    from otterapi.codegen.types import Parameter, RequestBodyInfo

# Type alias for import dictionaries
ImportDict = dict[str, set[str]]

__all__ = ['FunctionSignature', 'FunctionSignatureBuilder']


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
        >>> # Use signature.args, signature.kwonlyargs, etc.
    """

    def __init__(self):
        """Initialize an empty signature builder."""
        self._args: list[ast.arg] = []
        self._kwonlyargs: list[ast.arg] = []
        self._kw_defaults: list[ast.expr] = []
        self._imports: ImportDict = {}

    def _add_import(self, module: str, name: str) -> None:
        """Add a single import to the collection.

        Args:
            module: The module to import from.
            name: The name to import.
        """
        if module not in self._imports:
            self._imports[module] = set()
        self._imports[module].add(name)

    def _merge_imports(self, imports: ImportDict) -> None:
        """Merge imports from another ImportDict.

        Args:
            imports: Dictionary of imports to merge.
        """
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
        # Insert at the beginning
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
