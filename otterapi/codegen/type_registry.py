"""Type registry for managing generated types during code generation.

This module provides the TypeRegistry class for tracking, organizing, and
managing generated types throughout the code generation process.
"""

import ast
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from otterapi.codegen.types import Type


@dataclass
class TypeInfo:
    """Information about a registered type.

    Attributes:
        name: The Python name for this type.
        reference: The original OpenAPI reference (e.g., '#/components/schemas/Pet').
        type_obj: The Type object containing AST and metadata.
        dependencies: Set of type names this type depends on.
        is_root_model: Whether this is a Pydantic RootModel.
        is_generated: Whether the AST has been generated for this type.
    """

    name: str
    reference: str | None
    type_obj: 'Type'
    dependencies: set[str] = field(default_factory=set)
    is_root_model: bool = False
    is_generated: bool = False


class TypeRegistry:
    """Registry for managing generated types during code generation.

    This class provides a centralized location for tracking all types generated
    from an OpenAPI schema, handling dependencies between types, and ensuring
    types are generated in the correct order.

    Example:
        >>> registry = TypeRegistry()
        >>> registry.register(type_obj, name='Pet', reference='#/components/schemas/Pet')
        >>> if registry.has_type('Pet'):
        ...     pet_type = registry.get_type('Pet')
        >>> for type_info in registry.get_types_in_dependency_order():
        ...     generate_code(type_info)
    """

    def __init__(self):
        """Initialize an empty type registry."""
        self._types: dict[str, TypeInfo] = {}
        self._by_reference: dict[str, str] = {}  # reference -> name mapping
        self._primitive_types: set[str] = {'str', 'int', 'float', 'bool', 'None'}

    def register(
        self,
        type_obj: 'Type',
        name: str,
        reference: str | None = None,
        dependencies: set[str] | None = None,
        is_root_model: bool = False,
    ) -> TypeInfo:
        """Register a new type in the registry.

        Args:
            type_obj: The Type object containing the type information.
            name: The Python name for this type.
            reference: The OpenAPI reference string, if applicable.
            dependencies: Set of type names this type depends on.
            is_root_model: Whether this is a Pydantic RootModel.

        Returns:
            The TypeInfo object for the registered type.

        Raises:
            ValueError: If a type with the same name is already registered.
        """
        if name in self._types:
            raise ValueError(f"Type '{name}' is already registered")

        type_info = TypeInfo(
            name=name,
            reference=reference,
            type_obj=type_obj,
            dependencies=dependencies or set(),
            is_root_model=is_root_model,
        )

        self._types[name] = type_info

        if reference:
            self._by_reference[reference] = name

        return type_info

    def register_or_get(
        self,
        type_obj: 'Type',
        name: str,
        reference: str | None = None,
        dependencies: set[str] | None = None,
        is_root_model: bool = False,
    ) -> TypeInfo:
        """Register a type if not exists, otherwise return the existing one.

        Args:
            type_obj: The Type object containing the type information.
            name: The Python name for this type.
            reference: The OpenAPI reference string, if applicable.
            dependencies: Set of type names this type depends on.
            is_root_model: Whether this is a Pydantic RootModel.

        Returns:
            The TypeInfo object (existing or newly registered).
        """
        if name in self._types:
            return self._types[name]
        return self.register(type_obj, name, reference, dependencies, is_root_model)

    def has_type(self, name: str) -> bool:
        """Check if a type is registered.

        Args:
            name: The type name to check.

        Returns:
            True if the type is registered, False otherwise.
        """
        return name in self._types

    def has_reference(self, reference: str) -> bool:
        """Check if a reference has been registered.

        Args:
            reference: The OpenAPI reference string.

        Returns:
            True if the reference has been registered, False otherwise.
        """
        return reference in self._by_reference

    def get_type(self, name: str) -> TypeInfo | None:
        """Get a registered type by name.

        Args:
            name: The type name.

        Returns:
            The TypeInfo object, or None if not found.
        """
        return self._types.get(name)

    def get_type_by_reference(self, reference: str) -> TypeInfo | None:
        """Get a registered type by its OpenAPI reference.

        Args:
            reference: The OpenAPI reference string.

        Returns:
            The TypeInfo object, or None if not found.
        """
        name = self._by_reference.get(reference)
        if name:
            return self._types.get(name)
        return None

    def get_name_for_reference(self, reference: str) -> str | None:
        """Get the registered name for an OpenAPI reference.

        Args:
            reference: The OpenAPI reference string.

        Returns:
            The type name, or None if not found.
        """
        return self._by_reference.get(reference)

    def get_all_types(self) -> dict[str, TypeInfo]:
        """Get all registered types.

        Returns:
            Dictionary mapping type names to TypeInfo objects.
        """
        return dict(self._types)

    def get_type_names(self) -> list[str]:
        """Get all registered type names.

        Returns:
            List of type names, sorted alphabetically.
        """
        return sorted(self._types.keys())

    def add_dependency(self, type_name: str, depends_on: str) -> None:
        """Add a dependency relationship between types.

        Args:
            type_name: The type that has the dependency.
            depends_on: The type that is depended upon.

        Raises:
            KeyError: If the type_name is not registered.
        """
        if type_name not in self._types:
            raise KeyError(f"Type '{type_name}' is not registered")
        self._types[type_name].dependencies.add(depends_on)

    def get_dependencies(self, type_name: str) -> set[str]:
        """Get all dependencies for a type.

        Args:
            type_name: The type name.

        Returns:
            Set of type names that the type depends on.

        Raises:
            KeyError: If the type is not registered.
        """
        if type_name not in self._types:
            raise KeyError(f"Type '{type_name}' is not registered")
        return self._types[type_name].dependencies.copy()

    def get_types_in_dependency_order(self) -> list[TypeInfo]:
        """Get all types sorted in dependency order.

        Types are sorted so that dependencies come before the types that
        depend on them. This ensures proper import/definition order in
        generated code.

        Returns:
            List of TypeInfo objects in dependency order.
        """
        result: list[TypeInfo] = []
        visited: set[str] = set()
        visiting: set[str] = set()  # For cycle detection

        def visit(name: str) -> None:
            if name in visited:
                return
            if name in visiting:
                # Cycle detected - just skip to avoid infinite recursion
                # This can happen with mutually recursive types
                return
            if name in self._primitive_types:
                return
            if name not in self._types:
                return

            visiting.add(name)
            type_info = self._types[name]

            # Visit dependencies first
            for dep in type_info.dependencies:
                visit(dep)

            visiting.remove(name)
            visited.add(name)
            result.append(type_info)

        # Visit all types
        for name in sorted(self._types.keys()):
            visit(name)

        return result

    def mark_generated(self, name: str) -> None:
        """Mark a type as having its AST generated.

        Args:
            name: The type name.

        Raises:
            KeyError: If the type is not registered.
        """
        if name not in self._types:
            raise KeyError(f"Type '{name}' is not registered")
        self._types[name].is_generated = True

    def get_ungenerated_types(self) -> list[TypeInfo]:
        """Get all types that haven't been generated yet.

        Returns:
            List of TypeInfo objects that haven't been marked as generated.
        """
        return [t for t in self._types.values() if not t.is_generated]

    def clear(self) -> None:
        """Clear all registered types."""
        self._types.clear()
        self._by_reference.clear()

    def __len__(self) -> int:
        """Return the number of registered types."""
        return len(self._types)

    def __iter__(self) -> Iterator[TypeInfo]:
        """Iterate over all registered types."""
        return iter(self._types.values())

    def __contains__(self, name: str) -> bool:
        """Check if a type name is registered."""
        return name in self._types

    def get_root_models(self) -> list[TypeInfo]:
        """Get all registered root models.

        Returns:
            List of TypeInfo objects that are root models.
        """
        return [t for t in self._types.values() if t.is_root_model]

    def get_regular_models(self) -> list[TypeInfo]:
        """Get all registered non-root models.

        Returns:
            List of TypeInfo objects that are not root models.
        """
        return [t for t in self._types.values() if not t.is_root_model]

    def merge(self, other: 'TypeRegistry') -> None:
        """Merge another registry into this one.

        Types from the other registry are added to this one.
        If a type with the same name already exists, it is skipped.

        Args:
            other: The registry to merge from.
        """
        for type_info in other:
            if type_info.name not in self._types:
                self._types[type_info.name] = type_info
                if type_info.reference:
                    self._by_reference[type_info.reference] = type_info.name
