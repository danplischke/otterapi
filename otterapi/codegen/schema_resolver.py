"""Schema resolution utilities for OpenAPI documents.

This module provides the SchemaResolver class for resolving $ref references
and managing schema lookups in OpenAPI documents.
"""

from typing import Any

from otterapi.exceptions import SchemaReferenceError
from otterapi.openapi.v3_2 import Reference, Schema
from otterapi.openapi.v3_2.v3_2 import OpenAPI


class SchemaResolver:
    """Resolves $ref references and manages schema lookups in OpenAPI documents.

    This class provides a centralized way to resolve JSON References ($ref) in
    OpenAPI documents, supporting both local references (within the same document)
    and caching resolved references for performance.

    Example:
        >>> resolver = SchemaResolver(openapi_doc)
        >>> schema, name = resolver.resolve_reference(ref)
        >>> all_schemas = resolver.get_all_schemas()
    """

    def __init__(self, openapi: OpenAPI):
        """Initialize the schema resolver.

        Args:
            openapi: The OpenAPI document to resolve references from.
        """
        self.openapi = openapi
        self._cache: dict[str, tuple[Schema, str]] = {}

    def resolve_reference(self, reference: Reference | Schema) -> tuple[Schema, str]:
        """Resolve a $ref reference to its schema and name.

        If the input is already a Schema (not a Reference), it is returned as-is
        with its title as the name (if available).

        Args:
            reference: A Reference or Schema object to resolve.

        Returns:
            Tuple of (resolved Schema, schema name).

        Raises:
            SchemaReferenceError: If the reference cannot be resolved.
        """
        if isinstance(reference, Schema):
            name = (
                reference.title
                if hasattr(reference, 'title') and reference.title
                else None
            )
            return reference, self._sanitize_name(name) if name else None

        ref_str = reference.ref

        # Check cache first
        if ref_str in self._cache:
            return self._cache[ref_str]

        # Resolve the reference
        schema, name = self._resolve_ref_string(ref_str)

        # Cache the result
        self._cache[ref_str] = (schema, name)

        return schema, name

    def _resolve_ref_string(self, ref: str) -> tuple[Schema, str]:
        """Resolve a $ref string to its schema.

        Args:
            ref: The $ref string (e.g., '#/components/schemas/Pet').

        Returns:
            Tuple of (resolved Schema, schema name).

        Raises:
            SchemaReferenceError: If the reference format is unsupported or
                                  the referenced schema doesn't exist.
        """
        # Handle local references
        if ref.startswith('#/'):
            return self._resolve_local_reference(ref)

        # Handle external references (not yet supported)
        if ref.startswith('http://') or ref.startswith('https://'):
            raise SchemaReferenceError(
                ref,
                'External URL references are not yet supported. '
                'Consider inlining the referenced schema.',
            )

        if ref.startswith('./') or ref.startswith('../'):
            raise SchemaReferenceError(
                ref,
                'Relative file references are not yet supported. '
                'Consider using a tool to bundle your OpenAPI spec.',
            )

        raise SchemaReferenceError(ref, 'Unknown reference format')

    def _resolve_local_reference(self, ref: str) -> tuple[Schema, str]:
        """Resolve a local JSON Pointer reference.

        Args:
            ref: The local reference (e.g., '#/components/schemas/Pet').

        Returns:
            Tuple of (resolved Schema, schema name).

        Raises:
            SchemaReferenceError: If the reference path is invalid or
                                  the schema doesn't exist.
        """
        # Parse the JSON Pointer
        if not ref.startswith('#/'):
            raise SchemaReferenceError(ref, 'Local reference must start with #/')

        path_parts = ref[2:].split('/')

        # Currently only support components/schemas references
        if (
            len(path_parts) >= 3
            and path_parts[0] == 'components'
            and path_parts[1] == 'schemas'
        ):
            schema_name = path_parts[2]
            return self._get_component_schema(schema_name, ref)

        # Support components/parameters references
        if (
            len(path_parts) >= 3
            and path_parts[0] == 'components'
            and path_parts[1] == 'parameters'
        ):
            raise SchemaReferenceError(
                ref,
                'Parameter references should be resolved through the parameter resolver',
            )

        # Support components/responses references
        if (
            len(path_parts) >= 3
            and path_parts[0] == 'components'
            and path_parts[1] == 'responses'
        ):
            raise SchemaReferenceError(
                ref,
                'Response references should be resolved through the response resolver',
            )

        raise SchemaReferenceError(
            ref,
            f'Unsupported reference path. Only #/components/schemas/... is currently supported.',
        )

    def _get_component_schema(self, schema_name: str, ref: str) -> tuple[Schema, str]:
        """Get a schema from the components/schemas section.

        Args:
            schema_name: The name of the schema in components/schemas.
            ref: The original reference string (for error messages).

        Returns:
            Tuple of (Schema, sanitized schema name).

        Raises:
            SchemaReferenceError: If the schema doesn't exist.
        """
        if not self.openapi.components:
            raise SchemaReferenceError(
                ref,
                'Document has no components section',
            )

        if not self.openapi.components.schemas:
            raise SchemaReferenceError(
                ref,
                'Document has no schemas in components',
            )

        schemas = self.openapi.components.schemas

        if schema_name not in schemas:
            available = ', '.join(sorted(schemas.keys())[:10])
            if len(schemas) > 10:
                available += f', ... ({len(schemas)} total)'
            raise SchemaReferenceError(
                ref,
                f"Schema '{schema_name}' not found. Available schemas: {available}",
            )

        return schemas[schema_name], self._sanitize_name(schema_name)

    def get_all_schemas(self) -> dict[str, Schema]:
        """Get all schemas defined in the components/schemas section.

        Returns:
            Dictionary mapping schema names to Schema objects.
            Returns an empty dict if no schemas are defined.
        """
        if not self.openapi.components or not self.openapi.components.schemas:
            return {}
        return dict(self.openapi.components.schemas)

    def get_schema_names(self) -> list[str]:
        """Get the names of all schemas in the document.

        Returns:
            List of schema names, sorted alphabetically.
        """
        return sorted(self.get_all_schemas().keys())

    def has_schema(self, name: str) -> bool:
        """Check if a schema exists in the document.

        Args:
            name: The schema name to check.

        Returns:
            True if the schema exists, False otherwise.
        """
        return name in self.get_all_schemas()

    def clear_cache(self) -> None:
        """Clear the reference resolution cache."""
        self._cache.clear()

    @staticmethod
    def _sanitize_name(name: str | None) -> str | None:
        """Sanitize a name to be a valid Python identifier.

        Args:
            name: The name to sanitize.

        Returns:
            Sanitized name, or None if input is None.
        """
        if name is None:
            return None

        # Import here to avoid circular imports
        from otterapi.codegen.utils import sanitize_identifier

        return sanitize_identifier(name)

    def resolve_all_refs_in_schema(self, schema: Schema | Reference) -> Schema:
        """Recursively resolve all $ref references in a schema.

        This is useful when you need a fully resolved schema without
        any remaining references.

        Args:
            schema: The schema (potentially containing references) to resolve.

        Returns:
            The resolved schema with all references expanded.

        Note:
            This does not modify the original schema; references are
            resolved on access. For deeply nested schemas, this may
            result in multiple resolutions of the same reference.
        """
        if isinstance(schema, Reference):
            resolved, _ = self.resolve_reference(schema)
            return resolved
        return schema

    def get_reference_target(self, ref: str) -> str:
        """Extract the target name from a $ref string.

        Args:
            ref: The $ref string (e.g., '#/components/schemas/Pet').

        Returns:
            The target name (e.g., 'Pet').

        Raises:
            SchemaReferenceError: If the reference format is invalid.
        """
        if not ref.startswith('#/components/schemas/'):
            raise SchemaReferenceError(
                ref,
                'Can only extract target from #/components/schemas/ references',
            )
        return ref.split('/')[-1]
