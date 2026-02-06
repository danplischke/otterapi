"""Schema loading and resolution utilities for OpenAPI documents.

This module provides utilities for:
- Loading OpenAPI schemas from various sources (URLs, local files, YAML/JSON)
- Automatic version detection and upgrading of older OpenAPI/Swagger specifications
- Resolving $ref references and managing schema lookups in OpenAPI documents
"""

import json
import logging
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

import httpx
import yaml
from pydantic import TypeAdapter

from otterapi.exceptions import (
    SchemaLoadError,
    SchemaReferenceError,
    SchemaValidationError,
)
from otterapi.openapi import UniversalOpenAPI
from otterapi.openapi.v3_2 import Reference, Schema
from otterapi.openapi.v3_2.v3_2 import OpenAPI as OpenAPIv3

logger = logging.getLogger(__name__)

__all__ = [
    'SchemaLoader',
    'SchemaResolver',
]


# =============================================================================
# Schema Loader
# =============================================================================


class SchemaLoader:
    """Loads OpenAPI schemas from URLs or file paths.

    This class provides a unified interface for loading OpenAPI schemas
    from different sources (HTTP URLs or local files), supporting both
    JSON and YAML formats. It also handles automatic version detection
    and upgrading of Swagger 2.0 and older OpenAPI 3.x specifications.

    Features:
        - Load from URLs (http/https) or local file paths
        - Support for both JSON and YAML formats
        - Automatic version detection (Swagger 2.0, OpenAPI 3.0, 3.1, 3.2)
        - Automatic upgrade to OpenAPI 3.2 for code generation
        - External $ref resolution for URLs and relative files
        - Caching of loaded external schemas

    Example:
        >>> loader = SchemaLoader()
        >>> schema = loader.load('https://api.example.com/openapi.json')
        >>> # or
        >>> schema = loader.load('/path/to/openapi.yaml')
        >>> # or with external ref resolution
        >>> loader = SchemaLoader(resolve_external_refs=True)
        >>> schema = loader.load('./api.yaml')
    """

    def __init__(
        self,
        http_client: httpx.Client | None = None,
        resolve_external_refs: bool = False,
        base_path: str | Path | None = None,
    ):
        """Initialize the schema loader.

        Args:
            http_client: Optional HTTP client to use for URL requests.
                        If not provided, a default client will be created.
            resolve_external_refs: Whether to resolve external $ref references.
                                  If True, external URLs and relative file paths
                                  will be loaded and inlined.
            base_path: Base path for resolving relative file references.
                      Defaults to current working directory.
        """
        self._http_client = http_client
        self._resolve_external_refs = resolve_external_refs
        self._base_path = Path(base_path) if base_path else Path.cwd()
        self._external_cache: dict[str, dict] = {}
        self._upgrade_warnings: list[str] = []

    def load(self, source: str) -> OpenAPIv3:
        """Load and validate an OpenAPI schema from a URL or file path.

        This method:
        1. Loads the content from the source (URL or file)
        2. Parses JSON or YAML content
        3. Detects the OpenAPI version
        4. Optionally resolves external $ref references
        5. Upgrades older versions to OpenAPI 3.2
        6. Validates the final schema

        Args:
            source: URL or file path to the OpenAPI schema.

        Returns:
            Validated OpenAPIv3 (3.2) object.

        Raises:
            SchemaLoadError: If the schema cannot be loaded from the source.
            SchemaValidationError: If the schema is not valid OpenAPI.
        """
        try:
            if self._is_url(source):
                content = self._load_from_url(source)
                self._current_base_url = source.rsplit('/', 1)[0] + '/'
            else:
                content = self._load_from_file(source)
                self._current_base_url = None
                source_path = Path(source)
                if source_path.is_absolute():
                    self._base_path = source_path.parent
                else:
                    self._base_path = (Path.cwd() / source_path).parent

            if self._resolve_external_refs:
                content = self._resolve_refs(content, source)

            return self._validate_and_upgrade(content, source)

        except SchemaLoadError:
            raise
        except SchemaValidationError:
            raise
        except Exception as e:
            raise SchemaLoadError(source, cause=e)

    def get_upgrade_warnings(self) -> list[str]:
        """Get any warnings generated during schema upgrade."""
        return self._upgrade_warnings.copy()

    def get_detected_version(self, content: dict) -> str:
        """Detect the OpenAPI/Swagger version from schema content."""
        if 'swagger' in content:
            return content.get('swagger', '2.0')
        openapi_version = content.get('openapi', '3.0.0')
        if openapi_version.startswith('3.0'):
            return '3.0'
        elif openapi_version.startswith('3.1'):
            return '3.1'
        elif openapi_version.startswith('3.2'):
            return '3.2'
        return openapi_version

    def _is_url(self, text: str) -> bool:
        """Check if a string is a URL."""
        try:
            result = urlparse(text)
            return result.scheme in ('http', 'https')
        except ValueError:
            return False

    def _load_from_url(self, url: str) -> dict:
        """Load schema content from a URL."""
        try:
            if self._http_client:
                response = self._http_client.get(url)
            else:
                response = httpx.get(url, follow_redirects=True, timeout=30.0)

            response.raise_for_status()
            content_type = response.headers.get('content-type', '')
            content = response.text

            if 'yaml' in content_type or url.endswith(('.yaml', '.yml')):
                return yaml.safe_load(content)
            else:
                return json.loads(content)

        except httpx.HTTPError as e:
            raise SchemaLoadError(url, cause=e)
        except (json.JSONDecodeError, yaml.YAMLError) as e:
            raise SchemaLoadError(url, cause=e)

    def _load_from_file(self, file_path: str) -> dict:
        """Load schema content from a file."""
        path = Path(file_path)
        if not path.is_absolute():
            path = self._base_path / path

        if not path.exists():
            raise SchemaLoadError(
                str(file_path), cause=FileNotFoundError(f'File not found: {path}')
            )

        try:
            content = path.read_text(encoding='utf-8')
            if path.suffix.lower() in ('.yaml', '.yml'):
                return yaml.safe_load(content)
            else:
                return json.loads(content)
        except (json.JSONDecodeError, yaml.YAMLError) as e:
            raise SchemaLoadError(str(file_path), cause=e)
        except OSError as e:
            raise SchemaLoadError(str(file_path), cause=e)

    def _resolve_refs(self, content: dict, source: str) -> dict:
        """Resolve all external $ref references in the schema."""
        return self._resolve_refs_recursive(content, source, set())

    def _resolve_refs_recursive(self, obj: Any, base: str, visited: set[str]) -> Any:
        """Recursively resolve external references."""
        if isinstance(obj, dict):
            if '$ref' in obj:
                ref = obj['$ref']
                if not ref.startswith('#'):
                    return self._resolve_external_ref(ref, base, visited)
            return {
                k: self._resolve_refs_recursive(v, base, visited)
                for k, v in obj.items()
            }
        elif isinstance(obj, list):
            return [self._resolve_refs_recursive(item, base, visited) for item in obj]
        return obj

    def _resolve_external_ref(self, ref: str, base: str, visited: set[str]) -> dict:
        """Resolve an external $ref reference."""
        if '#' in ref:
            file_part, pointer = ref.split('#', 1)
        else:
            file_part, pointer = ref, ''

        if self._is_url(file_part):
            location = file_part
        elif self._is_url(base):
            location = urljoin(base, file_part)
        else:
            base_path = Path(base).parent if not Path(base).is_dir() else Path(base)
            location = str(base_path / file_part)

        cache_key = f'{location}#{pointer}'
        if cache_key in visited:
            logger.warning(f'Circular reference detected: {cache_key}')
            return {'$ref': ref}

        visited = visited | {cache_key}

        if location not in self._external_cache:
            try:
                if self._is_url(location):
                    self._external_cache[location] = self._load_from_url(location)
                else:
                    self._external_cache[location] = self._load_from_file(location)
            except SchemaLoadError:
                logger.warning(f'Failed to resolve external reference: {ref}')
                return {'$ref': ref}

        content = self._external_cache[location]

        if pointer:
            content = self._resolve_json_pointer(content, pointer)

        return self._resolve_refs_recursive(content, location, visited)

    def _resolve_json_pointer(self, obj: Any, pointer: str) -> Any:
        """Resolve a JSON pointer within an object."""
        if not pointer or pointer == '/':
            return obj

        parts = pointer.strip('/').split('/')
        current = obj

        for part in parts:
            part = part.replace('~1', '/').replace('~0', '~')
            if isinstance(current, dict):
                if part not in current:
                    raise ValueError(f'JSON pointer path not found: {pointer}')
                current = current[part]
            elif isinstance(current, list):
                try:
                    index = int(part)
                    current = current[index]
                except (ValueError, IndexError):
                    raise ValueError(f'JSON pointer path not found: {pointer}')
            else:
                raise ValueError(f'JSON pointer path not found: {pointer}')

        return current

    def _validate_and_upgrade(self, content: dict, source: str) -> OpenAPIv3:
        """Validate and upgrade schema content to OpenAPI 3.2."""
        self._upgrade_warnings = []

        try:
            universal: UniversalOpenAPI = TypeAdapter(UniversalOpenAPI).validate_python(
                content
            )
            schema = universal.root

            if isinstance(schema, OpenAPIv3):
                return schema

            while not isinstance(schema, OpenAPIv3):
                schema, warnings = schema.upgrade()
                self._upgrade_warnings.extend(warnings)

            return schema

        except Exception as e:
            raise SchemaValidationError(source, errors=[str(e)])


# =============================================================================
# Schema Resolver
# =============================================================================


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

    def __init__(self, openapi: OpenAPIv3):
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
            'Unsupported reference path. Only #/components/schemas/... is currently supported.',
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
