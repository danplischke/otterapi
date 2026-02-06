"""Type definitions and generation for OtterAPI code generation.

This module provides:
- Type dataclasses for representing generated types, parameters, responses, and endpoints
- TypeGenerator for creating Pydantic models from OpenAPI schemas
- TypeRegistry for managing generated types and their dependencies
- ModelNameCollector for tracking model usage in generated code
"""

import ast
import dataclasses
from collections.abc import Iterable, Iterator
from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field, RootModel

from otterapi.codegen.ast_utils import _call, _name, _subscript, _union_expr
from otterapi.codegen.utils import (
    OpenAPIProcessor,
    sanitize_identifier,
    sanitize_parameter_field_name,
)
from otterapi.openapi.v3_2 import Reference, Schema, Type as DataType

__all__ = [
    'Type',
    'Parameter',
    'ResponseInfo',
    'RequestBodyInfo',
    'Endpoint',
    'TypeGenerator',
    'TypeInfo',
    'TypeRegistry',
    'ModelNameCollector',
    'collect_used_model_names',
]

_PRIMITIVE_TYPE_MAP = {
    ('string', None): str,
    ('string', 'date-time'): datetime,
    ('string', 'date'): datetime,
    ('string', 'uuid'): UUID,
    ('integer', None): int,
    ('integer', 'int32'): int,
    ('integer', 'int64'): int,
    ('number', None): float,
    ('number', 'float'): float,
    ('number', 'double'): float,
    ('boolean', None): bool,
    ('null', None): None,
    (None, None): None,
}


@dataclasses.dataclass
class Type:
    reference: str | None  # reference is None if type is 'primitive'
    name: str | None
    type: Literal['primitive', 'root', 'model']
    annotation_ast: ast.expr | ast.stmt | None = dataclasses.field(default=None)
    implementation_ast: ast.expr | ast.stmt | None = dataclasses.field(default=None)
    dependencies: set[str] = dataclasses.field(default_factory=set)
    implementation_imports: dict[str, set[str]] = dataclasses.field(
        default_factory=dict
    )
    annotation_imports: dict[str, set[str]] = dataclasses.field(default_factory=dict)

    def __hash__(self):
        """Make Type hashable based on its name (for use in sets/dicts)."""
        # We only hash based on name since we use name as the key in the types dict
        return hash(self.name)

    def add_dependency(self, type_: 'Type') -> None:
        self.dependencies.add(type_.name)
        for dep in type_.dependencies:
            self.dependencies.add(dep)

    def add_implementation_import(self, module: str, name: str | Iterable[str]) -> None:
        # Skip builtins - they don't need to be imported
        if module == 'builtins':
            return

        if isinstance(name, str):
            name = [name]

        if module not in self.implementation_imports:
            self.implementation_imports[module] = set()

        for n in name:
            self.implementation_imports[module].add(n)

    def add_annotation_import(self, module: str, name: str | Iterable[str]) -> None:
        # Skip builtins - they don't need to be imported
        if module == 'builtins':
            return

        if isinstance(name, str):
            name = [name]

        if module not in self.annotation_imports:
            self.annotation_imports[module] = set()

        for n in name:
            self.annotation_imports[module].add(n)

    def copy_imports_from_sub_types(self, types: Iterable['Type']):
        for t in types:
            for module, names in t.annotation_imports.items():
                self.add_annotation_import(module, names)

            for module, names in t.implementation_imports.items():
                self.add_implementation_import(module, names)

    def __eq__(self, other):
        """Deep comparison of Type objects, including AST nodes."""
        if not isinstance(other, Type):
            return False

        # Compare simple fields
        if (
            self.reference != other.reference
            or self.name != other.name
            or self.type != other.type
        ):
            return False

        # Compare AST nodes by dumping them to strings
        if ast.dump(self.annotation_ast) != ast.dump(other.annotation_ast):
            return False

        # Compare implementation AST (can be None)
        if self.implementation_ast is None and other.implementation_ast is None:
            pass  # Both None, equal
        elif self.implementation_ast is None or other.implementation_ast is None:
            return False  # One is None, other isn't
        else:
            if ast.dump(self.implementation_ast) != ast.dump(other.implementation_ast):
                return False

        # Compare imports and dependencies
        if (
            self.dependencies != other.dependencies
            or self.implementation_imports != other.implementation_imports
            or self.annotation_imports != other.annotation_imports
        ):
            return False

        return True


@dataclasses.dataclass
class Parameter:
    name: str
    name_sanitized: str
    location: Literal['query', 'path', 'header', 'cookie', 'body']
    required: bool
    type: Type | None = None
    description: str | None = None


@dataclasses.dataclass
class ResponseInfo:
    """Information about a response for a specific status code.

    Attributes:
        status_code: The HTTP status code for this response.
        content_type: The content type (e.g., 'application/json', 'application/octet-stream').
        type: The Type object for JSON responses, or None for raw responses.
    """

    status_code: int
    content_type: str
    type: Type | None = None

    @property
    def is_json(self) -> bool:
        """Check if this is a JSON response."""
        return self.content_type in (
            'application/json',
            'text/json',
        ) or self.content_type.endswith('+json')

    @property
    def is_binary(self) -> bool:
        """Check if this is a binary response (file download)."""
        binary_types = (
            'application/octet-stream',
            'application/pdf',
            'application/zip',
            'application/gzip',
            'application/x-tar',
            'application/x-rar-compressed',
        )
        binary_prefixes = ('image/', 'audio/', 'video/', 'application/vnd.')
        return self.content_type in binary_types or any(
            self.content_type.startswith(p) for p in binary_prefixes
        )

    @property
    def is_text(self) -> bool:
        """Check if this is a plain text response."""
        return self.content_type.startswith('text/') and not self.is_json

    @property
    def is_raw(self) -> bool:
        """Check if this is an unknown content type that should return the raw httpx.Response."""
        return not (self.is_json or self.is_binary or self.is_text)


@dataclasses.dataclass
class RequestBodyInfo:
    """Information about a request body including its content type.

    Attributes:
        content_type: The content type (e.g., 'application/json', 'multipart/form-data').
        type: The Type object for the body schema, or None if no schema.
        required: Whether the request body is required.
        description: Optional description of the request body.
    """

    content_type: str
    type: Type | None = None
    required: bool = False
    description: str | None = None

    @property
    def is_json(self) -> bool:
        """Check if this is a JSON request body."""
        return self.content_type in (
            'application/json',
            'text/json',
        ) or self.content_type.endswith('+json')

    @property
    def is_form(self) -> bool:
        """Check if this is a form-encoded request body."""
        return self.content_type == 'application/x-www-form-urlencoded'

    @property
    def is_multipart(self) -> bool:
        """Check if this is a multipart form data request body."""
        return self.content_type == 'multipart/form-data'

    @property
    def is_binary(self) -> bool:
        """Check if this is a binary request body."""
        return self.content_type in ('application/octet-stream',)

    @property
    def httpx_param_name(self) -> str:
        """Get the httpx parameter name for this content type.

        Returns:
            The appropriate httpx parameter name: 'json', 'data', 'files', or 'content'.
        """
        if self.is_json:
            return 'json'
        elif self.is_form:
            return 'data'
        elif self.is_multipart:
            return 'files'
        elif self.is_binary:
            return 'content'
        else:
            return 'content'


@dataclasses.dataclass
class Endpoint:
    """Represents a generated API endpoint with sync and async functions."""

    # AST nodes
    sync_ast: ast.FunctionDef
    async_ast: ast.AsyncFunctionDef

    # Function names
    sync_fn_name: str
    async_fn_name: str

    # Endpoint metadata
    name: str
    method: str = ''
    path: str = ''
    description: str | None = None
    tags: list[str] | None = None  # OpenAPI tags for module splitting

    # Parameters and body
    parameters: list['Parameter'] | None = None
    request_body: 'RequestBodyInfo | None' = None

    # Response info
    response_type: 'Type | None' = None
    response_infos: list['ResponseInfo'] | None = None

    # Imports needed
    imports: dict[str, set[str]] = dataclasses.field(default_factory=dict)

    @property
    def fn(self) -> ast.FunctionDef:
        """Alias for sync_ast."""
        return self.sync_ast

    @property
    def async_fn(self) -> ast.AsyncFunctionDef:
        """Alias for async_ast."""
        return self.async_ast

    def add_imports(self, imports: list[dict[str, set[str]]]):
        for imports_ in imports:
            for module, names in imports_.items():
                if module not in self.imports:
                    self.imports[module] = set()
                self.imports[module].update(names)


@dataclasses.dataclass
class TypeGenerator(OpenAPIProcessor):
    types: dict[str, Type] = dataclasses.field(default_factory=dict)

    def add_type(self, type_: Type, base_name: str | None = None) -> Type:
        """Add a type to the registry. If a type with the same name but different definition
        already exists, generate a unique name using the base_name prefix.
        Returns the type (potentially with a modified name).
        """
        # Skip types without names (primitive types, inline types, etc.)
        if not type_.name:
            return type_

        # If type with same name and same definition exists, just return the existing one
        if type_.name in self.types:
            existing = self.types[type_.name]
            if existing == type_:
                # Same type already registered, return the existing one
                # This avoids creating Detail20, Detail21 when they're identical
                return existing
            else:
                # Different definition with same name - generate a unique name
                if base_name:
                    # Use base_name as prefix for endpoint-specific types
                    unique_name = f'{base_name}{type_.name}'
                    if unique_name not in self.types:
                        type_.name = unique_name
                        type_.annotation_ast = _name(unique_name)
                        # Update the implementation_ast name if it's a ClassDef
                        if isinstance(type_.implementation_ast, ast.ClassDef):
                            type_.implementation_ast.name = unique_name
                    else:
                        # Check if even the base_name version is the same
                        if (
                            unique_name in self.types
                            and self.types[unique_name] == type_
                        ):
                            return self.types[unique_name]
                        # If even that exists with different def, add a counter
                        counter = 1
                        while f'{unique_name}{counter}' in self.types:
                            candidate = f'{unique_name}{counter}'
                            if self.types[candidate] == type_:
                                return self.types[candidate]
                            counter += 1
                        unique_name = f'{unique_name}{counter}'
                        type_.name = unique_name
                        type_.annotation_ast = _name(unique_name)
                        if isinstance(type_.implementation_ast, ast.ClassDef):
                            type_.implementation_ast.name = unique_name
                else:
                    # No base_name provided, just add a counter
                    counter = 1
                    original_name = type_.name
                    while f'{original_name}{counter}' in self.types:
                        candidate = f'{original_name}{counter}'
                        if self.types[candidate] == type_:
                            # Found identical type with numbered name
                            return self.types[candidate]
                        counter += 1
                    unique_name = f'{original_name}{counter}'
                    type_.name = unique_name
                    type_.annotation_ast = _name(unique_name)
                    if isinstance(type_.implementation_ast, ast.ClassDef):
                        type_.implementation_ast.name = unique_name

        self.types[type_.name] = type_
        return type_

    def _resolve_reference(self, reference: Reference | Schema) -> tuple[Schema, str]:
        if hasattr(reference, 'ref'):
            if not reference.ref.startswith('#/components/schemas/'):
                raise ValueError(f'Unsupported reference format: {reference.ref}')

            schema_name = reference.ref.split('/')[-1]
            schemas = self.openapi.components.schemas

            if schema_name not in schemas:
                raise ValueError(
                    f"Referenced schema '{schema_name}' not found in components.schemas"
                )

            return schemas[schema_name], sanitize_identifier(schema_name)
        return reference, sanitize_identifier(
            reference.title
        ) if reference.title else None

    def _create_enum_type(
        self,
        schema: Schema,
        name: str | None = None,
        base_name: str | None = None,
        field_name: str | None = None,
    ) -> Type:
        """Create an Enum class for schema with enum values.

        Args:
            schema: The schema containing enum values.
            name: Optional explicit name for the enum.
            base_name: Optional base name prefix (e.g., parent model name).
            field_name: Optional field name this enum is used for (e.g., 'status').

        Returns:
            A Type representing the generated Enum class.
        """
        # Determine enum name - prefer schema title, then derive from context
        enum_name = name or (
            sanitize_identifier(schema.title) if schema.title else None
        )
        if not enum_name:
            # Generate name from field_name with base_name context for uniqueness
            if field_name:
                # e.g., Pet + status -> 'PetStatus', Order + status -> 'OrderStatus'
                # Capitalize the field part to ensure proper PascalCase
                field_part = sanitize_identifier(field_name)
                # Ensure first letter is capitalized for PascalCase
                if field_part:
                    field_part = field_part[0].upper() + field_part[1:]
                if base_name:
                    base_part = sanitize_identifier(base_name)
                    enum_name = f'{base_part}{field_part}'
                else:
                    enum_name = field_part
            elif base_name:
                enum_name = f'{sanitize_identifier(base_name)}Enum'
            else:
                enum_name = 'AutoEnum'

        # Create a hashable key from enum values to detect duplicates
        enum_values_key = tuple(sorted(str(v) for v in schema.enum if v is not None))

        # Check if an identical enum already exists
        for existing_name, existing_type in self.types.items():
            if existing_type.type == 'model' and isinstance(
                existing_type.implementation_ast, ast.ClassDef
            ):
                # Check if it's an Enum class with same values
                existing_class = existing_type.implementation_ast
                if any(
                    isinstance(base, ast.Name) and base.id == 'Enum'
                    for base in existing_class.bases
                ):
                    # Extract values from existing enum
                    existing_values = []
                    for node in existing_class.body:
                        if isinstance(node, ast.Assign) and node.value:
                            if isinstance(node.value, ast.Constant):
                                existing_values.append(str(node.value.value))
                    if tuple(sorted(existing_values)) == enum_values_key:
                        # Reuse existing enum
                        return existing_type

        # Ensure the name is unique
        if enum_name in self.types:
            counter = 1
            original_name = enum_name
            while f'{original_name}{counter}' in self.types:
                counter += 1
            enum_name = f'{original_name}{counter}'

        # Build enum members: NAME = 'value'
        # For string enums, use the value as the member name (sanitized)
        enum_body = []
        seen_member_names: dict[str, int] = {}  # Track seen names to handle duplicates
        for value in schema.enum:
            if value is None:
                continue  # Skip None values in enums
            # Create a valid Python identifier for the enum member
            if isinstance(value, str):
                member_name = sanitize_identifier(value).upper()
                # If the sanitized name starts with a digit, prefix with underscore
                if member_name and member_name[0].isdigit():
                    member_name = f'_{member_name}'
            else:
                # For numeric enums, create VALUE_X names
                member_name = f'VALUE_{value}'

            # Handle duplicate member names (e.g., 'mesoderm' and 'Mesoderm' both -> 'MESODERM')
            if member_name in seen_member_names:
                seen_member_names[member_name] += 1
                member_name = f'{member_name}_{seen_member_names[member_name]}'
            else:
                seen_member_names[member_name] = 0

            enum_body.append(
                ast.Assign(
                    targets=[ast.Name(id=member_name, ctx=ast.Store())],
                    value=ast.Constant(value=value),
                )
            )

        # If no valid members, fall back to Literal
        if not enum_body:
            return self._create_literal_type(schema)

        # Create the Enum class
        # class EnumName(str, Enum):  # str mixin for string enums
        #     MEMBER = 'value'
        bases = (
            [_name('str'), _name('Enum')]
            if schema.type and schema.type.value == 'string'
            else [_name('Enum')]
        )

        enum_class = ast.ClassDef(
            name=enum_name,
            bases=bases,
            keywords=[],
            body=enum_body,
            decorator_list=[],
            type_params=[],
        )

        type_ = Type(
            reference=None,
            name=enum_name,
            annotation_ast=_name(enum_name),
            implementation_ast=enum_class,
            type='model',  # Treat as model so it gets included in models.py
        )
        type_.add_implementation_import('enum', 'Enum')

        # Register the type
        self.types[enum_name] = type_

        return type_

    def _create_literal_type(self, schema: Schema) -> Type:
        """Create a Literal type for enum values (fallback)."""
        literal_values = [ast.Constant(value=v) for v in schema.enum]
        type_ = Type(
            None,
            sanitize_identifier(schema.title) if schema.title else None,
            annotation_ast=_subscript(
                'Literal', ast.Tuple(elts=literal_values, ctx=ast.Load())
            ),
            implementation_ast=None,
            type='primitive',
        )
        type_.add_annotation_import('typing', 'Literal')
        return type_

    def _is_nullable(self, schema: Schema) -> bool:
        """Check if a schema represents a nullable type.

        In OpenAPI 3.1+, nullable is expressed via type arrays like ["string", "null"].
        """
        if isinstance(schema.type, list):
            return any(
                t == DataType.null or (hasattr(t, 'value') and t.value == 'null')
                for t in schema.type
            )
        return False

    def _get_non_null_type(self, schema: Schema) -> DataType | None:
        """Extract the non-null type from a potentially nullable schema."""
        if isinstance(schema.type, list):
            for t in schema.type:
                if t != DataType.null and (
                    not hasattr(t, 'value') or t.value != 'null'
                ):
                    return t
            return None
        return schema.type

    def _make_nullable_type(self, base_type: Type) -> Type:
        """Wrap a type annotation to make it nullable (T | None)."""
        nullable_ast = _union_expr([base_type.annotation_ast, ast.Constant(value=None)])

        type_ = Type(
            reference=base_type.reference,
            name=base_type.name,
            annotation_ast=nullable_ast,
            implementation_ast=base_type.implementation_ast,
            type=base_type.type,
            dependencies=base_type.dependencies.copy(),
            implementation_imports=base_type.implementation_imports.copy(),
            annotation_imports=base_type.annotation_imports.copy(),
        )
        return type_

    def _get_primitive_type_ast(
        self,
        schema: Schema,
        base_name: str | None = None,
        field_name: str | None = None,
    ) -> Type:
        # Handle enum types - generate Enum class
        if schema.enum:
            return self._create_enum_type(
                schema, base_name=base_name, field_name=field_name
            )

        # Check for nullable type (type array with null)
        is_nullable = self._is_nullable(schema)
        actual_type = self._get_non_null_type(schema)

        # Fix: schema.type is a Type enum, need to use .value for string lookup
        type_value = actual_type.value if actual_type else None
        key = (type_value, schema.format or None)
        mapped = _PRIMITIVE_TYPE_MAP.get(key, Any)

        type_ = Type(
            None,
            sanitize_identifier(schema.title) if schema.title else None,
            annotation_ast=_name(mapped.__name__ if mapped is not None else 'None'),
            implementation_ast=None,
            type='primitive',
        )

        if mapped is not None and mapped.__module__ != 'builtins':
            type_.add_annotation_import(mapped.__module__, mapped.__name__)

        # Wrap in Union with None if nullable
        if is_nullable:
            type_ = self._make_nullable_type(type_)

        return type_

    def _create_pydantic_field(
        self,
        field_name: str,
        field_schema: Schema,
        field_type: Type,
        is_required: bool = False,
        is_nullable: bool = False,
    ) -> str:
        if hasattr(field_schema, 'ref'):
            field_schema, _ = self._resolve_reference(field_schema)

        field_keywords = list()

        sanitized_field_name = sanitize_parameter_field_name(field_name)

        # Determine the annotation - wrap in Union with None if nullable
        annotation_ast = field_type.annotation_ast
        if is_nullable and not self._type_already_nullable(field_type):
            annotation_ast = _union_expr(
                [field_type.annotation_ast, ast.Constant(value=None)]
            )

        value = None
        if field_schema.default is not None and isinstance(
            field_schema.default, (str, int, float, bool)
        ):
            field_keywords.append(
                ast.keyword(arg='default', value=ast.Constant(field_schema.default))
            )
        elif field_schema.default is None and not is_required:
            # Only add default=None for optional (not required) fields
            # Nullable but required fields should NOT have a default
            field_keywords.append(ast.keyword(arg='default', value=ast.Constant(None)))

        if sanitized_field_name != field_name:
            field_keywords.append(
                ast.keyword(
                    arg='alias',
                    value=ast.Constant(field_name),  # original name before adding _
                )
            )
            field_name = sanitized_field_name

        if field_keywords:
            value = _call(
                func=_name(Field.__name__),
                keywords=field_keywords,
            )

            field_type.add_implementation_import(
                module=Field.__module__, name=Field.__name__
            )

        return ast.AnnAssign(
            target=_name(field_name),
            annotation=annotation_ast,
            value=value,
            simple=1,
        )

    def _type_already_nullable(self, type_: Type) -> bool:
        """Check if a type annotation already includes None."""
        if isinstance(type_.annotation_ast, ast.Subscript):
            # Check if it's Union[..., None]
            if isinstance(type_.annotation_ast.value, ast.Name):
                if type_.annotation_ast.value.id == 'Union':
                    if isinstance(type_.annotation_ast.slice, ast.Tuple):
                        for elt in type_.annotation_ast.slice.elts:
                            if isinstance(elt, ast.Constant) and elt.value is None:
                                return True
        return False

    def _create_pydantic_root_model(
        self,
        schema: Schema,
        item_type: Type | None = None,
        name: str | None = None,
        base_name: str | None = None,
    ) -> Type:
        name = (
            name
            or base_name
            or (sanitize_identifier(schema.title) if schema.title else None)
        )
        if not name:
            raise ValueError('Root model must have a name')

        model = ast.ClassDef(
            name=name,
            bases=[_subscript(RootModel.__name__, item_type.annotation_ast)],
            keywords=[],
            body=[ast.Pass()],
            decorator_list=[],
            type_params=[],
        )

        type_ = Type(
            reference=None,
            name=name,
            annotation_ast=_name(name),
            implementation_ast=model,
            type='root',
        )
        type_.add_implementation_import(
            module=RootModel.__module__, name=RootModel.__name__
        )
        type_.copy_imports_from_sub_types([item_type] if item_type else [])
        if item_type is not None:
            type_.add_dependency(item_type)
        type_ = self.add_type(type_, base_name=base_name)

        return type_

    def _create_pydantic_model(
        self, schema: Schema, name: str | None = None, base_name: str | None = None
    ) -> Type:
        base_bases = []
        if schema.allOf:
            for base_schema in schema.allOf:
                base = self._create_object_type(schema=base_schema, base_name=base_name)
                base_bases.append(base)

        if schema.anyOf or schema.oneOf:
            # Use schema_to_type for each variant to properly handle primitives, objects, etc.
            types_ = [
                self.schema_to_type(t, base_name=base_name)
                for t in (schema.anyOf or schema.oneOf)
            ]

            union_type = Type(
                reference=None,
                name=None,  # Union type doesn't need a name, it's used inline
                annotation_ast=_union_expr(types=[t.annotation_ast for t in types_]),
                implementation_ast=None,
                type='primitive',
            )
            union_type.copy_imports_from_sub_types(types_)
            return union_type

        name = name or (
            sanitize_identifier(schema.title) if schema.title else 'UnnamedModel'
        )

        bases = [b.name for b in base_bases] or [BaseModel.__name__]
        bases = [_name(base) for base in bases]

        body = []
        field_types = []
        # Fix: Get the required fields from the parent schema's required array
        required_fields = set(schema.required or [])
        for property_name, property_schema in (schema.properties or {}).items():
            # Resolve reference to check for nullable
            resolved_schema = property_schema
            if hasattr(property_schema, 'ref') and property_schema.ref:
                resolved_schema, _ = self._resolve_reference(property_schema)

            # Check if field is nullable (type array with null)
            is_nullable = (
                self._is_nullable(resolved_schema) if resolved_schema else False
            )

            type_ = self.schema_to_type(
                property_schema, base_name=base_name, field_name=property_name
            )
            is_required = property_name in required_fields
            field = self._create_pydantic_field(
                property_name, property_schema, type_, is_required, is_nullable
            )

            body.append(field)
            field_types.append(type_)

        # Add deprecation docstring if schema is deprecated
        if schema.deprecated:
            deprecation_doc = ast.Expr(
                value=ast.Constant(
                    value=f'{name} is deprecated.\n\n.. deprecated::\n    This model is deprecated.'
                )
            )
            body = [deprecation_doc] + body if body else [deprecation_doc]

        model = ast.ClassDef(
            name=name,
            bases=bases,
            keywords=[],
            body=body or [ast.Pass()],
            decorator_list=[],
            type_params=[],
        )

        type_ = Type(
            reference=None,
            name=name,
            annotation_ast=_name(name),
            implementation_ast=model,
            dependencies=set(),
            type='model',
        )

        # Add base class dependencies
        if base_bases:
            for base in base_bases:
                type_.add_dependency(base)

        # Add field type dependencies
        for field_type in field_types:
            if field_type.name:
                type_.dependencies.add(field_type.name)
                type_.dependencies.update(field_type.dependencies)

        type_.add_implementation_import(
            module=BaseModel.__module__, name=BaseModel.__name__
        )
        type_.add_implementation_import(module=Field.__module__, name=Field.__name__)
        type_.copy_imports_from_sub_types(field_types)

        type_ = self.add_type(type_, base_name=base_name)
        return type_

    def _create_array_type(
        self, schema: Schema, name: str | None = None, base_name: str | None = None
    ) -> Type:
        if schema.type != DataType.array:
            raise ValueError('Schema is not an array')

        if not schema.items:
            type_ = Type(
                None,
                None,
                _subscript(
                    list.__name__,
                    ast.Name(id=Any.__name__, ctx=ast.Load()),
                ),
                'primitive',
            )

            type_.add_annotation_import(module=list.__module__, name=list.__name__)
            type_.add_annotation_import(module=Any.__module__, name=Any.__name__)

            return type_

        item_type = self.schema_to_type(schema.items, base_name=base_name)

        type_ = Type(
            None,
            None,
            annotation_ast=_subscript(
                list.__name__,
                item_type.annotation_ast,
            ),
            implementation_ast=None,
            type='primitive',
        )

        type_.add_annotation_import(list.__module__, list.__name__)
        type_.copy_imports_from_sub_types([item_type])

        if item_type:
            type_.add_dependency(item_type)

        return type_

    def _create_object_type(
        self,
        schema: Schema | Reference,
        name: str | None = None,
        base_name: str | None = None,
    ) -> Type:
        schema, schema_name = self._resolve_reference(schema)

        # Handle additionalProperties for dict-like types
        if (
            not schema.properties
            and not schema.allOf
            and not schema.anyOf
            and not schema.oneOf
        ):
            # Check for additionalProperties to determine value type
            value_type_ast = ast.Name(id=Any.__name__, ctx=ast.Load())
            value_type_imports: dict[str, set[str]] = {Any.__module__: {Any.__name__}}

            if (
                schema.additionalProperties is not None
                and schema.additionalProperties is not True
            ):
                if schema.additionalProperties is False:
                    # No additional properties allowed - still generate dict[str, Any]
                    pass
                elif isinstance(schema.additionalProperties, (Schema, Reference)):
                    # additionalProperties has a schema - use it for value type
                    additional_type = self.schema_to_type(
                        schema.additionalProperties, base_name=base_name
                    )
                    value_type_ast = additional_type.annotation_ast
                    value_type_imports = additional_type.annotation_imports.copy()

            type_ = Type(
                None,
                None,
                annotation_ast=_subscript(
                    dict.__name__,
                    ast.Tuple(
                        elts=[
                            ast.Name(id=str.__name__, ctx=ast.Load()),
                            value_type_ast,
                        ]
                    ),
                ),
                implementation_ast=None,
                type='primitive',
            )

            type_.add_annotation_import(dict.__module__, dict.__name__)
            for module, names in value_type_imports.items():
                for name_import in names:
                    type_.add_annotation_import(module, name_import)

            return type_

        return self._create_pydantic_model(
            schema, schema_name or name, base_name=base_name
        )

    def schema_to_type(
        self,
        schema: Schema | Reference,
        base_name: str | None = None,
        field_name: str | None = None,
    ) -> Type:
        if isinstance(schema, Reference):
            ref_name = schema.ref.split('/')[-1]
            sanitized_ref_name = sanitize_identifier(ref_name)
            if sanitized_ref_name in self.types:
                return self.types[sanitized_ref_name]

        schema, schema_name = self._resolve_reference(schema)

        # Use schema_name (from $ref) as base_name for nested types if available
        # This ensures enums inside Pet get names like "PetStatus" not "addPetRequestBodyStatus"
        effective_base_name = schema_name or base_name

        # TODO: schema.type can be array?
        if schema.type == DataType.array:
            type_ = self._create_array_type(
                schema=schema, name=schema_name, base_name=effective_base_name
            )
        elif schema.type == DataType.object or schema.type is None:
            type_ = self._create_object_type(
                schema, name=schema_name, base_name=effective_base_name
            )
        else:
            type_ = self._get_primitive_type_ast(
                schema, base_name=effective_base_name, field_name=field_name
            )

        return type_

    def get_sorted_types(self) -> list[Type]:
        """Returns the types sorted in dependency order using topological sort.
        Types with no dependencies come first.
        """
        sorted_types: list[Type] = []
        temp_mark: set[str] = set()
        perm_mark: set[str] = set()

        def visit(type_: Type):
            if type_.name in perm_mark:
                return
            if type_.name in temp_mark:
                raise ValueError(f'Cyclic dependency detected for type: {type_.name}')

            temp_mark.add(type_.name)

            for dep_name in type_.dependencies:
                if dep_name in self.types:
                    visit(self.types[dep_name])

            perm_mark.add(type_.name)
            temp_mark.remove(type_.name)
            sorted_types.append(type_)

        for type_ in self.types.values():
            if type_.name not in perm_mark:
                visit(type_)

        return list(reversed(sorted_types))


# =============================================================================
# Type Registry
# =============================================================================


@dataclasses.dataclass
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
    dependencies: set[str] = dataclasses.field(default_factory=set)
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
        self._by_reference: dict[str, str] = {}
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
        """Check if a type is registered."""
        return name in self._types

    def has_reference(self, reference: str) -> bool:
        """Check if a reference has been registered."""
        return reference in self._by_reference

    def get_type(self, name: str) -> TypeInfo | None:
        """Get a registered type by name."""
        return self._types.get(name)

    def get_type_by_reference(self, reference: str) -> TypeInfo | None:
        """Get a registered type by its OpenAPI reference."""
        name = self._by_reference.get(reference)
        if name:
            return self._types.get(name)
        return None

    def get_name_for_reference(self, reference: str) -> str | None:
        """Get the registered name for an OpenAPI reference."""
        return self._by_reference.get(reference)

    def get_all_types(self) -> dict[str, TypeInfo]:
        """Get all registered types."""
        return dict(self._types)

    def get_type_names(self) -> list[str]:
        """Get all registered type names, sorted alphabetically."""
        return sorted(self._types.keys())

    def add_dependency(self, type_name: str, depends_on: str) -> None:
        """Add a dependency relationship between types."""
        if type_name not in self._types:
            raise KeyError(f"Type '{type_name}' is not registered")
        self._types[type_name].dependencies.add(depends_on)

    def get_dependencies(self, type_name: str) -> set[str]:
        """Get all dependencies for a type."""
        if type_name not in self._types:
            raise KeyError(f"Type '{type_name}' is not registered")
        return self._types[type_name].dependencies.copy()

    def get_types_in_dependency_order(self) -> list[TypeInfo]:
        """Get all types sorted in dependency order.

        Types are sorted so that dependencies come before the types that
        depend on them.
        """
        result: list[TypeInfo] = []
        visited: set[str] = set()
        visiting: set[str] = set()

        def visit(name: str) -> None:
            if name in visited:
                return
            if name in visiting:
                return
            if name in self._primitive_types:
                return
            if name not in self._types:
                return

            visiting.add(name)
            type_info = self._types[name]

            for dep in type_info.dependencies:
                visit(dep)

            visiting.remove(name)
            visited.add(name)
            result.append(type_info)

        for name in sorted(self._types.keys()):
            visit(name)

        return result

    def mark_generated(self, name: str) -> None:
        """Mark a type as having its AST generated."""
        if name not in self._types:
            raise KeyError(f"Type '{name}' is not registered")
        self._types[name].is_generated = True

    def get_ungenerated_types(self) -> list[TypeInfo]:
        """Get all types that haven't been generated yet."""
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
        """Get all registered root models."""
        return [t for t in self._types.values() if t.is_root_model]

    def get_regular_models(self) -> list[TypeInfo]:
        """Get all registered non-root models."""
        return [t for t in self._types.values() if not t.is_root_model]

    def merge(self, other: 'TypeRegistry') -> None:
        """Merge another registry into this one."""
        for type_info in other:
            if type_info.name not in self._types:
                self._types[type_info.name] = type_info
                if type_info.reference:
                    self._by_reference[type_info.reference] = type_info.name


# =============================================================================
# Model Name Collector
# =============================================================================


class ModelNameCollector(ast.NodeVisitor):
    """AST visitor that collects model names from function definitions.

    This visitor walks AST nodes and identifies Name nodes that match
    a set of available model names, allowing us to determine which
    models are actually referenced in generated code.

    Example:
        >>> available = {'Pet', 'User', 'Order'}
        >>> collector = ModelNameCollector(available)
        >>> collector.visit(some_function_ast)
        >>> print(collector.used_models)
        {'Pet', 'User'}
    """

    def __init__(self, available_models: set[str]):
        """Initialize the collector.

        Args:
            available_models: Set of model names that are available for import.
        """
        self.available_models = available_models
        self.used_models: set[str] = set()

    def visit_Name(self, node: ast.Name) -> None:
        """Visit a Name node and check if it's an available model."""
        if node.id in self.available_models:
            self.used_models.add(node.id)
        self.generic_visit(node)

    @classmethod
    def collect_from_endpoints(
        cls,
        endpoints: list['Endpoint'],
        available_models: set[str],
    ) -> set[str]:
        """Collect model names used across multiple endpoints.

        Args:
            endpoints: List of Endpoint objects to scan.
            available_models: Set of model names that are available.

        Returns:
            Set of model names that are actually used in the endpoints.
        """
        collector = cls(available_models)
        for endpoint in endpoints:
            collector.visit(endpoint.sync_ast)
            collector.visit(endpoint.async_ast)
        return collector.used_models


def collect_used_model_names(
    endpoints: list['Endpoint'],
    typegen_types: dict[str, 'Type'],
) -> set[str]:
    """Collect model names that are actually used in endpoint signatures.

    Only collects models that have implementations (defined in models.py)
    and are referenced in endpoint parameters, request bodies, or responses.

    Args:
        endpoints: List of Endpoint objects to check for model usage.
        typegen_types: Dictionary mapping type names to Type objects.

    Returns:
        Set of model names actually used in endpoints.
    """
    available_models = {
        type_.name
        for type_ in typegen_types.values()
        if type_.name and type_.implementation_ast
    }

    return ModelNameCollector.collect_from_endpoints(endpoints, available_models)
