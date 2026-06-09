"""Type definitions and generation for OtterAPI code generation.

This module provides:
- Type dataclasses for representing generated types, parameters, responses, and endpoints
- TypeGenerator for creating Pydantic models from OpenAPI schemas
- ModelNameCollector for tracking model usage in generated code
"""

import ast
import dataclasses
import logging
from collections.abc import Iterable
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field

from otterapi.codegen.ast_utils import _call, _name, _subscript, _union_expr
from otterapi.codegen.utils import (
    OpenAPIProcessor,
    sanitize_identifier,
    sanitize_parameter_field_name,
)
from otterapi.openapi.constants import MediaType
from otterapi.openapi.v3_2 import Reference, Schema, Type as DataType

__all__ = [
    'Type',
    'Parameter',
    'ResponseInfo',
    'RequestBodyInfo',
    'Endpoint',
    'TypeGenerator',
    'ModelNameCollector',
    'collect_used_model_names',
]

_PRIMITIVE_TYPE_MAP = {
    # string formats
    ('string', None): str,
    ('string', 'date-time'): datetime,
    ('string', 'date'): date,
    ('string', 'time'): time,
    ('string', 'duration'): timedelta,
    ('string', 'uuid'): UUID,
    ('string', 'byte'): bytes,  # base64-encoded
    ('string', 'binary'): bytes,  # raw binary
    # unknown/semantic string formats that are still strings at runtime
    ('string', 'email'): str,
    ('string', 'idn-email'): str,
    ('string', 'uri'): str,
    ('string', 'uri-reference'): str,
    ('string', 'uri-template'): str,
    ('string', 'url'): str,
    ('string', 'hostname'): str,
    ('string', 'idn-hostname'): str,
    ('string', 'ipv4'): str,
    ('string', 'ipv6'): str,
    ('string', 'password'): str,
    # integer formats
    ('integer', None): int,
    ('integer', 'int32'): int,
    ('integer', 'int64'): int,
    ('integer', 'uint32'): int,
    ('integer', 'uint64'): int,
    # number formats
    ('number', None): float,
    ('number', 'float'): float,
    ('number', 'double'): float,
    ('number', 'decimal'): Decimal,
    # misc
    ('boolean', None): bool,
    ('null', None): None,
    (None, None): None,
}

# Base-type fallbacks for known OpenAPI types with unrecognised formats.
# Keeps generated code typed rather than falling back to Any.
_TYPE_FALLBACK = {
    'string': str,
    'integer': int,
    'number': float,
    'boolean': bool,
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
        self.dependencies.update(type_.dependencies)

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

    @staticmethod
    def _ast_nodes_equal(a: ast.AST | None, b: ast.AST | None) -> bool:
        """Compare two optional AST nodes for structural equality via ast.dump."""
        if a is None and b is None:
            return True
        if a is None or b is None:
            return False
        return ast.dump(a) == ast.dump(b)

    def __eq__(self, other):
        """Deep comparison of Type objects, including AST nodes."""
        if not isinstance(other, Type):
            return False

        if (
            self.reference != other.reference
            or self.name != other.name
            or self.type != other.type
        ):
            return False

        if not self._ast_nodes_equal(
            self.annotation_ast, other.annotation_ast
        ) or not self._ast_nodes_equal(
            self.implementation_ast, other.implementation_ast
        ):
            return False

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
            MediaType.JSON,
            MediaType.TEXT_JSON,
        ) or self.content_type.endswith('+json')

    @property
    def is_binary(self) -> bool:
        """Check if this is a binary response (file download)."""
        binary_types = (
            MediaType.OCTET_STREAM,
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
            MediaType.JSON,
            MediaType.TEXT_JSON,
        ) or self.content_type.endswith('+json')

    @property
    def is_form(self) -> bool:
        """Check if this is a form-encoded request body."""
        return self.content_type == MediaType.FORM_URLENCODED

    @property
    def is_multipart(self) -> bool:
        """Check if this is a multipart form data request body."""
        return self.content_type == MediaType.MULTIPART

    @property
    def is_binary(self) -> bool:
        """Check if this is a binary request body."""
        return self.content_type in (MediaType.OCTET_STREAM,)

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

    def add_imports(self, imports: list[dict[str, set[str]]]):
        for imports_ in imports:
            for module, names in imports_.items():
                if module not in self.imports:
                    self.imports[module] = set()
                self.imports[module].update(names)


def _schema_constraints_to_field_kwargs(schema: 'Schema') -> list[ast.keyword]:
    """Translate OpenAPI validation constraints into Pydantic ``Field()`` kwargs.

    Maps the JSON-Schema-derived validators on ``schema`` (minLength, pattern,
    minimum, multipleOf, ...) into the equivalent Pydantic v2 ``Field``
    arguments. Returns an empty list when the schema carries no constraints
    (or only carries OpenAPI defaults like ``minLength=0``), so callers can
    safely concatenate the result onto an existing keyword list.

    The OpenAPI 3.1+ semantics for ``exclusiveMinimum`` / ``exclusiveMaximum``
    (the value itself is the bound) are honored; OpenAPI 3.0's boolean form
    is upgraded earlier in the pipeline.
    """
    kwargs: list[ast.keyword] = []

    def _add(name: str, value):
        kwargs.append(ast.keyword(arg=name, value=ast.Constant(value=value)))

    # String constraints. minLength has an OpenAPI default of 0 which we treat
    # as "no constraint" -- only emit when the spec explicitly tightens it.
    min_length = getattr(schema, 'minLength', None)
    if min_length is not None and min_length > 0:
        _add('min_length', min_length)
    max_length = getattr(schema, 'maxLength', None)
    if max_length is not None:
        _add('max_length', max_length)
    pattern = getattr(schema, 'pattern', None)
    if pattern:
        _add('pattern', pattern)

    # Numeric constraints (Pydantic uses ge/le/gt/lt to disambiguate
    # inclusive vs. exclusive bounds, which matches OpenAPI 3.1+ semantics).
    minimum = getattr(schema, 'minimum', None)
    if minimum is not None:
        _add('ge', minimum)
    maximum = getattr(schema, 'maximum', None)
    if maximum is not None:
        _add('le', maximum)
    excl_min = getattr(schema, 'exclusiveMinimum', None)
    if excl_min is not None:
        _add('gt', excl_min)
    excl_max = getattr(schema, 'exclusiveMaximum', None)
    if excl_max is not None:
        _add('lt', excl_max)
    multiple_of = getattr(schema, 'multipleOf', None)
    if multiple_of is not None:
        _add('multiple_of', multiple_of)

    # Array constraints. Same minItems-default-of-0 caveat as minLength.
    min_items = getattr(schema, 'minItems', None)
    if min_items is not None and min_items > 0:
        _add('min_length', min_items)
    max_items = getattr(schema, 'maxItems', None)
    if max_items is not None:
        _add('max_length', max_items)

    return kwargs


def _binop_includes_none(node: ast.expr) -> bool:
    """Return True if a BitOr chain contains ``None`` as one of the operands."""
    if isinstance(node, ast.Constant) and node.value is None:
        return True
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return _binop_includes_none(node.left) or _binop_includes_none(node.right)
    return False


@dataclasses.dataclass
class TypeGenerator(OpenAPIProcessor):
    types: dict[str, Type] = dataclasses.field(default_factory=dict)

    @staticmethod
    def _rename_type(type_: Type, new_name: str) -> None:
        """Rename ``type_`` in place, syncing its annotation and implementation ASTs."""
        type_.name = new_name
        type_.annotation_ast = _name(new_name)
        if isinstance(type_.implementation_ast, ast.ClassDef):
            type_.implementation_ast.name = new_name

    def _find_unique_type_name(
        self, prefix: str, try_bare: bool, type_: Type
    ) -> str | Type:
        """Find a registry slot for ``type_`` starting from ``prefix``.

        Tries ``prefix`` itself first when ``try_bare``, then ``prefix1``,
        ``prefix2``, etc. Returns a free name to rename ``type_`` to, or an
        already-registered ``Type`` with an identical definition to reuse
        (this avoids generating near-duplicates like ``Detail20``/``Detail21``).
        """
        candidate = prefix if try_bare else None
        counter = 0
        while True:
            if candidate is not None:
                existing = self.types.get(candidate)
                if existing is None:
                    return candidate
                if existing == type_:
                    return existing
            counter += 1
            candidate = f'{prefix}{counter}'

    def add_type(self, type_: Type, base_name: str | None = None) -> Type:
        """Add a type to the registry. If a type with the same name but different definition
        already exists, generate a unique name using the base_name prefix.
        Returns the type (potentially with a modified name).
        """
        # Skip types without names (primitive types, inline types, etc.)
        if not type_.name:
            return type_

        if type_.name in self.types:
            existing = self.types[type_.name]
            if existing == type_:
                # Same type already registered, return the existing one
                return existing

            # Different definition with same name - find a unique slot.
            # With base_name, prefer "{base_name}{name}" before falling back
            # to a numbered suffix; without it, go straight to numbering.
            prefix = f'{base_name}{type_.name}' if base_name else type_.name
            result = self._find_unique_type_name(
                prefix, try_bare=bool(base_name), type_=type_
            )
            if isinstance(result, Type):
                return result
            self._rename_type(type_, result)

        self.types[type_.name] = type_
        return type_

    @staticmethod
    def _resolve_enum_name(
        schema: Schema,
        name: str | None,
        base_name: str | None,
        field_name: str | None,
    ) -> str:
        """Derive an enum class name from the explicit name, title, or surrounding context."""
        enum_name = name or (
            sanitize_identifier(schema.title) if schema.title else None
        )
        if enum_name:
            return enum_name

        if field_name:
            # e.g., Pet + status -> 'PetStatus', Order + status -> 'OrderStatus'
            field_part = sanitize_identifier(field_name)
            if field_part:
                field_part = field_part[0].upper() + field_part[1:]
            if base_name:
                return f'{sanitize_identifier(base_name)}{field_part}'
            return field_part

        if base_name:
            return f'{sanitize_identifier(base_name)}Enum'

        return 'AutoEnum'

    def _find_existing_identical_enum(
        self, enum_values_key: tuple[str, ...]
    ) -> Type | None:
        """Find a previously-registered Enum class with the same value set.

        Reusing an identical enum avoids generating near-duplicates like
        ``Status``/``Status1`` for schemas that share the same value set.
        """
        for existing_type in self.types.values():
            if existing_type.type != 'model' or not isinstance(
                existing_type.implementation_ast, ast.ClassDef
            ):
                continue
            existing_class = existing_type.implementation_ast
            if not any(
                isinstance(base, ast.Name) and base.id == 'Enum'
                for base in existing_class.bases
            ):
                continue
            existing_values = [
                str(node.value.value)
                for node in existing_class.body
                if isinstance(node, ast.Assign)
                and node.value
                and isinstance(node.value, ast.Constant)
            ]
            if tuple(sorted(existing_values)) == enum_values_key:
                return existing_type
        return None

    def _make_unique_enum_name(self, name: str) -> str:
        """Append a numeric suffix to ``name`` until it is free in the registry."""
        if name not in self.types:
            return name
        counter = 1
        while f'{name}{counter}' in self.types:
            counter += 1
        return f'{name}{counter}'

    @staticmethod
    def _build_enum_member_assignments(schema: Schema) -> list[ast.stmt]:
        """Build ``NAME = 'value'`` assignments for each non-null enum value.

        Sanitizes values into valid identifiers and disambiguates collisions
        (e.g. 'mesoderm' and 'Mesoderm' both sanitize to 'MESODERM').
        """
        enum_body: list[ast.stmt] = []
        seen_member_names: dict[str, int] = {}
        for value in schema.enum:
            if value is None:
                continue
            if isinstance(value, str):
                member_name = sanitize_identifier(value).upper()
                if member_name and member_name[0].isdigit():
                    member_name = f'_{member_name}'
            else:
                member_name = f'VALUE_{value}'

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
        return enum_body

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
        enum_name = self._resolve_enum_name(schema, name, base_name, field_name)

        enum_values_key = tuple(sorted(str(v) for v in schema.enum if v is not None))
        existing = self._find_existing_identical_enum(enum_values_key)
        if existing is not None:
            return existing

        enum_name = self._make_unique_enum_name(enum_name)

        enum_body = self._build_enum_member_assignments(schema)
        if not enum_body:
            return self._create_literal_type(schema)

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
        mapped = _PRIMITIVE_TYPE_MAP.get(
            key,
            _TYPE_FALLBACK.get(type_value, Any),  # unknown format → base type, not Any
        )

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
    ) -> ast.AnnAssign:
        if hasattr(field_schema, 'ref'):
            field_schema, _ = self._resolve_reference(field_schema)

        field_keywords = []

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

        # Forward OpenAPI validation constraints to Pydantic Field()
        field_keywords.extend(_schema_constraints_to_field_kwargs(field_schema))

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

    @staticmethod
    def _type_already_nullable(type_: Type) -> bool:
        """Check if a type annotation already includes None.

        Handles both the ``Union[X, None]`` (Subscript) form and the
        ``X | None`` (BinOp chain) form generated by ``_union_expr``.
        """
        annotation = type_.annotation_ast
        # Union[..., None] form
        if isinstance(annotation, ast.Subscript):
            if (
                isinstance(annotation.value, ast.Name)
                and annotation.value.id == 'Union'
            ):
                if isinstance(annotation.slice, ast.Tuple):
                    return any(
                        isinstance(elt, ast.Constant) and elt.value is None
                        for elt in annotation.slice.elts
                    )
        # X | None  (or X | Y | None) form — produced by _union_expr
        if isinstance(annotation, ast.BinOp) and isinstance(annotation.op, ast.BitOr):
            return _binop_includes_none(annotation)
        return False

    def _build_allof_base_types(
        self, schema: Schema, base_name: str | None
    ) -> list[Type]:
        """Create object types for each ``allOf`` member to use as model bases."""
        return [
            self._create_object_type(schema=base_schema, base_name=base_name)
            for base_schema in schema.allOf or []
        ]

    def _build_union_variant_type(self, schema: Schema, base_name: str | None) -> Type:
        """Build an inline union type for an ``anyOf``/``oneOf`` schema.

        Uses ``schema_to_type`` for each variant to properly handle primitives,
        objects, etc., then wraps the union as ``Annotated[Union[...],
        Field(discriminator=...)]`` when the schema declares a discriminator —
        so Pydantic v2 enforces tag-based dispatch at validation time. Without
        this, polymorphic responses fall back to "first union member that
        validates wins", which silently mis-routes payloads when variants share
        field shapes.
        """
        types_ = [
            self.schema_to_type(t, base_name=base_name)
            for t in (schema.anyOf or schema.oneOf)
        ]

        union_ast = _union_expr(types=[t.annotation_ast for t in types_])

        extra_imports: dict[str, set[str]] = {}
        if schema.discriminator is not None:
            discriminator_kw = ast.keyword(
                arg='discriminator',
                value=ast.Constant(value=schema.discriminator.propertyName),
            )
            annotation_ast = ast.Subscript(
                value=_name('Annotated'),
                slice=ast.Tuple(
                    elts=[
                        union_ast,
                        _call(
                            func=_name(Field.__name__),
                            keywords=[discriminator_kw],
                        ),
                    ],
                    ctx=ast.Load(),
                ),
                ctx=ast.Load(),
            )
            extra_imports['typing'] = {'Annotated'}
            extra_imports.setdefault(Field.__module__, set()).add(Field.__name__)
        else:
            annotation_ast = union_ast

        union_type = Type(
            reference=None,
            name=None,  # Union type doesn't need a name, it's used inline
            annotation_ast=annotation_ast,
            implementation_ast=None,
            type='primitive',
        )
        union_type.copy_imports_from_sub_types(types_)
        for module, names in extra_imports.items():
            for n in names:
                union_type.add_annotation_import(module=module, name=n)
                union_type.add_implementation_import(module=module, name=n)
        return union_type

    def _build_pydantic_model_fields(
        self, schema: Schema, base_name: str | None
    ) -> tuple[list[ast.AnnAssign], list[Type]]:
        """Build field assignments and resolve their types for each schema property."""
        body: list[ast.AnnAssign] = []
        field_types: list[Type] = []
        required_fields = set(schema.required or [])
        for property_name, property_schema in (schema.properties or {}).items():
            # Resolve reference to check for nullable
            resolved_schema = property_schema
            if hasattr(property_schema, 'ref') and property_schema.ref:
                resolved_schema, _ = self._resolve_reference(property_schema)

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
        return body, field_types

    @staticmethod
    def _prepend_model_config_and_docstring(
        body: list[ast.stmt], schema: Schema, name: str
    ) -> list[ast.stmt]:
        """Prepend ``model_config``/deprecation-docstring statements to a model body.

        When the schema declares ``additionalProperties: false``, this mirrors
        that constraint via Pydantic's ``model_config = {'extra': 'forbid'}`` so
        generated models actually reject unexpected fields at runtime. The
        deprecation docstring (if any) is prepended after ``model_config`` so it
        stays the first statement in the class body.
        """
        if schema.additionalProperties is False:
            model_config_assign = ast.Assign(
                targets=[_name('model_config')],
                value=ast.Dict(
                    keys=[ast.Constant(value='extra')],
                    values=[ast.Constant(value='forbid')],
                ),
            )
            body = [model_config_assign] + body

        if schema.deprecated:
            deprecation_doc = ast.Expr(
                value=ast.Constant(
                    value=f'{name} is deprecated.\n\n.. deprecated::\n    This model is deprecated.'
                )
            )
            body = [deprecation_doc] + body if body else [deprecation_doc]

        return body

    @staticmethod
    def _attach_model_dependencies(
        type_: Type, base_bases: list[Type], field_types: list[Type]
    ) -> None:
        """Register base-class and field-type dependencies on a generated model Type."""
        for base in base_bases:
            type_.add_dependency(base)

        for field_type in field_types:
            if field_type.name:
                type_.dependencies.add(field_type.name)
                type_.dependencies.update(field_type.dependencies)

    def _create_pydantic_model(
        self, schema: Schema, name: str | None = None, base_name: str | None = None
    ) -> Type:
        base_bases = self._build_allof_base_types(schema, base_name)

        if schema.anyOf or schema.oneOf:
            return self._build_union_variant_type(schema, base_name)

        name = name or (
            sanitize_identifier(schema.title) if schema.title else 'UnnamedModel'
        )

        bases = [b.name for b in base_bases] or [BaseModel.__name__]
        bases = [_name(base) for base in bases]

        body, field_types = self._build_pydantic_model_fields(schema, base_name)
        body = self._prepend_model_config_and_docstring(body, schema, name)

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

        self._attach_model_dependencies(type_, base_bases, field_types)

        type_.add_implementation_import(
            module=BaseModel.__module__, name=BaseModel.__name__
        )
        type_.add_implementation_import(module=Field.__module__, name=Field.__name__)
        type_.copy_imports_from_sub_types(field_types)

        type_ = self.add_type(type_, base_name=base_name)
        return type_

    def _create_array_type(self, schema: Schema, base_name: str | None = None) -> Type:
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

        # ``not`` schemas describe what a value *isn't* -- Pydantic doesn't have
        # a 1:1 translation, so silently treating them as whatever the outer
        # type says would produce a model that accepts forbidden values. Log a
        # warning and fall back to ``Any`` so the payload is still accepted but
        # the looseness is visible. (Issue #3 follow-up, audit item "not
        # schemas silently pass through".)
        if getattr(schema, 'not_', None) is not None:
            logging.warning(
                'OpenAPI ``not`` schema encountered for %s; falling back to '
                'Any (Pydantic has no direct equivalent).',
                effective_base_name or field_name or '<inline>',
            )
            any_type = Type(
                reference=None,
                name=None,
                annotation_ast=_name('Any'),
                implementation_ast=None,
                type='primitive',
            )
            any_type.add_annotation_import('typing', 'Any')
            return any_type

        if schema.type == DataType.array:
            type_ = self._create_array_type(
                schema=schema, base_name=effective_base_name
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
