import ast
import dataclasses
from datetime import datetime
from typing import Literal, Iterable, Any
from uuid import UUID

from pydantic import Field, RootModel, BaseModel

from otterapi.codegen.ast_utils import _name
from otterapi.codegen_v2.ast_utils import _subscript, _call, _union_expr
from otterapi.codegen_v2.utils import OpenAPIProcessor, sanitize_identifier, sanitize_parameter_field_name
from otterapi.openapi.v3_2 import Schema, Reference, Type as DataType

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
    annotation_ast: ast.expr | ast.stmt | None = dataclasses.field(
        default=None
    )
    implementation_ast: ast.expr | ast.stmt | None = dataclasses.field(
        default=None
    )
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
        if isinstance(name, str):
            name = [name]

        if module not in self.implementation_imports:
            self.implementation_imports[module] = set()

        for n in name:
            self.implementation_imports[module].add(n)

    def add_annotation_import(self, module: str, name: str | Iterable[str]) -> None:
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
    location: Literal["query", "path", "header", "cookie", "body"]
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
        return self.content_type in ('application/json', 'text/json') or self.content_type.endswith('+json')

    @property
    def is_binary(self) -> bool:
        """Check if this is a binary response."""
        return self.content_type in ('application/octet-stream',) or self.content_type.startswith('image/')

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
        return self.content_type in ('application/json', 'text/json') or self.content_type.endswith('+json')

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
    sync_ast: ast.FunctionDef
    async_ast: ast.AsyncFunctionDef

    sync_fn_name: str
    async_fn_name: str

    name: str
    imports: dict[str, set[str]] = dataclasses.field(default_factory=dict)

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

    def _get_primitive_type_ast(self, schema: Schema) -> Type:
        key = (schema.type, schema.format or None)
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
        return type_

    def _create_pydantic_field(
        self, field_name: str, field_schema: Schema, field_type: Type
    ) -> str:
        if hasattr(field_schema, 'ref'):
            field_schema, _ = self._resolve_reference(field_schema)

        field_keywords = list()

        sanitized_field_name = sanitize_parameter_field_name(field_name)

        value = None
        if field_schema.default is not None and isinstance(
            field_schema.default, (str, int, float, bool)
        ):
            field_keywords.append(
                ast.keyword(arg='default', value=ast.Constant(field_schema.default))
            )
        elif field_schema.default is None and not field_schema.required:
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
            annotation=field_type.annotation_ast,
            value=value,
            simple=1,
        )

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
            types_ = [
                self._create_object_type(schema=t, base_name=base_name)
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
            union_type.add_annotation_import('typing', 'Union')
            return union_type

        name = name or (
            sanitize_identifier(schema.title) if schema.title else 'UnnamedModel'
        )

        bases = [b.name for b in base_bases] or [BaseModel.__name__]
        bases = [_name(base) for base in bases]

        body = []
        field_types = []
        for property_name, property_schema in schema.properties.items():
            type_ = self.schema_to_type(property_schema, base_name=base_name)
            field = self._create_pydantic_field(property_name, property_schema, type_)

            body.append(field)
            field_types.append(type_)

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
        if (
            not schema.properties
            and not schema.allOf
            and not schema.anyOf
            and not schema.oneOf
        ):
            type_ = Type(
                None,
                None,
                annotation_ast=_subscript(
                    dict.__name__,
                    ast.Tuple(
                        elts=[
                            ast.Name(id=str.__name__, ctx=ast.Load()),
                            ast.Name(id=Any.__name__, ctx=ast.Load()),
                        ]
                    ),
                ),
                implementation_ast=None,
                type='primitive',
            )

            type_.add_annotation_import(dict.__module__, dict.__name__)
            type_.add_annotation_import(Any.__module__, Any.__name__)

            return type_

        return self._create_pydantic_model(
            schema, schema_name or name, base_name=base_name
        )

    def schema_to_type(
        self, schema: Schema | Reference, base_name: str | None = None
    ) -> Type:
        if isinstance(schema, Reference):
            ref_name = schema.ref.split('/')[-1]
            sanitized_ref_name = sanitize_identifier(ref_name)
            if sanitized_ref_name in self.types:
                return self.types[sanitized_ref_name]

        schema, schema_name = self._resolve_reference(schema)

        # TODO: schema.type can be array?
        if schema.type == DataType.array:
            type_ = self._create_array_type(
                schema=schema, name=schema_name, base_name=base_name
            )
        elif schema.type == DataType.object or schema.type is None:
            type_ = self._create_object_type(
                schema, name=schema_name, base_name=base_name
            )
        else:
            type_ = self._get_primitive_type_ast(schema)

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


