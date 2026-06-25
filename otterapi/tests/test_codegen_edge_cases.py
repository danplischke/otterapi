"""Regression tests for recently added codegen edge-case behaviors."""

from __future__ import annotations

import ast

from pydantic import TypeAdapter

from otterapi.codegen.types import TypeGenerator
from otterapi.codegen.utils import sanitize_identifier, sanitize_parameter_field_name
from otterapi.openapi import UniversalOpenAPI
from otterapi.openapi.v3_2.v3_2 import Discriminator, OpenAPI, Schema

MINIMAL_SPEC = {
    'openapi': '3.0.0',
    'info': {'title': 'Test', 'version': '1.0.0'},
    'paths': {},
}


def _typegen() -> TypeGenerator:
    openapi = TypeAdapter(UniversalOpenAPI).validate_python(MINIMAL_SPEC)
    schema = openapi.root
    while not isinstance(schema, OpenAPI):
        schema, _ = schema.upgrade()
    return TypeGenerator(schema)


def _unparse(node: ast.AST) -> str:
    return ast.unparse(ast.fix_missing_locations(node))


def _exec_module(source: str) -> dict:
    namespace: dict = {}
    exec(compile(source, 'generated', 'exec'), namespace)
    return namespace


def _animal_union_schema(*, discriminator_is_literal: bool) -> Schema:
    dog_kind = (
        Schema(type='string', enum=['dog'])
        if discriminator_is_literal
        else Schema(type='string')
    )
    cat_kind = (
        Schema(type='string', enum=['cat'])
        if discriminator_is_literal
        else Schema(type='string')
    )
    return Schema(
        oneOf=[
            Schema(
                type='object',
                title='Dog',
                required=['kind', 'bark'],
                properties={'kind': dog_kind, 'bark': Schema(type='string')},
            ),
            Schema(
                type='object',
                title='Cat',
                required=['kind', 'meow'],
                properties={'kind': cat_kind, 'meow': Schema(type='string')},
            ),
        ],
        discriminator=Discriminator(
            propertyName='kind',
            mapping={'dog': '#/Dog', 'cat': '#/Cat'},
        ),
    )


def test_discriminator_without_literal_constraints_stays_plain_union_and_validates():
    typegen = _typegen()
    union_type = typegen.schema_to_type(
        _animal_union_schema(discriminator_is_literal=False), 'Animal'
    )

    union_source = _unparse(union_type.annotation_ast)

    assert 'Annotated[' not in union_source
    assert 'discriminator=' not in union_source
    assert 'Dog | Cat' in union_source

    generated_models = '\n\n'.join(
        _unparse(type_.implementation_ast)
        for type_ in typegen.get_sorted_types()
        if type_.implementation_ast is not None
    )
    namespace = _exec_module(
        'from __future__ import annotations\n'
        'from pydantic import BaseModel, Field, TypeAdapter\n\n'
        f'{generated_models}\n\n'
        f'AnimalAdapter = TypeAdapter({union_source})\n'
    )

    parsed = namespace['AnimalAdapter'].validate_python({'kind': 'dog', 'bark': 'woof'})

    assert type(parsed).__name__ == 'Dog'


def test_discriminator_with_single_value_enums_emits_annotated_union():
    typegen = _typegen()
    union_type = typegen.schema_to_type(
        _animal_union_schema(discriminator_is_literal=True), 'Animal'
    )

    union_source = _unparse(union_type.annotation_ast)

    assert 'Annotated[' in union_source
    assert "Field(discriminator='kind')" in union_source
    assert 'Dog | Cat' in union_source


def test_builtin_type_field_name_is_suffixed_and_preserves_alias_round_trip():
    typegen = _typegen()
    model_type = typegen.schema_to_type(
        Schema(
            type='object',
            required=['int', 'count'],
            properties={
                'int': Schema(type='boolean'),
                'count': Schema(type='integer'),
            },
        ),
        'BuiltinShadowModel',
    )

    model_source = _unparse(model_type.implementation_ast)

    assert "int_: bool = Field(alias='int')" in model_source
    assert 'count: int' in model_source

    namespace = _exec_module(
        'from __future__ import annotations\n'
        'from pydantic import BaseModel, Field\n\n'
        f'{model_source}\n'
    )
    Model = namespace[model_type.name]

    instance = Model.model_validate({'int': True, 'count': 3})

    assert instance.int_ is True
    assert instance.count == 3
    assert instance.model_dump(by_alias=True) == {'int': True, 'count': 3}


def test_sanitize_identifier_suffixes_reserved_generated_class_names():
    assert sanitize_identifier('Field') == 'Field_'
    assert sanitize_identifier('BaseModel') == 'BaseModel_'
    assert sanitize_identifier('Optional') == 'Optional_'
    assert sanitize_identifier('Customer') == 'Customer'


def test_sanitize_parameter_field_name_prefixes_digit_leading_names_with_field():
    assert sanitize_parameter_field_name('123abc') == 'field_123abc'


def test_nullable_string_enum_type_array_generates_str_enum_base():
    typegen = _typegen()
    enum_type = typegen.schema_to_type(
        Schema(type=['string', 'null'], enum=['draft', 'published']),
        'NullableStatus',
    )

    enum_source = _unparse(enum_type.implementation_ast)

    # The generator suffixes enum class names with "Enum"; the behavior under
    # test is the `str, Enum` base emitted for a nullable (type-array) string
    # enum, so assert on the base list rather than the exact class name.
    assert 'class NullableStatusEnum(str, Enum):' in enum_source
