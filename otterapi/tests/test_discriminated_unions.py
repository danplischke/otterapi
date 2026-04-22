"""Tests that ``oneOf`` / ``anyOf`` schemas with a ``discriminator`` are
emitted as ``Annotated[Union[...], Field(discriminator=...)]`` so Pydantic
v2 dispatches by the tag field at validation time.
"""

from __future__ import annotations

import ast

import pytest
from pydantic import TypeAdapter, ValidationError

from otterapi.codegen.types import TypeGenerator
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


def _dog_cat_schema(*, with_discriminator: bool, use_oneof: bool = True) -> Schema:
    """Build a polymorphic schema with two variants sharing a 'kind' tag."""
    dog = Schema(
        type='object',
        title='Dog',
        required=['kind', 'bark'],
        properties={
            'kind': Schema(type='string'),
            'bark': Schema(type='string'),
        },
    )
    cat = Schema(
        type='object',
        title='Cat',
        required=['kind', 'meow'],
        properties={
            'kind': Schema(type='string'),
            'meow': Schema(type='string'),
        },
    )
    kwargs = {'oneOf': [dog, cat]} if use_oneof else {'anyOf': [dog, cat]}
    if with_discriminator:
        kwargs['discriminator'] = Discriminator(
            propertyName='kind', mapping={'dog': '#/Dog', 'cat': '#/Cat'}
        )
    return Schema(**kwargs)


class TestDiscriminatorASTEmission:
    def test_oneof_with_discriminator_emits_annotated(self):
        typegen = _typegen()
        result = typegen.schema_to_type(
            _dog_cat_schema(with_discriminator=True), 'Animal'
        )
        source = ast.unparse(ast.fix_missing_locations(result.annotation_ast))
        assert 'Annotated[' in source
        assert "Field(discriminator='kind')" in source
        assert 'Dog' in source
        assert 'Cat' in source

    def test_anyof_with_discriminator_emits_annotated(self):
        typegen = _typegen()
        result = typegen.schema_to_type(
            _dog_cat_schema(with_discriminator=True, use_oneof=False), 'Animal'
        )
        source = ast.unparse(ast.fix_missing_locations(result.annotation_ast))
        assert 'Annotated[' in source
        assert "Field(discriminator='kind')" in source

    def test_oneof_without_discriminator_stays_plain_union(self):
        typegen = _typegen()
        result = typegen.schema_to_type(
            _dog_cat_schema(with_discriminator=False), 'Animal'
        )
        source = ast.unparse(ast.fix_missing_locations(result.annotation_ast))
        assert 'Annotated' not in source
        assert 'discriminator' not in source

    def test_imports_include_annotated_and_field(self):
        typegen = _typegen()
        result = typegen.schema_to_type(
            _dog_cat_schema(with_discriminator=True), 'Animal'
        )
        # Imports get tracked on the result Type.
        ann_imports = result.annotation_imports
        assert 'Annotated' in ann_imports.get('typing', set())
        # ``Field`` is exported from ``pydantic`` but lives in
        # ``pydantic.fields`` -- the import collector tracks the canonical
        # module. Either is acceptable.
        impl_imports = result.implementation_imports
        assert any(
            'Field' in impl_imports.get(mod, set())
            for mod in ('pydantic', 'pydantic.fields')
        )


class TestDiscriminatorRuntimeDispatch:
    """End-to-end: hand-build variants with ``Literal`` discriminator fields and
    confirm the generated ``Annotated[Union[...], Field(discriminator=...)]``
    actually drives Pydantic's tag-based dispatch.
    """

    def test_pydantic_uses_emitted_discriminator(self):
        # Generate just the union AST -- the variant ``Dog`` / ``Cat`` classes
        # are hand-rolled here with ``Literal['dog']`` / ``Literal['cat']`` on
        # the ``kind`` field so Pydantic's discriminator can dispatch
        # (single-value Enums on the discriminator field would error with
        # "discriminator needs literal" at adapter rebuild time).
        typegen = _typegen()
        union_type = typegen.schema_to_type(
            _dog_cat_schema(with_discriminator=True), 'Animal'
        )
        union_source = ast.unparse(ast.fix_missing_locations(union_type.annotation_ast))

        full = (
            'from typing import Annotated, Literal\n'
            'from pydantic import BaseModel, Field, TypeAdapter\n'
            'class Dog(BaseModel):\n'
            "    kind: Literal['dog']\n"
            '    bark: str\n'
            'class Cat(BaseModel):\n'
            "    kind: Literal['cat']\n"
            '    meow: str\n'
            f'AnimalAdapter = TypeAdapter({union_source})\n'
        )
        ns: dict = {}
        exec(compile(full, 'union', 'exec'), ns)
        adapter = ns['AnimalAdapter']
        # Variants live in the dynamic exec namespace; rebuild the adapter
        # against that scope so Pydantic can resolve them.
        adapter.rebuild(_types_namespace=ns)

        parsed_dog = adapter.validate_python({'kind': 'dog', 'bark': 'woof'})
        assert type(parsed_dog).__name__ == 'Dog'

        parsed_cat = adapter.validate_python({'kind': 'cat', 'meow': 'mrr'})
        assert type(parsed_cat).__name__ == 'Cat'

        # Unknown tag is rejected by the discriminator -- without it, Pydantic
        # would silently fall back to "first union member that validates wins".
        with pytest.raises(ValidationError):
            adapter.validate_python({'kind': 'goldfish', 'bark': 'glub'})
