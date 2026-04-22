"""Tests that OpenAPI validation constraints are forwarded to Pydantic Field()."""

from __future__ import annotations

import ast

import pytest
from pydantic import TypeAdapter, ValidationError

from otterapi.codegen.types import (
    TypeGenerator,
    _schema_constraints_to_field_kwargs,
)
from otterapi.openapi import UniversalOpenAPI
from otterapi.openapi.v3_2.v3_2 import OpenAPI, Schema

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


def _kw_pairs(kwargs: list[ast.keyword]) -> dict[str, object]:
    return {kw.arg: kw.value.value for kw in kwargs}


class TestSchemaConstraintsToFieldKwargs:
    def test_no_constraints_returns_empty(self):
        assert _schema_constraints_to_field_kwargs(Schema(type='string')) == []

    def test_string_constraints(self):
        schema = Schema(type='string', minLength=3, maxLength=20, pattern=r'^[a-z]+$')
        assert _kw_pairs(_schema_constraints_to_field_kwargs(schema)) == {
            'min_length': 3,
            'max_length': 20,
            'pattern': r'^[a-z]+$',
        }

    def test_min_length_zero_is_skipped(self):
        # OpenAPI defaults minLength to 0 -- treat as "no constraint".
        schema = Schema(type='string', minLength=0)
        assert _schema_constraints_to_field_kwargs(schema) == []

    def test_numeric_constraints_inclusive(self):
        schema = Schema(type='integer', minimum=1, maximum=100)
        assert _kw_pairs(_schema_constraints_to_field_kwargs(schema)) == {
            'ge': 1,
            'le': 100,
        }

    def test_numeric_constraints_exclusive(self):
        schema = Schema(type='number', exclusiveMinimum=0.0, exclusiveMaximum=1.0)
        assert _kw_pairs(_schema_constraints_to_field_kwargs(schema)) == {
            'gt': 0.0,
            'lt': 1.0,
        }

    def test_multiple_of(self):
        schema = Schema(type='integer', multipleOf=5.0)
        assert _kw_pairs(_schema_constraints_to_field_kwargs(schema)) == {
            'multiple_of': 5.0,
        }

    def test_array_constraints(self):
        schema = Schema(type='array', minItems=1, maxItems=10)
        assert _kw_pairs(_schema_constraints_to_field_kwargs(schema)) == {
            'min_length': 1,
            'max_length': 10,
        }

    def test_min_items_zero_is_skipped(self):
        schema = Schema(type='array', minItems=0)
        assert _schema_constraints_to_field_kwargs(schema) == []


class TestPydanticFieldEmissionAtCodegen:
    """Generates a model AST from a constrained spec and verifies the emitted Field()."""

    def test_string_field_emits_pattern_and_lengths(self):
        typegen = _typegen()
        schema = Schema(
            type='object',
            required=['username'],
            properties={
                'username': Schema(
                    type='string', minLength=3, maxLength=20, pattern='^[a-z]+$'
                ),
            },
        )
        model_type = typegen.schema_to_type(schema, 'User')
        assert model_type is not None
        source = ast.unparse(model_type.implementation_ast)
        assert 'min_length=3' in source
        assert 'max_length=20' in source
        assert "pattern='^[a-z]+$'" in source

    def test_numeric_field_emits_bounds(self):
        typegen = _typegen()
        schema = Schema(
            type='object',
            required=['age'],
            properties={'age': Schema(type='integer', minimum=0, maximum=150)},
        )
        model_type = typegen.schema_to_type(schema, 'Person')
        source = ast.unparse(model_type.implementation_ast)
        assert 'ge=0' in source
        assert 'le=150' in source

    def test_array_field_emits_item_bounds(self):
        typegen = _typegen()
        schema = Schema(
            type='object',
            required=['tags'],
            properties={
                'tags': Schema(
                    type='array', items=Schema(type='string'), minItems=1, maxItems=5
                ),
            },
        )
        model_type = typegen.schema_to_type(schema, 'TagSet')
        source = ast.unparse(model_type.implementation_ast)
        assert 'min_length=1' in source
        assert 'max_length=5' in source


class TestGeneratedModelEnforcesConstraints:
    """End-to-end: compile the generated model and confirm Pydantic actually validates."""

    def _compile_model(self, schema: Schema, name: str):
        """Round-trip the generated AST through unparse + exec into a live class."""
        typegen = _typegen()
        # The TypeGenerator can rename inline schemas (e.g. ``UnnamedModel``).
        # Use whatever name it actually picked when looking up the result.
        model_type = typegen.schema_to_type(schema, name)
        source = 'from pydantic import BaseModel, Field\n' + ast.unparse(
            ast.fix_missing_locations(model_type.implementation_ast)
        )
        ns: dict = {}
        exec(compile(source, name, 'exec'), ns)
        return ns[model_type.name]

    def test_string_pattern_rejected_at_runtime(self):
        schema = Schema(
            type='object',
            required=['code'],
            properties={'code': Schema(type='string', pattern='^[A-Z]{3}$')},
        )
        Model = self._compile_model(schema, 'Coded')
        Model(code='ABC')  # passes
        with pytest.raises(ValidationError):
            Model(code='abc')

    def test_numeric_bounds_rejected_at_runtime(self):
        schema = Schema(
            type='object',
            required=['percent'],
            properties={'percent': Schema(type='integer', minimum=0, maximum=100)},
        )
        Model = self._compile_model(schema, 'Bounded')
        Model(percent=50)
        with pytest.raises(ValidationError):
            Model(percent=-1)
        with pytest.raises(ValidationError):
            Model(percent=101)

    def test_array_min_items_rejected_at_runtime(self):
        schema = Schema(
            type='object',
            required=['names'],
            properties={
                'names': Schema(type='array', items=Schema(type='string'), minItems=2),
            },
        )
        Model = self._compile_model(schema, 'NamedSet')
        Model(names=['a', 'b'])
        with pytest.raises(ValidationError):
            Model(names=['a'])
