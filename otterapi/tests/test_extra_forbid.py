"""Tests that ``additionalProperties: false`` is enforced via ``extra='forbid'``."""

from __future__ import annotations

import ast

import pytest
from pydantic import TypeAdapter, ValidationError

from otterapi.codegen.types import TypeGenerator
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


def _compile(schema: Schema, name: str):
    typegen = _typegen()
    model_type = typegen.schema_to_type(schema, name)
    source = 'from pydantic import BaseModel, Field\n' + ast.unparse(
        ast.fix_missing_locations(model_type.implementation_ast)
    )
    ns: dict = {}
    exec(compile(source, name, 'exec'), ns)
    return ns[model_type.name], source


class TestAdditionalPropertiesFalse:
    def test_emits_model_config_extra_forbid(self):
        schema = Schema(
            type='object',
            additionalProperties=False,
            required=['id'],
            properties={'id': Schema(type='integer')},
        )
        Model, source = _compile(schema, 'Strict')
        assert "model_config = {'extra': 'forbid'}" in source
        assert Model.model_config.get('extra') == 'forbid'

    def test_extra_field_rejected_at_runtime(self):
        schema = Schema(
            type='object',
            additionalProperties=False,
            required=['id'],
            properties={'id': Schema(type='integer')},
        )
        Model, _ = _compile(schema, 'Strict2')
        Model(id=1)
        with pytest.raises(ValidationError):
            Model(id=1, sneaky='nope')

    def test_default_additional_properties_allows_extras(self):
        # OpenAPI defaults additionalProperties to True; Pydantic's default
        # ``extra='ignore'`` is fine -- we just must NOT emit ``extra='forbid'``.
        schema = Schema(
            type='object',
            required=['id'],
            properties={'id': Schema(type='integer')},
        )
        Model, source = _compile(schema, 'Loose')
        assert 'forbid' not in source
        instance = Model.model_validate({'id': 1, 'extra_field': 'ignored'})
        assert instance.id == 1

    def test_explicit_true_is_not_treated_as_false(self):
        schema = Schema(
            type='object',
            additionalProperties=True,
            required=['id'],
            properties={'id': Schema(type='integer')},
        )
        _, source = _compile(schema, 'StillLoose')
        assert 'forbid' not in source

    def test_additional_properties_schema_not_treated_as_false(self):
        # ``additionalProperties: {type: string}`` means extras allowed but typed --
        # we don't translate that today, but the model must NOT be marked forbid.
        schema = Schema(
            type='object',
            additionalProperties=Schema(type='string'),
            required=['id'],
            properties={'id': Schema(type='integer')},
        )
        _, source = _compile(schema, 'Typed')
        assert 'forbid' not in source

    def test_works_alongside_deprecated_docstring(self):
        schema = Schema(
            type='object',
            deprecated=True,
            additionalProperties=False,
            required=['id'],
            properties={'id': Schema(type='integer')},
        )
        _, source = _compile(schema, 'StrictDeprecated')
        assert "model_config = {'extra': 'forbid'}" in source
        assert 'deprecated' in source.lower()
        # Docstring stays first; model_config follows.
        body_lines = [line for line in source.splitlines() if line.startswith('    ')]
        assert body_lines[0].strip().startswith("'") or body_lines[
            0
        ].strip().startswith('"')
