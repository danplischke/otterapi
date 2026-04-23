"""Tests for the ``not`` schema fallback in TypeGenerator.

Pydantic has no direct translation for "value must NOT match this subschema",
so when the generator encounters ``not``, it logs a warning and emits
``Any``. This is preferable to silently pretending the ``not`` isn't there
(which yields a model that accepts forbidden values).
"""

from __future__ import annotations

import ast
import logging

from pydantic import TypeAdapter

from otterapi.codegen.types import TypeGenerator
from otterapi.openapi import UniversalOpenAPI
from otterapi.openapi.v3_2.v3_2 import OpenAPI, Schema

MINIMAL_SPEC = {
    'openapi': '3.0.0',
    'info': {'title': 'not schema', 'version': '1.0.0'},
    'paths': {},
}


def _typegen() -> TypeGenerator:
    openapi = TypeAdapter(UniversalOpenAPI).validate_python(MINIMAL_SPEC)
    doc = openapi.root
    while not isinstance(doc, OpenAPI):
        doc, _ = doc.upgrade()
    return TypeGenerator(doc)


class TestNotSchemaFallback:
    def test_top_level_not_emits_any(self, caplog):
        """A ``not`` at the schema root short-circuits to ``Any`` + a warning."""
        schema = Schema.model_validate({'not': {'type': 'string'}})
        with caplog.at_level(logging.WARNING):
            result = _typegen().schema_to_type(schema, 'Forbidden')
        assert ast.unparse(result.annotation_ast) == 'Any'
        assert 'Any' in (result.annotation_imports.get('typing') or set())
        assert any('not' in rec.message for rec in caplog.records)

    def test_property_level_not_emits_any_field(self, caplog):
        """A ``not`` inside an object property falls back too."""
        schema = Schema.model_validate(
            {
                'type': 'object',
                'required': ['kind'],
                'properties': {
                    'kind': {'type': 'string'},
                    'forbidden': {'not': {'type': 'integer'}},
                },
            }
        )
        typegen = _typegen()
        with caplog.at_level(logging.WARNING):
            model_type = typegen.schema_to_type(schema, 'User')
        source = ast.unparse(ast.fix_missing_locations(model_type.implementation_ast))
        # The field resolves to Any because ``not`` hit the fallback path.
        assert 'forbidden: Any' in source
        # We warned so regenerations aren't silent about losing fidelity.
        assert any('not' in rec.message for rec in caplog.records)

    def test_plain_schema_without_not_is_unaffected(self):
        """Regression guard: the fallback must only fire when ``not`` is present."""
        schema = Schema(type='string')
        result = _typegen().schema_to_type(schema, 'Plain')
        assert ast.unparse(result.annotation_ast) == 'str'
