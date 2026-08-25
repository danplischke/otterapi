"""Tests for the empty-schema (``{}``) fallback in TypeGenerator.

An empty schema means "any value, no constraints" in JSON Schema / OpenAPI.
It must map to ``Any`` -- which accepts strings, numbers, null, arrays, and
objects -- not ``dict[str, Any]``, which rejects everything that isn't an
object. This is deliberately distinct from an explicit ``type: object``
(correctly ``dict[str, Any]``); only a *missing* type is treated as "any".

Regression guard for the bug where ``schema.type is None`` was conflated with
``type: object`` in the dispatch, so ``{}`` items yielded
``list[dict[str, Any]]`` instead of ``list[Any]`` -- even though an *absent*
``items`` already (correctly) yielded ``list[Any]``.
"""

from __future__ import annotations

import ast

from pydantic import TypeAdapter

from otterapi.codegen.types import TypeGenerator
from otterapi.openapi import UniversalOpenAPI
from otterapi.openapi.v3_2.v3_2 import OpenAPI, Schema

MINIMAL_SPEC = {
    'openapi': '3.0.0',
    'info': {'title': 'empty schema', 'version': '1.0.0'},
    'paths': {},
}


def _typegen() -> TypeGenerator:
    openapi = TypeAdapter(UniversalOpenAPI).validate_python(MINIMAL_SPEC)
    doc = openapi.root
    while not isinstance(doc, OpenAPI):
        doc, _ = doc.upgrade()
    return TypeGenerator(doc)


class TestEmptySchemaEmitsAny:
    def test_top_level_empty_schema_emits_any(self):
        """A bare ``{}`` resolves to ``Any``, not ``dict[str, Any]``."""
        schema = Schema.model_validate({})
        result = _typegen().schema_to_type(schema, 'Anything')
        assert ast.unparse(result.annotation_ast) == 'Any'
        assert 'Any' in (result.annotation_imports.get('typing') or set())

    def test_array_with_empty_items_emits_list_any(self):
        """``items: {}`` must yield ``list[Any]`` (the reported bug)."""
        schema = Schema.model_validate({'type': 'array', 'items': {}})
        result = _typegen().schema_to_type(schema, 'Items')
        assert ast.unparse(result.annotation_ast) == 'list[Any]'

    def test_array_with_empty_items_matches_absent_items(self):
        """An explicit empty ``items`` and an absent ``items`` must agree.

        Both mean "any element", so both yield ``list[Any]``.
        """
        explicit = _typegen().schema_to_type(
            Schema.model_validate({'type': 'array', 'items': {}}), 'Explicit'
        )
        absent = _typegen().schema_to_type(
            Schema.model_validate({'type': 'array'}), 'Absent'
        )
        assert (
            ast.unparse(explicit.annotation_ast)
            == ast.unparse(absent.annotation_ast)
            == 'list[Any]'
        )

    def test_empty_schema_property_field_is_any(self):
        """An empty-schema property renders as an ``Any`` field, not a dict."""
        schema = Schema.model_validate(
            {
                'type': 'object',
                'required': ['id'],
                'properties': {
                    'id': {'type': 'integer'},
                    'payload': {},
                },
            }
        )
        typegen = _typegen()
        model_type = typegen.schema_to_type(schema, 'Envelope')
        source = ast.unparse(ast.fix_missing_locations(model_type.implementation_ast))
        assert 'payload: Any' in source
        assert 'payload: dict[str, Any]' not in source


class TestObjectSchemasStillMapToDict:
    """Regression guard: the empty-schema fallback must not over-broaden.

    Schemas that genuinely describe an object/map keep their ``dict`` shape.
    """

    def test_explicit_type_object_is_dict(self):
        """An explicit ``type: object`` (no properties) stays ``dict[str, Any]``."""
        schema = Schema.model_validate({'type': 'object'})
        result = _typegen().schema_to_type(schema, 'Obj')
        assert ast.unparse(result.annotation_ast) == 'dict[str, Any]'

    def test_typeless_map_with_additional_properties_is_dict(self):
        """``additionalProperties`` with a schema is a map, not "any"."""
        schema = Schema.model_validate({'additionalProperties': {'type': 'string'}})
        result = _typegen().schema_to_type(schema, 'Map')
        assert ast.unparse(result.annotation_ast) == 'dict[str, str]'
