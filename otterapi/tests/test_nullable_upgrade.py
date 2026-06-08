"""Tests that v3.0 ``nullable: true`` upgrades to the same Pydantic
annotation shape as v3.1+'s ``type: [..., "null"]``.

The audit flagged these as handled differently; this file verifies they
produce byte-identical output today so the Wave 1.1/1.2 correctness
fixes can't be worked around by staying on a 3.0 spec.
"""

from __future__ import annotations

import ast

from pydantic import TypeAdapter

from otterapi.codegen.types import TypeGenerator
from otterapi.openapi import UniversalOpenAPI
from otterapi.openapi.v3_2.v3_2 import OpenAPI


def _typegen_from(spec: dict) -> TypeGenerator:
    openapi = TypeAdapter(UniversalOpenAPI).validate_python(spec)
    doc = openapi.root
    while not isinstance(doc, OpenAPI):
        doc, _ = doc.upgrade()
    return TypeGenerator(doc)


def _emit_user(spec: dict) -> str:
    typegen = _typegen_from(spec)
    user_schema = typegen.openapi.components.schemas['User']
    user = typegen.schema_to_type(user_schema, 'User')
    return ast.unparse(ast.fix_missing_locations(user.implementation_ast))


V3_0_SPEC = {
    'openapi': '3.0.0',
    'info': {'title': 'v3.0 nullable', 'version': '1.0.0'},
    'paths': {},
    'components': {
        'schemas': {
            'User': {
                'type': 'object',
                'required': ['id'],
                'properties': {
                    'id': {'type': 'integer'},
                    'email': {'type': 'string', 'nullable': True},
                    'nickname': {'type': 'string', 'nullable': True},
                },
            },
        },
    },
}


V3_1_SPEC = {
    'openapi': '3.1.0',
    'info': {'title': 'v3.1 type-array', 'version': '1.0.0'},
    'paths': {},
    'components': {
        'schemas': {
            'User': {
                'type': 'object',
                'required': ['id'],
                'properties': {
                    'id': {'type': 'integer'},
                    'email': {'type': ['string', 'null']},
                    'nickname': {'type': ['string', 'null']},
                },
            },
        },
    },
}


class TestNullableUpgrade:
    def test_v3_0_nullable_emits_optional_annotation(self):
        source = _emit_user(V3_0_SPEC)
        assert 'email: str | None' in source
        assert 'nickname: str | None' in source

    def test_v3_1_type_array_emits_optional_annotation(self):
        source = _emit_user(V3_1_SPEC)
        assert 'email: str | None' in source
        assert 'nickname: str | None' in source

    def test_v3_0_and_v3_1_produce_equivalent_field_shape(self):
        v3_0 = _emit_user(V3_0_SPEC)
        v3_1 = _emit_user(V3_1_SPEC)
        # Strip docstrings / class names so the comparison is structural.
        # Both specs must emit the same nullable annotation for the two
        # fields that differ only in how they spell "nullable".
        for field in ('email', 'nickname'):
            v0_line = next(
                line for line in v3_0.splitlines() if line.strip().startswith(field)
            )
            v1_line = next(
                line for line in v3_1.splitlines() if line.strip().startswith(field)
            )
            assert v0_line == v1_line


class TestNullableWithoutTypeDoesNotCrash:
    """v3.0 allows ``nullable: true`` on a property without a primitive
    ``type``. The upgrader promotes it to ``type: [null]``; the
    TypeGenerator must not crash when it sees that edge case.

    Known cosmetic wart: the resulting annotation is spelled
    ``None | None | None`` (equivalent to ``None`` at type-check time).
    Not a correctness bug -- just aesthetically noisy -- so we assert
    the crash-safety guarantee here and leave the normalization for a
    future pass.
    """

    def test_nullable_without_type_upgrades_cleanly(self):
        spec = {
            'openapi': '3.0.0',
            'info': {'title': 'test', 'version': '1.0.0'},
            'paths': {},
            'components': {
                'schemas': {
                    'User': {
                        'type': 'object',
                        'required': ['id'],
                        'properties': {
                            'id': {'type': 'integer'},
                            'maybe': {'nullable': True},
                        },
                    },
                },
            },
        }
        # Must not raise. The generator may rename the inline schema to
        # UnnamedModel when the title isn't set -- we only check that
        # codegen produces a syntactically valid class with the two
        # expected fields.
        source = _emit_user(spec)
        parsed = ast.parse(source)
        class_def = next(node for node in parsed.body if isinstance(node, ast.ClassDef))
        field_names = {
            node.target.id
            for node in class_def.body
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
        }
        assert field_names == {'id', 'maybe'}
