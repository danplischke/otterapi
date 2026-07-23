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
    assert typegen.openapi is not None and typegen.openapi.components is not None
    schemas = typegen.openapi.components.schemas
    assert schemas is not None
    user_schema = schemas['User']
    user = typegen.schema_to_type(user_schema, 'User')
    assert user.implementation_ast is not None
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


# A Swagger 2.0 spec whose scalar array items carry a JSON-Schema type
# array with a ``null`` member (``["string", "null"]``). Swagger 2.0 has no
# ``type`` array concept, so the v2 -> v3.0 converter must fold the ``null``
# into ``nullable: true`` instead of discarding it. Regression guard for the
# bug where the item type collapsed to the first non-null member and the
# nullability was silently dropped, which would have forced a ``list[Any]``
# workaround instead of the properly-typed ``list[str | None]``.
V2_NULLABLE_ITEM_SPEC = {
    'swagger': '2.0',
    'info': {'title': 'v2 nullable array items', 'version': '1.0.0'},
    'paths': {},
    'definitions': {
        'User': {
            'type': 'object',
            'required': ['id'],
            'properties': {
                'id': {'type': 'integer'},
                'tags': {
                    'type': 'array',
                    'items': {'type': ['string', 'null']},
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


class TestV2NullableArrayItems:
    """Regression guard: a Swagger 2.0 scalar array whose items carry a
    ``["string", "null"]`` type array must survive the v2 -> v3.0 -> v3.1
    upgrade as ``list[str | None]``.

    The v2 -> v3.0 converter previously collapsed the item type to the first
    non-null member and discarded ``null`` outright, which meant a
    null-polluted vendor array could only be tolerated by degrading the item
    type to ``Any`` (``list[Any]``). Folding ``null`` into ``nullable: true``
    keeps full typing while remaining null-tolerant.
    """

    def test_v2_nullable_scalar_array_emits_list_optional(self):
        source = _emit_user(V2_NULLABLE_ITEM_SPEC)
        assert 'tags: list[str | None]' in source

    def test_v2_nullable_scalar_array_does_not_degrade_to_any(self):
        source = _emit_user(V2_NULLABLE_ITEM_SPEC)
        assert 'list[Any]' not in source
        # The element type must be preserved, not silently dropped to ``str``.
        assert 'tags: list[str]' not in source
