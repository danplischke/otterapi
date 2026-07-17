"""Tests for opt-in lenient validation and vendor-extension preservation.

Lenient mode (``--lenient`` / ``lenient: true``) drops unrecognized,
structurally-invalid fields instead of failing. These tests cover the layers
that implement it:

* ``LenientRefModel`` -- the shared base for the OpenAPI 3.0/3.1/3.2 object
  models. It preserves ``x-*`` vendor extensions (``extra='allow'``) while still
  rejecting other unknown keys in strict mode and keeping ``$ref`` discrimination.
* The v2 bases ``LenientModelWithVendorExtensions`` (strips) and
  ``UnionVariantModel`` (always raises, to keep union discrimination correct).
* ``SchemaLoader`` end-to-end: strict raises with a ``--lenient`` hint, lenient
  loads and records a warning naming the dropped keys.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from otterapi.codegen.schema import SchemaLoader
from otterapi.exceptions import SchemaValidationError
from otterapi.openapi._lenient import LenientRefModel
from otterapi.openapi.v2 import v2
from otterapi.openapi.v3 import v3 as v3_0
from otterapi.openapi.v3_1 import v3_1
from otterapi.openapi.v3_2 import v3_2


def _lenient_ctx() -> tuple[dict, set[str]]:
    """Return a lenient validation context and the ``dropped_keys`` sink it writes to."""
    dropped: set[str] = set()
    return {'lenient': True, 'dropped_keys': dropped}, dropped


# -----------------------------------------------------------------------------
# LenientRefModel -- the v3.x object-model base (extra='allow', preserves x-*)
# -----------------------------------------------------------------------------


class TestLenientRefModelUnit:
    def test_strict_rejects_unknown_field(self):
        with pytest.raises(ValidationError):
            v3_2.Info.model_validate({'title': 't', 'version': '1', 'bogus': 1})

    def test_lenient_drops_unknown_and_records_it(self):
        ctx, dropped = _lenient_ctx()
        info = v3_2.Info.model_validate(
            {'title': 't', 'version': '1', 'bogus': 1}, context=ctx
        )
        assert info.title == 't'
        assert info.version == '1'
        assert 'bogus' in dropped

    def test_lenient_without_sink_still_strips(self):
        info = v3_2.Info.model_validate(
            {'title': 't', 'version': '1', 'bogus': 1}, context={'lenient': True}
        )
        assert info.title == 't'

    def test_lenient_false_context_is_a_no_op(self):
        with pytest.raises(ValidationError):
            v3_2.Info.model_validate(
                {'title': 't', 'version': '1', 'bogus': 1},
                context={'lenient': False},
            )

    @pytest.mark.parametrize(
        ('model', 'base'),
        [
            (v3_2.Info, {'title': 't', 'version': '1'}),
            (v3_1.Contact, {'name': 'Acme'}),
            (v3_0.MediaType, {}),
        ],
    )
    def test_lenient_covers_all_three_v3_files(self, model, base):
        # Sanity: every model under test is a LenientRefModel.
        assert issubclass(model, LenientRefModel)

        with pytest.raises(ValidationError):
            model.model_validate({**base, 'junk': 1})

        ctx, dropped = _lenient_ctx()
        instance = model.model_validate({**base, 'junk': 1}, context=ctx)
        assert instance is not None
        assert 'junk' in dropped


# -----------------------------------------------------------------------------
# Reference discrimination must survive lenient stripping
# -----------------------------------------------------------------------------


class TestRefDiscriminationUnderLenient:
    def test_ref_wins_over_object_branch_strict(self):
        mt = v3_2.MediaType.model_validate(
            {'schema': {'$ref': '#/components/schemas/Pet'}}
        )
        assert isinstance(mt.schema_, v3_2.Reference)
        assert mt.schema_.ref == '#/components/schemas/Pet'

    def test_ref_still_wins_in_lenient_mode(self):
        ctx, _ = _lenient_ctx()
        mt = v3_2.MediaType.model_validate(
            {'schema': {'$ref': '#/components/schemas/Pet', 'bogus': 1}}, context=ctx
        )
        # An undeclared $ref makes the object branch fail, so the Reference
        # branch continues to win -- even in lenient mode.
        assert isinstance(mt.schema_, v3_2.Reference)

    def test_object_schema_still_parses_as_schema(self):
        ctx, _ = _lenient_ctx()
        mt = v3_2.MediaType.model_validate({'schema': {'type': 'object'}}, context=ctx)
        assert isinstance(mt.schema_, v3_2.Schema)


# -----------------------------------------------------------------------------
# Vendor extensions (x-*) are preserved, never rejected
# -----------------------------------------------------------------------------


class TestVendorExtensionsPreserved:
    def test_x_star_preserved_in_strict_mode(self):
        info = v3_2.Info.model_validate(
            {'title': 't', 'version': '1', 'x-internal': 'keep', 'x-team': {'a': 1}}
        )
        assert info.model_extra == {'x-internal': 'keep', 'x-team': {'a': 1}}

    def test_x_star_kept_while_junk_dropped_in_lenient(self):
        ctx, dropped = _lenient_ctx()
        info = v3_2.Info.model_validate(
            {'title': 't', 'version': '1', 'x-ok': 9, 'bogus': 1}, context=ctx
        )
        assert dropped == {'bogus'}
        assert (info.model_extra or {}).get('x-ok') == 9

    @pytest.mark.parametrize(
        ('model', 'base'),
        [
            (v3_2.Info, {'title': 't', 'version': '1'}),
            (v3_1.Contact, {'name': 'Acme'}),
            (v3_0.MediaType, {}),
        ],
    )
    def test_x_star_preserved_across_all_three_v3_files(self, model, base):
        instance = model.model_validate({**base, 'x-vendor': 'v'})
        assert (instance.model_extra or {}).get('x-vendor') == 'v'

    def test_object_schema_keeps_x_star(self):
        mt = v3_2.MediaType.model_validate(
            {'schema': {'type': 'object', 'x-vendor': 'v'}}
        )
        assert isinstance(mt.schema_, v3_2.Schema)
        assert (mt.schema_.model_extra or {}).get('x-vendor') == 'v'


# -----------------------------------------------------------------------------
# v2 bases: LenientModelWithVendorExtensions (strips) vs UnionVariantModel (raises)
# -----------------------------------------------------------------------------


class TestLenientV2Bases:
    _OP = {'responses': {'200': {'description': 'ok'}}}

    def test_operation_strict_rejects_stray_key(self):
        with pytest.raises(ValidationError):
            v2.Operation.model_validate({**self._OP, 'bogus': 1})

    def test_operation_lenient_drops_stray_but_keeps_vendor_extension(self):
        ctx, dropped = _lenient_ctx()
        op = v2.Operation.model_validate(
            {**self._OP, 'bogus': 1, 'x-internal': 'keep-me'}, context=ctx
        )
        assert 'bogus' in dropped
        assert 'x-internal' not in dropped
        assert (op.model_extra or {}).get('x-internal') == 'keep-me'

    def test_union_variant_always_raises_even_in_lenient_mode(self):
        # UnionVariantModel sets LENIENT_STRIPPABLE=False so an unknown key can
        # still discriminate the correct variant instead of being silently dropped.
        assert v2.ApiKeySecurity.LENIENT_STRIPPABLE is False
        ctx, _ = _lenient_ctx()
        with pytest.raises(ValidationError):
            v2.ApiKeySecurity.model_validate(
                {'type': 'apiKey', 'name': 'k', 'in': 'header', 'bogus': 1},
                context=ctx,
            )


# -----------------------------------------------------------------------------
# SchemaLoader end-to-end integration
# -----------------------------------------------------------------------------


OPENAPI_31_WITH_JUNK = {
    'openapi': '3.1.0',
    'info': {'title': 'Junk API', 'version': '1.0.0', 'bogusInfoKey': True},
    'paths': {},
}

SWAGGER_20_WITH_STRAY_KEY = {
    'swagger': '2.0',
    'info': {'title': 'NSwag API', 'version': '1.0.0'},
    'paths': {
        '/pets': {
            'get': {
                'operationId': 'listPets',
                'responses': {'200': {'description': 'ok'}},
                'bogusOperationKey': {'nonsense': True},
            }
        }
    },
}


class TestSchemaLoaderLenient:
    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def _write(self, temp_dir: Path, spec: dict) -> str:
        spec_file = temp_dir / 'spec.json'
        spec_file.write_text(json.dumps(spec))
        return str(spec_file)

    def test_strict_load_raises_with_lenient_hint(self, temp_dir):
        path = self._write(temp_dir, OPENAPI_31_WITH_JUNK)
        with pytest.raises(SchemaValidationError) as exc:
            SchemaLoader().load(path)
        assert any('--lenient' in message for message in exc.value.errors)

    def test_lenient_load_succeeds_and_warns(self, temp_dir):
        path = self._write(temp_dir, OPENAPI_31_WITH_JUNK)
        loader = SchemaLoader(lenient=True)
        schema = loader.load(path)
        assert schema.info.title == 'Junk API'
        warnings = loader.get_upgrade_warnings()
        assert any('bogusInfoKey' in w for w in warnings)

    def test_lenient_swagger2_strips_stray_operation_key(self, temp_dir):
        path = self._write(temp_dir, SWAGGER_20_WITH_STRAY_KEY)

        with pytest.raises(SchemaValidationError):
            SchemaLoader().load(path)

        loader = SchemaLoader(lenient=True)
        schema = loader.load(path)
        assert schema.openapi.startswith('3.')
        warnings = loader.get_upgrade_warnings()
        assert any('bogusOperationKey' in w for w in warnings)
