"""Tests for spreading a request body into individual parameters.

``request_body.flatten`` turns ``place_order(body=Order(id=7))`` into
``place_order(id=7)``.  The wire payload must come out identical either way,
which is what most of these assert: flattening is a signature change, not a
protocol change.
"""

from __future__ import annotations

import datetime
import importlib
import json
import sys
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml

from otterapi.codegen.codegen import Codegen
from otterapi.config import (
    DocumentConfig,
    EndpointRequestBodyConfig,
    RequestBodyConfig,
)

_OK = {
    '200': {
        'description': 'ok',
        'content': {'application/json': {'schema': {'type': 'object'}}},
    }
}


def _post(operation_id: str, ref: str, required: bool = True, **extra) -> dict:
    body = {
        'operationId': operation_id,
        'requestBody': {
            'required': required,
            'content': {'application/json': {'schema': {'$ref': ref}}},
        },
        'responses': _OK,
    }
    body.update(extra)
    return {'post': body}


SPEC: dict = {
    'openapi': '3.1.0',
    'info': {'title': 'Flat', 'version': '1.0'},
    'servers': [{'url': 'https://example.test'}],
    'paths': {
        '/orders': _post('placeOrder', '#/components/schemas/Order', required=False),
        '/pets': _post('addPet', '#/components/schemas/Pet'),
        '/pets/{petId}/notes': _post(
            'addNote',
            '#/components/schemas/Note',
            parameters=[
                {
                    'name': 'petId',
                    'in': 'path',
                    'required': True,
                    'schema': {'type': 'integer'},
                },
                {'name': 'status', 'in': 'query', 'schema': {'type': 'string'}},
            ],
        ),
        '/composed': _post('createComposed', '#/components/schemas/Derived'),
    },
    'components': {
        'schemas': {
            'Order': {
                'type': 'object',
                'properties': {
                    'id': {'type': 'integer'},
                    'shipDate': {'type': 'string', 'format': 'date-time'},
                    'note': {'type': 'string'},
                },
            },
            'Pet': {
                'type': 'object',
                'required': ['name'],
                'properties': {
                    'name': {'type': 'string'},
                    'tag': {'type': 'string'},
                },
            },
            'Note': {
                'type': 'object',
                'required': ['status', 'x-source'],
                'properties': {
                    'status': {'type': 'string'},
                    'x-source': {'type': 'string'},
                },
            },
            'Base': {
                'type': 'object',
                'required': ['baseRequired'],
                'properties': {
                    'baseRequired': {'type': 'string'},
                    'baseOptional': {'type': 'integer'},
                },
            },
            'Derived': {
                'allOf': [
                    {'$ref': '#/components/schemas/Base'},
                    {
                        'type': 'object',
                        'required': ['own'],
                        'properties': {'own': {'type': 'string'}},
                    },
                ]
            },
        }
    },
}


def _generate(tmp_path: Path, package: str, **config_kwargs: Any):
    spec_file = tmp_path / 'openapi.yaml'
    spec_file.write_text(yaml.safe_dump(SPEC), encoding='utf-8')
    Codegen(
        DocumentConfig(
            source=str(spec_file),
            output=str(tmp_path / package),
            **config_kwargs,
        )
    ).generate()

    sys.path.insert(0, str(tmp_path))
    try:
        for name in [
            m for m in list(sys.modules) if m == package or m.startswith(f'{package}.')
        ]:
            sys.modules.pop(name, None)
        return importlib.import_module(package)
    finally:
        try:
            sys.path.remove(str(tmp_path))
        except ValueError:
            pass


class Sent:
    """The last request a generated client produced."""

    def __init__(self) -> None:
        self.request: httpx.Request | None = None

    def bind(self, client: Any) -> Any:
        def handler(request: httpx.Request) -> httpx.Response:
            self.request = request
            return httpx.Response(200, json={})

        client._sync_client = httpx.Client(transport=httpx.MockTransport(handler))
        return client

    @property
    def body(self) -> str:
        assert self.request is not None
        return self.request.content.decode()

    @property
    def url(self) -> httpx.URL:
        assert self.request is not None
        return self.request.url


@pytest.fixture
def flat(tmp_path: Path):
    return _generate(tmp_path, 'flat_on', request_body=RequestBodyConfig(flatten=True))


# ---------------------------------------------------------------------------
# Signature shape
# ---------------------------------------------------------------------------


class TestSignature:
    def test_body_parameter_is_replaced_by_the_fields(self, flat):
        import inspect

        params = inspect.signature(flat.place_order).parameters
        assert 'body' not in params
        assert {'id', 'shipDate', 'note'} <= set(params)

    def test_required_fields_are_positional(self, flat):
        import inspect

        params = inspect.signature(flat.add_pet).parameters
        assert params['name'].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        assert params['name'].default is inspect.Parameter.empty
        assert params['tag'].kind is inspect.Parameter.KEYWORD_ONLY
        assert params['tag'].default is flat.NOTSET

    def test_inherited_fields_are_included(self, flat):
        """AllOf composition puts required fields on a base class."""
        import inspect

        params = inspect.signature(flat.create_composed).parameters
        assert {'baseRequired', 'own', 'baseOptional'} <= set(params)

    def test_off_by_default(self, tmp_path: Path):
        import inspect

        mod = _generate(tmp_path, 'flat_default')
        assert 'body' in inspect.signature(mod.place_order).parameters


# ---------------------------------------------------------------------------
# Wire payload
# ---------------------------------------------------------------------------


class TestPayload:
    def test_only_supplied_fields_are_sent(self, flat):
        sent = Sent()
        flat.place_order(id=7, client=sent.bind(flat.Client()))
        assert sent.body == '{"id":7}'

    def test_matches_the_unflattened_payload(self, tmp_path: Path):
        """Flattening changes the signature, not the request."""
        flat_mod = _generate(
            tmp_path, 'flat_cmp_on', request_body=RequestBodyConfig(flatten=True)
        )
        plain_mod = _generate(tmp_path, 'flat_cmp_off')

        flat_sent, plain_sent = Sent(), Sent()
        when = datetime.datetime(2024, 1, 2, 3, 4, 5)
        flat_mod.place_order(
            id=7, shipDate=when, client=flat_sent.bind(flat_mod.Client())
        )
        plain_mod.place_order(
            body=plain_mod.models.Order(id=7, shipDate=when),
            client=plain_sent.bind(plain_mod.Client()),
        )
        assert json.loads(flat_sent.body) == json.loads(plain_sent.body)

    def test_json_types_still_serialize(self, flat):
        sent = Sent()
        flat.place_order(
            shipDate=datetime.datetime(2024, 1, 2, 3, 4, 5),
            client=sent.bind(flat.Client()),
        )
        assert sent.body == '{"shipDate":"2024-01-02T03:04:05"}'

    def test_optional_body_with_no_arguments_sends_nothing(self, flat):
        """As the unflattened form does, rather than an empty object."""
        sent = Sent()
        flat.place_order(client=sent.bind(flat.Client()))
        assert sent.body == ''

    def test_field_alias_is_used_on_the_wire(self, flat):
        """``x-source`` is not an identifier, so the model carries an alias."""
        sent = Sent()
        flat.add_note(1, 'open', 'api', client=sent.bind(flat.Client()))
        assert json.loads(sent.body) == {'status': 'open', 'x-source': 'api'}

    def test_a_missing_required_field_still_fails(self, flat):
        with pytest.raises(Exception):
            flat.add_pet()  # name is required


class TestParameterCollisions:
    """A body field and a real parameter can share a name."""

    def test_the_body_field_is_renamed(self, flat):
        import inspect

        params = inspect.signature(flat.add_note).parameters
        # The query parameter keeps `status`; the body field yields.
        assert params['status'].kind is inspect.Parameter.KEYWORD_ONLY
        assert 'status_body' in params

    def test_each_value_reaches_its_own_place(self, flat):
        sent = Sent()
        flat.add_note(
            1,
            'body-status',
            'api',
            status='query-status',
            client=sent.bind(flat.Client()),
        )
        assert sent.url.params['status'] == 'query-status'
        assert json.loads(sent.body)['status'] == 'body-status'


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class TestPathOverrides:
    def test_a_path_can_opt_in(self, tmp_path: Path):
        import inspect

        mod = _generate(
            tmp_path,
            'flat_path_in',
            request_body=RequestBodyConfig(
                flatten=False,
                paths={'/orders': EndpointRequestBodyConfig(flatten=True)},
            ),
        )
        assert 'body' not in inspect.signature(mod.place_order).parameters
        assert 'body' in inspect.signature(mod.add_pet).parameters

    def test_a_path_can_opt_out(self, tmp_path: Path):
        import inspect

        mod = _generate(
            tmp_path,
            'flat_path_out',
            request_body=RequestBodyConfig(
                flatten=True,
                paths={'/orders': EndpointRequestBodyConfig(flatten=False)},
            ),
        )
        assert 'body' in inspect.signature(mod.place_order).parameters
        assert 'body' not in inspect.signature(mod.add_pet).parameters

    def test_globs_match_and_the_longest_wins(self, tmp_path: Path):
        import inspect

        mod = _generate(
            tmp_path,
            'flat_path_glob',
            request_body=RequestBodyConfig(
                flatten=False,
                paths={
                    '/pets*': EndpointRequestBodyConfig(flatten=True),
                    '/pets/{petId}/notes': EndpointRequestBodyConfig(flatten=False),
                },
            ),
        )
        assert 'body' not in inspect.signature(mod.add_pet).parameters
        assert 'body' in inspect.signature(mod.add_note).parameters

    def test_exclude_unset_is_overridable_per_path(self, tmp_path: Path):
        mod = _generate(
            tmp_path,
            'flat_path_unset',
            request_body=RequestBodyConfig(
                flatten=True,
                paths={'/orders': EndpointRequestBodyConfig(exclude_unset=False)},
            ),
        )
        sent = Sent()
        mod.place_order(id=7, client=sent.bind(mod.Client()))
        assert json.loads(sent.body) == {'id': 7, 'shipDate': None, 'note': None}


def test_unflattenable_bodies_keep_the_body_parameter(tmp_path: Path):
    """An array body has no fields to spread; generation must not fail."""
    import inspect

    spec = dict(SPEC)
    spec['paths'] = dict(SPEC['paths'])
    spec['paths']['/bulk'] = {
        'post': {
            'operationId': 'bulkCreate',
            'requestBody': {
                'required': True,
                'content': {
                    'application/json': {
                        'schema': {
                            'type': 'array',
                            'items': {'$ref': '#/components/schemas/Pet'},
                        }
                    }
                },
            },
            'responses': _OK,
        }
    }
    spec_file = tmp_path / 'openapi.yaml'
    spec_file.write_text(yaml.safe_dump(spec), encoding='utf-8')
    output = tmp_path / 'flat_array'
    Codegen(
        DocumentConfig(
            source=str(spec_file),
            output=str(output),
            request_body=RequestBodyConfig(flatten=True),
        )
    ).generate()

    sys.path.insert(0, str(tmp_path))
    try:
        mod = importlib.import_module('flat_array')
    finally:
        sys.path.remove(str(tmp_path))
    assert 'body' in inspect.signature(mod.bulk_create).parameters


# ---------------------------------------------------------------------------
# The NOTSET sentinel
# ---------------------------------------------------------------------------


class TestNotSet:
    """Optional flattened fields default to NOTSET, not None.

    ``None`` already means "send a JSON null" for a nullable field, so it
    cannot also mean "leave this out" -- the two need different defaults or
    one of them is unreachable.
    """

    def test_omitting_a_field_leaves_it_out(self, flat):
        sent = Sent()
        flat.place_order(id=7, client=sent.bind(flat.Client()))
        assert json.loads(sent.body) == {'id': 7}

    def test_an_explicit_none_sends_a_json_null(self, flat):
        sent = Sent()
        flat.place_order(id=7, note=None, client=sent.bind(flat.Client()))
        assert json.loads(sent.body) == {'id': 7, 'note': None}

    def test_passing_notset_is_the_same_as_omitting(self, flat):
        """So a caller can decide field-by-field from a variable."""
        sent = Sent()
        flat.place_order(id=7, note=flat.NOTSET, client=sent.bind(flat.Client()))
        assert json.loads(sent.body) == {'id': 7}

    def test_notset_is_exported_from_the_package(self, flat):
        assert flat.NOTSET is flat.NotSet.NOTSET
        assert repr(flat.NOTSET) == 'NOTSET'
        # Falsy, so `if value:` guards treat it as absent.
        assert not flat.NOTSET

    def test_the_sentinel_never_reaches_the_wire(self, flat):
        """Even if one is handed to a query parameter by mistake."""
        sent = Sent()
        flat.add_note(
            1, 'open', 'api', status=flat.NOTSET, client=sent.bind(flat.Client())
        )
        assert 'status' not in sent.url.params

    def test_required_fields_take_no_sentinel(self, flat):
        import inspect

        params = inspect.signature(flat.add_pet).parameters
        assert params['name'].default is inspect.Parameter.empty
