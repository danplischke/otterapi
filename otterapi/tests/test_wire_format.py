"""End-to-end wire-format tests for generated clients.

Every test here generates a real client from a spec, imports it, and drives it
through an ``httpx.MockTransport`` to assert on the bytes that would go on the
wire.  The AST-level tests elsewhere prove the right *code* is emitted; these
prove the emitted code sends the right *request*, which is the failure mode
that reaches users.

Each case corresponds to a defect where httpx's own defaults disagree with the
OpenAPI wire format -- ``str()`` on an enum or datetime, ``None`` header
values, unencoded path parameters, ``repr()`` on object query parameters, and
Python-mode ``model_dump()`` on request bodies.
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
from otterapi.config import DocumentConfig

# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


def _generate(tmp_path: Path, spec: dict, package: str, **config_kwargs: Any):
    """Generate *spec* into an importable package and return the module."""
    spec_file = tmp_path / 'openapi.yaml'
    spec_file.write_text(yaml.safe_dump(spec), encoding='utf-8')

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
            m for m in sys.modules if m == package or m.startswith(f'{package}.')
        ]:
            sys.modules.pop(name, None)
        return importlib.import_module(package)
    finally:
        try:
            sys.path.remove(str(tmp_path))
        except ValueError:
            pass


class Recorder:
    """Captures the request a generated client actually sends."""

    def __init__(self, payload: Any = None) -> None:
        self.payload = payload if payload is not None else {}
        self.request: httpx.Request | None = None

    def transport(self) -> httpx.MockTransport:
        def handler(request: httpx.Request) -> httpx.Response:
            self.request = request
            return httpx.Response(200, json=self.payload)

        return httpx.MockTransport(handler)

    def bind(self, client: Any) -> Any:
        client._sync_client = httpx.Client(transport=self.transport())
        return client

    @property
    def url(self) -> httpx.URL:
        assert self.request is not None, 'no request was sent'
        return self.request.url

    @property
    def query(self) -> str:
        return self.url.query.decode()

    @property
    def body(self) -> str:
        assert self.request is not None, 'no request was sent'
        return self.request.content.decode()


def _sole_enum(mod: Any) -> Any:
    """Return the one enum a single-enum spec generated.

    Looked up by shape rather than by name so the test keeps working if the
    generator's naming for inline parameter enums changes.
    """
    import enum

    found = [
        obj
        for obj in vars(mod.models).values()
        if isinstance(obj, type)
        and issubclass(obj, enum.Enum)
        # Skip the imported `enum.Enum` base itself.
        and obj.__module__ == mod.models.__name__
    ]
    assert len(found) == 1, f'expected exactly one generated enum, got {found}'
    return found[0]


def _spec(paths: dict, schemas: dict | None = None) -> dict:
    spec: dict = {
        'openapi': '3.1.0',
        'info': {'title': 'Wire', 'version': '1.0'},
        'servers': [{'url': 'https://example.test'}],
        'paths': paths,
    }
    if schemas:
        spec['components'] = {'schemas': schemas}
    return spec


_OK = {
    '200': {
        'description': 'ok',
        'content': {'application/json': {'schema': {'type': 'object'}}},
    }
}


# ---------------------------------------------------------------------------
# Query parameters
# ---------------------------------------------------------------------------


class TestQueryParams:
    def test_enum_sends_its_value_not_its_repr(self, tmp_path: Path):
        """A ``str``-mixin enum stringifies to 'Status.ACTIVE'; the wire wants 'active'."""
        mod = _generate(
            tmp_path,
            _spec(
                {
                    '/items': {
                        'get': {
                            'operationId': 'listItems',
                            'parameters': [
                                {
                                    'name': 'status',
                                    'in': 'query',
                                    'required': True,
                                    'schema': {
                                        'type': 'string',
                                        'enum': ['active', 'archived'],
                                    },
                                }
                            ],
                            'responses': _OK,
                        }
                    }
                }
            ),
            'wire_enum',
        )
        rec = Recorder()
        status_enum = _sole_enum(mod)
        mod.list_items(status_enum.ACTIVE, client=rec.bind(mod.Client()))
        assert rec.query == 'status=active'

    def test_datetime_uses_rfc3339_separator(self, tmp_path: Path):
        """``str(datetime)`` uses a space; ``format: date-time`` wants a 'T'."""
        mod = _generate(
            tmp_path,
            _spec(
                {
                    '/items': {
                        'get': {
                            'operationId': 'listItems',
                            'parameters': [
                                {
                                    'name': 'since',
                                    'in': 'query',
                                    'required': True,
                                    'schema': {
                                        'type': 'string',
                                        'format': 'date-time',
                                    },
                                }
                            ],
                            'responses': _OK,
                        }
                    }
                }
            ),
            'wire_dt',
        )
        rec = Recorder()
        mod.list_items(
            datetime.datetime(2024, 1, 2, 3, 4, 5), client=rec.bind(mod.Client())
        )
        assert rec.url.params['since'] == '2024-01-02T03:04:05'

    def test_unset_optional_params_are_omitted(self, tmp_path: Path):
        mod = _generate(
            tmp_path,
            _spec(
                {
                    '/items': {
                        'get': {
                            'operationId': 'listItems',
                            'parameters': [
                                {
                                    'name': 'a',
                                    'in': 'query',
                                    'schema': {'type': 'string'},
                                },
                                {
                                    'name': 'b',
                                    'in': 'query',
                                    'schema': {'type': 'string'},
                                },
                            ],
                            'responses': _OK,
                        }
                    }
                }
            ),
            'wire_optional',
        )
        rec = Recorder()
        mod.list_items(b='set', client=rec.bind(mod.Client()))
        assert rec.query == 'b=set'

    def test_object_param_expands_instead_of_repr(self, tmp_path: Path):
        """The OpenAPI default for an object is form/explode -> 'a=1&b=2'."""
        mod = _generate(
            tmp_path,
            _spec(
                {
                    '/search': {
                        'get': {
                            'operationId': 'search',
                            'parameters': [
                                {
                                    'name': 'filter',
                                    'in': 'query',
                                    'schema': {'type': 'object'},
                                }
                            ],
                            'responses': _OK,
                        }
                    }
                }
            ),
            'wire_object',
        )
        rec = Recorder()
        mod.search(filter={'a': 1, 'b': 2}, client=rec.bind(mod.Client()))
        assert rec.query == 'a=1&b=2'

    @pytest.mark.parametrize(
        ('style', 'explode', 'value', 'expected'),
        [
            ('form', True, ['a', 'b'], 'tag=a&tag=b'),
            ('form', False, ['a', 'b'], 'tag=a%2Cb'),
            ('pipeDelimited', False, ['a', 'b'], 'tag=a%7Cb'),
            ('spaceDelimited', False, ['a', 'b'], 'tag=a+b'),
        ],
    )
    def test_array_styles(
        self, tmp_path: Path, style: str, explode: bool, value: list, expected: str
    ):
        mod = _generate(
            tmp_path,
            _spec(
                {
                    '/items': {
                        'get': {
                            'operationId': 'listItems',
                            'parameters': [
                                {
                                    'name': 'tag',
                                    'in': 'query',
                                    'style': style,
                                    'explode': explode,
                                    'schema': {
                                        'type': 'array',
                                        'items': {'type': 'string'},
                                    },
                                }
                            ],
                            'responses': _OK,
                        }
                    }
                }
            ),
            f'wire_style_{style.lower()}_{int(explode)}',
        )
        rec = Recorder()
        mod.list_items(tag=value, client=rec.bind(mod.Client()))
        assert rec.query == expected

    def test_deep_object_style(self, tmp_path: Path):
        mod = _generate(
            tmp_path,
            _spec(
                {
                    '/search': {
                        'get': {
                            'operationId': 'search',
                            'parameters': [
                                {
                                    'name': 'filter',
                                    'in': 'query',
                                    'style': 'deepObject',
                                    'explode': True,
                                    'schema': {'type': 'object'},
                                }
                            ],
                            'responses': _OK,
                        }
                    }
                }
            ),
            'wire_deepobject',
        )
        rec = Recorder()
        mod.search(filter={'k': 'v'}, client=rec.bind(mod.Client()))
        assert rec.query == 'filter%5Bk%5D=v'


# ---------------------------------------------------------------------------
# Header parameters
# ---------------------------------------------------------------------------


class TestHeaderParams:
    HEADER_SPEC = _spec(
        {
            '/thing': {
                'get': {
                    'operationId': 'getThing',
                    'parameters': [
                        {
                            'name': 'X-Trace',
                            'in': 'header',
                            'schema': {'type': 'string'},
                        }
                    ],
                    'responses': _OK,
                }
            }
        }
    )

    def test_unset_optional_header_does_not_raise(self, tmp_path: Path):
        """Httpx rejects a None header value; an unset header must be dropped."""
        mod = _generate(tmp_path, self.HEADER_SPEC, 'wire_hdr_unset')
        rec = Recorder()
        mod.get_thing(client=rec.bind(mod.Client()))
        assert rec.request is not None
        assert 'x-trace' not in rec.request.headers

    def test_set_header_is_sent(self, tmp_path: Path):
        mod = _generate(tmp_path, self.HEADER_SPEC, 'wire_hdr_set')
        rec = Recorder()
        mod.get_thing(X_Trace='abc', client=rec.bind(mod.Client()))
        assert rec.request is not None
        assert rec.request.headers['X-Trace'] == 'abc'

    def test_non_string_header_is_stringified(self, tmp_path: Path):
        """Httpx requires str/bytes, so an integer header must be rendered."""
        mod = _generate(
            tmp_path,
            _spec(
                {
                    '/thing': {
                        'get': {
                            'operationId': 'getThing',
                            'parameters': [
                                {
                                    'name': 'X-Count',
                                    'in': 'header',
                                    'schema': {'type': 'integer'},
                                }
                            ],
                            'responses': _OK,
                        }
                    }
                }
            ),
            'wire_hdr_int',
        )
        rec = Recorder()
        mod.get_thing(X_Count=7, client=rec.bind(mod.Client()))
        assert rec.request is not None
        assert rec.request.headers['X-Count'] == '7'


# ---------------------------------------------------------------------------
# Path parameters
# ---------------------------------------------------------------------------


class TestPathParams:
    PATH_SPEC = _spec(
        {
            '/items/{name}': {
                'get': {
                    'operationId': 'getItem',
                    'parameters': [
                        {
                            'name': 'name',
                            'in': 'path',
                            'required': True,
                            'schema': {'type': 'string'},
                        }
                    ],
                    'responses': _OK,
                }
            }
        }
    )

    def test_path_param_cannot_escape_its_segment(self, tmp_path: Path):
        """An unencoded value would inject extra path segments and a query string."""
        mod = _generate(tmp_path, self.PATH_SPEC, 'wire_path_escape')
        rec = Recorder()
        mod.get_item('a/b?evil=1', client=rec.bind(mod.Client()))
        # .path decodes; .raw_path is what actually goes on the wire.
        assert rec.url.raw_path == b'/items/a%2Fb%3Fevil%3D1'
        assert rec.query == ''

    def test_ordinary_path_param_is_unchanged(self, tmp_path: Path):
        mod = _generate(tmp_path, self.PATH_SPEC, 'wire_path_plain')
        rec = Recorder()
        mod.get_item('widget', client=rec.bind(mod.Client()))
        assert rec.url.raw_path == b'/items/widget'

    def test_enum_path_param_uses_its_value(self, tmp_path: Path):
        mod = _generate(
            tmp_path,
            _spec(
                {
                    '/items/{status}': {
                        'get': {
                            'operationId': 'getItem',
                            'parameters': [
                                {
                                    'name': 'status',
                                    'in': 'path',
                                    'required': True,
                                    'schema': {
                                        'type': 'string',
                                        'enum': ['active', 'archived'],
                                    },
                                }
                            ],
                            'responses': _OK,
                        }
                    }
                }
            ),
            'wire_path_enum',
        )
        rec = Recorder()
        status_enum = _sole_enum(mod)
        mod.get_item(status_enum.ACTIVE, client=rec.bind(mod.Client()))
        assert rec.url.raw_path == b'/items/active'


# ---------------------------------------------------------------------------
# Request bodies
# ---------------------------------------------------------------------------


_BODY_SCHEMAS = {
    'Order': {
        'type': 'object',
        'properties': {
            'id': {'type': 'integer'},
            'shipDate': {'type': 'string', 'format': 'date-time'},
            'note': {'type': 'string'},
        },
    }
}


def _body_spec(required: bool) -> dict:
    return _spec(
        {
            '/orders': {
                'post': {
                    'operationId': 'placeOrder',
                    'requestBody': {
                        'required': required,
                        'content': {
                            'application/json': {
                                'schema': {'$ref': '#/components/schemas/Order'}
                            }
                        },
                    },
                    'responses': _OK,
                }
            }
        },
        _BODY_SCHEMAS,
    )


class TestRequestBody:
    def test_datetime_field_is_json_serializable(self, tmp_path: Path):
        """Python-mode model_dump() leaves a datetime object that json= cannot encode."""
        mod = _generate(tmp_path, _body_spec(required=True), 'wire_body_dt')
        rec = Recorder()
        mod.place_order(
            mod.models.Order(id=1, shipDate=datetime.datetime(2024, 1, 2, 3, 4, 5)),
            client=rec.bind(mod.Client()),
        )
        assert rec.body == '{"id":1,"shipDate":"2024-01-02T03:04:05"}'

    def test_unset_fields_are_not_sent_as_null(self, tmp_path: Path):
        mod = _generate(tmp_path, _body_spec(required=True), 'wire_body_unset')
        rec = Recorder()
        mod.place_order(mod.models.Order(id=7), client=rec.bind(mod.Client()))
        assert rec.body == '{"id":7}'

    def test_omitted_optional_body_sends_no_payload(self, tmp_path: Path):
        """An optional body defaults to None, and None has no model_dump()."""
        mod = _generate(tmp_path, _body_spec(required=False), 'wire_body_omitted')
        rec = Recorder()
        mod.place_order(client=rec.bind(mod.Client()))
        assert rec.body == ''

    def test_optional_body_still_sends_when_given(self, tmp_path: Path):
        mod = _generate(tmp_path, _body_spec(required=False), 'wire_body_present')
        rec = Recorder()
        mod.place_order(body=mod.models.Order(id=3), client=rec.bind(mod.Client()))
        assert rec.body == '{"id":3}'


# ---------------------------------------------------------------------------
# Model annotations
# ---------------------------------------------------------------------------


def test_optional_fields_are_annotated_optional(tmp_path: Path):
    """A field defaulting to None must not claim a non-optional type."""
    mod = _generate(tmp_path, _body_spec(required=True), 'wire_annotations')
    fields = mod.models.Order.model_fields
    for name in ('id', 'shipDate', 'note'):
        assert not fields[name].is_required()
    source = (tmp_path / 'wire_annotations' / 'models.py').read_text(encoding='utf-8')
    assert 'id: int | None' in source


# ---------------------------------------------------------------------------
# Raw (unparsed) response types
# ---------------------------------------------------------------------------


def _scalar_body_spec(content_type: str) -> dict:
    return _spec(
        {
            '/blob': {
                'get': {
                    'operationId': 'getBlob',
                    'responses': {
                        '200': {
                            'description': 'ok',
                            'content': {content_type: {'schema': {'type': 'string'}}},
                        }
                    },
                }
            }
        }
    )


class TestRawResponseTypes:
    def test_string_response_returns_the_text_not_the_response(self, tmp_path: Path):
        """A function annotated ``-> str`` used to hand back an httpx Response.

        The Petstore's ``/user/login`` is exactly this shape: a JSON body whose
        schema is a bare ``type: string``.
        """
        mod = _generate(tmp_path, _scalar_body_spec('application/json'), 'wire_raw_str')
        client = mod.Client()
        client._sync_client = httpx.Client(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, text='hello')
            )
        )
        assert mod.get_blob(client=client) == 'hello'

    @pytest.mark.parametrize('content_type', ['text/plain', 'application/octet-stream'])
    def test_unparsed_content_still_yields_the_response(
        self, tmp_path: Path, content_type: str
    ):
        """Content types with no model are typed ``-> Response`` and stay raw."""
        package = 'wire_raw_' + content_type.split('/')[1].replace('-', '_')
        mod = _generate(tmp_path, _scalar_body_spec(content_type), package)
        client = mod.Client()
        client._sync_client = httpx.Client(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, text='hello')
            )
        )
        result = mod.get_blob(client=client)
        assert isinstance(result, httpx.Response)
        assert result.text == 'hello'


# ---------------------------------------------------------------------------
# request_body.exclude_unset
# ---------------------------------------------------------------------------


class TestExcludeUnsetConfig:
    """Whether an untouched optional field goes out as an explicit null.

    Most APIs read a null as "clear this field", so omitting unset fields is
    the safer default -- but a PUT that replaces a whole resource may need
    them, which is what turning this off is for.
    """

    def test_default_omits_unset_fields(self, tmp_path: Path):
        mod = _generate(tmp_path, _body_spec(required=True), 'wire_unset_default')
        rec = Recorder()
        mod.place_order(mod.models.Order(id=7), client=rec.bind(mod.Client()))
        assert rec.body == '{"id":7}'

    def test_disabled_sends_explicit_nulls(self, tmp_path: Path):
        from otterapi.config import RequestBodyConfig

        mod = _generate(
            tmp_path,
            _body_spec(required=True),
            'wire_unset_off',
            request_body=RequestBodyConfig(exclude_unset=False),
        )
        rec = Recorder()
        mod.place_order(mod.models.Order(id=7), client=rec.bind(mod.Client()))
        # Parsed rather than compared as text: field order follows the model,
        # which is not what this test is about.
        assert json.loads(rec.body) == {'id': 7, 'shipDate': None, 'note': None}

    def test_disabled_still_serializes_json_types(self, tmp_path: Path):
        """Turning off exclude_unset must not undo the mode='json' fix."""
        from otterapi.config import RequestBodyConfig

        mod = _generate(
            tmp_path,
            _body_spec(required=True),
            'wire_unset_off_dt',
            request_body=RequestBodyConfig(exclude_unset=False),
        )
        rec = Recorder()
        mod.place_order(
            mod.models.Order(id=1, shipDate=datetime.datetime(2024, 1, 2, 3, 4, 5)),
            client=rec.bind(mod.Client()),
        )
        assert '"shipDate":"2024-01-02T03:04:05"' in rec.body
