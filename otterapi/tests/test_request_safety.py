"""Tests for the guards that keep a generated request on the URL the call names.

Two inputs can steer a request somewhere the caller did not intend:

* a path-parameter value -- ``..``, ``.`` or an empty string survives
  percent-encoding and is then resolved away by the HTTP stack, so
  ``remove_member('acme', '..')`` used to send ``DELETE /orgs/acme``;
* a path from the spec -- appended to the base URL by plain concatenation, a key
  such as ``.attacker.example/x`` or ``@attacker.example/x`` rewrote the host
  and carried the caller's credentials along.
"""

from __future__ import annotations

import importlib
import sys
from typing import Any

import httpx
import pytest
import yaml

from otterapi.codegen.codegen import Codegen
from otterapi.codegen.runtime._serialization import format_path_param, join_url
from otterapi.config import DocumentConfig

SPEC: dict[str, Any] = {
    'openapi': '3.0.0',
    'info': {'title': 'Org API', 'version': '1.0.0'},
    'servers': [{'url': 'https://api.example.test/v1'}],
    'paths': {
        '/orgs/{org_id}/members/{member_id}': {
            'delete': {
                'operationId': 'removeMember',
                'parameters': [
                    {
                        'in': 'path',
                        'name': 'org_id',
                        'required': True,
                        'schema': {'type': 'string'},
                    },
                    {
                        'in': 'path',
                        'name': 'member_id',
                        'required': True,
                        'schema': {'type': 'string'},
                    },
                ],
                'responses': {'204': {'description': 'gone'}},
            }
        },
        '/labels/{label}': {
            'get': {
                'operationId': 'getLabel',
                'parameters': [
                    {
                        'in': 'path',
                        'name': 'label',
                        'required': True,
                        'style': 'label',
                        'schema': {'type': 'string'},
                    }
                ],
                'responses': {'200': {'description': 'ok'}},
            }
        },
    },
}


@pytest.fixture(scope='module')
def pkg(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp('request_safety')
    spec_file = tmp_path / 'openapi.yaml'
    spec_file.write_text(yaml.safe_dump(SPEC), encoding='utf-8')
    Codegen(
        DocumentConfig(source=str(spec_file), output=str(tmp_path / 'safety_pkg'))
    ).generate()
    sys.path.insert(0, str(tmp_path))
    try:
        for name in [m for m in sys.modules if m.split('.')[0] == 'safety_pkg']:
            sys.modules.pop(name, None)
        yield importlib.import_module('safety_pkg')
    finally:
        sys.path.remove(str(tmp_path))


class Wire:
    """A client whose requests are recorded instead of sent."""

    def __init__(self, pkg: Any, **client_kwargs: Any) -> None:
        self.requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            self.requests.append(request)
            return httpx.Response(200, json={})

        self.client = pkg.Client(
            http_client=httpx.Client(transport=httpx.MockTransport(handler)),
            **client_kwargs,
        )


class TestFormatPathParam:
    @pytest.mark.parametrize('value', ['..', '.', ''])
    def test_dot_segments_and_empty_values_are_rejected(self, value):
        with pytest.raises(ValueError, match='path parameter'):
            format_path_param(value)

    def test_none_is_rejected_instead_of_becoming_the_text_none(self):
        with pytest.raises(ValueError, match='None'):
            format_path_param(None)

    @pytest.mark.parametrize('value', [[], {}, ['..'], ['.']])
    def test_containers_that_render_to_a_dot_segment_are_rejected(self, value):
        with pytest.raises(ValueError):
            format_path_param(value)

    @pytest.mark.parametrize(
        ('value', 'expected'),
        [
            ('plain', 'plain'),
            ('v1.0', 'v1.0'),
            ('a..b', 'a..b'),
            ('...', '...'),
            ('..hidden', '..hidden'),
            ('a/b', 'a%2Fb'),
            ('a?x=1', 'a%3Fx%3D1'),
            ('a#frag', 'a%23frag'),
            ('../admin', '..%2Fadmin'),
            # Already-encoded input is encoded again, so it stays literal.
            ('%2e%2e', '%252e%252e'),
            (0, '0'),
            (['a', '..'], 'a,..'),
        ],
    )
    def test_everything_else_is_unchanged(self, value, expected):
        assert format_path_param(value) == expected

    def test_label_style_keeps_its_prefix(self):
        assert format_path_param('red', style='label') == '.red'

    def test_matrix_style_allows_an_empty_value(self):
        assert format_path_param('', style='matrix', name='color') == ';color='


class TestJoinUrl:
    BASE = 'https://api.example.test/v1'

    def test_joins_a_normal_path(self):
        assert join_url(self.BASE, '/users/7') == f'{self.BASE}/users/7'

    def test_an_empty_path_addresses_the_base_url(self):
        assert join_url(self.BASE, '') == self.BASE

    @pytest.mark.parametrize(
        'path',
        [
            'users',
            '.attacker.example/collect',
            '@attacker.example/collect',
            ':8443/x',
            'attacker.example/x',
        ],
    )
    def test_a_path_without_a_leading_slash_is_rejected(self, path):
        with pytest.raises(ValueError, match='start with'):
            join_url('https://api.example.test', path)

    @pytest.mark.parametrize(
        'path',
        [
            '/a/../b',
            '/a/./b',
            '/..',
            '/a/..',
            '/a/%2e%2e/b',
            '/a/%2E%2E/b',
            '/a/%2e/b',
            '/a/..?x=1',
            '/a/../b#frag',
        ],
    )
    def test_dot_segments_are_rejected_even_when_percent_encoded(self, path):
        with pytest.raises(ValueError, match='segments'):
            join_url(self.BASE, path)

    @pytest.mark.parametrize(
        'path',
        [
            '/a/b..c',
            '/a/..b',
            '/v1.0/x',
            '/a/...',
            '/search?q=../x',
            '/search?q=a#../frag',
            '//double/slash',
        ],
    )
    def test_look_alikes_are_allowed(self, path):
        assert join_url(self.BASE, path) == f'{self.BASE}{path}'

    def test_a_leading_double_slash_cannot_change_the_host(self):
        url = join_url('https://api.example.test', '//attacker.example/x')
        assert httpx.URL(url).host == 'api.example.test'


class TestGeneratedClient:
    def test_dot_segment_value_never_reaches_the_wire(self, pkg):
        wire = Wire(pkg)
        for value in ('..', '.', ''):
            with pytest.raises(ValueError):
                pkg.remove_member('acme', value, client=wire.client)
        assert wire.requests == []

    def test_a_value_with_a_slash_stays_in_its_segment(self, pkg):
        wire = Wire(pkg)
        pkg.remove_member('acme', '../admin', client=wire.client)
        assert wire.requests[0].url.raw_path == b'/v1/orgs/acme/members/..%2Fadmin'

    def test_label_style_empty_value_is_caught_on_the_assembled_path(self, pkg):
        wire = Wire(pkg)
        with pytest.raises(ValueError, match='segments'):
            pkg.get_label('', client=wire.client)
        assert wire.requests == []

    def test_request_refuses_a_path_that_would_change_the_host(self, pkg):
        wire = Wire(pkg, base_url='https://api.trusted.example')
        for path in ('.attacker.example/collect', '@attacker.example/collect'):
            with pytest.raises(ValueError, match='start with'):
                wire.client._request('get', path)
        assert wire.requests == []

    def test_request_refuses_a_dot_segment_in_the_path(self, pkg):
        wire = Wire(pkg)
        with pytest.raises(ValueError, match='segments'):
            wire.client._request('get', '/orgs/../admin')
        assert wire.requests == []

    @pytest.mark.asyncio
    async def test_async_requests_are_guarded_too(self, pkg):
        calls: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request)
            return httpx.Response(200, json={})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            client = pkg.Client(async_http_client=http)
            with pytest.raises(ValueError):
                await pkg.async_remove_member('acme', '..', client=client)
            with pytest.raises(ValueError, match='start with'):
                await client._request_async('get', '@attacker.example/x')
        assert calls == []
