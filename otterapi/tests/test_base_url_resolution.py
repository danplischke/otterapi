"""Tests for how the generated client's default base URL is chosen.

The default base URL is written into a source file that gets committed, so two
inputs must not leak into it: credentials carried by the URL the document was
fetched from, and server URLs the HTTP client cannot use.  A Swagger 2.0
document that does not list ``schemes`` must not silently turn into a plain-http
client either.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pytest

from otterapi.codegen.codegen import Codegen
from otterapi.codegen.schema import SchemaLoader
from otterapi.config import DocumentConfig

_OK = {'200': {'description': 'ok'}}


def _openapi(server_url: str) -> dict[str, Any]:
    return {
        'openapi': '3.1.0',
        'info': {'title': 'Base', 'version': '1.0'},
        'servers': [{'url': server_url}],
        'paths': {'/thing': {'get': {'operationId': 'getThing', 'responses': _OK}}},
    }


def _codegen(tmp_path: Path, spec: dict[str, Any], source: str | None = None):
    spec_file = tmp_path / 'openapi.json'
    spec_file.write_text(json.dumps(spec), encoding='utf-8')
    codegen = Codegen(
        DocumentConfig(source=str(spec_file), output=str(tmp_path / 'client'))
    )
    codegen._load_schema()
    if source is not None:
        # Loaded over HTTP the source is a URL; swap it in after loading so the
        # test needs no server.
        codegen.config.source = source
    return codegen


class TestSourceUrlSecretsAreNotBakedIn:
    @pytest.mark.parametrize(
        ('source', 'server', 'expected'),
        [
            (
                'https://alice:hunter2@api.example.test/docs/openapi.json',
                '/',
                'https://api.example.test/',
            ),
            (
                'https://api.example.test/docs/openapi.json?api_key=SECRET123',
                '/v1',
                'https://api.example.test/v1',
            ),
            (
                'https://api.example.test/docs/openapi.json?api_key=SECRET123',
                '?',
                'https://api.example.test/docs/openapi.json',
            ),
            (
                'https://api.example.test/docs/openapi.json#token=SECRET123',
                '#',
                'https://api.example.test/docs/openapi.json',
            ),
            (
                'https://user@api.example.test:8443/docs/openapi.json',
                '../v2',
                'https://api.example.test:8443/v2',
            ),
            (
                'http://bob:pw@[::1]:8000/openapi.json',
                '/api',
                'http://[::1]:8000/api',
            ),
        ],
    )
    def test_resolved_default_carries_no_userinfo_query_or_fragment(
        self, tmp_path, source, server, expected
    ):
        codegen = _codegen(tmp_path, _openapi(server), source)
        resolved = codegen._resolve_base_url()
        assert resolved == expected
        for secret in ('hunter2', 'SECRET123', 'alice', 'bob', 'pw@'):
            assert secret not in resolved

    def test_the_log_line_does_not_repeat_the_secrets(self, tmp_path, caplog):
        codegen = _codegen(
            tmp_path,
            _openapi('/v1'),
            'https://alice:hunter2@api.example.test/openapi.json?api_key=SECRET123',
        )
        with caplog.at_level(logging.DEBUG, logger='otterapi.codegen.codegen'):
            codegen._resolve_base_url()
        logged = '\n'.join(record.getMessage() for record in caplog.records)
        assert 'api.example.test' in logged
        assert 'hunter2' not in logged
        assert 'SECRET123' not in logged

    def test_generated_client_has_no_secret_anywhere(self, tmp_path):
        spec_file = tmp_path / 'openapi.json'
        spec_file.write_text(json.dumps(_openapi('/v1')), encoding='utf-8')
        output = tmp_path / 'client'
        codegen = Codegen(DocumentConfig(source=str(spec_file), output=str(output)))
        codegen._load_schema()
        codegen.config.source = (
            'https://alice:hunter2@api.example.test/o.json?k=SECRET123'
        )
        # Resolve exactly as generate() would, then write the real package.
        codegen.config.base_url = codegen._resolve_base_url()
        codegen.config.source = str(spec_file)
        codegen.generate()
        for path in output.rglob('*.py'):
            text = path.read_text(encoding='utf-8')
            assert 'hunter2' not in text
            assert 'SECRET123' not in text


class TestServerSchemes:
    @pytest.mark.parametrize(
        'server',
        [
            'file:///etc/passwd',
            'javascript:alert(1)',
            'ftp://files.example.test/api',
            'wss://stream.example.test/api',
        ],
    )
    def test_a_document_server_that_is_not_http_is_rejected(self, tmp_path, server):
        codegen = _codegen(tmp_path, _openapi(server))
        with pytest.raises(ValueError, match='unsupported scheme') as excinfo:
            codegen._resolve_base_url()
        assert '-b/--base-url' in str(excinfo.value)

    @pytest.mark.parametrize('server', ['javascript:alert(1)', 'file:///etc/hosts'])
    def test_a_relative_server_cannot_smuggle_in_a_scheme(self, tmp_path, server):
        codegen = _codegen(
            tmp_path, _openapi(server), 'https://api.example.test/openapi.json'
        )
        with pytest.raises(ValueError, match='unsupported scheme'):
            codegen._resolve_base_url()

    @pytest.mark.parametrize(
        'server',
        ['https://api.example.test/v1', 'http://localhost:8000', 'HTTPS://X.test'],
    )
    def test_http_and_https_servers_are_accepted(self, tmp_path, server):
        assert _codegen(tmp_path, _openapi(server))._resolve_base_url() == server

    def test_a_relative_server_from_a_file_is_still_kept_as_is(self, tmp_path):
        assert _codegen(tmp_path, _openapi('/api/v3'))._resolve_base_url() == '/api/v3'

    def test_the_users_own_base_url_is_not_second_guessed(self, tmp_path):
        spec_file = tmp_path / 'openapi.json'
        spec_file.write_text(json.dumps(_openapi('/v1')), encoding='utf-8')
        codegen = Codegen(
            DocumentConfig(
                source=str(spec_file),
                output=str(tmp_path / 'client'),
                base_url='http://127.0.0.1:9000',
            )
        )
        assert codegen._resolve_base_url() == 'http://127.0.0.1:9000'


SWAGGER_NO_SCHEMES: dict[str, Any] = {
    'swagger': '2.0',
    'info': {'title': 'Legacy', 'version': '1.0'},
    'host': 'api.legacy.example',
    'basePath': '/v1',
    'securityDefinitions': {'key': {'type': 'apiKey', 'in': 'header', 'name': 'X-Key'}},
    'security': [{'key': []}],
    'paths': {'/things': {'get': {'operationId': 'listThings', 'responses': _OK}}},
}


class TestSwagger2DefaultScheme:
    def _load(self, tmp_path: Path, spec: dict[str, Any]):
        spec_file = tmp_path / 'swagger.json'
        spec_file.write_text(json.dumps(spec), encoding='utf-8')
        loader = SchemaLoader()
        return loader.load(str(spec_file)), loader.get_upgrade_warnings()

    def test_no_schemes_defaults_to_https(self, tmp_path):
        document, _ = self._load(tmp_path, SWAGGER_NO_SCHEMES)
        assert [server.url for server in document.servers] == [
            'https://api.legacy.example/v1'
        ]

    def test_the_default_is_reported(self, tmp_path):
        _, warnings = self._load(tmp_path, SWAGGER_NO_SCHEMES)
        assert any('schemes' in warning and 'https' in warning for warning in warnings)

    def test_an_explicit_http_scheme_is_respected_and_not_reported(self, tmp_path):
        document, warnings = self._load(
            tmp_path, {**SWAGGER_NO_SCHEMES, 'schemes': ['http']}
        )
        assert [server.url for server in document.servers] == [
            'http://api.legacy.example/v1'
        ]
        assert not any("No 'schemes'" in warning for warning in warnings)

    def test_an_explicit_https_scheme_is_unchanged(self, tmp_path):
        document, _ = self._load(tmp_path, {**SWAGGER_NO_SCHEMES, 'schemes': ['https']})
        assert [server.url for server in document.servers] == [
            'https://api.legacy.example/v1'
        ]
