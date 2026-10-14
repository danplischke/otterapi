"""Tests for security-scheme (authentication) code generation.

``components.securitySchemes`` used to be parsed and then ignored -- the only
mention in the whole generator was a count printed by ``otter validate``.  Each
test here generates a client for one scheme type and drives it through an
``httpx.MockTransport`` to assert the credential lands where the spec says.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml

from otterapi.codegen.auth import (
    DEFAULT_ENV_PREFIX,
    collect_auth_schemes,
    default_env_prefix,
)
from otterapi.codegen.codegen import Codegen
from otterapi.config import AuthConfig, DocumentConfig


def _spec(schemes: dict) -> dict:
    return {
        'openapi': '3.1.0',
        'info': {'title': 'Auth', 'version': '1.0'},
        'servers': [{'url': 'https://example.test'}],
        'paths': {
            '/thing': {
                'get': {
                    'operationId': 'getThing',
                    'responses': {
                        '200': {
                            'description': 'ok',
                            'content': {
                                'application/json': {'schema': {'type': 'object'}}
                            },
                        }
                    },
                }
            }
        },
        'components': {'securitySchemes': schemes},
    }


def _generate(tmp_path: Path, spec: dict, package: str, **kwargs: Any):
    spec_file = tmp_path / 'openapi.yaml'
    spec_file.write_text(yaml.safe_dump(spec), encoding='utf-8')
    Codegen(
        DocumentConfig(source=str(spec_file), output=str(tmp_path / package), **kwargs)
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


def _send(mod, client):
    """Call the one endpoint and return the request it produced."""
    captured: dict[str, httpx.Request] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured['request'] = request
        return httpx.Response(200, json={})

    client._sync_client = httpx.Client(transport=httpx.MockTransport(handler))
    mod.get_thing(client=client)
    return captured['request']


# ---------------------------------------------------------------------------
# Scheme collection
# ---------------------------------------------------------------------------


class TestCollectAuthSchemes:
    def test_no_document_yields_nothing(self):
        assert collect_auth_schemes(None) == []

    def test_unsupported_scheme_is_skipped_not_fatal(self, tmp_path: Path):
        """An unknown scheme must not stop the rest of the client generating."""
        mod = _generate(
            tmp_path,
            _spec(
                {
                    'weird': {'type': 'http', 'scheme': 'negotiate'},
                    'key': {'type': 'apiKey', 'name': 'X-Key', 'in': 'header'},
                }
            ),
            'auth_skip',
        )
        client = mod.Client(key='k')
        assert not hasattr(client, 'weird')
        assert _send(mod, client).headers['X-Key'] == 'k'


# ---------------------------------------------------------------------------
# Per-scheme wire behaviour
# ---------------------------------------------------------------------------


class TestApiKeySchemes:
    def test_header_api_key(self, tmp_path: Path):
        mod = _generate(
            tmp_path,
            _spec({'api_key': {'type': 'apiKey', 'name': 'X-Key', 'in': 'header'}}),
            'auth_hdr',
        )
        request = _send(mod, mod.Client(api_key='secret'))
        assert request.headers['X-Key'] == 'secret'

    def test_query_api_key(self, tmp_path: Path):
        mod = _generate(
            tmp_path,
            _spec({'api_key': {'type': 'apiKey', 'name': 'key', 'in': 'query'}}),
            'auth_query',
        )
        request = _send(mod, mod.Client(api_key='secret'))
        assert request.url.params['key'] == 'secret'

    def test_cookie_api_key(self, tmp_path: Path):
        mod = _generate(
            tmp_path,
            _spec({'session': {'type': 'apiKey', 'name': 'sid', 'in': 'cookie'}}),
            'auth_cookie',
        )
        request = _send(mod, mod.Client(session='abc'))
        assert request.headers['Cookie'] == 'sid=abc'

    def test_unset_credential_is_not_sent(self, tmp_path: Path):
        mod = _generate(
            tmp_path,
            _spec({'api_key': {'type': 'apiKey', 'name': 'X-Key', 'in': 'header'}}),
            'auth_unset',
        )
        request = _send(mod, mod.Client())
        assert 'x-key' not in request.headers


class TestHttpSchemes:
    def test_bearer(self, tmp_path: Path):
        mod = _generate(
            tmp_path,
            _spec({'bearerAuth': {'type': 'http', 'scheme': 'bearer'}}),
            'auth_bearer',
        )
        request = _send(mod, mod.Client(bearer_auth='tok'))
        assert request.headers['Authorization'] == 'Bearer tok'

    def test_basic(self, tmp_path: Path):
        mod = _generate(
            tmp_path,
            _spec({'basicAuth': {'type': 'http', 'scheme': 'basic'}}),
            'auth_basic',
        )
        request = _send(mod, mod.Client(basic_auth=('user', 'pass')))
        # base64('user:pass')
        assert request.headers['Authorization'] == 'Basic dXNlcjpwYXNz'

    def test_oauth2_is_treated_as_a_bearer_token(self, tmp_path: Path):
        mod = _generate(
            tmp_path,
            _spec(
                {
                    'oauth': {
                        'type': 'oauth2',
                        'flows': {
                            'implicit': {
                                'authorizationUrl': 'https://example.test/auth',
                                'scopes': {},
                            }
                        },
                    }
                }
            ),
            'auth_oauth',
        )
        request = _send(mod, mod.Client(oauth='tok'))
        assert request.headers['Authorization'] == 'Bearer tok'


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class TestAuthConfig:
    SPEC = _spec({'api_key': {'type': 'apiKey', 'name': 'X-Key', 'in': 'header'}})

    def test_env_var_supplies_an_omitted_credential(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        mod = _generate(tmp_path, self.SPEC, 'auth_env')
        # The default prefix is the output package's name.
        monkeypatch.setenv('AUTH_ENV_API_KEY', 'from-env')
        request = _send(mod, mod.Client())
        assert request.headers['X-Key'] == 'from-env'

    def test_explicit_argument_beats_the_environment(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        mod = _generate(tmp_path, self.SPEC, 'auth_env_override')
        monkeypatch.setenv('AUTH_ENV_OVERRIDE_API_KEY', 'from-env')
        request = _send(mod, mod.Client(api_key='explicit'))
        assert request.headers['X-Key'] == 'explicit'

    def test_env_prefix_is_configurable(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Two clients in one project need distinct variables."""
        mod = _generate(
            tmp_path,
            self.SPEC,
            'auth_prefix',
            auth=AuthConfig(env_prefix='MYAPI'),
        )
        monkeypatch.setenv('MYAPI_API_KEY', 'scoped')
        request = _send(mod, mod.Client())
        assert request.headers['X-Key'] == 'scoped'

    def test_disabled_emits_no_credential_parameters(self, tmp_path: Path):
        mod = _generate(
            tmp_path, self.SPEC, 'auth_disabled', auth=AuthConfig(enabled=False)
        )
        with pytest.raises(TypeError):
            mod.Client(api_key='secret')
        assert not hasattr(mod.Client(), '_apply_auth')

    def test_no_schemes_means_no_apply_auth_method(self, tmp_path: Path):
        spec = _spec({})
        spec['components'] = {}
        mod = _generate(tmp_path, spec, 'auth_none')
        assert not hasattr(mod.Client(), '_apply_auth')


class TestEnvPrefixIsPerClient:
    """The prefix must not be shared, and must not be chosen by the document.

    The scheme name is the document's; the rest of the variable name is what
    keeps the document from picking a variable it has no business reading.
    """

    SPEC = _spec({'api_key': {'type': 'apiKey', 'name': 'X-Key', 'in': 'header'}})

    def test_one_clients_variable_is_not_read_by_another(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        (tmp_path / 'a').mkdir()
        (tmp_path / 'b').mkdir()
        internal = _generate(tmp_path / 'a', self.SPEC, 'internal_api')
        external = _generate(tmp_path / 'b', self.SPEC, 'partner_api')
        monkeypatch.setenv('INTERNAL_API_API_KEY', 'internal-secret')

        assert _send(internal, internal.Client()).headers['X-Key'] == 'internal-secret'
        assert 'X-Key' not in _send(external, external.Client()).headers

    def test_the_old_shared_prefix_is_no_longer_read(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        mod = _generate(tmp_path, self.SPEC, 'auth_noshare')
        monkeypatch.setenv('OTTER_API_KEY', 'shared')
        assert 'X-Key' not in _send(mod, mod.Client()).headers

    def test_the_document_title_does_not_choose_the_prefix(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        spec = _spec({'api_key': {'type': 'apiKey', 'name': 'X-Key', 'in': 'header'}})
        spec['info']['title'] = 'Stripe'
        mod = _generate(tmp_path, spec, 'auth_title')
        monkeypatch.setenv('STRIPE_API_KEY', 'sk_live')
        assert 'X-Key' not in _send(mod, mod.Client()).headers

    def test_empty_prefix_reads_only_the_schemes_named_in_env_vars(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        spec = _spec(
            {
                'AWS_SECRET_ACCESS_KEY': {
                    'type': 'apiKey',
                    'name': 'X-Aws',
                    'in': 'header',
                },
                'GITHUB_TOKEN': {'type': 'http', 'scheme': 'bearer'},
            }
        )
        mod = _generate(
            tmp_path,
            spec,
            'auth_empty_prefix',
            auth=AuthConfig(env_prefix='', env_vars={'GITHUB_TOKEN': 'MY_GITHUB'}),
        )
        monkeypatch.setenv('AWS_SECRET_ACCESS_KEY', 'aws-secret')
        monkeypatch.setenv('GITHUB_TOKEN', 'wrong-variable')
        monkeypatch.setenv('MY_GITHUB', 'named-variable')

        headers = _send(mod, mod.Client()).headers
        assert headers['Authorization'] == 'Bearer named-variable'
        assert 'X-Aws' not in headers
        source = (tmp_path / 'auth_empty_prefix' / '_client.py').read_text(
            encoding='utf-8'
        )
        assert 'environ.get("AWS_SECRET_ACCESS_KEY")' not in source

    def test_empty_prefix_warns_about_schemes_with_no_variable(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ):
        with caplog.at_level('WARNING', logger='otterapi.codegen.auth'):
            _generate(
                tmp_path, self.SPEC, 'auth_empty_warn', auth=AuthConfig(env_prefix='')
            )
        assert any('auth.env_vars' in record.getMessage() for record in caplog.records)

    def test_an_explicit_prefix_still_applies_to_every_scheme(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        mod = _generate(
            tmp_path, self.SPEC, 'auth_explicit', auth=AuthConfig(env_prefix='SHARED')
        )
        monkeypatch.setenv('SHARED_API_KEY', 'ok')
        assert _send(mod, mod.Client()).headers['X-Key'] == 'ok'


class TestDefaultEnvPrefix:
    @pytest.mark.parametrize(
        ('output', 'expected'),
        [
            ('./petstore_client', 'PETSTORE_CLIENT'),
            ('out/PetStoreClient', 'PET_STORE_CLIENT'),
            ('my-api.v2', 'MY_API_V2'),
            ('some/dir/billing/', 'BILLING'),
            ('123abc', '_123ABC'),
        ],
    )
    def test_derived_from_the_output_package_name(self, output, expected):
        assert default_env_prefix(output) == expected

    def test_a_dot_uses_the_current_directory_name(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        project = tmp_path / 'my_project'
        project.mkdir()
        monkeypatch.chdir(project)
        assert default_env_prefix('.') == 'MY_PROJECT'

    def test_a_nameless_path_falls_back(self):
        assert default_env_prefix('/') == DEFAULT_ENV_PREFIX


def test_before_request_hook_can_override_generated_auth(tmp_path: Path):
    """_apply_auth runs first so a hook can still refresh or replace a token."""
    mod = _generate(
        tmp_path,
        _spec({'bearerAuth': {'type': 'http', 'scheme': 'bearer'}}),
        'auth_hook',
    )

    class Refreshing(mod.Client):
        def _before_request(self, request: dict) -> dict:
            request['headers']['Authorization'] = 'Bearer refreshed'
            return request

    request = _send(mod, Refreshing(bearer_auth='stale'))
    assert request.headers['Authorization'] == 'Bearer refreshed'


class TestEnvVarNames:
    """``env_vars`` pins the exact environment variable per scheme.

    For credentials that already live somewhere the prefix rule would never
    derive.
    """

    TWO_SCHEMES = _spec(
        {
            'api_key': {'type': 'apiKey', 'name': 'X-Key', 'in': 'header'},
            'bearerAuth': {'type': 'http', 'scheme': 'bearer'},
        }
    )

    def test_named_variable_is_used(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        mod = _generate(
            tmp_path,
            self.TWO_SCHEMES,
            'auth_envvar',
            auth=AuthConfig(env_vars={'api_key': 'STRIPE_SECRET_KEY'}),
        )
        monkeypatch.setenv('STRIPE_SECRET_KEY', 'sk_test')
        assert _send(mod, mod.Client()).headers['X-Key'] == 'sk_test'

    def test_unnamed_schemes_keep_the_derived_default(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Naming one scheme must not disturb the others."""
        mod = _generate(
            tmp_path,
            self.TWO_SCHEMES,
            'auth_envvar_partial',
            auth=AuthConfig(env_vars={'api_key': 'STRIPE_SECRET_KEY'}),
        )
        monkeypatch.setenv('AUTH_ENVVAR_PARTIAL_BEARER_AUTH', 'tok')
        assert _send(mod, mod.Client()).headers['Authorization'] == 'Bearer tok'

    def test_named_variable_beats_the_prefix(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        mod = _generate(
            tmp_path,
            self.TWO_SCHEMES,
            'auth_envvar_precedence',
            auth=AuthConfig(env_prefix='MYAPI', env_vars={'api_key': 'EXACT_NAME'}),
        )
        monkeypatch.setenv('MYAPI_API_KEY', 'from-prefix')
        monkeypatch.setenv('EXACT_NAME', 'from-exact')
        assert _send(mod, mod.Client()).headers['X-Key'] == 'from-exact'

    def test_generated_parameter_name_works_as_a_key(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """``bearerAuth`` in the spec is ``bearer_auth`` in the client."""
        mod = _generate(
            tmp_path,
            self.TWO_SCHEMES,
            'auth_envvar_alias',
            auth=AuthConfig(env_vars={'bearer_auth': 'MY_TOKEN'}),
        )
        monkeypatch.setenv('MY_TOKEN', 'tok')
        assert _send(mod, mod.Client()).headers['Authorization'] == 'Bearer tok'

    def test_explicit_argument_still_wins(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        mod = _generate(
            tmp_path,
            self.TWO_SCHEMES,
            'auth_envvar_arg_wins',
            auth=AuthConfig(env_vars={'api_key': 'STRIPE_SECRET_KEY'}),
        )
        monkeypatch.setenv('STRIPE_SECRET_KEY', 'from-env')
        request = _send(mod, mod.Client(api_key='explicit'))
        assert request.headers['X-Key'] == 'explicit'
