"""Per-call raw response access and request logging (issue #29)."""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

import httpx2
import pytest

from otterapi.codegen.client import log_env_var_name
from otterapi.tests.helpers import generate_package

SPEC = {
    'openapi': '3.1.0',
    'info': {'title': 'Pet Store', 'version': '1'},
    'servers': [{'url': 'https://example.test'}],
    'components': {
        'securitySchemes': {
            'apiKeyQuery': {'type': 'apiKey', 'in': 'query', 'name': 'sig'},
            'bearer': {'type': 'http', 'scheme': 'bearer'},
        },
        'schemas': {
            'User': {
                'type': 'object',
                'required': ['id'],
                'properties': {'id': {'type': 'integer'}, 'name': {'type': 'string'}},
            },
            'Login': {
                'type': 'object',
                'properties': {
                    'username': {'type': 'string'},
                    'password': {'type': 'string'},
                },
            },
        },
    },
    'security': [{'apiKeyQuery': []}, {'bearer': []}],
    'paths': {
        '/users/{user_id}': {
            'get': {
                'operationId': 'get_user',
                'parameters': [
                    {
                        'name': 'user_id',
                        'in': 'path',
                        'required': True,
                        'schema': {'type': 'integer'},
                    }
                ],
                'responses': {
                    '200': {
                        'description': 'ok',
                        'content': {
                            'application/json': {
                                'schema': {'$ref': '#/components/schemas/User'}
                            }
                        },
                    }
                },
            }
        },
        '/login': {
            'post': {
                'operationId': 'login',
                'requestBody': {
                    'content': {
                        'application/json': {
                            'schema': {'$ref': '#/components/schemas/Login'}
                        }
                    }
                },
                'responses': {
                    '200': {
                        'description': 'ok',
                        'content': {
                            'application/json': {
                                'schema': {'$ref': '#/components/schemas/User'}
                            }
                        },
                    }
                },
            }
        },
        '/raw': {
            'get': {
                'operationId': 'get_raw',
                'responses': {
                    '200': {
                        'description': 'ok',
                        'content': {'text/plain': {'schema': {'type': 'string'}}},
                    }
                },
            }
        },
    },
}


def handler(request: httpx2.Request) -> httpx2.Response:
    if request.url.path == '/raw':
        return httpx2.Response(200, text='plain')
    if request.url.path == '/users/404':
        return httpx2.Response(404, json={'detail': 'nope'}, headers={'x-req': 'e1'})
    return httpx2.Response(
        200,
        json={'id': 1, 'name': 'Ada'},
        headers={'x-ratelimit-remaining': '41', 'set-cookie': 'session=abc'},
    )


def _generate(tmp_path: Path, package: str, **config: Any) -> Any:
    return generate_package(tmp_path, package, SPEC, **config)


def _client(pkg: Any, **kwargs: Any) -> Any:
    transport = httpx2.MockTransport(handler)
    return pkg.Client(
        http_client=httpx2.Client(transport=transport),
        async_http_client=httpx2.AsyncClient(transport=transport),
        max_retries=0,
        **kwargs,
    )


@pytest.fixture(scope='module')
def fn_pkg(tmp_path_factory: pytest.TempPathFactory) -> Any:
    return _generate(tmp_path_factory.mktemp('raw_fn'), 'raw_fn', raw_response=True)


class TestRawResponse:
    def test_headers_status_and_parse(self, fn_pkg: Any) -> None:
        client = _client(fn_pkg)
        raw = fn_pkg.get_user_with_response(user_id=1, client=client)
        assert raw.status_code == 200
        assert raw.headers['x-ratelimit-remaining'] == '41'
        assert raw.http_response.request.url.path == '/users/1'
        assert raw.parse() == fn_pkg.get_user(user_id=1, client=client)
        assert raw.parse() is raw.parse()  # parsed once

    @pytest.mark.asyncio
    async def test_async(self, fn_pkg: Any) -> None:
        raw = await fn_pkg.async_get_user_with_response(
            user_id=1, client=_client(fn_pkg)
        )
        assert raw.headers['x-ratelimit-remaining'] == '41'
        user = await raw.parse()
        assert user.name == 'Ada'

    def test_errors_still_raise_with_the_response(self, fn_pkg: Any) -> None:
        with pytest.raises(fn_pkg.NotFoundError) as err:
            fn_pkg.get_user_with_response(user_id=404, client=_client(fn_pkg))
        assert err.value.response.headers['x-req'] == 'e1'

    def test_raw_endpoints_get_no_variant(self, fn_pkg: Any) -> None:
        assert hasattr(fn_pkg, 'get_raw')
        assert not hasattr(fn_pkg, 'get_raw_with_response')

    def test_off_by_default(self, tmp_path: Path) -> None:
        pkg = _generate(tmp_path, 'raw_off')
        assert not hasattr(pkg, 'get_user_with_response')
        assert not (tmp_path / 'raw_off' / '_raw_response.py').exists()

    @pytest.mark.parametrize('style', ['client', 'resource'])
    def test_class_clients(self, tmp_path: Path, style: str) -> None:
        pkg = _generate(tmp_path, f'raw_{style}', raw_response=True, client_style=style)
        client = _client(pkg)
        target = client if style == 'client' else client.users
        raw = target.get_user_with_response(user_id=1)
        assert raw.headers['x-ratelimit-remaining'] == '41'
        assert raw.parse().id == 1


class TestLogging:
    def test_info_summary(self, fn_pkg: Any, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.INFO, logger='raw_fn._client'):
            fn_pkg.get_user(user_id=1, client=_client(fn_pkg, api_key_query='s3cr3t'))
        [record] = [r for r in caplog.records if r.name == 'raw_fn._client']
        message = record.getMessage()
        assert message.startswith('GET https://example.test/users/1?sig=')
        assert '-> 200 (' in message and 'ms)' in message
        assert 's3cr3t' not in message and 'sig=[REDACTED]' in message

    def test_debug_details_are_redacted(
        self, fn_pkg: Any, caplog: pytest.LogCaptureFixture
    ) -> None:
        client = _client(fn_pkg, bearer='tok-123', api_key_query='s3cr3t')
        with caplog.at_level(logging.DEBUG, logger='raw_fn._client'):
            fn_pkg.login(
                body=fn_pkg.Login(username='ada', password='hunter2'), client=client
            )
        text = '\n'.join(
            r.getMessage() for r in caplog.records if r.name == 'raw_fn._client'
        )
        assert '"username": "ada"' in text
        assert 'x-ratelimit-remaining' in text
        for secret in ('hunter2', 'tok-123', 's3cr3t', 'session=abc'):
            assert secret not in text
        assert text.count('[REDACTED]') >= 3

    def test_silent_by_default(self, fn_pkg: Any, capfd: pytest.CaptureFixture) -> None:
        fn_pkg.get_user(user_id=1, client=_client(fn_pkg))
        out, err = capfd.readouterr()
        assert out == '' and err == ''

    def test_env_var_turns_logging_on(self, tmp_path: Path) -> None:
        _generate(tmp_path, 'env_log')
        script = (
            'import httpx2, env_log\n'
            'c = env_log.Client(http_client=httpx2.Client('
            'transport=httpx2.MockTransport(lambda r: httpx2.Response(200, '
            'json={"id": 1}))))\n'
            'env_log.get_user(user_id=1, client=c)\n'
        )
        env_var = log_env_var_name('PetStoreClient')
        assert env_var == 'PET_STORE_LOG'
        result = subprocess.run(
            [sys.executable, '-c', script],
            cwd=tmp_path,
            env={'PATH': '', env_var: 'debug'},
            capture_output=True,
            text=True,
            check=True,
        )
        assert 'env_log._client INFO GET https://example.test/users/1' in result.stderr
        assert 'DEBUG Response: 200' in result.stderr


@pytest.mark.parametrize(
    ('name', 'expected'),
    [
        ('PetStoreClient', 'PET_STORE_LOG'),
        ('Client', 'API_LOG'),
        ('HTTPBinClient', 'HTTPBIN_LOG'),
        ('3dApiClient', 'API_3D_API_LOG'),
    ],
)
def test_env_var_name(name: str, expected: str) -> None:
    assert log_env_var_name(name) == expected
