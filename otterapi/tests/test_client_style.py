"""Tests for ``client_style: 'client'`` (Client / AsyncClient method surface).

Generates a client-style package, imports it, and exercises the delegating
methods against a mock transport -- the end-to-end proof that a method call
flows through the free function to the shared request infrastructure.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import httpx
import pytest

from otterapi.codegen.codegen import Codegen
from otterapi.config import DocumentConfig

SPEC = Path(__file__).parent / 'fixtures' / 'golden' / 'constraints' / 'spec.yaml'


def _mock_handler(request: httpx.Request) -> httpx.Response:
    if request.url.path == '/users' and request.method == 'GET':
        return httpx.Response(
            200,
            json=[
                {'id': 1, 'username': 'alice', 'age': 30, 'tags': ['admin']},
                {'id': 2, 'username': 'bob_2', 'age': 28, 'tags': []},
            ],
        )
    return httpx.Response(404, json={'detail': 'unknown route'})


def _generate(target: Path, **overrides) -> None:
    config = DocumentConfig.model_validate(
        {
            'source': str(SPEC),
            'output': str(target),
            'base_url': 'https://example.test',
            **overrides,
        }
    )
    Codegen(config).generate()


def _import_fresh(parent: Path, package: str):
    sys.path.insert(0, str(parent))
    try:
        for mod in [
            m for m in list(sys.modules) if m == package or m.startswith(package + '.')
        ]:
            sys.modules.pop(mod, None)
        return importlib.import_module(package)
    finally:
        try:
            sys.path.remove(str(parent))
        except ValueError:
            pass


@pytest.fixture(scope='module')
def client_style_module(tmp_path_factory):
    parent = tmp_path_factory.mktemp('client_style')
    _generate(parent / 'cs_pkg', client_style='client')
    return _import_fresh(parent, 'cs_pkg')


class TestPublicSurface:
    def test_exports_client_classes_not_functions(self, client_style_module):
        m = client_style_module
        assert 'Client' in m.__all__
        assert 'AsyncClient' in m.__all__
        # Free functions are the implementation, not the public surface.
        assert 'list_users' not in m.__all__
        assert 'async_list_users' not in m.__all__

    def test_async_methods_have_no_async_prefix(self, client_style_module):
        ac = client_style_module.AsyncClient
        assert hasattr(ac, 'list_users')
        assert not hasattr(ac, 'async_list_users')


class TestSyncClient:
    def test_method_delegates_and_returns_models(self, client_style_module):
        Client = client_style_module.Client
        User = client_style_module.User
        with httpx.Client(transport=httpx.MockTransport(_mock_handler)) as http:
            client = Client(http_client=http)
            users = client.list_users()
        assert [u.username for u in users] == ['alice', 'bob_2']
        assert all(isinstance(u, User) for u in users)

    def test_error_propagates(self, client_style_module):
        Client = client_style_module.Client
        BaseAPIError = client_style_module.BaseAPIError

        def fail(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(404, json={'detail': 'nope'})

        with httpx.Client(transport=httpx.MockTransport(fail)) as http:
            with pytest.raises(BaseAPIError):
                Client(http_client=http).list_users()


class TestAsyncClient:
    @pytest.mark.asyncio
    async def test_method_delegates_and_returns_models(self, client_style_module):
        AsyncClient = client_style_module.AsyncClient
        User = client_style_module.User
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(_mock_handler)
        ) as http:
            client = AsyncClient(async_http_client=http)
            users = await client.list_users()
        assert [u.username for u in users] == ['alice', 'bob_2']
        assert all(isinstance(u, User) for u in users)


class TestConfigGuards:
    def test_default_style_still_exports_functions(self, tmp_path):
        _generate(tmp_path / 'fn_pkg')
        mod = _import_fresh(tmp_path, 'fn_pkg')
        assert 'list_users' in mod.__all__
        assert 'Client' in mod.__all__

    def test_client_style_with_split_raises(self, tmp_path):
        with pytest.raises(ValueError, match='not yet supported'):
            _generate(
                tmp_path / 'split_pkg',
                client_style='client',
                module_split={'enabled': True, 'strategy': 'tag'},
            )
