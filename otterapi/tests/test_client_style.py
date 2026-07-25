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
TAGGED_SPEC = Path(__file__).parent / 'fixtures' / 'golden' / 'tagged' / 'spec.yaml'


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


def _tagged_handler(request: httpx.Request) -> httpx.Response:
    if request.url.path == '/users':
        return httpx.Response(200, json=[{'id': 1, 'name': 'alice'}])
    if request.url.path == '/orders':
        return httpx.Response(200, json=[{'id': 7, 'total': 9.5}])
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


@pytest.fixture(scope='module')
def resource_style_module(tmp_path_factory):
    parent = tmp_path_factory.mktemp('resource_style')
    config = DocumentConfig.model_validate(
        {
            'source': str(TAGGED_SPEC),
            'output': str(parent / 'rs_pkg'),
            'base_url': 'https://example.test',
            'client_style': 'resource',
        }
    )
    Codegen(config).generate()
    return _import_fresh(parent, 'rs_pkg')


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


class TestResourceStyle:
    def test_resources_grouped_with_stripped_method_names(self, resource_style_module):
        Client = resource_style_module.Client
        client = Client(base_url='https://example.test')
        # Grouped under resource sub-clients ...
        assert hasattr(client, 'users')
        assert hasattr(client, 'orders')
        # ... with the resource token stripped from the method name.
        assert hasattr(client.users, 'list')
        assert hasattr(client.users, 'get')
        assert not hasattr(client.users, 'list_users')

    def test_sync_resource_call_returns_models(self, resource_style_module):
        Client = resource_style_module.Client
        User = resource_style_module.User
        with httpx.Client(transport=httpx.MockTransport(_tagged_handler)) as http:
            client = Client(http_client=http)
            users = client.users.list()
            orders = client.orders.list()
        assert [u.name for u in users] == ['alice']
        assert all(isinstance(u, User) for u in users)
        assert orders[0].total == 9.5

    @pytest.mark.asyncio
    async def test_async_resource_call_returns_models(self, resource_style_module):
        AsyncClient = resource_style_module.AsyncClient
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(_tagged_handler)
        ) as http:
            client = AsyncClient(async_http_client=http)
            users = await client.users.list()
        assert [u.name for u in users] == ['alice']

    def test_exports_client_classes(self, resource_style_module):
        m = resource_style_module
        assert 'Client' in m.__all__ and 'AsyncClient' in m.__all__
        assert 'list_users' not in m.__all__


def _generate_tagged(target: Path, **overrides) -> None:
    config = DocumentConfig.model_validate(
        {
            'source': str(TAGGED_SPEC),
            'output': str(target),
            'base_url': 'https://example.test',
            **overrides,
        }
    )
    Codegen(config).generate()


class TestSplitClientStyle:
    """client_style combined with module_split: functions live in split modules,
    the client methods route to them."""

    def test_split_resource_calls_across_modules(self, tmp_path):
        _generate_tagged(
            tmp_path / 'sr',
            client_style='resource',
            module_split={'enabled': True, 'strategy': 'tag'},
        )
        # Functions live in per-tag module files, not a single endpoints.py.
        assert (tmp_path / 'sr' / 'users.py').exists()
        assert (tmp_path / 'sr' / 'orders.py').exists()
        assert not (tmp_path / 'sr' / 'endpoints.py').exists()

        mod = _import_fresh(tmp_path, 'sr')
        assert 'Client' in mod.__all__ and 'AsyncClient' in mod.__all__
        assert 'list_users' not in mod.__all__
        with httpx.Client(transport=httpx.MockTransport(_tagged_handler)) as http:
            client = mod.Client(http_client=http)
            assert [u.name for u in client.users.list()] == ['alice']
            assert client.orders.list()[0].total == 9.5

    def test_split_client_flat_calls_across_modules(self, tmp_path):
        _generate_tagged(
            tmp_path / 'sc',
            client_style='client',
            module_split={'enabled': True, 'strategy': 'tag'},
        )
        mod = _import_fresh(tmp_path, 'sc')
        assert hasattr(mod.Client, 'list_users')
        assert hasattr(mod.Client, 'list_orders')
        with httpx.Client(transport=httpx.MockTransport(_tagged_handler)) as http:
            client = mod.Client(http_client=http)
            assert [u.name for u in client.list_users()] == ['alice']

    @pytest.mark.asyncio
    async def test_split_resource_async(self, tmp_path):
        _generate_tagged(
            tmp_path / 'sra',
            client_style='resource',
            module_split={'enabled': True, 'strategy': 'tag'},
        )
        mod = _import_fresh(tmp_path, 'sra')
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(_tagged_handler)
        ) as http:
            client = mod.AsyncClient(async_http_client=http)
            users = await client.users.list()
        assert [u.name for u in users] == ['alice']


class TestConfigGuards:
    def test_default_style_still_exports_functions(self, tmp_path):
        _generate(tmp_path / 'fn_pkg')
        mod = _import_fresh(tmp_path, 'fn_pkg')
        assert 'list_users' in mod.__all__
        assert 'Client' in mod.__all__
