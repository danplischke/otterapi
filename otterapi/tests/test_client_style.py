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
from pydantic import ValidationError

from otterapi.codegen.codegen import Codegen
from otterapi.config import DocumentConfig

GOLDEN = Path(__file__).parent / 'fixtures' / 'golden'
SPEC = GOLDEN / 'constraints' / 'spec.yaml'
TAGGED_SPEC = GOLDEN / 'tagged' / 'spec.yaml'
NESTED_SPEC = GOLDEN / 'nested' / 'spec.yaml'
PAGINATED_SPEC = GOLDEN / 'paginated' / 'spec.yaml'
ENVELOPE_SPEC = GOLDEN / 'envelope' / 'spec.yaml'
SHADOW_SPEC = GOLDEN / 'shadow' / 'spec.yaml'


def _gen(target: Path, source: Path, **overrides) -> None:
    Codegen(
        DocumentConfig.model_validate(
            {
                'source': str(source),
                'output': str(target),
                'base_url': 'https://example.test',
                **overrides,
            }
        )
    ).generate()


_PAGES = {
    0: [{'id': 1, 'name': 'a'}, {'id': 2, 'name': 'b'}],
    2: [{'id': 3, 'name': 'c'}],
}


def _paginated_handler(request: httpx.Request) -> httpx.Response:
    offset = int(dict(request.url.params).get('offset', 0))
    return httpx.Response(200, json=_PAGES.get(offset, []))


def _nested_handler(request: httpx.Request) -> httpx.Response:
    if request.url.path == '/identity/users':
        return httpx.Response(200, json=[{'id': 1, 'name': 'alice'}])
    if request.url.path == '/identity/users/1':
        return httpx.Response(200, json={'id': 1, 'name': 'alice'})
    if request.url.path == '/billing/invoices':
        return httpx.Response(200, json=[{'id': 9, 'amount': 4.2}])
    return httpx.Response(404, json={'detail': 'unknown route'})


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
    if request.url.path == '/users/1':
        return httpx.Response(200, json={'id': 1, 'name': 'alice'})
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


def _generate_nested(target: Path, resource_naming: str, **overrides) -> None:
    config = DocumentConfig.model_validate(
        {
            'source': str(NESTED_SPEC),
            'output': str(target),
            'base_url': 'https://example.test',
            'client_style': 'resource',
            'resource_naming': resource_naming,
            **overrides,
        }
    )
    Codegen(config).generate()


class TestNestedResources:
    @pytest.mark.parametrize('naming', ['path', 'operation_id'])
    def test_nested_chain_calls(self, tmp_path, naming):
        _generate_nested(tmp_path / f'nst_{naming}', naming)
        mod = _import_fresh(tmp_path, f'nst_{naming}')
        with httpx.Client(transport=httpx.MockTransport(_nested_handler)) as http:
            client = mod.Client(http_client=http)
            # Two levels of nesting: client.identity.users.<method>
            users = client.identity.users.list()
            one = client.identity.users.get(userId=1)
            invoices = client.billing.invoices.list()
        assert [u.name for u in users] == ['alice']
        assert one.id == 1
        assert invoices[0].amount == 4.2

    @pytest.mark.asyncio
    async def test_nested_async(self, tmp_path):
        _generate_nested(tmp_path / 'nst_a', 'path')
        mod = _import_fresh(tmp_path, 'nst_a')
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(_nested_handler)
        ) as http:
            client = mod.AsyncClient(async_http_client=http)
            users = await client.identity.users.list()
        assert [u.name for u in users] == ['alice']

    def test_tag_mode_stays_flat(self, tmp_path):
        # Default resource_naming keeps the single-level grouping.
        _generate_nested(tmp_path / 'nst_flat', 'tag')
        mod = _import_fresh(tmp_path, 'nst_flat')
        client = mod.Client(base_url='https://example.test')
        # No nested identity.users chain; the tag/hybrid strategy groups flatly.
        assert not hasattr(getattr(client, 'identity', object()), 'users')


class TestResultObjects:
    def _generate_ro(self, target: Path, source: Path, **overrides) -> None:
        config = DocumentConfig.model_validate(
            {
                'source': str(source),
                'output': str(target),
                'base_url': 'https://example.test',
                'client_style': 'resource',
                'result_objects': True,
                **overrides,
            }
        )
        Codegen(config).generate()

    def test_query_terminals(self, tmp_path):
        self._generate_ro(
            tmp_path / 'ro',
            PAGINATED_SPEC,
            pagination={'enabled': True, 'auto_detect': True, 'default_page_size': 2},
            dataframe={'enabled': True, 'pandas': True},
            export={'enabled': True, 'formats': ['csv']},
        )
        mod = _import_fresh(tmp_path, 'ro')
        out = tmp_path / 'out.csv'
        with httpx.Client(transport=httpx.MockTransport(_paginated_handler)) as http:
            client = mod.Client(http_client=http)
            assert [r.id for r in client.items.list().all()] == [1, 2, 3]
            assert client.items.list().to_pandas().shape == (3, 2)
            assert [r.id for r in client.items.list().iter()] == [1, 2, 3]
            assert client.items.list().export(str(out)) == 3
        assert out.exists()

    @pytest.mark.asyncio
    async def test_query_terminals_async(self, tmp_path):
        self._generate_ro(
            tmp_path / 'roa',
            PAGINATED_SPEC,
            pagination={'enabled': True, 'auto_detect': True, 'default_page_size': 2},
            dataframe={'enabled': True, 'pandas': True},
        )
        mod = _import_fresh(tmp_path, 'roa')
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(_paginated_handler)
        ) as http:
            client = mod.AsyncClient(async_http_client=http)
            assert [r.id for r in await client.items.list().all()] == [1, 2, 3]
            assert [r.id async for r in client.items.list().iter()] == [1, 2, 3]

    def test_list_returns_query_scalar_stays_plain(self, tmp_path):
        # A list endpoint collapses its variants into a Query; a scalar endpoint
        # keeps returning the model directly.
        self._generate_ro(
            tmp_path / 'ros',
            TAGGED_SPEC,
            dataframe={'enabled': True, 'pandas': True},
        )
        mod = _import_fresh(tmp_path, 'ros')
        with httpx.Client(transport=httpx.MockTransport(_tagged_handler)) as http:
            client = mod.Client(http_client=http)
            listed = client.users.list()
            assert type(listed).__name__ == 'Query'
            assert hasattr(listed, 'to_pandas')
            assert [u.name for u in listed.all()] == ['alice']
            user = client.users.get(userId=1)
            assert type(user).__name__ == 'User'
        # The separate _df variant method is gone (collapsed into the Query).
        assert not hasattr(mod.Client(base_url='x').users, 'list_df')

    def test_flat_client_style_result_objects(self, tmp_path):
        # result_objects also works with the flat client style.
        self._generate_ro(
            tmp_path / 'roc',
            PAGINATED_SPEC,
            client_style='client',
            pagination={'enabled': True, 'auto_detect': True, 'default_page_size': 2},
            dataframe={'enabled': True, 'pandas': True},
        )
        mod = _import_fresh(tmp_path, 'roc')
        with httpx.Client(transport=httpx.MockTransport(_paginated_handler)) as http:
            q = mod.Client(http_client=http).list_items()
            assert type(q).__name__ == 'Query'
            assert [r.id for r in q.all()] == [1, 2, 3]

    def test_requires_class_style(self, tmp_path):
        # result_objects is rejected at config load with the functions style.
        with pytest.raises(ValidationError, match='requires client_style'):
            DocumentConfig.model_validate(
                {
                    'source': str(TAGGED_SPEC),
                    'output': str(tmp_path / 'bad'),
                    'base_url': 'https://example.test',
                    'client_style': 'functions',
                    'result_objects': True,
                }
            )


class TestNameCollisions:
    def test_resource_name_does_not_shadow_client_member(self, tmp_path):
        # A resource named like a base-client member (here 'close') must not
        # override it; it is renamed so client.close() stays callable.
        _gen(
            tmp_path / 'sh',
            SHADOW_SPEC,
            client_style='resource',
            resource_naming='path',
        )
        mod = _import_fresh(tmp_path, 'sh')
        client = mod.Client(base_url='https://example.test')
        assert callable(client.close)
        assert not isinstance(type(client).__dict__.get('close'), property)
        assert hasattr(type(client), 'close_')  # the resource moved aside


class TestFeatureCombinations:
    """Combinations that compose across features -- guard against regressions."""

    def test_split_plus_result_objects(self, tmp_path):
        _gen(
            tmp_path / 'sro',
            PAGINATED_SPEC,
            client_style='resource',
            result_objects=True,
            module_split={'enabled': True, 'strategy': 'path'},
            pagination={'enabled': True, 'auto_detect': True, 'default_page_size': 2},
            dataframe={'enabled': True, 'pandas': True},
        )
        mod = _import_fresh(tmp_path, 'sro')
        with httpx.Client(transport=httpx.MockTransport(_paginated_handler)) as http:
            client = mod.Client(http_client=http)
            q = client.items.list()
            assert type(q).__name__ == 'Query'
            assert [r.id for r in q.all()] == [1, 2, 3]

    def test_response_unwrap_plus_resource(self, tmp_path):
        # The method should return the unwrapped list, not the envelope.
        _gen(
            tmp_path / 'unw',
            ENVELOPE_SPEC,
            client_style='resource',
            response_unwrap={'enabled': True, 'data_path': 'data'},
        )
        mod = _import_fresh(tmp_path, 'unw')

        def handler(_r: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200, json={'data': [{'id': 1}, {'id': 2}], 'status': 'ok'}
            )

        with httpx.Client(transport=httpx.MockTransport(handler)) as http:
            result = mod.Client(http_client=http).items.list()
        assert [i.id for i in result] == [1, 2]

    def test_reexport_models_plus_resource(self, tmp_path):
        _gen(
            tmp_path / 'rx', TAGGED_SPEC, client_style='resource', reexport_models=True
        )
        mod = _import_fresh(tmp_path, 'rx')
        assert 'Client' in mod.__all__
        assert 'User' in mod.__all__


class TestConfigGuards:
    def test_default_style_still_exports_functions(self, tmp_path):
        _generate(tmp_path / 'fn_pkg')
        mod = _import_fresh(tmp_path, 'fn_pkg')
        assert 'list_users' in mod.__all__
        assert 'Client' in mod.__all__
