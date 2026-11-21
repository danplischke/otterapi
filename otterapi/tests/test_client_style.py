"""Tests for ``client_style: 'client'`` (Client / AsyncClient method surface).

Generates a client-style package, imports it, and exercises the delegating
methods against a mock transport -- the end-to-end proof that a method call
flows through the free function to the shared request infrastructure.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import json
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
    def test_resources_grouped_keeping_function_names(self, resource_style_module):
        Client = resource_style_module.Client
        client = Client(base_url='https://example.test')
        # Grouped under resource sub-clients ...
        assert hasattr(client, 'users')
        assert hasattr(client, 'orders')
        # ... with each method named exactly like its function.
        assert hasattr(client.users, 'list_users')
        assert hasattr(client.users, 'get_user')
        assert not hasattr(client.users, 'list')
        assert not hasattr(client.users, 'get')

    def test_sync_resource_call_returns_models(self, resource_style_module):
        Client = resource_style_module.Client
        User = resource_style_module.User
        with httpx.Client(transport=httpx.MockTransport(_tagged_handler)) as http:
            client = Client(http_client=http)
            users = client.users.list_users()
            orders = client.orders.list_orders()
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
            users = await client.users.list_users()
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
    """client_style combined with module_split.

    Functions live in split modules; the client methods route to them.
    """

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
            assert [u.name for u in client.users.list_users()] == ['alice']
            assert client.orders.list_orders()[0].total == 9.5

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
            users = await client.users.list_users()
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
            # Two levels of nesting: client.identity.users.<function name>
            users = client.identity.users.identity_users_list()
            one = client.identity.users.identity_users_get(userId=1)
            invoices = client.billing.invoices.billing_invoices_list()
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
            users = await client.identity.users.identity_users_list()
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
            assert [r.id for r in client.items.list_items().all()] == [1, 2, 3]
            assert client.items.list_items().to_pandas().shape == (3, 2)
            assert [r.id for r in client.items.list_items().iter()] == [1, 2, 3]
            assert client.items.list_items().export(str(out)) == 3
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
            assert [r.id for r in await client.items.list_items().all()] == [1, 2, 3]
            assert [r.id async for r in client.items.list_items().iter()] == [1, 2, 3]

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
            listed = client.users.list_users()
            assert type(listed).__name__ == 'Query'
            assert hasattr(listed, 'to_pandas')
            assert [u.name for u in listed.all()] == ['alice']
            user = client.users.get_user(userId=1)
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
            q = client.items.list_items()
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
            result = mod.Client(http_client=http).items.list_items()
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


def _delegated_endpoints(package_dir: Path) -> dict[str, set[str]]:
    """Endpoint functions the generated class surface delegates to, per side.

    Walks every class in ``_clients.py`` (``Client`` / ``AsyncClient`` and any
    resource sub-clients) and collects the ``<endpoints alias>.<fn>`` references
    inside it. That is exactly the set of operations a facade composed over the
    surface can reach, so two layouts with equal sets offer the same reach.
    """
    tree = ast.parse((package_dir / '_clients.py').read_text())
    # Non-split: ``from . import endpoints as _endpoints``; split: one
    # ``from . import <module> as _ep_<module>`` per endpoints module.
    aliases: set[str] = set()
    for node in tree.body:
        if not isinstance(node, ast.ImportFrom) or node.level != 1:
            continue
        if node.module is not None and not node.module.startswith('endpoints'):
            continue
        aliases.update(a.asname for a in node.names if a.asname)
    reach: dict[str, set[str]] = {'sync': set(), 'async': set()}
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        # ``AsyncClient`` / ``_<Path>ResourceAsync`` are the async side.
        is_async = node.name.startswith('Async') or node.name.endswith('Async')
        side = 'async' if is_async else 'sync'
        for sub in ast.walk(node):
            if (
                isinstance(sub, ast.Attribute)
                and isinstance(sub.value, ast.Name)
                and sub.value.id in aliases
            ):
                reach[side].add(sub.attr)
    return reach


def _own_public_methods(cls: type) -> set[str]:
    return {n for n, v in vars(cls).items() if not n.startswith('_') and callable(v)}


class TestComposedFacade:
    """The ``client`` layout as the base for a hand-written, user-facing SDK.

    The recommended way to ship your own API on top of a generated one is to
    *compose* -- hold a generated ``Client`` privately and expose your own
    methods -- rather than subclass. That only works if the flat surface has
    feature parity with the other layouts, which these tests pin down.
    """

    _FEATURES = {
        'pagination': {'enabled': True, 'auto_detect': True, 'default_page_size': 2},
        'dataframe': {'enabled': True, 'pandas': True},
        'export': {'enabled': True, 'formats': ['csv']},
    }

    @pytest.mark.parametrize('split', [False, True], ids=['flat', 'split'])
    @pytest.mark.parametrize('result_objects', [False, True], ids=['methods', 'query'])
    def test_flat_surface_reaches_every_resource_operation(
        self, tmp_path, split, result_objects
    ):
        # Whatever a ``resource``-style user can call, a facade composed over
        # the ``client`` style can call too -- on both the sync and async side.
        overrides = dict(self._FEATURES, result_objects=result_objects)
        if split:
            overrides['module_split'] = {'enabled': True, 'strategy': 'path'}
        reach = {}
        for style in ('client', 'resource'):
            target = tmp_path / f'{style}_{int(split)}_{int(result_objects)}'
            _generate_tagged(target, client_style=style, **overrides)
            reach[style] = _delegated_endpoints(target)
        assert reach['client']['sync'], 'no delegations found'
        assert reach['client']['sync'] == reach['resource']['sync']
        assert reach['client']['async'] == reach['resource']['async']

    def test_sync_and_async_surfaces_match(self, tmp_path):
        _gen(
            tmp_path / 'sa',
            PAGINATED_SPEC,
            client_style='client',
            result_objects=True,
            **self._FEATURES,
        )
        mod = _import_fresh(tmp_path, 'sa')
        assert _own_public_methods(mod.Client) == _own_public_methods(mod.AsyncClient)
        reach = _delegated_endpoints(tmp_path / 'sa')
        assert {n.removeprefix('async_') for n in reach['async']} == reach['sync']

    def test_composed_facade_reaches_every_terminal(self, tmp_path):
        _gen(
            tmp_path / 'cf',
            PAGINATED_SPEC,
            client_style='client',
            result_objects=True,
            **self._FEATURES,
        )
        mod = _import_fresh(tmp_path, 'cf')

        class Catalog:
            """A hand-written facade; only what it defines is public."""

            def __init__(self, api):
                self._api = api

            def items(self):
                return self._api.list_items()

            def item_names(self) -> list[str]:
                return [i.name for i in self.items().all()]

        out = tmp_path / 'items.csv'
        with httpx.Client(transport=httpx.MockTransport(_paginated_handler)) as http:
            catalog = Catalog(mod.Client(http_client=http))
            assert isinstance(catalog.items(), mod.Query)
            assert catalog.item_names() == ['a', 'b', 'c']
            assert [r.id for r in catalog.items().iter()] == [1, 2, 3]
            assert catalog.items().to_pandas().shape == (3, 2)
            assert catalog.items().export(str(out)) == 3
        assert out.exists()

    @pytest.mark.asyncio
    async def test_composed_facade_async(self, tmp_path):
        _gen(
            tmp_path / 'cfa',
            PAGINATED_SPEC,
            client_style='client',
            result_objects=True,
            **self._FEATURES,
        )
        mod = _import_fresh(tmp_path, 'cfa')

        class Catalog:
            def __init__(self, api):
                self._api = api

            def items(self):
                return self._api.list_items()

        out = tmp_path / 'items_async.csv'
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(_paginated_handler)
        ) as http:
            catalog = Catalog(mod.AsyncClient(async_http_client=http))
            assert isinstance(catalog.items(), mod.AsyncQuery)
            assert [r.id for r in await catalog.items().all()] == [1, 2, 3]
            assert [r.id async for r in catalog.items().iter()] == [1, 2, 3]
            assert (await catalog.items().to_pandas()).shape == (3, 2)
            assert await catalog.items().export(str(out)) == 3
        assert out.exists()

    def test_composed_facade_can_translate_errors(self, client_style_module):
        # A user-facing SDK maps the generated error type onto its own.
        mod = client_style_module

        class DirectoryError(Exception):
            pass

        class Directory:
            def __init__(self, api):
                self._api = api

            def users(self):
                try:
                    return self._api.list_users()
                except mod.BaseAPIError as e:
                    raise DirectoryError(e.status_code) from e

        def fail(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(503, json={'detail': 'down'})

        with httpx.Client(transport=httpx.MockTransport(fail)) as http:
            with pytest.raises(DirectoryError) as info:
                Directory(mod.Client(http_client=http)).users()
        assert info.value.args == (503,)
        assert isinstance(info.value.__cause__, mod.BaseAPIError)


# ---------------------------------------------------------------------------
# Regressions for the review of the class-style layouts (see PR discussion).
# ---------------------------------------------------------------------------

_ITEM_SCHEMA = {
    'type': 'object',
    'required': ['id', 'name'],
    'properties': {'id': {'type': 'integer'}, 'name': {'type': 'string'}},
}
_ITEM_LIST_RESPONSE = {
    '200': {
        'description': 'ok',
        'content': {
            'application/json': {
                'schema': {
                    'type': 'array',
                    'items': {'$ref': '#/components/schemas/Item'},
                }
            }
        },
    }
}
_ITEM_RESPONSE = {
    '200': {
        'description': 'ok',
        'content': {
            'application/json': {'schema': {'$ref': '#/components/schemas/Item'}}
        },
    }
}


def _write_spec(target: Path, paths: dict, schemas: dict | None = None) -> Path:
    spec = {
        'openapi': '3.0.0',
        'info': {'title': 'T', 'version': '1'},
        'paths': paths,
        'components': {'schemas': {'Item': _ITEM_SCHEMA, **(schemas or {})}},
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(spec))
    return target


def _list_op(operation_id: str, params: list | None = None) -> dict:
    op: dict = {'operationId': operation_id, 'responses': _ITEM_LIST_RESPONSE}
    if params:
        op['parameters'] = params
    return op


def _function(module_src: str, name: str) -> ast.FunctionDef | ast.AsyncFunctionDef:
    for node in ast.parse(module_src).body:
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == name
        ):
            return node
    raise AssertionError(f'{name} not generated')


class TestReviewRegressions:
    def test_class_styles_import_signature_types(self, tmp_path):
        # Imports come from the endpoint sink, so a date-time / uuid parameter
        # no longer leaves ``datetime`` / ``UUID`` undefined in _clients.py.
        spec = _write_spec(
            tmp_path / 'spec.json',
            {
                '/items': {
                    'get': _list_op(
                        'listItems',
                        [
                            {
                                'in': 'query',
                                'name': 'since',
                                'schema': {'type': 'string', 'format': 'date-time'},
                            },
                            {
                                'in': 'query',
                                'name': 'owner',
                                'schema': {'type': 'string', 'format': 'uuid'},
                            },
                        ],
                    )
                }
            },
        )
        for style in ('client', 'resource'):
            _gen(tmp_path / f'imp_{style}', spec, client_style=style)
        mod = _import_fresh(tmp_path, 'imp_client')
        params = inspect.signature(mod.Client.list_items).parameters
        assert 'datetime' in str(params['since'].annotation)
        assert 'UUID' in str(params['owner'].annotation)

    def test_variant_suffix_operation_ids_survive_result_objects(self, tmp_path):
        # Families are grouped by owning endpoint, so an operation whose own
        # name ends in a variant suffix is a core method, not a dropped stray.
        spec = _write_spec(
            tmp_path / 'spec.json',
            {
                '/items': {'get': _list_op('listItems')},
                '/export': {
                    'post': {'operationId': 'createExport', 'responses': _ITEM_RESPONSE}
                },
                '/iter': {
                    'get': {'operationId': 'getIter', 'responses': _ITEM_RESPONSE}
                },
            },
        )
        _gen(
            tmp_path / 'vs',
            spec,
            client_style='client',
            result_objects=True,
            dataframe={'enabled': True, 'pandas': True},
        )
        mod = _import_fresh(tmp_path, 'vs')
        expected = {'create_export', 'get_iter', 'list_items'}
        assert expected <= _own_public_methods(mod.Client)
        assert expected <= _own_public_methods(mod.AsyncClient)

    def test_every_list_endpoint_gets_its_variants(self, tmp_path):
        # The file-level feature flags must not short-circuit emission: the
        # second and third list endpoints get DataFrame / export variants too.
        spec = _write_spec(
            tmp_path / 'spec.json',
            {f'/{n}': {'get': _list_op(f'list{n.upper()}')} for n in ('a', 'b', 'c')},
        )
        features = {
            'dataframe': {'enabled': True, 'pandas': True},
            'export': {'enabled': True, 'formats': ['csv']},
        }
        _gen(tmp_path / 'ev', spec, **features)
        src = (tmp_path / 'ev' / 'endpoints.py').read_text()
        for name in ('list_a', 'list_b', 'list_c'):
            assert f'def {name}_df(' in src, name
            assert f'def {name}_export(' in src, name

        _gen(
            tmp_path / 'evq',
            spec,
            client_style='client',
            result_objects=True,
            **features,
        )
        mod = _import_fresh(tmp_path, 'evq')
        client = mod.Client(base_url='https://example.test')
        for name in ('list_a', 'list_b', 'list_c'):
            assert isinstance(getattr(client, name)(), mod.Query), name

    def test_optional_unwrapped_list_keeps_query_item_type(self, tmp_path):
        envelope = {
            'type': 'object',
            'properties': {
                'data': {
                    'type': 'array',
                    'items': {'$ref': '#/components/schemas/Item'},
                },
                'status': {'type': 'string'},
            },
        }
        spec = _write_spec(
            tmp_path / 'spec.json',
            {
                '/items': {
                    'get': {
                        'operationId': 'listItems',
                        'responses': {
                            '200': {
                                'description': 'ok',
                                'content': {
                                    'application/json': {
                                        'schema': {
                                            '$ref': '#/components/schemas/Envelope'
                                        }
                                    }
                                },
                            }
                        },
                    }
                }
            },
            {'Envelope': envelope},
        )
        _gen(
            tmp_path / 'oq',
            spec,
            client_style='client',
            result_objects=True,
            response_unwrap={'enabled': True, 'data_path': 'data'},
            dataframe={'enabled': True, 'pandas': True},
        )
        src = (tmp_path / 'oq' / '_clients.py').read_text()
        assert 'Query[Item]' in src
        assert 'Query[Any]' not in src

    @pytest.mark.asyncio
    async def test_async_query_iterates_without_pagination(self, tmp_path):
        # ``async for`` over a result object falls back to the materialized
        # list, exactly like ``for`` over the sync one.
        _gen(
            tmp_path / 'aq',
            TAGGED_SPEC,
            client_style='resource',
            result_objects=True,
            dataframe={'enabled': True, 'pandas': True},
        )
        mod = _import_fresh(tmp_path, 'aq')
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(_tagged_handler)
        ) as http:
            client = mod.AsyncClient(async_http_client=http)
            assert [u.name async for u in client.users.list_users()] == ['alice']

    def test_resources_mirror_paths_and_keep_function_names(self, tmp_path):
        spec = _write_spec(
            tmp_path / 'spec.json',
            {
                '/pet/findByStatus': {'get': _list_op('findPetsByStatus')},
                '/pet/{petId}': {
                    'get': {
                        'operationId': 'getPetById',
                        'parameters': [
                            {
                                'in': 'path',
                                'name': 'petId',
                                'required': True,
                                'schema': {'type': 'integer'},
                            }
                        ],
                        'responses': _ITEM_RESPONSE,
                    }
                },
                '/user/login': {
                    'get': {'operationId': 'loginUser', 'responses': _ITEM_RESPONSE}
                },
                '/billing/invoices': {'get': _list_op('listInvoices')},
            },
        )
        _gen(
            tmp_path / 'nofold',
            spec,
            client_style='resource',
            resource_naming='path',
            dataframe={'enabled': True, 'pandas': True},
        )
        mod = _import_fresh(tmp_path, 'nofold')
        client = mod.Client(base_url='https://example.test')
        # Every path segment is a resource -- single-operation "action"
        # segments are not folded into their parent -- and every method keeps
        # its function's name, variants included.
        assert _own_public_methods(type(client.pet.find_by_status)) == {
            'find_pets_by_status',
            'find_pets_by_status_df',
        }
        assert 'get_pet_by_id' in _own_public_methods(type(client.pet))
        assert _own_public_methods(type(client.user.login)) == {'login_user'}
        assert 'list_invoices' in _own_public_methods(type(client.billing.invoices))

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == '/pet/findByStatus':
                return httpx.Response(200, json=[{'id': 1, 'name': 'rex'}])
            return httpx.Response(404, json={'detail': 'unknown route'})

        with httpx.Client(transport=httpx.MockTransport(handler)) as http:
            pets = mod.Client(http_client=http).pet.find_by_status.find_pets_by_status()
        assert [p.name for p in pets] == ['rex']

    def test_export_wrapper_survives_colliding_parameter(self, tmp_path):
        # An endpoint parameter named ``format`` no longer produces a duplicate
        # argument: it is aliased ``format_`` in the wrapper and forwarded under
        # its real name, and Query.export applies the same aliasing.
        spec = _write_spec(
            tmp_path / 'spec.json',
            {
                '/items': {
                    'get': _list_op(
                        'listItems',
                        [
                            {
                                'in': 'query',
                                'name': 'format',
                                'schema': {'type': 'string'},
                            }
                        ],
                    )
                }
            },
        )
        _gen(
            tmp_path / 'xf',
            spec,
            client_style='client',
            result_objects=True,
            export={'enabled': True, 'formats': ['csv']},
        )
        mod = _import_fresh(tmp_path, 'xf')
        endpoints = importlib.import_module('xf.endpoints')
        seen: list[str | None] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request.url.params.get('format'))
            return httpx.Response(200, json=[{'id': 1, 'name': 'a'}])

        out = tmp_path / 'items.csv'
        with httpx.Client(transport=httpx.MockTransport(handler)) as http:
            client = mod.Client(http_client=http)
            assert (
                endpoints.list_items_export(str(out), format_='wide', client=client)
                == 1
            )
            assert client.list_items(format='wide').export(str(out)) == 1
        assert seen == ['wide', 'wide']
        assert out.exists()

    def test_export_wrapper_forwards_writer_options(self, tmp_path):
        spec = _write_spec(
            tmp_path / 'spec.json', {'/items': {'get': _list_op('listItems')}}
        )
        _gen(tmp_path / 'xw', spec, export={'enabled': True, 'formats': ['csv']})
        src = (tmp_path / 'xw' / 'endpoints.py').read_text()
        wrapper = ast.unparse(_function(src, 'list_items_export'))
        assert '**format_kwargs' in wrapper

    def test_export_wrapper_matches_iter_without_page_size(self, tmp_path):
        # Cursor pagination with send_page_size=false has no page_size knob on
        # ``_iter``; the export wrapper must neither declare nor forward one.
        spec = _write_spec(
            tmp_path / 'spec.json',
            {
                '/items': {
                    'get': _list_op(
                        'listItems',
                        [
                            {
                                'in': 'query',
                                'name': 'cursor',
                                'schema': {'type': 'string'},
                            }
                        ],
                    )
                }
            },
        )
        _gen(
            tmp_path / 'xc',
            spec,
            pagination={
                'enabled': True,
                'endpoints': {
                    'list_items': {
                        'style': 'cursor',
                        'cursor_param': 'cursor',
                        'send_page_size': False,
                    }
                },
            },
            export={'enabled': True, 'formats': ['csv']},
        )
        src = (tmp_path / 'xc' / 'endpoints.py').read_text()

        def accepted(name: str) -> set[str]:
            fn = _function(src, name)
            return {a.arg for a in (*fn.args.args, *fn.args.kwonlyargs)}

        assert 'page_size' not in accepted('list_items_iter')
        assert 'cursor' in accepted('list_items_iter')
        # The wrapper declares exactly _iter's knobs plus its own writer args.
        assert accepted('list_items_export') == accepted('list_items_iter') | {
            'output_path',
            'format',
            'batch_size',
        }
        assert 'page_size' not in ast.unparse(_function(src, 'list_items_export'))

    def test_class_styles_emit_each_endpoint_once(self, tmp_path, monkeypatch):
        # The layout writer wraps the sink the endpoints file already produced
        # instead of running every feature builder a second time.
        import otterapi.codegen.codegen as codegen_module
        import otterapi.codegen.splitting as splitting_module

        calls: list[int] = []
        real = codegen_module.build_endpoint_sink

        def counting(endpoints, *args, **kwargs):
            calls.append(len(endpoints))
            return real(endpoints, *args, **kwargs)

        # Both writers build sinks; neither the flat nor the split path may
        # build a second one for the layout file.
        monkeypatch.setattr(codegen_module, 'build_endpoint_sink', counting)
        monkeypatch.setattr(splitting_module, 'build_endpoint_sink', counting)
        _generate_tagged(
            tmp_path / 'once',
            client_style='client',
            dataframe={'enabled': True, 'pandas': True},
        )
        assert calls == [4]  # the four tagged-spec operations, emitted once

        calls.clear()
        _generate_tagged(
            tmp_path / 'once_split',
            client_style='client',
            module_split={'enabled': True, 'strategy': 'path'},
        )
        assert sorted(calls) == [2, 2]  # one sink per split module, no re-emit

    def test_nested_split_class_style_resolves_imports(self, tmp_path):
        # A nested split module has its sibling imports re-pointed (``..models``);
        # _clients.py lives at the package root and must normalize them back.
        spec = _write_spec(
            tmp_path / 'spec.json',
            {
                '/identity/users': {
                    'get': _list_op(
                        'listUsers',
                        [
                            {
                                'in': 'query',
                                'name': 'since',
                                'schema': {'type': 'string', 'format': 'date-time'},
                            }
                        ],
                    )
                },
                '/billing/invoices': {'get': _list_op('listInvoices')},
            },
        )
        _gen(
            tmp_path / 'nsi',
            spec,
            client_style='client',
            module_split={
                'enabled': True,
                'strategy': 'path',
                'path_depth': 2,
                'min_endpoints': 1,
            },
        )
        src = (tmp_path / 'nsi' / '_clients.py').read_text()
        assert 'from ..' not in src
        mod = _import_fresh(tmp_path, 'nsi')
        params = inspect.signature(mod.Client.list_users).parameters
        assert 'datetime' in str(params['since'].annotation)

    def test_export_wrapper_aliases_colliding_body_field(self, tmp_path):
        # The same aliasing covers a flattened request-body field named like
        # one of the wrapper's own arguments.
        spec = _write_spec(
            tmp_path / 'spec.json',
            {
                '/reports': {
                    'post': {
                        'operationId': 'searchReports',
                        'requestBody': {
                            'required': True,
                            'content': {
                                'application/json': {
                                    'schema': {'$ref': '#/components/schemas/Filter'}
                                }
                            },
                        },
                        'responses': _ITEM_LIST_RESPONSE,
                    }
                }
            },
            {
                'Filter': {
                    'type': 'object',
                    'required': ['format', 'name'],
                    'properties': {
                        'format': {'type': 'string'},
                        'name': {'type': 'string'},
                    },
                }
            },
        )
        _gen(
            tmp_path / 'xb',
            spec,
            request_body={'flatten': True},
            export={'enabled': True, 'formats': ['csv']},
        )
        mod = _import_fresh(tmp_path, 'xb')
        endpoints = importlib.import_module('xb.endpoints')
        seen: list[dict] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(json.loads(request.content))
            return httpx.Response(200, json=[{'id': 1, 'name': 'a'}])

        out = tmp_path / 'reports.csv'
        with httpx.Client(transport=httpx.MockTransport(handler)) as http:
            client = mod.Client(http_client=http)
            assert (
                endpoints.search_reports_export(
                    str(out), format_='wide', name='q', client=client
                )
                == 1
            )
        assert seen == [{'format': 'wide', 'name': 'q'}]

    def test_family_grouping_falls_back_to_suffix_without_owner(self):
        from otterapi.codegen.client_layout import _family_of

        fn = ast.parse('def list_users_df(): ...').body[0]
        assert isinstance(fn, ast.FunctionDef)
        assert _family_of(fn, {}) == ('list_users', 'to_pandas')
        core = ast.parse('def list_users(): ...').body[0]
        assert isinstance(core, ast.FunctionDef)
        assert _family_of(core, {}) == ('list_users', 'fetch')


def _two_ops(prefix: str, *, tag: str | None = None, params: list | None = None):
    """A path item with a list and a create operation (so it never folds)."""
    ops = {
        'get': _list_op(f'list_{prefix}', params),
        'post': {
            'operationId': f'create_{prefix}',
            'responses': _ITEM_RESPONSE,
            **({'parameters': params} if params else {}),
        },
    }
    if tag:
        for op in ops.values():
            op['tags'] = [tag]
    return ops


def _path_recorder(seen: list[str]):
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        if request.method == 'POST':
            return httpx.Response(200, json={'id': 1, 'name': 'x'})
        return httpx.Response(200, json=[])

    return handler


class TestGeneratedIdentifiers:
    """Names derived from the spec must always be valid, distinct Python."""

    def test_method_names_are_never_shortened(self, tmp_path):
        # Methods keep the function name even where dropping the resource
        # word would read shorter (and here would even leave a keyword).
        spec = _write_spec(
            tmp_path / 'spec.json',
            {
                '/users': {
                    'get': {**_list_op('list_users'), 'tags': ['users']},
                    'post': {
                        'operationId': 'import_users',
                        'tags': ['users'],
                        'responses': _ITEM_RESPONSE,
                    },
                }
            },
        )
        _gen(tmp_path / 'kw_pkg', spec, client_style='resource')
        mod = _import_fresh(tmp_path, 'kw_pkg')
        users = mod.Client(base_url='https://example.test').users
        assert _own_public_methods(type(users)) == {'import_users', 'list_users'}

    def test_path_segments_become_valid_attributes(self, tmp_path):
        # ``/2fa`` starts with a digit and ``import`` is a keyword; both are
        # property names under resource_naming="path".
        id_param = [
            {
                'in': 'path',
                'name': 'id',
                'required': True,
                'schema': {'type': 'integer'},
            }
        ]
        spec = _write_spec(
            tmp_path / 'spec.json',
            {
                '/2fa/codes': _two_ops('codes'),
                '/repos/{id}/import': _two_ops('imports', params=id_param),
            },
        )
        _gen(
            tmp_path / 'seg_pkg',
            spec,
            client_style='resource',
            resource_naming='path',
        )
        mod = _import_fresh(tmp_path, 'seg_pkg')
        seen: list[str] = []
        with httpx.Client(transport=httpx.MockTransport(_path_recorder(seen))) as http:
            client = mod.Client(http_client=http)
            client.n2fa.codes.list_codes()
            client.repos.import_.list_imports(1)
        assert seen == ['/2fa/codes', '/repos/1/import']

    @pytest.mark.parametrize('is_async', [False, True])
    def test_resource_class_names_never_collide(self, tmp_path, is_async):
        # ('user', 'groups') vs ('user_groups',) both flatten to UserGroups,
        # and a sync ``async_jobs`` node reads like the async side of ``jobs``.
        spec = _write_spec(
            tmp_path / 'spec.json',
            {
                '/user_groups': _two_ops('flat_groups'),
                '/user/groups': _two_ops('nested_groups'),
                '/async_jobs': _two_ops('async_jobs'),
                '/jobs': _two_ops('jobs'),
            },
        )
        pkg = f'cls_pkg_{int(is_async)}'
        _gen(tmp_path / pkg, spec, client_style='resource', resource_naming='path')
        src = (tmp_path / pkg / '_clients.py').read_text()
        class_names = [
            node.name for node in ast.parse(src).body if isinstance(node, ast.ClassDef)
        ]
        assert len(class_names) == len(set(class_names)), class_names

        mod = _import_fresh(tmp_path, pkg)
        seen: list[str] = []
        handler = _path_recorder(seen)
        if is_async:

            async def run() -> None:
                async with httpx.AsyncClient(
                    transport=httpx.MockTransport(handler)
                ) as http:
                    client = mod.AsyncClient(async_http_client=http)
                    await client.user.groups.list_nested_groups()
                    await client.user_groups.list_flat_groups()
                    await client.async_jobs.list_async_jobs()
                    await client.jobs.list_jobs()

            import asyncio

            asyncio.run(run())
        else:
            with httpx.Client(transport=httpx.MockTransport(handler)) as http:
                client = mod.Client(http_client=http)
                client.user.groups.list_nested_groups()
                client.user_groups.list_flat_groups()
                client.async_jobs.list_async_jobs()
                client.jobs.list_jobs()
        assert seen == ['/user/groups', '/user_groups', '/async_jobs', '/jobs']

    def test_class_style_without_endpoints_is_importable(self, tmp_path):
        # _clients.py is only written alongside the endpoints; __init__ must
        # not re-export it when endpoint generation is off.
        config = DocumentConfig.model_validate(
            {
                'source': str(SPEC),
                'output': str(tmp_path / 'noep_pkg'),
                'base_url': 'https://example.test',
                'client_style': 'client',
                'result_objects': True,
            }
        )
        Codegen(config, generate_endpoints=False).generate()
        assert not (tmp_path / 'noep_pkg' / '_clients.py').exists()
        mod = _import_fresh(tmp_path, 'noep_pkg')
        assert 'Client' in mod.__all__


class TestQueryIteration:
    def _query_module(self, tmp_path):
        _gen(
            tmp_path / 'q_pkg',
            SPEC,
            client_style='client',
            result_objects=True,
        )
        return _import_fresh(tmp_path, 'q_pkg')

    def test_iter_streams_when_paginated(self, tmp_path):
        mod = self._query_module(tmp_path)

        def fetch(**_kw):
            raise AssertionError('plain iteration must not materialize')

        q = mod.Query(None, {}, fetch=fetch, iterate=lambda **_kw: iter([1, 2]))
        assert list(q) == [1, 2]

    def test_iter_falls_back_to_fetch(self, tmp_path):
        mod = self._query_module(tmp_path)
        q = mod.Query(None, {}, fetch=lambda **_kw: [3, 4])
        assert list(q) == [3, 4]


def _class_surface(package_dir: Path) -> dict[tuple[str, str], tuple[str, ...]]:
    """``{(class, member): (return annotation, *functions it calls)}``.

    Covers every class in _clients.py. Resource accessors are members too
    (annotated with their resource class), so a renamed resource class shows
    up as a changed annotation; the called endpoint functions catch a member
    name that silently moved to a different operation.
    """
    tree = ast.parse((package_dir / '_clients.py').read_text())
    surface: dict[tuple[str, str], tuple[str, ...]] = {}
    for cls in tree.body:
        if not isinstance(cls, ast.ClassDef):
            continue
        for member in cls.body:
            if not isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            returns = ast.unparse(member.returns) if member.returns else ''
            calls = sorted(
                {
                    node.attr
                    for node in ast.walk(member)
                    if isinstance(node, ast.Attribute)
                    and isinstance(node.value, ast.Name)
                    and node.value.id.startswith(('_endpoints', '_ep_'))
                }
            )
            surface[(cls.name, member.name)] = (returns, *calls)
    return surface


def _surface_regressions(
    before: dict[tuple[str, str], tuple[str, ...]],
    after: dict[tuple[str, str], tuple[str, ...]],
) -> dict[tuple[str, str], tuple]:
    """Members that vanished, changed return type, or lost a delegated call.

    Gaining calls is fine: a ``Query`` method binds a new terminal when its
    feature is switched on.
    """
    regressions: dict[tuple[str, str], tuple] = {}
    for key, (returns, *calls) in before.items():
        now = after.get(key)
        if now is None or now[0] != returns or not set(calls) <= set(now[1:]):
            regressions[key] = (before[key], now)
    return regressions


_OFFSET_LIMIT = [
    {'in': 'query', 'name': 'offset', 'schema': {'type': 'integer'}},
    {'in': 'query', 'name': 'limit', 'schema': {'type': 'integer'}},
]
_ID_PARAM = [
    {'in': 'path', 'name': 'id', 'required': True, 'schema': {'type': 'integer'}}
]


def _tagged(op: dict, tag: str) -> dict:
    return {**op, 'tags': [tag]}


def _stability_paths() -> dict:
    return {
        '/users': {
            'get': _tagged(_list_op('list_users', _OFFSET_LIMIT), 'users'),
            'post': _tagged(
                {'operationId': 'create_user', 'responses': _ITEM_RESPONSE}, 'users'
            ),
        },
        '/users/{id}': {
            'get': _tagged(
                {
                    'operationId': 'get_user',
                    'parameters': _ID_PARAM,
                    'responses': _ITEM_RESPONSE,
                },
                'users',
            )
        },
        '/user_groups': {'get': _tagged(_list_op('list_user_groups'), 'groups')},
    }


def _grown_paths() -> dict:
    """The same spec plus operations that used to rename existing members.

    Under the old naming rules each addition renamed or reshaped a method that
    already existed: a shortened-name clash (``list``), a second operation on a
    folded action segment, a colliding resource class (``/user/groups``).
    """
    paths = _stability_paths()
    paths['/user/groups'] = {
        'get': _tagged(_list_op('list_group_members'), 'groups'),
        'post': _tagged(
            {'operationId': 'add_group_member', 'responses': _ITEM_RESPONSE}, 'groups'
        ),
    }
    paths['/users/search'] = {'get': _tagged(_list_op('search_users'), 'users')}
    paths['/users/{id}/import'] = {
        'post': _tagged(
            {
                'operationId': 'import_user',
                'parameters': _ID_PARAM,
                'responses': _ITEM_RESPONSE,
            },
            'users',
        )
    }
    paths['/list'] = {'get': _tagged(_list_op('list'), 'users')}
    paths['/user_groups']['post'] = _tagged(
        {'operationId': 'create_user_group', 'responses': _ITEM_RESPONSE}, 'groups'
    )
    return paths


_LAYOUTS = [('client', 'tag'), ('resource', 'tag'), ('resource', 'path')]


class TestSurfaceStability:
    """Spec growth and feature flags only ever *add* to the class surface.

    Every method name and return annotation generated before an operation is
    added -- or a feature switched on -- must come out identical afterwards.
    """

    def _surface(self, target: Path, paths: dict, **overrides):
        spec = _write_spec(target.parent / f'{target.name}.json', paths)
        _gen(target, spec, **overrides)
        return _class_surface(target)

    @pytest.mark.parametrize('result_objects', [False, True])
    @pytest.mark.parametrize(('style', 'naming'), _LAYOUTS)
    def test_adding_operations_never_changes_existing_members(
        self, tmp_path, style, naming, result_objects
    ):
        config = dict(
            client_style=style, resource_naming=naming, result_objects=result_objects
        )
        before = self._surface(tmp_path / 'before', _stability_paths(), **config)
        after = self._surface(tmp_path / 'after', _grown_paths(), **config)
        assert before
        assert not _surface_regressions(before, after)

    @pytest.mark.parametrize('result_objects', [False, True])
    @pytest.mark.parametrize(('style', 'naming'), _LAYOUTS)
    def test_enabling_features_never_changes_existing_members(
        self, tmp_path, style, naming, result_objects
    ):
        config = dict(
            client_style=style, resource_naming=naming, result_objects=result_objects
        )
        plain = self._surface(tmp_path / 'plain', _stability_paths(), **config)
        featured = self._surface(
            tmp_path / 'featured',
            _stability_paths(),
            pagination={'enabled': True, 'auto_detect': True},
            dataframe={'enabled': True, 'pandas': True, 'polars': True},
            export={'enabled': True, 'formats': ['csv']},
            **config,
        )
        assert plain
        assert not _surface_regressions(plain, featured)

    def test_list_endpoint_returns_query_without_any_feature(self, tmp_path):
        # Query-ness follows the return type, not the enabled features: a
        # list endpoint is a Query even when no terminal beyond .all() exists.
        surface = self._surface(
            tmp_path / 'q',
            _stability_paths(),
            client_style='client',
            result_objects=True,
        )
        assert surface[('Client', 'list_users')][0] == 'Query[Item]'
        assert surface[('AsyncClient', 'list_users')][0] == 'AsyncQuery[Item]'
        assert surface[('Client', 'get_user')][0] == 'Item'
