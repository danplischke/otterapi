"""Path-versioned APIs (issue #30): unwrap, body cursor pagination, naming.

Each test generates a real client from the spec in the issue (or a focused
variant of it), imports it, and drives it through ``httpx2.MockTransport``.
"""

from __future__ import annotations

import copy
import importlib
import json
import sys
from pathlib import Path
from typing import Any

import httpx2
import pytest
import yaml

from otterapi.codegen.codegen import Codegen
from otterapi.config import DocumentConfig
from otterapi.exceptions import ConfigurationError

# ---------------------------------------------------------------------------
# Spec from the issue
# ---------------------------------------------------------------------------


def _json(ref: str) -> dict:
    return {'application/json': {'schema': {'$ref': f'#/components/schemas/{ref}'}}}


def _body_op(operation_id: str, response: str) -> dict:
    return {
        'post': {
            'operationId': operation_id,
            'requestBody': {'required': True, 'content': _json('ByIdRequest')},
            'responses': {'200': {'description': 'ok', 'content': _json(response)}},
        }
    }


SPEC: dict = {
    'openapi': '3.1.0',
    'info': {'title': 'Widgets', 'version': '1'},
    'paths': {
        '/api/widgets/v1/by-id': {
            'get': {
                'operationId': 'get_widgets_by_id',
                'parameters': [
                    {
                        'name': 'id',
                        'in': 'query',
                        'required': True,
                        'schema': {'type': 'string'},
                    }
                ],
                'responses': {
                    '200': {'description': 'ok', 'content': _json('WidgetEnvelope')}
                },
            }
        },
        '/api/widgets/v2/by-id': _body_op('get_widgets_by_id_v2', 'WidgetV2Page'),
        '/api/widgets/v2/by-id/count': _body_op(
            'get_widgets_by_id_v2_count', 'CountEnvelope'
        ),
    },
    'components': {
        'schemas': {
            'Widget': {'type': 'object', 'properties': {'id': {'type': 'string'}}},
            'WidgetEnvelope': {
                'type': 'object',
                'required': ['data'],
                'properties': {
                    'data': {
                        'type': 'array',
                        'items': {'$ref': '#/components/schemas/Widget'},
                    }
                },
            },
            'WidgetV2': {
                'type': 'object',
                'required': ['id', 'cursor'],
                'properties': {
                    'id': {'type': 'string'},
                    'score': {'type': 'number'},
                    'cursor': {'type': 'string'},
                },
            },
            'WidgetV2Page': {
                'type': 'object',
                'required': ['data'],
                'properties': {
                    'data': {
                        'type': 'array',
                        'items': {'$ref': '#/components/schemas/WidgetV2'},
                    },
                    'total': {'type': ['integer', 'null']},
                },
            },
            'ByIdRequest': {
                'type': 'object',
                'required': ['id'],
                'properties': {
                    'id': {'type': 'string'},
                    'min_score': {'type': ['number', 'null']},
                    'limit': {'type': 'integer', 'default': 100},
                    'after': {'type': ['string', 'null']},
                },
            },
            'CountEnvelope': {
                'type': 'object',
                'required': ['total'],
                'properties': {'total': {'type': 'integer'}},
            },
        }
    },
}

MODULE_SPLIT = {
    'enabled': True,
    'strategy': 'path',
    'path_depth': 1,
    'min_endpoints': 1,
    'global_strip_prefixes': ['/api'],
}

TOTAL_ROWS = 250


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


def _generate(
    tmp_path: Path, package: str, spec: dict | None = None, **config: Any
) -> Any:
    """Generate ``spec`` into ``tmp_path/package`` and import the package."""
    spec_file = tmp_path / 'openapi.yaml'
    spec_file.write_text(yaml.safe_dump(spec or SPEC), encoding='utf-8')
    Codegen(
        DocumentConfig(
            source=str(spec_file),
            output=str(tmp_path / package),
            base_url='https://example.test',
            **config,
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
        sys.path.remove(str(tmp_path))


class WidgetServer:
    """Serves ``TOTAL_ROWS`` widgets, paging on the body's ``after``/``limit``.

    ``max_limit`` caps the page size the way a real server does, to prove the
    client does not stop on a page that is merely shorter than it asked for.
    """

    def __init__(self, max_limit: int | None = None) -> None:
        self.max_limit = max_limit
        self.bodies: list[dict] = []

    def handler(self, request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content or b'{}')
        self.bodies.append(body)
        if request.url.path.endswith('/count'):
            return httpx2.Response(200, json={'total': TOTAL_ROWS})
        start = int(body.get('after') or 0)
        limit = int(body.get('limit') or 100)
        if self.max_limit is not None:
            limit = min(limit, self.max_limit)
        rows = [
            {'id': body.get('id', 'x'), 'score': 0.5, 'cursor': str(i + 1)}
            for i in range(start, min(start + limit, TOTAL_ROWS))
        ]
        return httpx2.Response(200, json={'data': rows, 'total': TOTAL_ROWS})

    def client(self, client_cls: type) -> Any:
        return client_cls(
            http_client=httpx2.Client(
                base_url='https://example.test',
                transport=httpx2.MockTransport(self.handler),
            )
        )


def _source(tmp_path: Path, package: str, module: str) -> str:
    return (tmp_path / package / f'{module}.py').read_text(encoding='utf-8')


# ---------------------------------------------------------------------------
# 3. response_unwrap only where the response has the field
# ---------------------------------------------------------------------------


class TestUnwrapNeedsTheField:
    def test_count_endpoint_returns_model_not_data(self, tmp_path: Path) -> None:
        pkg = _generate(
            tmp_path,
            'unwrap_skip',
            module_split=MODULE_SPLIT,
            response_unwrap={'enabled': True, 'data_path': 'data'},
        )
        src = _source(tmp_path, 'unwrap_skip', 'widgets')
        assert 'result.data' in src  # the v1 envelope still unwraps
        count_src = src[src.index('def get_widgets_by_id_v2_count(') :]
        count_src = count_src[: count_src.index('\n\n\n')]
        assert '.data' not in count_src
        assert '-> CountEnvelope' in count_src

        from unwrap_skip.client import Client  # type: ignore[import-not-found]

        widgets = importlib.import_module('unwrap_skip.widgets')
        models = importlib.import_module('unwrap_skip.models')
        server = WidgetServer()
        result = widgets.get_widgets_by_id_v2_count(
            models.ByIdRequest(id='x'), client=server.client(Client)
        )
        assert result.total == TOTAL_ROWS
        assert pkg is not None

    def test_explicit_override_unwraps_scalar(self, tmp_path: Path) -> None:
        _generate(
            tmp_path,
            'unwrap_total',
            module_split=MODULE_SPLIT,
            response_unwrap={
                'enabled': True,
                'data_path': 'data',
                'endpoints': {'get_widgets_by_id_v2_count': {'data_path': 'total'}},
            },
        )
        from unwrap_total.client import Client  # type: ignore[import-not-found]

        widgets = importlib.import_module('unwrap_total.widgets')
        models = importlib.import_module('unwrap_total.models')
        src = _source(tmp_path, 'unwrap_total', 'widgets')
        assert 'def get_widgets_by_id_v2_count(' in src
        assert ') -> int:' in src
        total = widgets.get_widgets_by_id_v2_count(
            models.ByIdRequest(id='x'), client=WidgetServer().client(Client)
        )
        assert total == TOTAL_ROWS

    def test_explicit_missing_field_is_an_error(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigurationError, match='does not exist'):
            _generate(
                tmp_path,
                'unwrap_bad',
                response_unwrap={
                    'enabled': True,
                    'endpoints': {'get_widgets_by_id_v2_count': {'data_path': 'nope'}},
                },
            )

    def test_dotted_path_resolves_nested_type(self, tmp_path: Path) -> None:
        spec = copy.deepcopy(SPEC)
        schemas = spec['components']['schemas']
        schemas['Nested'] = {
            'type': 'object',
            'required': ['result'],
            'properties': {'result': {'$ref': '#/components/schemas/CountEnvelope'}},
        }
        spec['paths']['/api/nested'] = {
            'get': {
                'operationId': 'get_nested',
                'responses': {'200': {'description': 'ok', 'content': _json('Nested')}},
            }
        }
        _generate(
            tmp_path,
            'unwrap_nested',
            spec=spec,
            response_unwrap={
                'enabled': True,
                'endpoints': {'get_nested': {'data_path': 'result.total'}},
            },
        )
        src = (tmp_path / 'unwrap_nested' / 'endpoints.py').read_text()
        fn = src[src.index('def get_nested(') :]
        assert fn[: fn.index(':\n')].endswith('-> int')
        assert 'result.result.total' in fn


class TestLooksPaginatedWarning:
    PAGINATION = {
        'enabled': True,
        'default_style': 'offset',
        'default_data_path': 'data',
        'default_total_path': 'total',
    }

    def test_warns_for_unconfigured_body_endpoint(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level('WARNING', logger='otterapi.codegen.emit'):
            _generate(tmp_path, 'warn_pag', pagination=self.PAGINATION)
        messages = [r.getMessage() for r in caplog.records]
        assert any(
            'get_widgets_by_id_v2 ' in m and 'only the first page' in m
            for m in messages
        )
        # The count endpoint has a total but no items list: no warning.
        assert not any('get_widgets_by_id_v2_count ' in m for m in messages)

    def test_explicitly_disabled_endpoint_is_silent(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        pagination = {
            **self.PAGINATION,
            'endpoints': {'get_widgets_by_id_v2': {'enabled': False}},
        }
        with caplog.at_level('WARNING', logger='otterapi.codegen.emit'):
            _generate(tmp_path, 'warn_off', pagination=pagination)
        assert not any('only the first page' in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------
# 2. Cursor pagination through the request body
# ---------------------------------------------------------------------------

BODY_PAGINATION: dict = {
    'enabled': True,
    'default_style': 'offset',
    'default_data_path': 'data',
    'default_total_path': 'total',
    'endpoints': {
        'get_widgets_by_id_v2': {
            'style': 'cursor',
            'cursor_param': 'after',
            'cursor_location': 'body',
            'next_cursor_from': 'last_item',
        }
    },
}


@pytest.fixture(scope='module')
def body_pkg(tmp_path_factory: pytest.TempPathFactory) -> Any:
    tmp_path = tmp_path_factory.mktemp('body_pag')
    _generate(
        tmp_path,
        'body_pag',
        module_split=MODULE_SPLIT,
        pagination=BODY_PAGINATION,
        response_unwrap={'enabled': True, 'data_path': 'data'},
        dataframe={'enabled': True, 'pandas': True, 'default_path': 'data'},
        export={'enabled': True},
    )
    sys.path.insert(0, str(tmp_path))
    try:
        yield (
            importlib.import_module('body_pag.widgets'),
            importlib.import_module('body_pag.models'),
            importlib.import_module('body_pag.client').Client,
        )
    finally:
        sys.path.remove(str(tmp_path))


class TestBodyCursorPagination:
    def test_df_fetches_every_page(self, body_pkg: Any) -> None:
        widgets, models, client_cls = body_pkg
        server = WidgetServer()
        df = widgets.get_widgets_by_id_v2_df(
            models.ByIdRequest(id='x', min_score=0.5),
            client=server.client(client_cls),
        )
        assert len(df) == TOTAL_ROWS
        # total reached after the third page: no empty fourth request.
        assert [b.get('after') for b in server.bodies] == [None, '100', '200']
        assert all(b['id'] == 'x' and b['min_score'] == 0.5 for b in server.bodies)
        assert all(b['limit'] == 100 for b in server.bodies)

    def test_server_capping_the_page_size_is_not_truncation(
        self, body_pkg: Any
    ) -> None:
        widgets, models, client_cls = body_pkg
        server = WidgetServer(max_limit=40)
        items = widgets.get_widgets_by_id_v2(
            models.ByIdRequest(id='x'), client=server.client(client_cls)
        )
        assert len(items) == TOTAL_ROWS
        assert len(server.bodies) == 7  # ceil(250 / 40)

    def test_without_total_stops_on_the_empty_page(self, body_pkg: Any) -> None:
        widgets, models, client_cls = body_pkg

        class NoTotal(WidgetServer):
            def handler(self, request: httpx2.Request) -> httpx2.Response:
                response = super().handler(request)
                payload = json.loads(response.content)
                payload['total'] = None
                return httpx2.Response(200, json=payload)

        server = NoTotal()
        items = list(
            widgets.get_widgets_by_id_v2_iter(
                models.ByIdRequest(id='x'), client=server.client(client_cls)
            )
        )
        assert len(items) == TOTAL_ROWS
        assert len(server.bodies) == 4

    def test_server_ignoring_the_cursor_does_not_loop_forever(
        self, body_pkg: Any
    ) -> None:
        widgets, models, client_cls = body_pkg
        bodies: list[dict] = []

        def handler(request: httpx2.Request) -> httpx2.Response:
            bodies.append(json.loads(request.content))
            rows = [{'id': 'x', 'cursor': 'same'}]
            return httpx2.Response(200, json={'data': rows, 'total': None})

        client = client_cls(
            http_client=httpx2.Client(
                base_url='https://example.test',
                transport=httpx2.MockTransport(handler),
            )
        )
        items = widgets.get_widgets_by_id_v2(models.ByIdRequest(id='x'), client=client)
        assert len(items) == 2
        assert [b.get('after') for b in bodies] == [None, 'same']

    def test_start_cursor_and_max_items(self, body_pkg: Any) -> None:
        widgets, models, client_cls = body_pkg
        server = WidgetServer()
        items = widgets.get_widgets_by_id_v2(
            models.ByIdRequest(id='x'),
            cursor='200',
            page_size=20,
            max_items=30,
            client=server.client(client_cls),
        )
        assert [i.cursor for i in items][:2] == ['201', '202']
        assert len(items) == 30
        assert [b['after'] for b in server.bodies] == ['200', '220']

    def test_cursor_in_body_is_used_when_no_start_cursor(self, body_pkg: Any) -> None:
        widgets, models, client_cls = body_pkg
        server = WidgetServer()
        items = widgets.get_widgets_by_id_v2(
            models.ByIdRequest(id='x', after='240'), client=server.client(client_cls)
        )
        assert len(items) == 10
        assert server.bodies[0]['after'] == '240'

    def test_export_fetches_every_page(self, body_pkg: Any, tmp_path: Path) -> None:
        widgets, models, client_cls = body_pkg
        out = tmp_path / 'rows.jsonl'
        widgets.get_widgets_by_id_v2_export(
            str(out),
            body=models.ByIdRequest(id='x'),
            format='jsonl',
            client=WidgetServer().client(client_cls),
        )
        assert len(out.read_text().splitlines()) == TOTAL_ROWS

    @pytest.mark.asyncio
    async def test_async_df_fetches_every_page(self, body_pkg: Any) -> None:
        widgets, models, client_cls = body_pkg
        server = WidgetServer()
        client = client_cls(
            async_http_client=httpx2.AsyncClient(
                base_url='https://example.test',
                transport=httpx2.MockTransport(server.handler),
            )
        )
        df = await widgets.async_get_widgets_by_id_v2_df(
            models.ByIdRequest(id='x'), client=client
        )
        assert len(df) == TOTAL_ROWS
        assert len(server.bodies) == 3

    def test_body_location_needs_a_json_object_body(self, tmp_path: Path) -> None:
        pagination = copy.deepcopy(BODY_PAGINATION)
        pagination['endpoints'] = {
            'get_widgets_by_id': {'style': 'cursor', 'cursor_location': 'body'}
        }
        with pytest.raises(ConfigurationError, match='JSON object request body'):
            _generate(tmp_path, 'body_bad', pagination=pagination)


class TestPaginatedRequestsCarryHeadersAndBody:
    """Paginated functions used to accept header params and bodies but drop them."""

    def test_query_paginated_post_sends_body_and_headers(self, tmp_path: Path) -> None:
        spec = copy.deepcopy(SPEC)
        spec['paths'] = {
            '/search': {
                'post': {
                    'operationId': 'search',
                    'parameters': [
                        {
                            'name': 'offset',
                            'in': 'query',
                            'schema': {'type': 'integer'},
                        },
                        {'name': 'limit', 'in': 'query', 'schema': {'type': 'integer'}},
                        {
                            'name': 'X-Tenant',
                            'in': 'header',
                            'required': True,
                            'schema': {'type': 'string'},
                        },
                    ],
                    'requestBody': {'required': True, 'content': _json('ByIdRequest')},
                    'responses': {
                        '200': {'description': 'ok', 'content': _json('WidgetV2Page')}
                    },
                }
            }
        }
        _generate(
            tmp_path,
            'hdr_pag',
            spec=spec,
            pagination={'enabled': True, 'default_data_path': 'data'},
        )
        sys.path.insert(0, str(tmp_path))
        try:
            endpoints = importlib.import_module('hdr_pag.endpoints')
            models = importlib.import_module('hdr_pag.models')
            client_cls = importlib.import_module('hdr_pag.client').Client
        finally:
            sys.path.remove(str(tmp_path))
        requests: list[httpx2.Request] = []

        def handler(request: httpx2.Request) -> httpx2.Response:
            requests.append(request)
            offset = int(request.url.params['offset'])
            rows = [
                {'id': 'x', 'cursor': str(i)} for i in range(offset, min(offset + 2, 3))
            ]
            return httpx2.Response(200, json={'data': rows})

        client = client_cls(
            http_client=httpx2.Client(
                base_url='https://example.test',
                transport=httpx2.MockTransport(handler),
            )
        )
        items = endpoints.search(
            body=models.ByIdRequest(id='q'), X_Tenant='acme', page_size=2, client=client
        )
        assert len(items) == 3
        assert all(r.headers['X-Tenant'] == 'acme' for r in requests)
        assert all(json.loads(r.content) == {'id': 'q'} for r in requests)


def test_extract_path_walks_model_attributes() -> None:
    from pydantic import BaseModel

    from otterapi.codegen.runtime._pagination import extract_path

    class Inner(BaseModel):
        items: list[int]

    class Outer(BaseModel):
        data: Inner

    assert extract_path(Outer(data=Inner(items=[1, 2])), 'data.items') == [1, 2]


# ---------------------------------------------------------------------------
# 4. Flattened bodies hide the fields the paginator drives
# ---------------------------------------------------------------------------


class TestFlattenedBodyPagination:
    @pytest.fixture(scope='class')
    def flat_pkg(self, tmp_path_factory: pytest.TempPathFactory) -> Any:
        spec = copy.deepcopy(SPEC)
        # A required paging field must still validate on the first page.
        spec['components']['schemas']['ByIdRequest']['required'] = ['id', 'limit']
        tmp_path = tmp_path_factory.mktemp('flat_pag')
        _generate(
            tmp_path,
            'flat_pag',
            spec=spec,
            module_split=MODULE_SPLIT,
            pagination=BODY_PAGINATION,
            request_body={'flatten': True},
            dataframe={'enabled': True, 'pandas': True, 'default_path': 'data'},
            export={'enabled': True},
        )
        sys.path.insert(0, str(tmp_path))
        try:
            yield (
                importlib.import_module('flat_pag.widgets'),
                importlib.import_module('flat_pag.client').Client,
            )
        finally:
            sys.path.remove(str(tmp_path))

    @pytest.mark.parametrize(
        'suffix', ['', '_iter', '_df', '_export', '_page', '_pages']
    )
    def test_paging_fields_are_not_parameters(self, flat_pkg: Any, suffix: str) -> None:
        import inspect

        widgets, _ = flat_pkg
        for prefix in ('', 'async_'):
            fn = getattr(widgets, f'{prefix}get_widgets_by_id_v2{suffix}')
            params = inspect.signature(fn).parameters
            assert 'after' not in params and 'limit' not in params
            assert {'id', 'min_score', 'cursor', 'page_size'} <= set(params)

    def test_non_paginated_endpoint_keeps_every_field(self, flat_pkg: Any) -> None:
        import inspect

        widgets, _ = flat_pkg
        params = inspect.signature(widgets.get_widgets_by_id_v2_count).parameters
        assert {'id', 'min_score', 'limit', 'after'} <= set(params)

    def test_df_fetches_every_page(self, flat_pkg: Any) -> None:
        widgets, client_cls = flat_pkg
        server = WidgetServer()
        df = widgets.get_widgets_by_id_v2_df(
            'x', min_score=0.5, client=server.client(client_cls)
        )
        assert len(df) == TOTAL_ROWS
        assert server.bodies == [
            {'id': 'x', 'min_score': 0.5, 'limit': 100},
            {'id': 'x', 'min_score': 0.5, 'limit': 100, 'after': '100'},
            {'id': 'x', 'min_score': 0.5, 'limit': 100, 'after': '200'},
        ]

    def test_export_fetches_every_page(self, flat_pkg: Any, tmp_path: Path) -> None:
        widgets, client_cls = flat_pkg
        out = tmp_path / 'rows.jsonl'
        widgets.get_widgets_by_id_v2_export(
            str(out), 'x', format='jsonl', client=WidgetServer().client(client_cls)
        )
        assert len(out.read_text().splitlines()) == TOTAL_ROWS


# ---------------------------------------------------------------------------
# 1. Version-aware naming and legacy routing
# ---------------------------------------------------------------------------

NAMING = {'rules': [{'pattern': r'_v\d+(?=_|$)', 'replace': ''}]}
VERSIONED = {
    'module_split': MODULE_SPLIT,
    'naming': NAMING,
    'versioning': {'path_pattern': '/v(?P<version>\\d+)/'},
    'response_unwrap': {'enabled': True, 'data_path': 'data'},
    'dataframe': {'enabled': True, 'pandas': True, 'default_path': 'data'},
}


class TestVersionRouting:
    @pytest.fixture(scope='class')
    def versioned(self, tmp_path_factory: pytest.TempPathFactory) -> Any:
        tmp_path = tmp_path_factory.mktemp('versioned')
        pagination = copy.deepcopy(BODY_PAGINATION)
        # Keyed by the new name: the latest (v2) endpoint.
        pagination['endpoints'] = {
            'get_widgets_by_id': pagination['endpoints']['get_widgets_by_id_v2']
        }
        _generate(tmp_path, 'versioned', pagination=pagination, **VERSIONED)
        sys.path.insert(0, str(tmp_path))
        try:
            yield (
                tmp_path,
                (
                    importlib.import_module('versioned'),
                    importlib.import_module('versioned.widgets'),
                    importlib.import_module('versioned.widgets.v1'),
                    importlib.import_module('versioned.models'),
                ),
            )
        finally:
            sys.path.remove(str(tmp_path))

    def test_layout(self, versioned: Any) -> None:
        tmp_path, _ = versioned
        root = tmp_path / 'versioned'
        assert (root / 'widgets' / '__init__.py').exists()
        assert (root / 'widgets' / 'v1.py').exists()
        assert not (root / 'widgets.py').exists()

    def test_latest_is_unsuffixed_everywhere(self, versioned: Any) -> None:
        _, (pkg, widgets, _v1, _) = versioned
        for name in (
            'get_widgets_by_id',
            'async_get_widgets_by_id',
            'get_widgets_by_id_df',
            'get_widgets_by_id_iter',
            'get_widgets_by_id_page',
            'get_widgets_by_id_count',
        ):
            assert name in widgets.__all__
            assert getattr(pkg, name) is getattr(widgets, name)
        assert not any('_v2' in name for name in widgets.__all__)
        assert not any('_v2' in name for name in pkg.__all__)

    def test_legacy_is_reachable_but_not_reexported(self, versioned: Any) -> None:
        _, (pkg, widgets, v1, _) = versioned
        assert set(v1.__all__) >= {'get_widgets_by_id', 'async_get_widgets_by_id'}
        assert v1.get_widgets_by_id is not widgets.get_widgets_by_id
        assert pkg.get_widgets_by_id is widgets.get_widgets_by_id

    def test_wire(self, versioned: Any) -> None:
        _, (pkg, widgets, v1, models) = versioned
        from versioned.client import Client  # type: ignore[import-not-found]

        requests: list[httpx2.Request] = []
        server = WidgetServer()

        def handler(request: httpx2.Request) -> httpx2.Response:
            requests.append(request)
            if request.method == 'GET':
                return httpx2.Response(200, json={'data': [{'id': 'old'}]})
            return server.handler(request)

        client = Client(
            http_client=httpx2.Client(
                base_url='https://example.test',
                transport=httpx2.MockTransport(handler),
            )
        )
        rows = widgets.get_widgets_by_id_df(models.ByIdRequest(id='x'), client=client)
        assert len(rows) == TOTAL_ROWS
        assert {r.url.path for r in requests} == {'/api/widgets/v2/by-id'}
        old = v1.get_widgets_by_id(id='x', client=client)
        assert [w.id for w in old] == ['old']
        assert requests[-1].url.path == '/api/widgets/v1/by-id'
        assert (
            widgets.get_widgets_by_id_count(
                models.ByIdRequest(id='x'), client=client
            ).total
            == TOTAL_ROWS
        )

    def test_suffix_mode_without_module_split(self, tmp_path: Path) -> None:
        pkg = _generate(
            tmp_path,
            'suffixed',
            naming=NAMING,
            versioning={'legacy': 'suffix'},
        )
        names = set(pkg.__all__)
        assert {'get_widgets_by_id', 'get_widgets_by_id_v1'} <= names
        src = (tmp_path / 'suffixed' / 'endpoints.py').read_text()
        legacy_fn = src.split('def get_widgets_by_id_v1(')[1].split('\ndef ')[0]
        assert '/api/widgets/v1/by-id' in legacy_fn

    def test_submodule_needs_module_split(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigurationError, match='module_split'):
            _generate(tmp_path, 'nosplit', naming=NAMING, versioning={})

    def test_submodule_needs_functions_style(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigurationError, match='client_style'):
            _generate(
                tmp_path,
                'classy',
                naming=NAMING,
                versioning={},
                module_split=MODULE_SPLIT,
                client_style='client',
            )

    def test_suffix_mode_works_with_class_clients(self, tmp_path: Path) -> None:
        pkg = _generate(
            tmp_path,
            'classy_sfx',
            naming=NAMING,
            versioning={'legacy': 'suffix'},
            client_style='resource',
            module_split=MODULE_SPLIT,
        )
        widgets = pkg.Client().widgets
        assert callable(widgets.get_widgets_by_id)
        assert callable(widgets.get_widgets_by_id_v1)

    def test_rename_collision_without_versioning_is_an_error(
        self, tmp_path: Path
    ) -> None:
        with pytest.raises(ConfigurationError, match='get_widgets_by_id'):
            _generate(tmp_path, 'collide', naming=NAMING)

    def test_legacy_follows_latest_into_the_fallback_module(
        self, tmp_path: Path
    ) -> None:
        split = {**MODULE_SPLIT, 'min_endpoints': 10}
        _generate(tmp_path, 'fallback', **{**VERSIONED, 'module_split': split})
        root = tmp_path / 'fallback'
        modules = sorted(
            str(p.relative_to(root))
            for p in root.rglob('*.py')
            if '_' not in p.name[:1]
        )
        assert any(m.endswith('/v1.py') for m in modules), modules
        latest_pkg = next(m for m in modules if m.endswith('/v1.py')).rsplit('/', 1)[0]
        assert (root / latest_pkg / '__init__.py').exists()


class TestConfigKeys:
    def test_operation_id_key_reaches_a_renamed_endpoint(self, tmp_path: Path) -> None:
        pagination = copy.deepcopy(BODY_PAGINATION)  # keyed get_widgets_by_id_v2
        pkg = _generate(
            tmp_path,
            'opid_key',
            pagination=pagination,
            **{**VERSIONED, 'response_unwrap': {'enabled': False}},
        )
        assert (
            'get_widgets_by_id_iter'
            in importlib.import_module('opid_key.widgets').__all__
        )
        assert pkg is not None

    def test_unknown_key_warns(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level('WARNING', logger='otterapi.codegen.codegen'):
            _generate(
                tmp_path,
                'typo_key',
                pagination={
                    'enabled': True,
                    'endpoints': {'get_widgets_by_idd': {'style': 'cursor'}},
                },
            )
        assert any(
            'pagination.endpoints.get_widgets_by_idd matches no generated endpoint'
            in r.getMessage()
            for r in caplog.records
        )

    def test_legacy_endpoint_is_keyed_with_its_version(self, tmp_path: Path) -> None:
        _generate(
            tmp_path,
            'legacy_key',
            **{
                **VERSIONED,
                'dataframe': {
                    'enabled': True,
                    'pandas': True,
                    'default_path': 'data',
                    'endpoints': {'get_widgets_by_id_v1': {'enabled': False}},
                },
            },
        )
        v1 = (tmp_path / 'legacy_key' / 'widgets' / 'v1.py').read_text()
        latest = (tmp_path / 'legacy_key' / 'widgets' / '__init__.py').read_text()
        assert 'def get_widgets_by_id_df(' not in v1
        assert 'def get_widgets_by_id_df(' in latest


def test_variant_colliding_with_an_operation_is_an_error(tmp_path: Path) -> None:
    spec = copy.deepcopy(SPEC)
    page_op = copy.deepcopy(spec['paths']['/api/widgets/v2/by-id'])
    page_op['post']['operationId'] = 'get_widgets_by_id_v2_page'
    spec['paths']['/api/widgets/v2/page'] = page_op
    with pytest.raises(ConfigurationError, match="named 'get_widgets_by_id_v2_page'"):
        _generate(tmp_path, 'dup_variant', spec=spec, pagination=BODY_PAGINATION)
