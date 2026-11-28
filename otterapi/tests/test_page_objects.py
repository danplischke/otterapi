"""Page objects for paginated endpoints (issue #28).

Every test generates a client, drives it through ``httpx2.MockTransport`` and
checks the ``_page`` / ``_pages`` variants: items, total, next position,
envelope fields, resuming, and parity with the item-level ``_iter``.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

import httpx2
import pytest
import yaml

from otterapi.codegen.codegen import Codegen
from otterapi.config import DocumentConfig

TOTAL = 23


def _q(name: str, type_: str) -> dict:
    return {'name': name, 'in': 'query', 'schema': {'type': type_}}


def _list_op(operation_id: str, params: list[dict], envelope: str) -> dict:
    return {
        'get': {
            'operationId': operation_id,
            'parameters': params,
            'responses': {
                '200': {
                    'description': 'ok',
                    'content': {
                        'application/json': {
                            'schema': {'$ref': f'#/components/schemas/{envelope}'}
                        }
                    },
                }
            },
        }
    }


SPEC = {
    'openapi': '3.1.0',
    'info': {'title': 'Users', 'version': '1'},
    'paths': {
        '/users': _list_op(
            'list_users', [_q('offset', 'integer'), _q('limit', 'integer')], 'Offset'
        ),
        '/feed': _list_op(
            'list_feed', [_q('cursor', 'string'), _q('limit', 'integer')], 'Cursor'
        ),
        '/pages': _list_op(
            'list_pages', [_q('page', 'integer'), _q('per_page', 'integer')], 'Numbered'
        ),
    },
    'components': {
        'schemas': {
            'User': {
                'type': 'object',
                'required': ['id'],
                'properties': {'id': {'type': 'integer'}},
            },
            'Offset': {
                'type': 'object',
                'required': ['data'],
                'properties': {
                    'data': {
                        'type': 'array',
                        'items': {'$ref': '#/components/schemas/User'},
                    },
                    'total': {'type': 'integer'},
                    'region': {'type': 'string'},
                },
            },
            'Cursor': {
                'type': 'object',
                'required': ['data'],
                'properties': {
                    'data': {
                        'type': 'array',
                        'items': {'$ref': '#/components/schemas/User'},
                    },
                    'next': {'type': ['string', 'null']},
                },
            },
            'Numbered': {
                'type': 'object',
                'required': ['data'],
                'properties': {
                    'data': {
                        'type': 'array',
                        'items': {'$ref': '#/components/schemas/User'},
                    },
                    'pages': {'type': 'integer'},
                },
            },
        }
    },
}

PAGINATION = {
    'enabled': True,
    'default_data_path': 'data',
    'endpoints': {
        'list_users': {'style': 'offset', 'total_path': 'total'},
        'list_feed': {'style': 'cursor', 'next_cursor_path': 'next'},
        'list_pages': {'style': 'page', 'total_pages_path': 'pages'},
    },
}


class Server:
    def __init__(self) -> None:
        self.requests: list[httpx2.Request] = []

    def handler(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        params = request.url.params
        if request.url.path == '/users':
            start, limit = int(params['offset']), int(params['limit'])
            rows = list(range(start, min(start + limit, TOTAL)))
            body: dict = {'total': TOTAL, 'region': 'eu'}
        elif request.url.path == '/feed':
            start = int(params.get('cursor') or 0)
            limit = int(params['limit'])
            rows = list(range(start, min(start + limit, TOTAL)))
            nxt = start + limit
            body = {'next': str(nxt) if nxt < TOTAL else None}
        else:
            page, per = int(params['page']), int(params['per_page'])
            start = (page - 1) * per
            rows = list(range(start, min(start + per, TOTAL)))
            body = {'pages': -(-TOTAL // per)}
        body['data'] = [{'id': i} for i in rows]
        return httpx2.Response(200, json=body)


def _generate(tmp_path: Path, package: str, **config: Any) -> None:
    spec_file = tmp_path / 'openapi.yaml'
    spec_file.write_text(yaml.safe_dump(SPEC), encoding='utf-8')
    Codegen(
        DocumentConfig(
            source=str(spec_file),
            output=str(tmp_path / package),
            base_url='https://example.test',
            pagination=PAGINATION,
            **config,
        )
    ).generate()


def _import(tmp_path: Path, name: str) -> Any:
    sys.path.insert(0, str(tmp_path))
    try:
        return importlib.import_module(name)
    finally:
        sys.path.remove(str(tmp_path))


@pytest.fixture(scope='module')
def fn_pkg(tmp_path_factory: pytest.TempPathFactory) -> Any:
    tmp_path = tmp_path_factory.mktemp('pages_fn')
    _generate(tmp_path, 'pages_fn')
    endpoints = _import(tmp_path, 'pages_fn.endpoints')
    client_cls = _import(tmp_path, 'pages_fn.client').Client
    return endpoints, client_cls


def _client(client_cls: type, server: Server) -> Any:
    transport = httpx2.MockTransport(server.handler)
    return client_cls(
        http_client=httpx2.Client(base_url='https://example.test', transport=transport),
        async_http_client=httpx2.AsyncClient(
            base_url='https://example.test', transport=transport
        ),
    )


class TestOffsetPages:
    def test_page_exposes_total_next_offset_and_envelope(self, fn_pkg: Any) -> None:
        endpoints, client_cls = fn_pkg
        page = endpoints.list_users_page(
            page_size=10, client=_client(client_cls, Server())
        )
        assert [u.id for u in page.items] == list(range(10))
        assert page.total == TOTAL
        assert page.next_offset == 10
        assert page.data.region == 'eu'
        assert page.has_next_page()
        second = page.get_next_page()
        assert second is not None and second.items[0].id == 10

    def test_pages_walks_every_page_and_matches_iter(self, fn_pkg: Any) -> None:
        endpoints, client_cls = fn_pkg
        client = _client(client_cls, Server())
        pages = list(endpoints.list_users_pages(page_size=10, client=client))
        assert [len(p) for p in pages] == [10, 10, 3]
        assert not pages[-1].has_next_page() and pages[-1].next_offset is None
        assert pages[-1].get_next_page() is None
        flat = [u.id for p in pages for u in p]
        assert flat == [
            u.id for u in endpoints.list_users_iter(page_size=10, client=client)
        ]

    def test_resume_from_saved_offset(self, fn_pkg: Any) -> None:
        endpoints, client_cls = fn_pkg
        server = Server()
        page = endpoints.list_users_page(
            offset=20, page_size=10, client=_client(client_cls, server)
        )
        assert [u.id for u in page] == [20, 21, 22]
        assert server.requests[0].url.params['offset'] == '20'

    def test_max_pages(self, fn_pkg: Any) -> None:
        endpoints, client_cls = fn_pkg
        server = Server()
        pages = list(
            endpoints.list_users_pages(
                page_size=5, max_pages=2, client=_client(client_cls, server)
            )
        )
        assert len(pages) == 2
        assert len(server.requests) == 2


class TestCursorPages:
    def test_checkpoint_and_resume_from_cursor(self, fn_pkg: Any) -> None:
        endpoints, client_cls = fn_pkg
        client = _client(client_cls, Server())
        first = endpoints.list_feed_page(page_size=10, client=client)
        assert first.next_cursor == '10'
        saved = first.next_cursor
        resumed = endpoints.list_feed_page(cursor=saved, page_size=10, client=client)
        assert resumed.items[0].id == 10
        rest = list(
            endpoints.list_feed_pages(cursor=saved, page_size=10, client=client)
        )
        assert [len(p) for p in rest] == [10, 3]
        assert rest[-1].next_cursor is None

    @pytest.mark.asyncio
    async def test_async_pages(self, fn_pkg: Any) -> None:
        endpoints, client_cls = fn_pkg
        client = _client(client_cls, Server())
        sizes = [
            len(page)
            async for page in endpoints.async_list_feed_pages(
                page_size=10, client=client
            )
        ]
        assert sizes == [10, 10, 3]
        page = await endpoints.async_list_feed_page(page_size=20, client=client)
        nxt = await page.get_next_page()
        assert nxt is not None and len(nxt) == 3
        assert await nxt.get_next_page() is None


class TestNumberedPages:
    def test_total_pages_and_next_page(self, fn_pkg: Any) -> None:
        endpoints, client_cls = fn_pkg
        pages = list(
            endpoints.list_pages_pages(
                page_size=10, client=_client(client_cls, Server())
            )
        )
        assert [p.next_page for p in pages] == [2, 3, None]
        assert pages[0].total_pages == 3


def test_page_variants_can_be_disabled(tmp_path: Path) -> None:
    pagination = {**PAGINATION, 'page_variants': False}
    spec_file = tmp_path / 'openapi.yaml'
    spec_file.write_text(yaml.safe_dump(SPEC), encoding='utf-8')
    Codegen(
        DocumentConfig(
            source=str(spec_file),
            output=str(tmp_path / 'nopages'),
            base_url='https://example.test',
            pagination=pagination,
        )
    ).generate()
    src = (tmp_path / 'nopages' / 'endpoints.py').read_text()
    assert 'def list_users_iter(' in src
    assert 'def list_users_page(' not in src and 'def list_users_pages(' not in src


@pytest.mark.parametrize('result_objects', [False, True])
def test_class_client_exposes_pages(tmp_path: Path, result_objects: bool) -> None:
    package = f'pages_cls_{int(result_objects)}'
    _generate(tmp_path, package, client_style='client', result_objects=result_objects)
    pkg = _import(tmp_path, package)
    server = Server()
    client = pkg.Client(
        http_client=httpx2.Client(
            base_url='https://example.test',
            transport=httpx2.MockTransport(server.handler),
        )
    )
    if result_objects:
        query = client.list_users(page_size=10)
        assert query.page().total == TOTAL
        assert [len(p) for p in query.pages()] == [10, 10, 3]
    else:
        assert client.list_users_page(page_size=10).total == TOTAL
        assert [len(p) for p in client.list_users_pages(page_size=10)] == [10, 10, 3]
