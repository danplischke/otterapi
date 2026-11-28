"""Pagination through the request body, flattened or not (issue #30)."""

from __future__ import annotations

import copy
import importlib
import json
from pathlib import Path
from typing import Any

import httpx2
import pytest

from otterapi.exceptions import ConfigurationError
from otterapi.tests.fixtures.widgets import (
    BODY_PAGINATION,
    MODULE_SPLIT,
    SPEC,
    TOTAL_ROWS,
    WidgetServer,
    _json,
    generate,
)


def _source(tmp_path: Path, package: str, module: str) -> str:
    return (tmp_path / package / f'{module}.py').read_text(encoding='utf-8')


@pytest.fixture(scope='module')
def body_pkg(tmp_path_factory: pytest.TempPathFactory) -> Any:
    tmp_path = tmp_path_factory.mktemp('body_pag')
    generate(
        tmp_path,
        'body_pag',
        module_split=MODULE_SPLIT,
        pagination=BODY_PAGINATION,
        response_unwrap={'enabled': True, 'data_path': 'data'},
        dataframe={'enabled': True, 'pandas': True, 'default_path': 'data'},
        export={'enabled': True},
    )
    yield (
        importlib.import_module('body_pag.widgets'),
        importlib.import_module('body_pag.models'),
        importlib.import_module('body_pag.client').Client,
    )


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
            generate(tmp_path, 'body_bad', pagination=pagination)


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
        generate(
            tmp_path,
            'hdr_pag',
            spec=spec,
            pagination={'enabled': True, 'default_data_path': 'data'},
        )
        endpoints = importlib.import_module('hdr_pag.endpoints')
        models = importlib.import_module('hdr_pag.models')
        client_cls = importlib.import_module('hdr_pag.client').Client
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
        generate(
            tmp_path,
            'flat_pag',
            spec=spec,
            module_split=MODULE_SPLIT,
            pagination=BODY_PAGINATION,
            request_body={'flatten': True},
            dataframe={'enabled': True, 'pandas': True, 'default_path': 'data'},
            export={'enabled': True},
        )
        yield (
            importlib.import_module('flat_pag.widgets'),
            importlib.import_module('flat_pag.client').Client,
        )

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
