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
