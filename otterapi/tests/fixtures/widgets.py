"""The path-versioned widgets API from issue #30, and a server for it."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx2

from otterapi.tests.helpers import generate_package


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

NAMING = {'rules': [{'pattern': r'_v\d+(?=_|$)', 'replace': ''}]}
VERSIONED = {
    'module_split': MODULE_SPLIT,
    'naming': NAMING,
    'versioning': {'path_pattern': '/v(?P<version>\\d+)/'},
    'response_unwrap': {'enabled': True, 'data_path': 'data'},
    'dataframe': {'enabled': True, 'pandas': True, 'default_path': 'data'},
}


def generate(
    tmp_path: Path, package: str, spec: dict | None = None, **config: Any
) -> Any:
    """Generate the widgets spec (or ``spec``) and import the package."""
    return generate_package(
        tmp_path, package, spec or SPEC, base_url='https://example.test', **config
    )
