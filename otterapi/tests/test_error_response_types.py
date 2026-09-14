"""Error responses must not shape an endpoint's return type.

The generated client raises on ``response.is_error`` -- httpx's name for 4xx
and 5xx -- *before* it parses anything, so a 4xx/5xx schema never reaches the
parser. Folding it into the return type is not merely redundant: pydantic's
smart union picks the arm that needs the least coercion, so an error response
typed ``{'type': 'object'}`` contributes a ``dict[str, Any]`` arm that is an
exact match for any JSON object and beats the success model every time. The
endpoint then returns a plain dict for every 200, and the typed model the spec
promises is never constructed.
"""

from __future__ import annotations

import ast
import copy
import importlib
import json
import sys
from pathlib import Path
from typing import Any

import httpx
import pytest

from otterapi.codegen.codegen import Codegen
from otterapi.config import DocumentConfig

CORE_ENTRY = {
    'type': 'object',
    'properties': {'rcsb_id': {'type': 'string'}, 'title': {'type': 'string'}},
}

# The shape reported in the issue: a typed 200 alongside an untyped 404 body.
# Real example: RCSB PDB Data API, /rest/v1/core/entry/{entry_id}.
UNTYPED_ERROR_SPEC: dict[str, Any] = {
    'openapi': '3.0.1',
    'info': {'title': 'Entries', 'version': '1.0'},
    'servers': [{'url': 'https://data.example.test'}],
    'paths': {
        '/entry/{entry_id}': {
            'get': {
                'operationId': 'getEntryById',
                'parameters': [
                    {
                        'name': 'entry_id',
                        'in': 'path',
                        'required': True,
                        'schema': {'type': 'string'},
                    }
                ],
                'responses': {
                    '200': {
                        'description': 'ok',
                        'content': {
                            'application/json': {
                                'schema': {'$ref': '#/components/schemas/CoreEntry'}
                            }
                        },
                    },
                    '404': {
                        'description': 'not found',
                        'content': {'application/json': {'schema': {'type': 'object'}}},
                    },
                },
            }
        }
    },
    'components': {'schemas': {'CoreEntry': CORE_ENTRY}},
}


def _generate(tmp_path: Path, spec: dict, name: str = 'pkg') -> Path:
    """Generate *spec* into ``tmp_path/name`` and return the package directory."""
    spec_file = tmp_path / 'openapi.json'
    spec_file.write_text(json.dumps(spec), encoding='utf-8')
    output = tmp_path / name
    Codegen(DocumentConfig(source=str(spec_file), output=str(output))).generate()
    return output


def _return_annotation(output: Path, function: str) -> str:
    """The rendered return annotation of a generated endpoint function."""
    tree = ast.parse((output / 'endpoints.py').read_text(encoding='utf-8'))
    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == function
        ):
            assert node.returns is not None
            return ast.unparse(node.returns)
    raise AssertionError(f'no generated function named {function}')


def _parse_type(output: Path, function: str) -> str:
    """The type handed to ``_parse_response`` inside a generated endpoint."""
    tree = ast.parse((output / 'endpoints.py').read_text(encoding='utf-8'))
    for node in ast.walk(tree):
        if (
            not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            or node.name != function
        ):
            continue
        for call in ast.walk(node):
            if (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr.startswith('_parse_response')
            ):
                return ast.unparse(call.args[1])
    raise AssertionError(f'{function} never parses a response')


class TestReturnTypeIgnoresErrorResponses:
    def test_untyped_error_body_does_not_widen_the_return_type(self, tmp_path: Path):
        """The reported bug: ``CoreEntry | dict[str, Any]``."""
        output = _generate(tmp_path, UNTYPED_ERROR_SPEC)

        assert _return_annotation(output, 'get_entry_by_id') == 'CoreEntry'
        assert _return_annotation(output, 'async_get_entry_by_id') == 'CoreEntry'

    def test_the_parse_type_matches_the_annotation(self, tmp_path: Path):
        """A union arm that is never returned would still be parsed against."""
        output = _generate(tmp_path, UNTYPED_ERROR_SPEC)

        assert _parse_type(output, 'get_entry_by_id') == 'CoreEntry'
        assert _parse_type(output, 'async_get_entry_by_id') == 'CoreEntry'

    def test_a_typed_error_model_is_dropped_too(self, tmp_path: Path):
        """``Pet | Error`` is just as unreachable as ``Pet | dict``."""
        spec = {
            **copy.deepcopy(UNTYPED_ERROR_SPEC),
            'components': {
                'schemas': {
                    'CoreEntry': CORE_ENTRY,
                    'ApiError': {
                        'type': 'object',
                        'properties': {'message': {'type': 'string'}},
                    },
                }
            },
        }
        spec['paths']['/entry/{entry_id}']['get']['responses']['404']['content'][
            'application/json'
        ]['schema'] = {'$ref': '#/components/schemas/ApiError'}

        output = _generate(tmp_path, spec)

        assert _return_annotation(output, 'get_entry_by_id') == 'CoreEntry'

    def test_a_non_json_error_does_not_make_the_endpoint_return_response(
        self, tmp_path: Path
    ):
        """A ``text/plain`` 500 used to drag ``httpx.Response`` into the union.

        That arm also broke parsing outright: ``Response`` is not something a
        ``TypeAdapter`` can validate a JSON body against.
        """
        spec = copy.deepcopy(UNTYPED_ERROR_SPEC)
        spec['paths']['/entry/{entry_id}']['get']['responses']['500'] = {
            'description': 'boom',
            'content': {'text/plain': {'schema': {'type': 'string'}}},
        }

        output = _generate(tmp_path, spec)

        assert _return_annotation(output, 'get_entry_by_id') == 'CoreEntry'
        assert 'Response' not in _parse_type(output, 'get_entry_by_id')

    def test_several_success_responses_still_union(self, tmp_path: Path):
        """Only *error* responses are dropped -- 2xx variants are all reachable."""
        spec = {
            **copy.deepcopy(UNTYPED_ERROR_SPEC),
            'components': {
                'schemas': {
                    'CoreEntry': CORE_ENTRY,
                    'Accepted': {
                        'type': 'object',
                        'properties': {'job_id': {'type': 'string'}},
                    },
                }
            },
        }
        spec['paths']['/entry/{entry_id}']['get']['responses']['202'] = {
            'description': 'queued',
            'content': {
                'application/json': {
                    'schema': {'$ref': '#/components/schemas/Accepted'}
                }
            },
        }

        output = _generate(tmp_path, spec)

        assert _return_annotation(output, 'get_entry_by_id') == 'CoreEntry | Accepted'

    def test_a_3xx_response_is_still_parsed(self, tmp_path: Path):
        """``is_error`` is 4xx/5xx, so a redirect body still reaches the parser."""
        spec = {
            **copy.deepcopy(UNTYPED_ERROR_SPEC),
            'components': {
                'schemas': {
                    'CoreEntry': CORE_ENTRY,
                    'Moved': {
                        'type': 'object',
                        'properties': {'location': {'type': 'string'}},
                    },
                }
            },
        }
        spec['paths']['/entry/{entry_id}']['get']['responses']['302'] = {
            'description': 'moved',
            'content': {
                'application/json': {'schema': {'$ref': '#/components/schemas/Moved'}}
            },
        }

        output = _generate(tmp_path, spec)

        assert _return_annotation(output, 'get_entry_by_id') == 'CoreEntry | Moved'

    def test_error_responses_are_still_recorded_on_the_endpoint(self, tmp_path: Path):
        """Dropping them from the return type must not drop them from the model.

        ``response_infos`` still drives things like pagination unwrapping and
        the per-status error hierarchy.
        """
        spec_file = tmp_path / 'openapi.json'
        spec_file.write_text(json.dumps(UNTYPED_ERROR_SPEC), encoding='utf-8')
        codegen = Codegen(
            DocumentConfig(source=str(spec_file), output=str(tmp_path / 'pkg'))
        )
        codegen._load_schema()

        endpoints = codegen._generate_endpoints()
        statuses = {
            info.status_code
            for endpoint in endpoints
            for info in endpoint.response_infos
        }
        assert statuses == {200, 404}


class TestGeneratedClientReturnsTheTypedModel:
    """The end of the issue: ``isinstance(result, CoreEntry)`` has to hold."""

    @pytest.fixture(scope='class')
    def module(self, tmp_path_factory):
        package = 'error_union_pkg'
        parent = tmp_path_factory.mktemp('error_union')
        _generate(parent, UNTYPED_ERROR_SPEC, name=package)

        sys.path.insert(0, str(parent))
        try:
            for stale in [
                mod
                for mod in list(sys.modules)
                if mod == package or mod.startswith(package + '.')
            ]:
                sys.modules.pop(stale, None)
            yield importlib.import_module(package)
        finally:
            try:
                sys.path.remove(str(parent))
            except ValueError:
                pass

    @staticmethod
    def _handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == '/entry/4HHB':
            return httpx.Response(200, json={'rcsb_id': '4HHB', 'title': 'Hemoglobin'})
        return httpx.Response(404, json={'message': 'unknown entry'})

    def test_a_200_returns_the_model_not_a_dict(self, module):
        with httpx.Client(transport=httpx.MockTransport(self._handler)) as http:
            client = module.Client(http_client=http)
            result = module.get_entry_by_id('4HHB', client=client)

        assert isinstance(result, module.CoreEntry)
        assert result.rcsb_id == '4HHB'

    @pytest.mark.asyncio
    async def test_the_async_endpoint_returns_the_model_too(self, module):
        transport = httpx.MockTransport(self._handler)
        async with httpx.AsyncClient(transport=transport) as http:
            client = module.Client(async_http_client=http)
            result = await module.async_get_entry_by_id('4HHB', client=client)

        assert isinstance(result, module.CoreEntry)
        assert result.title == 'Hemoglobin'

    def test_a_404_still_raises_rather_than_parsing(self, module):
        with httpx.Client(transport=httpx.MockTransport(self._handler)) as http:
            client = module.Client(http_client=http, max_retries=0)
            with pytest.raises(module.NotFoundError):
                module.get_entry_by_id('0000', client=client)
