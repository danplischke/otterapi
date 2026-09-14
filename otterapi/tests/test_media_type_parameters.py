"""Content-type keys carrying media-type parameters still classify correctly.

A media type may carry parameters after a ``;`` (RFC 9110 8.3), and its type,
subtype and parameter names are case-insensitive. Springdoc-generated specs
routinely key their content on ``application/json;charset=utf-8``. Comparing
that key against the bare ``'application/json'`` made JSON detection fail, so a
schema reachable only through such a key was never realized into a model -- a
spec that uses the parametrized spelling throughout generated *zero* models.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import pytest

from otterapi.codegen.codegen import Codegen
from otterapi.codegen.types import RequestBodyInfo, ResponseInfo
from otterapi.config import DocumentConfig
from otterapi.openapi.constants import base_media_type, is_json_media_type


def _spec(response_key: str, body_key: str | None = None) -> dict[str, Any]:
    """A one-operation spec keyed on *response_key* (and optionally *body_key*)."""
    operation: dict[str, Any] = {
        'operationId': 'getEntry',
        'responses': {
            '200': {
                'description': 'ok',
                'content': {
                    response_key: {'schema': {'$ref': '#/components/schemas/CoreEntry'}}
                },
            }
        },
    }
    if body_key:
        operation['operationId'] = 'createEntry'
        operation['requestBody'] = {
            'required': True,
            'content': {
                body_key: {'schema': {'$ref': '#/components/schemas/NewEntry'}}
            },
        }
    method = 'post' if body_key else 'get'
    return {
        'openapi': '3.0.1',
        'info': {'title': 'Charset', 'version': '1.0'},
        'servers': [{'url': 'https://data.example.test'}],
        'paths': {'/entry': {method: operation}},
        'components': {
            'schemas': {
                'CoreEntry': {
                    'type': 'object',
                    'properties': {'rcsb_id': {'type': 'string'}},
                },
                'NewEntry': {
                    'type': 'object',
                    'properties': {'title': {'type': 'string'}},
                },
            }
        },
    }


def _generate(tmp_path: Path, spec: dict) -> Path:
    spec_file = tmp_path / 'openapi.json'
    spec_file.write_text(json.dumps(spec), encoding='utf-8')
    output = tmp_path / 'pkg'
    Codegen(DocumentConfig(source=str(spec_file), output=str(output))).generate()
    return output


def _endpoint(output: Path, name: str) -> ast.FunctionDef | ast.AsyncFunctionDef:
    tree = ast.parse((output / 'endpoints.py').read_text(encoding='utf-8'))
    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == name
        ):
            return node
    raise AssertionError(f'no generated function named {name}')


def _request_keywords(output: Path, name: str) -> set[str]:
    """The keyword arguments the endpoint passes to ``_request``."""
    for call in ast.walk(_endpoint(output, name)):
        if (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr.startswith('_request')
        ):
            return {kw.arg for kw in call.keywords if kw.arg}
    raise AssertionError(f'{name} never issues a request')


class TestMediaTypeHelpers:
    @pytest.mark.parametrize(
        'content_type',
        [
            'application/json',
            'application/json;charset=utf-8',
            'application/json; charset=UTF-8',
            'Application/JSON',
            '  application/json  ',
            'text/json;charset=iso-8859-1',
            'application/vnd.api+json;version=1',
        ],
    )
    def test_json_media_types(self, content_type: str):
        assert is_json_media_type(content_type)

    @pytest.mark.parametrize(
        'content_type',
        [
            'application/xml',
            'text/plain;charset=utf-8',
            'application/octet-stream',
            'multipart/form-data; boundary=x',
            'application/jsonl',
        ],
    )
    def test_non_json_media_types(self, content_type: str):
        assert not is_json_media_type(content_type)

    def test_base_media_type_strips_parameters_and_case(self):
        assert base_media_type('Application/JSON; charset=utf-8') == 'application/json'
        assert base_media_type('text/plain') == 'text/plain'


class TestResponseAndBodyClassification:
    def test_response_is_json_with_a_charset(self):
        info = ResponseInfo(200, 'application/json;charset=utf-8')
        assert info.is_json
        assert not info.is_text
        assert not info.is_binary

    def test_text_response_with_a_charset_is_still_text(self):
        info = ResponseInfo(200, 'text/plain; charset=utf-8')
        assert info.is_text
        assert not info.is_json

    def test_binary_response_with_parameters(self):
        assert ResponseInfo(200, 'application/pdf; version=1.7').is_binary

    def test_body_is_json_with_a_charset(self):
        info = RequestBodyInfo(content_type='application/json;charset=UTF-8')
        assert info.is_json
        assert info.httpx_param_name == 'json'

    def test_multipart_body_with_a_boundary(self):
        info = RequestBodyInfo(content_type='multipart/form-data; boundary=abc')
        assert info.is_multipart
        assert info.httpx_param_name == 'files'

    def test_form_body_with_a_charset(self):
        info = RequestBodyInfo(
            content_type='application/x-www-form-urlencoded;charset=utf-8'
        )
        assert info.is_form
        assert info.httpx_param_name == 'data'


class TestGenerationWithParametrizedJson:
    def test_a_parametrized_sole_content_type_still_emits_the_model(
        self, tmp_path: Path
    ):
        """The reported bug: a schema-rich spec generating no models at all."""
        output = _generate(tmp_path, _spec('application/json;charset=utf-8'))

        models = (output / 'models.py').read_text(encoding='utf-8')
        assert 'class CoreEntry(' in models

        endpoint = _endpoint(output, 'get_entry')
        assert endpoint.returns is not None
        assert ast.unparse(endpoint.returns) == 'CoreEntry'

    def test_a_parametrized_request_body_is_sent_as_json(self, tmp_path: Path):
        """An unclassified body used to fall back to a raw ``content=`` send."""
        output = _generate(
            tmp_path,
            _spec(
                'application/json; charset=UTF-8',
                body_key='application/json;charset=UTF-8',
            ),
        )

        assert 'json' in _request_keywords(output, 'create_entry')
        assert 'content' not in _request_keywords(output, 'create_entry')

    def test_a_parametrized_json_key_wins_over_an_earlier_non_json_one(
        self, tmp_path: Path
    ):
        """``_select_content_type`` prefers JSON, parameters or not."""
        spec = _spec('application/json;charset=utf-8')
        content = spec['paths']['/entry']['get']['responses']['200']['content']
        spec['paths']['/entry']['get']['responses']['200']['content'] = {
            'application/xml': {'schema': {'type': 'string'}},
            **content,
        }

        output = _generate(tmp_path, spec)

        endpoint = _endpoint(output, 'get_entry')
        assert endpoint.returns is not None
        assert ast.unparse(endpoint.returns) == 'CoreEntry'

    def test_an_uppercase_key_is_matched_too(self, tmp_path: Path):
        output = _generate(tmp_path, _spec('Application/JSON'))

        endpoint = _endpoint(output, 'get_entry')
        assert endpoint.returns is not None
        assert ast.unparse(endpoint.returns) == 'CoreEntry'

    def test_a_non_json_content_type_still_returns_the_raw_response(
        self, tmp_path: Path
    ):
        """Normalizing must not turn everything into JSON."""
        output = _generate(tmp_path, _spec('application/xml;charset=utf-8'))

        endpoint = _endpoint(output, 'get_entry')
        assert endpoint.returns is not None
        assert ast.unparse(endpoint.returns) == 'Response'
