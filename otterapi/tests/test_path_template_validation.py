"""Generation-time checks on the path keys of a document.

Spec paths are appended to the client's base URL and some of their text ends up
in generated source, so three shapes are refused instead of emitted:

* a key that does not start with ``/`` -- ``.attacker.example/x`` or
  ``@attacker.example/x`` becomes a different host once appended;
* a ``.``/``..`` segment, which HTTP stacks resolve before routing;
* a ``{placeholder}`` with no matching path parameter, which has no Python
  variable to fill it.  It used to be emitted as a bare name; with
  ``validate_output`` switched off that is source code taken from the spec.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from otterapi.codegen.codegen import Codegen
from otterapi.config import DocumentConfig
from otterapi.exceptions import CodeGenerationError

_OK = {'200': {'description': 'ok'}}


def _id_param(name: str = 'item_id') -> dict[str, Any]:
    return {'in': 'path', 'name': name, 'required': True, 'schema': {'type': 'string'}}


def _spec(paths: dict[str, Any]) -> dict[str, Any]:
    return {
        'openapi': '3.0.0',
        'info': {'title': 'Paths API', 'version': '1.0.0'},
        'servers': [{'url': 'https://api.example.test'}],
        'paths': paths,
    }


def _generate(tmp_path: Path, paths: dict[str, Any], **kwargs: Any) -> Path:
    spec_file = tmp_path / 'openapi.yaml'
    spec_file.write_text(yaml.safe_dump(_spec(paths)), encoding='utf-8')
    output = tmp_path / 'client'
    config_kwargs = {
        key: kwargs.pop(key)
        for key in ('include_paths', 'exclude_paths')
        if key in kwargs
    }
    Codegen(
        DocumentConfig(source=str(spec_file), output=str(output), **config_kwargs),
        **kwargs,
    ).generate()
    return output


class TestPathKeysMustStartWithASlash:
    @pytest.mark.parametrize(
        'path',
        ['.attacker.example/collect', '@attacker.example/collect', 'items', ':8443/x'],
    )
    def test_generation_fails_and_leaves_nothing_behind(self, tmp_path, path):
        with pytest.raises(CodeGenerationError, match='does not start with'):
            _generate(tmp_path, {path: {'get': {'operationId': 'x', 'responses': _OK}}})
        assert not (tmp_path / 'client').exists()

    def test_the_error_names_the_offending_path(self, tmp_path):
        with pytest.raises(CodeGenerationError) as excinfo:
            _generate(
                tmp_path,
                {
                    '@attacker.example/x': {
                        'get': {'operationId': 'x', 'responses': _OK}
                    }
                },
            )
        assert '@attacker.example/x' in str(excinfo.value)

    def test_an_excluded_path_is_not_checked(self, tmp_path):
        output = _generate(
            tmp_path,
            {
                '/ok': {'get': {'operationId': 'ok', 'responses': _OK}},
                'bad': {'get': {'operationId': 'bad', 'responses': _OK}},
            },
            exclude_paths=['bad'],
        )
        endpoints = (output / 'endpoints.py').read_text(encoding='utf-8')
        assert 'def ok(' in endpoints
        assert 'def bad(' not in endpoints


class TestDotSegments:
    @pytest.mark.parametrize('path', ['/a/../b', '/a/./b', '/..', '/a/%2e%2e/b'])
    def test_generation_fails(self, tmp_path, path):
        with pytest.raises(CodeGenerationError, match=r'"\." or "\.\."'):
            _generate(tmp_path, {path: {'get': {'operationId': 'x', 'responses': _OK}}})

    @pytest.mark.parametrize('path', ['/v1.0/items', '/a/b..c', '/a/..b', '/a/...'])
    def test_look_alikes_are_fine(self, tmp_path, path):
        output = _generate(
            tmp_path, {path: {'get': {'operationId': 'x', 'responses': _OK}}}
        )
        assert path in (output / 'endpoints.py').read_text(encoding='utf-8')


class TestUndeclaredPlaceholders:
    def test_a_placeholder_without_a_parameter_fails_generation(self, tmp_path):
        with pytest.raises(CodeGenerationError, match=r'\{item_id\}'):
            _generate(
                tmp_path,
                {'/items/{item_id}': {'get': {'operationId': 'x', 'responses': _OK}}},
            )

    def test_only_the_undeclared_placeholders_are_named(self, tmp_path):
        with pytest.raises(CodeGenerationError) as excinfo:
            _generate(
                tmp_path,
                {
                    '/items/{item_id}/parts/{part_id}': {
                        'get': {
                            'operationId': 'x',
                            'parameters': [_id_param('item_id')],
                            'responses': _OK,
                        }
                    }
                },
            )
        message = str(excinfo.value)
        assert '{part_id}' in message
        assert '{item_id}' not in message.replace("'/items/{item_id}/parts", '')

    @pytest.mark.parametrize(
        'path', ['/files/{secret.value}', '/files/{a + b}', '/files/{x();y}']
    )
    def test_nothing_derived_from_a_placeholder_is_emitted_as_code(
        self, tmp_path, path
    ):
        # Validation of the generated module is off, so only the generator's own
        # check stands between a placeholder and the emitted source.
        with pytest.raises(CodeGenerationError):
            _generate(
                tmp_path,
                {path: {'get': {'operationId': 'x', 'responses': _OK}}},
                validate_output=False,
            )
        assert not (tmp_path / 'client').exists()

    def test_a_path_item_level_parameter_counts_as_declared(self, tmp_path):
        output = _generate(
            tmp_path,
            {
                '/items/{item_id}': {
                    'parameters': [_id_param()],
                    'get': {'operationId': 'getItem', 'responses': _OK},
                }
            },
        )
        assert 'format_path_param(item_id)' in (output / 'endpoints.py').read_text(
            encoding='utf-8'
        )

    def test_an_excluded_path_is_not_checked(self, tmp_path):
        output = _generate(
            tmp_path,
            {
                '/ok': {'get': {'operationId': 'ok', 'responses': _OK}},
                '/items/{item_id}': {'get': {'operationId': 'bad', 'responses': _OK}},
            },
            exclude_paths=['/items/*'],
        )
        assert 'def ok(' in (output / 'endpoints.py').read_text(encoding='utf-8')
