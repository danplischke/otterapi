"""response_unwrap only where the response has the field (issue #30)."""

from __future__ import annotations

import copy
import importlib
from pathlib import Path

import pytest

from otterapi.exceptions import ConfigurationError
from otterapi.tests.fixtures.widgets import (
    MODULE_SPLIT,
    SPEC,
    TOTAL_ROWS,
    WidgetServer,
    _json,
    generate,
)


def _source(tmp_path: Path, package: str, module: str) -> str:
    return (tmp_path / package / f'{module}.py').read_text(encoding='utf-8')


class TestUnwrapNeedsTheField:
    def test_count_endpoint_returns_model_not_data(self, tmp_path: Path) -> None:
        pkg = generate(
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
        generate(
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
            generate(
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
        generate(
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
            generate(tmp_path, 'warn_pag', pagination=self.PAGINATION)
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
            generate(tmp_path, 'warn_off', pagination=pagination)
        assert not any('only the first page' in r.getMessage() for r in caplog.records)
