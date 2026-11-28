"""Naming rules and path-version routing (issue #30)."""

from __future__ import annotations

import copy
import importlib
from pathlib import Path
from typing import Any

import httpx2
import pytest

from otterapi.exceptions import ConfigurationError
from otterapi.tests.fixtures.widgets import (
    BODY_PAGINATION,
    MODULE_SPLIT,
    NAMING,
    SPEC,
    TOTAL_ROWS,
    VERSIONED,
    WidgetServer,
    generate,
)


def _source(tmp_path: Path, package: str, module: str) -> str:
    return (tmp_path / package / f'{module}.py').read_text(encoding='utf-8')


class TestVersionRouting:
    @pytest.fixture(scope='class')
    def versioned(self, tmp_path_factory: pytest.TempPathFactory) -> Any:
        tmp_path = tmp_path_factory.mktemp('versioned')
        pagination = copy.deepcopy(BODY_PAGINATION)
        # Keyed by the new name: the latest (v2) endpoint.
        pagination['endpoints'] = {
            'get_widgets_by_id': pagination['endpoints']['get_widgets_by_id_v2']
        }
        generate(tmp_path, 'versioned', pagination=pagination, **VERSIONED)
        yield (
            tmp_path,
            (
                importlib.import_module('versioned'),
                importlib.import_module('versioned.widgets'),
                importlib.import_module('versioned.widgets.v1'),
                importlib.import_module('versioned.models'),
            ),
        )

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
        pkg = generate(
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
            generate(tmp_path, 'nosplit', naming=NAMING, versioning={})

    def test_submodule_needs_functions_style(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigurationError, match='client_style'):
            generate(
                tmp_path,
                'classy',
                naming=NAMING,
                versioning={},
                module_split=MODULE_SPLIT,
                client_style='client',
            )

    def test_suffix_mode_works_with_class_clients(self, tmp_path: Path) -> None:
        pkg = generate(
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
            generate(tmp_path, 'collide', naming=NAMING)

    def test_legacy_follows_latest_into_the_fallback_module(
        self, tmp_path: Path
    ) -> None:
        split = {**MODULE_SPLIT, 'min_endpoints': 10}
        generate(tmp_path, 'fallback', **{**VERSIONED, 'module_split': split})
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
        pkg = generate(
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
            generate(
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
        generate(
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
        generate(tmp_path, 'dup_variant', spec=spec, pagination=BODY_PAGINATION)
