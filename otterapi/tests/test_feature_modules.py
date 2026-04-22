"""Tests for the FeatureModule pipeline that emits runtime helper modules."""

from __future__ import annotations

from pathlib import Path

import pytest

from otterapi.codegen._features import (
    DataFrameFeature,
    ExportFeature,
    FeatureModule,
    PaginationFeature,
    all_features,
    write_enabled_features,
)
from otterapi.config import (
    DataFrameConfig,
    DocumentConfig,
    ExportConfig,
    PaginationConfig,
)


def _doc(**overrides) -> DocumentConfig:
    base = {'source': 'spec.json', 'output': './out'}
    base.update(overrides)
    return DocumentConfig(**base)


class TestPaginationFeature:
    def test_disabled_by_default(self):
        assert PaginationFeature().is_enabled(_doc()) is False

    def test_enabled_when_config_enabled(self):
        doc = _doc(pagination=PaginationConfig(enabled=True))
        assert PaginationFeature().is_enabled(doc) is True

    def test_writes_pagination_module(self, tmp_path: Path):
        result = PaginationFeature().write(tmp_path)
        assert result.exists()
        contents = Path(result).read_text(encoding='utf-8')
        assert 'paginate_offset' in contents
        assert 'iterate_cursor' in contents


class TestDataFrameFeature:
    def test_disabled_by_default(self):
        assert DataFrameFeature().is_enabled(_doc()) is False

    def test_enabled_only_when_a_library_is_picked(self):
        # ``enabled=True`` but neither pandas nor polars selected -> off.
        doc = _doc(dataframe=DataFrameConfig(enabled=True, pandas=False, polars=False))
        assert DataFrameFeature().is_enabled(doc) is False

    def test_enabled_with_pandas(self):
        doc = _doc(dataframe=DataFrameConfig(enabled=True, pandas=True))
        assert DataFrameFeature().is_enabled(doc) is True

    def test_enabled_with_polars(self):
        doc = _doc(dataframe=DataFrameConfig(enabled=True, pandas=False, polars=True))
        assert DataFrameFeature().is_enabled(doc) is True


class TestExportFeature:
    def test_disabled_by_default(self):
        assert ExportFeature().is_enabled(_doc()) is False

    def test_enabled_when_config_enabled(self):
        assert ExportFeature().is_enabled(_doc(export=ExportConfig(enabled=True)))

    def test_writes_export_module_with_utf8(self, tmp_path: Path):
        result = ExportFeature().write(tmp_path)
        contents = Path(result).read_bytes()
        # Must be valid UTF-8 and ASCII-only at the byte level (the Windows
        # encoding bug from PR #2 hinged on a stray em-dash being written as
        # cp1252 0x97 -- regression-fence that exact failure mode here).
        assert all(b < 128 for b in contents), 'runtime _export.py must stay ASCII'


class TestWriteEnabledFeatures:
    def test_writes_only_enabled(self, tmp_path: Path):
        doc = _doc(
            pagination=PaginationConfig(enabled=True),
            export=ExportConfig(enabled=False),
        )
        written = write_enabled_features(doc, tmp_path, all_features())
        names = sorted(p.name for p in written)
        assert names == ['_pagination.py']

    def test_writes_all_when_all_enabled(self, tmp_path: Path):
        doc = _doc(
            pagination=PaginationConfig(enabled=True),
            dataframe=DataFrameConfig(enabled=True, pandas=True),
            export=ExportConfig(enabled=True),
        )
        written = write_enabled_features(doc, tmp_path, all_features())
        assert sorted(p.name for p in written) == [
            '_dataframe.py',
            '_export.py',
            '_pagination.py',
        ]

    def test_writes_none_when_all_disabled(self, tmp_path: Path):
        assert write_enabled_features(_doc(), tmp_path, all_features()) == []


class TestExtensibility:
    """A new feature should be a 3-line subclass, not a fork of codegen.py."""

    def test_subclass_contract_is_enough(self, tmp_path: Path):
        class MetricsFeature(FeatureModule):
            module_filename = '_metrics.py'
            module_content = '"""Stub metrics module."""\n'

            def is_enabled(self, config: DocumentConfig) -> bool:
                return True

        feature = MetricsFeature()
        result = feature.write(tmp_path)
        assert result.exists()
        assert (
            Path(result).read_text(encoding='utf-8') == '"""Stub metrics module."""\n'
        )

    def test_abstract_methods_enforced(self):
        class Incomplete(FeatureModule):
            pass

        with pytest.raises(TypeError):
            Incomplete()  # type: ignore[abstract]
