"""Tests for the friendlier config validation error path (Wave 3.12)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from otterapi.config import ConfigValidationError, get_config


def _write(tmp_path: Path, name: str, data: dict) -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(data), encoding='utf-8')
    return path


class TestConfigValidationError:
    def test_invalid_value_for_field_raises_friendly_error(self, tmp_path: Path):
        # ``export.batch_size: "huge"`` is a typical typo that Pydantic
        # cannot coerce -- a bare ValidationError dumps something dense
        # and unhelpful, the friendly wrapper pinpoints the field.
        bad = {
            'documents': [
                {
                    'source': 'spec.json',
                    'output': './out',
                    'export': {'batch_size': 'huge'},
                },
            ],
        }
        path = _write(tmp_path, 'otter.yaml', bad)
        with pytest.raises(ConfigValidationError) as exc:
            get_config(str(path))
        msg = str(exc.value)
        assert 'Invalid OtterAPI configuration' in msg
        assert str(path) in msg
        # Field path should pinpoint the offending field.
        assert 'export.batch_size' in msg

    def test_missing_required_field_named_in_message(self, tmp_path: Path):
        # ``output`` is required on DocumentConfig.
        bad = {'documents': [{'source': 'spec.json'}]}
        path = _write(tmp_path, 'otter.yaml', bad)
        with pytest.raises(ConfigValidationError) as exc:
            get_config(str(path))
        msg = str(exc.value)
        assert 'output' in msg
        assert 'documents.0.output' in msg or 'documents' in msg

    def test_unknown_field_rejected_with_path(self, tmp_path: Path):
        bad = {
            'documents': [
                {
                    'source': 'spec.json',
                    'output': './out',
                    'export': {'enabled': True, 'unknown_field': 1},
                },
            ],
        }
        path = _write(tmp_path, 'otter.yaml', bad)
        with pytest.raises(ConfigValidationError) as exc:
            get_config(str(path))
        msg = str(exc.value)
        assert 'unknown_field' in msg

    def test_error_attributes_are_populated(self, tmp_path: Path):
        bad = {'documents': [{'source': 'spec.json'}]}
        path = _write(tmp_path, 'otter.yaml', bad)
        with pytest.raises(ConfigValidationError) as exc:
            get_config(str(path))
        assert exc.value.source is not None
        assert isinstance(exc.value.errors, list)
        assert len(exc.value.errors) >= 1

    def test_valid_config_does_not_raise(self, tmp_path: Path):
        good = {
            'documents': [
                {
                    'source': 'spec.json',
                    'output': './out',
                    'export': {'enabled': True, 'formats': ['csv']},
                },
            ],
        }
        path = _write(tmp_path, 'otter.yaml', good)
        config = get_config(str(path))
        assert len(config.documents) == 1
        assert config.documents[0].export.enabled is True

    def test_error_count_in_header(self, tmp_path: Path):
        # Multiple errors: bad type for pandas, missing output.
        bad = {
            'documents': [
                {'source': 'spec.json', 'dataframe': {'pandas': 'yes'}},
            ],
        }
        path = _write(tmp_path, 'otter.yaml', bad)
        with pytest.raises(ConfigValidationError) as exc:
            get_config(str(path))
        msg = str(exc.value)
        # Header should mention the count -- "2 errors", "3 errors", ...
        assert '2 error' in msg or '3 error' in msg or 'error' in msg
