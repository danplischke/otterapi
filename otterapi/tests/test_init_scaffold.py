"""Tests for the optional-section scaffolding emitted by ``otterapi init``."""

from __future__ import annotations

import yaml

from otterapi.cli import _commented_optional_sections


class TestCommentedSections:
    def test_returns_a_string(self):
        scaffold = _commented_optional_sections()
        assert isinstance(scaffold, str)
        assert len(scaffold) > 0

    def test_mentions_every_optional_section(self):
        scaffold = _commented_optional_sections()
        for section in (
            'pagination',
            'export',
            'dataframe',
            'module_split',
            'response_unwrap',
        ):
            assert section in scaffold

    def test_all_lines_are_comments_or_blank(self):
        # The scaffold appends after a real config -- it MUST stay comments
        # so YAML still parses to the original dict.
        for line in _commented_optional_sections().splitlines():
            stripped = line.strip()
            assert stripped == '' or stripped.startswith('#'), (
                f'non-comment line in scaffold: {line!r}'
            )

    def test_appending_scaffold_does_not_change_parsed_config(self):
        base_yaml = yaml.safe_dump(
            {'documents': [{'source': 'spec.json', 'output': './out'}]},
            sort_keys=False,
        )
        combined = base_yaml + _commented_optional_sections()
        # Round-trip: the parsed combined YAML must equal the parsed base.
        assert yaml.safe_load(combined) == yaml.safe_load(base_yaml)

    def test_mentions_otterapi_parquet_extra(self):
        # New users discovering the export section should see the
        # "[parquet]" extra mentioned right next to the format list so
        # they know what to install.
        assert '[parquet]' in _commented_optional_sections()
