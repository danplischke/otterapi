"""Golden-file tests for the code generator.

For each subdirectory of ``tests/fixtures/golden/`` we:

1. Run ``Codegen`` against ``spec.yaml``.
2. Diff every emitted ``*.py`` against ``expected/<filename>``.
3. Fail on mismatch with a unified diff.

A fixture may place an optional ``config.yaml`` alongside ``spec.yaml``
carrying extra ``DocumentConfig`` fields (e.g. ``pagination.enabled: true``
or ``dataframe.pandas: true``) so the harness can cover feature-specific
emitters, not just the plain correctness paths.

Use ``OTTER_UPDATE_GOLDEN=1 uv run pytest otterapi/tests/test_golden.py``
to (re)write the ``expected/`` directory after intentional codegen changes.

The first run on a brand-new fixture creates ``expected/`` automatically and
asks the developer to commit and review it -- subsequent runs enforce
byte-for-byte equality.
"""

from __future__ import annotations

import difflib
import os
import shutil
from pathlib import Path

import pytest
import yaml

from otterapi.codegen.codegen import Codegen
from otterapi.config import DocumentConfig

GOLDEN_ROOT = Path(__file__).parent / 'fixtures' / 'golden'
UPDATE_ENV = 'OTTER_UPDATE_GOLDEN'


def _golden_specs() -> list[Path]:
    if not GOLDEN_ROOT.is_dir():
        return []
    return sorted(p for p in GOLDEN_ROOT.iterdir() if (p / 'spec.yaml').is_file())


def _generated_files(directory: Path) -> dict[str, str]:
    """Read every .py file in ``directory`` recursively, keyed by relative path."""
    out: dict[str, str] = {}
    for path in sorted(directory.rglob('*.py')):
        rel = path.relative_to(directory).as_posix()
        out[rel] = path.read_text(encoding='utf-8')
    return out


def _write_expected(expected_dir: Path, files: dict[str, str]) -> None:
    if expected_dir.exists():
        shutil.rmtree(expected_dir)
    expected_dir.mkdir(parents=True)
    for rel, content in files.items():
        target = expected_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding='utf-8')


def _diff(label: str, expected: str, actual: str) -> str:
    return ''.join(
        difflib.unified_diff(
            expected.splitlines(keepends=True),
            actual.splitlines(keepends=True),
            fromfile=f'expected/{label}',
            tofile=f'actual/{label}',
        )
    )


def _load_fixture_config(spec_dir: Path) -> dict:
    """Read the optional ``config.yaml`` carrying per-fixture overrides."""
    config_path = spec_dir / 'config.yaml'
    if not config_path.is_file():
        return {}
    return yaml.safe_load(config_path.read_text(encoding='utf-8')) or {}


@pytest.mark.parametrize('spec_dir', _golden_specs(), ids=lambda p: p.name)
def test_golden(spec_dir: Path, tmp_path: Path) -> None:
    spec_path = spec_dir / 'spec.yaml'
    expected_dir = spec_dir / 'expected'
    output_dir = tmp_path / spec_dir.name

    overrides = _load_fixture_config(spec_dir)
    config = DocumentConfig.model_validate(
        {
            'source': str(spec_path),
            'output': str(output_dir),
            'base_url': 'https://example.test',
            **overrides,
        }
    )
    Codegen(config).generate()

    actual_files = _generated_files(output_dir)
    assert actual_files, f'codegen produced no .py files in {output_dir}'

    if os.environ.get(UPDATE_ENV) == '1':
        _write_expected(expected_dir, actual_files)
        pytest.skip(
            f'Wrote {len(actual_files)} expected files for {spec_dir.name}; '
            'review and commit'
        )

    if not expected_dir.exists():
        # First run on a new fixture: bootstrap, then ask the dev to review.
        _write_expected(expected_dir, actual_files)
        pytest.fail(
            f'No expected/ directory for {spec_dir.name}; just bootstrapped '
            f'{len(actual_files)} files. Review the contents under '
            f'{expected_dir.relative_to(GOLDEN_ROOT.parent.parent.parent)} '
            'and commit them.'
        )

    expected_files = _generated_files(expected_dir)

    missing_in_actual = sorted(set(expected_files) - set(actual_files))
    missing_in_expected = sorted(set(actual_files) - set(expected_files))
    assert not missing_in_actual, (
        f'codegen no longer emits these files: {missing_in_actual}. '
        f'Re-run with {UPDATE_ENV}=1 if intentional.'
    )
    assert not missing_in_expected, (
        f'codegen now emits unexpected files: {missing_in_expected}. '
        f'Re-run with {UPDATE_ENV}=1 if intentional.'
    )

    mismatches = []
    for rel, expected_content in expected_files.items():
        actual_content = actual_files[rel]
        if actual_content != expected_content:
            mismatches.append(_diff(rel, expected_content, actual_content))
    assert not mismatches, (
        f'Generated output for {spec_dir.name} differs from goldens.\n'
        f'Re-run with {UPDATE_ENV}=1 if intentional.\n\n' + '\n'.join(mismatches)
    )
