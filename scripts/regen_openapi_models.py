#!/usr/bin/env python
"""Regenerate the OpenAPI Pydantic models in ``otterapi/openapi/v*/``.

The hand-maintained models in ``otterapi/openapi/{v2,v3,v3_1,v3_2}/*.py`` are
~4000 lines of Pydantic class definitions that mirror the canonical
JSON-Schema published by the OpenAPI Initiative. They have drifted in subtle
ways over time and are easy to mis-edit when a new field shows up upstream.

This script downloads the canonical schemas and re-emits each module via
``datamodel-code-generator``. Running it should be a *deliberate* refresh
step, not a part of the normal generate pipeline:

    uv run --with datamodel-code-generator python scripts/regen_openapi_models.py

After regeneration, **review the diff carefully** before committing -- the
generator may rename helper classes (e.g. ``Schema1`` vs ``Schema``) or
reorder inherited fields in ways that break callers in ``codegen``.

Issue: https://github.com/danplischke/otterapi/issues/3 (item 11).
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import NamedTuple

REPO_ROOT = Path(__file__).resolve().parent.parent


class SpecSource(NamedTuple):
    """A single OpenAPI version + where to source its schema + where to write."""

    version_label: str
    schema_url: str
    package: str  # subdir under otterapi/openapi/, e.g. 'v3_2'
    module_filename: str  # e.g. 'v3_2.py'


# Each entry tracks the canonical JSON-Schema URL for that OpenAPI revision.
# Updating the URL here is enough to point at a newer minor release once the
# OAI publishes one (e.g. 3.2.1 -> 3.2.2).
SOURCES: list[SpecSource] = [
    SpecSource(
        version_label='3.2.0',
        schema_url='https://spec.openapis.org/oas/3.2/schema/2025-09-15',
        package='v3_2',
        module_filename='v3_2.py',
    ),
    SpecSource(
        version_label='3.1.0',
        schema_url='https://spec.openapis.org/oas/3.1/schema/2022-10-07',
        package='v3_1',
        module_filename='v3_1.py',
    ),
    SpecSource(
        version_label='3.0.3',
        schema_url='https://spec.openapis.org/oas/3.0/schema/2021-09-28',
        package='v3',
        module_filename='v3.py',
    ),
    SpecSource(
        version_label='2.0',
        schema_url='https://raw.githubusercontent.com/OAI/OpenAPI-Specification/main/schemas/v2.0/schema.json',
        package='v2',
        module_filename='v2.py',
    ),
]


def _ensure_dmg() -> None:
    if shutil.which('datamodel-codegen') is None:
        sys.exit(
            'datamodel-codegen is not on PATH. Run with:\n'
            '    uv run --with datamodel-code-generator '
            'python scripts/regen_openapi_models.py'
        )


def _download(url: str, dest: Path) -> None:
    print(f'  downloading: {url}')
    with urllib.request.urlopen(url) as resp:
        dest.write_bytes(resp.read())


def _regen_one(source: SpecSource, work_dir: Path) -> None:
    print(f'== {source.version_label} -> otterapi/openapi/{source.package}/')
    schema_path = work_dir / f'{source.package}.schema.json'
    _download(source.schema_url, schema_path)

    target = (
        REPO_ROOT / 'otterapi' / 'openapi' / source.package / source.module_filename
    )
    target.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        'datamodel-codegen',
        '--input',
        str(schema_path),
        '--input-file-type',
        'jsonschema',
        '--output',
        str(target),
        '--output-model-type',
        'pydantic_v2.BaseModel',
        '--use-annotated',
        '--use-standard-collections',
        '--use-union-operator',
        '--target-python-version',
        '3.10',
        '--use-schema-description',
        '--snake-case-field',
        '--reuse-model',
        '--enable-version-header',
    ]
    print('  running:', ' '.join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    _ensure_dmg()
    work_dir = REPO_ROOT / '.cache' / 'openapi-schemas'
    work_dir.mkdir(parents=True, exist_ok=True)
    for source in SOURCES:
        _regen_one(source, work_dir)
    print(
        '\nDone. Review the diff in otterapi/openapi/, run the full test suite,\n'
        'and update any callers in otterapi/codegen/ that the regenerator\n'
        'renamed (typegen often relies on specific helper class names).'
    )


if __name__ == '__main__':
    main()
