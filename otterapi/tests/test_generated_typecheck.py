"""Type-check the *generated* clients with pyright.

The whole promise of the library is a fully typed client, yet only the generator
source was ever type-checked -- not its output. This gate generates a matrix of
clients spanning the feature surface and asserts pyright reports zero errors, so
a change that makes generated code type-incorrect (a missing import, a wrong
return type, ...) fails CI instead of shipping.

Runs only when the ``typecheck`` dependency group (pyright) is installed; the
normal test matrix skips it. Marked ``typecheck`` for a dedicated CI job.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from otterapi.codegen.codegen import Codegen
from otterapi.config import DocumentConfig

pytestmark = pytest.mark.typecheck

if importlib.util.find_spec('pyright') is None:  # pragma: no cover
    pytest.skip(
        'pyright not installed (uv sync --group typecheck)', allow_module_level=True
    )

GOLDEN = Path(__file__).parent / 'fixtures' / 'golden'

_ALL_FEATURES = {
    'dataframe': {'enabled': True, 'pandas': True, 'polars': True},
    'export': {'enabled': True, 'formats': ['csv', 'jsonl', 'parquet']},
    'pagination': {'enabled': True, 'auto_detect': True},
}
# Discriminated-union item types + export is a known pre-existing base-codegen
# issue (the union is passed as ``model=`` to export, which wants a BaseModel);
# it is style-independent, so the discriminator case exercises client styles
# without export.
_NO_EXPORT = {
    'dataframe': {'enabled': True, 'pandas': True, 'polars': True},
    'pagination': {'enabled': True, 'auto_detect': True},
}

# name -> (spec fixture, config overrides). Spans styles x split x result_objects
# x nesting x pydantic version x discriminated unions.
MATRIX: dict[str, tuple[str, dict]] = {
    'functions': ('paginated', _ALL_FEATURES),
    'functions_v1': ('paginated', {**_ALL_FEATURES, 'pydantic_version': 1}),
    'functions_split': (
        'tagged',
        {**_ALL_FEATURES, 'module_split': {'enabled': True, 'strategy': 'tag'}},
    ),
    'client': ('paginated', {**_ALL_FEATURES, 'client_style': 'client'}),
    'client_query': (
        'paginated',
        {**_ALL_FEATURES, 'client_style': 'client', 'result_objects': True},
    ),
    'resource': ('tagged', {**_ALL_FEATURES, 'client_style': 'resource'}),
    'resource_query': (
        'paginated',
        {**_ALL_FEATURES, 'client_style': 'resource', 'result_objects': True},
    ),
    'nested_query': (
        'nested',
        {
            **_ALL_FEATURES,
            'client_style': 'resource',
            'resource_naming': 'path',
            'result_objects': True,
        },
    ),
    'resource_split': (
        'tagged',
        {
            **_ALL_FEATURES,
            'client_style': 'resource',
            'module_split': {'enabled': True, 'strategy': 'tag'},
        },
    ),
    'discriminated_union': (
        'discriminator',
        {**_NO_EXPORT, 'client_style': 'resource'},
    ),
}


def _generate_matrix(root: Path) -> None:
    for name, (spec, overrides) in MATRIX.items():
        config = DocumentConfig.model_validate(
            {
                'source': str(GOLDEN / spec / 'spec.yaml'),
                'output': str(root / name),
                'base_url': 'https://example.test',
                **overrides,
            }
        )
        # Skip formatting -- pyright does not care and it keeps the test fast.
        Codegen(config, format_output=False).generate()


def _run_pyright(root: Path) -> list[dict]:
    result = subprocess.run(
        [sys.executable, '-m', 'pyright', '--outputjson', str(root)],
        capture_output=True,
        text=True,
    )
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:  # pragma: no cover
        raise AssertionError(
            f'pyright did not return JSON.\nstdout:\n{result.stdout}\n'
            f'stderr:\n{result.stderr}'
        ) from None
    return [
        d for d in data.get('generalDiagnostics', []) if d.get('severity') == 'error'
    ]


def test_generated_clients_typecheck_clean(tmp_path):
    _generate_matrix(tmp_path)
    errors = _run_pyright(tmp_path)
    if errors:
        lines = [
            f'{Path(d["file"]).relative_to(tmp_path)}:'
            f'{d["range"]["start"]["line"] + 1} {d["message"].splitlines()[0]}'
            for d in errors
        ]
        pytest.fail(
            f'{len(errors)} pyright error(s) in generated clients:\n' + '\n'.join(lines)
        )
