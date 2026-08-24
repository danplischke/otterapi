"""Generated code must not repeat ``isinstance`` calls on the same target.

Every module otterapi emits lands in a user's tree, where it is linted by
*their* ruff config -- not by this repository's.  ``isinstance(x, A) or
isinstance(x, B)`` is ruff's SIM101, and its fix is classed unsafe, so a
generated occurrence surfaces as a finding the user cannot auto-fix
(https://github.com/danplischke/otterapi/issues/19).

Two sources feed generated packages, so both are checked:

- the runtime helpers under ``otterapi/codegen/runtime``, copied verbatim;
- the modules codegen builds as AST (client, endpoints, models).

The check is a plain AST walk rather than a ruff subprocess, so the test
suite stays independent of the linter's version and availability.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import yaml

from otterapi.codegen.codegen import Codegen
from otterapi.config import DocumentConfig

RUNTIME_ROOT = Path(__file__).parent.parent / 'codegen' / 'runtime'
FIXTURES_ROOT = Path(__file__).parent / 'fixtures' / 'golden'


def _isinstance_target(node: ast.expr) -> str | None:
    """The dumped first argument of *node* when it is an ``isinstance`` call."""
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
        return None
    if node.func.id != 'isinstance' or len(node.args) != 2:
        return None
    return ast.dump(node.args[0])


def _duplicated_isinstance(source: str) -> list[str]:
    """Targets tested by more than one ``isinstance`` call in one ``or``."""
    offenders: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.BoolOp) or not isinstance(node.op, ast.Or):
            continue
        seen: dict[str, int] = {}
        for value in node.values:
            target = _isinstance_target(value)
            if target is not None:
                seen[target] = seen.get(target, 0) + 1
        offenders.extend(f'line {node.lineno}' for count in seen.values() if count > 1)
    return offenders


@pytest.mark.parametrize(
    'module', sorted(RUNTIME_ROOT.glob('*.py')), ids=lambda p: p.name
)
def test_runtime_helpers_merge_isinstance_calls(module: Path):
    offenders = _duplicated_isinstance(module.read_text(encoding='utf-8'))
    assert not offenders, (
        f'{module.name} repeats isinstance on one target at {offenders}; '
        f'merge the checks into a single call so SIM101 stays clean in '
        f'generated packages'
    )


# (scenario, extra DocumentConfig overrides)
_SCENARIOS: list[tuple[str, dict]] = [
    ('constraints', {}),
    ('discriminator', {}),
    (
        'dataframe',
        {
            'dataframe': {'enabled': True, 'pandas': True, 'polars': True},
            'export': {'enabled': True},
        },
    ),
    (
        'paginated',
        {
            'pagination': {'enabled': True, 'auto_detect': True},
        },
    ),
]


@pytest.mark.parametrize(
    ('fixture', 'overrides'),
    _SCENARIOS,
    ids=[f'{name}-{i}' for i, (name, _) in enumerate(_SCENARIOS)],
)
def test_generated_modules_merge_isinstance_calls(
    fixture: str, overrides: dict, tmp_path: Path
):
    spec_dir = FIXTURES_ROOT / fixture
    base: dict = {}
    config_path = spec_dir / 'config.yaml'
    if config_path.is_file():
        base = yaml.safe_load(config_path.read_text(encoding='utf-8')) or {}

    config = DocumentConfig.model_validate(
        {
            'source': str(spec_dir / 'spec.yaml'),
            'output': str(tmp_path),
            'base_url': 'https://example.test',
            **base,
            **overrides,
        }
    )
    Codegen(config).generate()

    offenders: dict[str, list[str]] = {}
    for path in sorted(tmp_path.rglob('*.py')):
        repeated = _duplicated_isinstance(path.read_text(encoding='utf-8'))
        if repeated:
            offenders[path.relative_to(tmp_path).as_posix()] = repeated

    assert not offenders, f'generated modules repeat isinstance calls: {offenders}'
