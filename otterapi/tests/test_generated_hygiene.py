"""Generated code must arrive clean under the linters users actually run.

Every module otterapi emits lands in a user's tree, where it is linted by
*their* ruff config -- not by this repository's.  Issue #19 is the shape of
the problem: ``isinstance(x, A) or isinstance(x, B)`` is SIM101, its fix is
classed unsafe, and so it surfaced as a finding the user could not clear.
The same went for imports in discovery order (I001), a ``pass`` next to a
docstring (PIE790) and ``ast.unparse``'s parenthesized generator arguments
(UP034).

Three sources feed generated packages, so all three are checked:

- the runtime helpers under ``otterapi/codegen/runtime``, copied verbatim;
- the modules codegen builds as AST (client, endpoints, models);
- ``_client_stub.py.tpl``, rendered by string substitution.

The structural checks are plain AST walks, so the suite stays independent of
the linter's version and availability; the ruff run at the bottom is the
end-to-end guarantee and skips when ruff is not installed.
"""

from __future__ import annotations

import ast
import shutil
import subprocess
import sys
from pathlib import Path
from typing import cast

import pytest
import yaml

from otterapi.codegen.ast_utils import sort_import_blocks
from otterapi.codegen.codegen import Codegen
from otterapi.codegen.utils import _apply_import_separators
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


def _generate(fixture: str, overrides: dict, output: Path) -> None:
    """Generate *fixture* into *output*, honouring its own config.yaml."""
    spec_dir = FIXTURES_ROOT / fixture
    base: dict = {}
    config_path = spec_dir / 'config.yaml'
    if config_path.is_file():
        base = yaml.safe_load(config_path.read_text(encoding='utf-8')) or {}

    config = DocumentConfig.model_validate(
        {
            'source': str(spec_dir / 'spec.yaml'),
            'output': str(output),
            'base_url': 'https://example.test',
            **base,
            **overrides,
        }
    )
    Codegen(config).generate()


@pytest.mark.parametrize(
    ('fixture', 'overrides'),
    _SCENARIOS,
    ids=[f'{name}-{i}' for i, (name, _) in enumerate(_SCENARIOS)],
)
def test_generated_modules_merge_isinstance_calls(
    fixture: str, overrides: dict, tmp_path: Path
):
    _generate(fixture, overrides, tmp_path)

    offenders: dict[str, list[str]] = {}
    for path in sorted(tmp_path.rglob('*.py')):
        repeated = _duplicated_isinstance(path.read_text(encoding='utf-8'))
        if repeated:
            offenders[path.relative_to(tmp_path).as_posix()] = repeated

    assert not offenders, f'generated modules repeat isinstance calls: {offenders}'


# ---------------------------------------------------------------------------
# PIE790: a docstring is a body -- no ``pass`` beside it
# ---------------------------------------------------------------------------


def _stray_pass(source: str) -> list[str]:
    """Lines where a ``pass`` sits in a body that holds other statements."""
    offenders: list[str] = []
    for node in ast.walk(ast.parse(source)):
        body = getattr(node, 'body', None)
        if not isinstance(body, list) or len(body) < 2:
            continue
        offenders.extend(
            f'line {stmt.lineno}' for stmt in body if isinstance(stmt, ast.Pass)
        )
    return offenders


@pytest.mark.parametrize(
    ('fixture', 'overrides'),
    _SCENARIOS,
    ids=[f'{name}-{i}' for i, (name, _) in enumerate(_SCENARIOS)],
)
def test_generated_modules_have_no_stray_pass(
    fixture: str, overrides: dict, tmp_path: Path
):
    _generate(fixture, overrides, tmp_path)

    offenders: dict[str, list[str]] = {}
    for path in sorted(tmp_path.rglob('*.py')):
        stray = _stray_pass(path.read_text(encoding='utf-8'))
        if stray:
            offenders[path.relative_to(tmp_path).as_posix()] = stray

    assert not offenders, f'generated modules keep a redundant pass: {offenders}'


# ---------------------------------------------------------------------------
# I001: every import run comes out in isort order
# ---------------------------------------------------------------------------


def _section(stmt: ast.stmt) -> int:
    """The isort section for *stmt*, classified independently of the emitter."""
    if isinstance(stmt, ast.ImportFrom):
        if stmt.level:
            return 3
        if stmt.module == '__future__':
            return 0
        root = (stmt.module or '').split('.')[0]
    else:
        root = cast('ast.Import', stmt).names[0].name.split('.')[0]
    return 1 if root in sys.stdlib_module_names else 2


def _order_key(stmt: ast.stmt) -> tuple:
    """Position of *stmt* within its section: plain imports first, then names."""
    if isinstance(stmt, ast.ImportFrom):
        return (_section(stmt), 1, -stmt.level, (stmt.module or '').lower())
    module = cast('ast.Import', stmt).names[0].name
    return (_section(stmt), 0, 0, module.lower())


def _import_runs(body: list[ast.stmt]) -> list[list[ast.stmt]]:
    """Every contiguous run of imports in *body*, ``TYPE_CHECKING`` included."""
    runs: list[list[ast.stmt]] = []
    run: list[ast.stmt] = []
    for stmt in body:
        if isinstance(stmt, ast.Import | ast.ImportFrom):
            run.append(stmt)
            continue
        if run:
            runs.append(run)
            run = []
        if isinstance(stmt, ast.If):
            runs.extend(_import_runs(stmt.body))
    if run:
        runs.append(run)
    return runs


def _unsorted_imports(source: str) -> list[str]:
    """Descriptions of every import run that is not in isort order."""
    offenders: list[str] = []
    for run in _import_runs(ast.parse(source).body):
        keys = [_order_key(stmt) for stmt in run]
        if keys != sorted(keys):
            offenders.append(f'run at line {run[0].lineno} is out of order')
        for stmt in run:
            if not isinstance(stmt, ast.ImportFrom):
                continue
            names = [alias.name for alias in stmt.names]
            if names != sorted(names, key=lambda n: (_member_rank(n), n.lower())):
                offenders.append(f'names on line {stmt.lineno} are out of order')
    return offenders


def _member_rank(name: str) -> int:
    """Constants sort before classes, which sort before everything else."""
    stripped = name.lstrip('_')
    if not stripped:
        return 2
    if stripped.isupper():
        return 0
    return 1 if stripped[0].isupper() else 2


@pytest.mark.parametrize(
    ('fixture', 'overrides'),
    _SCENARIOS,
    ids=[f'{name}-{i}' for i, (name, _) in enumerate(_SCENARIOS)],
)
def test_generated_modules_sort_their_imports(
    fixture: str, overrides: dict, tmp_path: Path
):
    _generate(fixture, overrides, tmp_path)

    offenders: dict[str, list[str]] = {}
    for path in sorted(tmp_path.rglob('*.py')):
        unsorted_runs = _unsorted_imports(path.read_text(encoding='utf-8'))
        if unsorted_runs:
            offenders[path.relative_to(tmp_path).as_posix()] = unsorted_runs

    assert not offenders, f'generated modules emit unsorted imports: {offenders}'


def test_import_sections_are_separated_by_a_blank_line(tmp_path: Path):
    """Sections need the blank line between them, or ruff still reports I001."""
    _generate('constraints', {}, tmp_path)

    source = (tmp_path / '_client.py').read_text(encoding='utf-8')
    lines = source.splitlines()
    last_import = max(
        index
        for index, line in enumerate(lines)
        if line.startswith(('import ', 'from '))
    )
    block = lines[: last_import + 1]

    assert '' in block, f'no section separator in the import block: {block}'
    assert '\n\n\n' not in '\n'.join(block), 'sections separated by more than one line'


# ---------------------------------------------------------------------------
# End-to-end: ruff itself, when it is installed
# ---------------------------------------------------------------------------

# Rules a generated package can be held to regardless of the user's own
# config.  Style preferences (quotes, line length, argument counts) are the
# user's call and are deliberately left out.
_RUFF_RULES = 'SIM,UP,I,C4,B,PIE,PERF,RUF'


@pytest.mark.skipif(shutil.which('ruff') is None, reason='ruff is not installed')
@pytest.mark.parametrize(
    ('fixture', 'overrides'),
    _SCENARIOS,
    ids=[f'{name}-{i}' for i, (name, _) in enumerate(_SCENARIOS)],
)
def test_generated_package_passes_ruff(fixture: str, overrides: dict, tmp_path: Path):
    _generate(fixture, overrides, tmp_path)

    result = subprocess.run(
        [
            'ruff',
            'check',
            '--isolated',
            '--select',
            _RUFF_RULES,
            '--target-version',
            'py310',
            '--output-format',
            'concise',
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, f'ruff findings in generated code:\n{result.stdout}'


# ---------------------------------------------------------------------------
# sort_import_blocks: the emitter-side contract
# ---------------------------------------------------------------------------


def _sorted_source(source: str) -> str:
    """Round-trip *source* through the sorter and render the separators."""
    module = ast.Module(
        body=sort_import_blocks(ast.parse(source).body), type_ignores=[]
    )
    ast.fix_missing_locations(module)
    return _apply_import_separators(ast.unparse(module))


class TestSortImportBlocks:
    """Tests for sort_import_blocks()."""

    def test_orders_sections_and_separates_them(self):
        source = 'from .models import Pet\nimport httpx\nfrom typing import Any\n'
        assert _sorted_source(source) == (
            'from typing import Any\n\nimport httpx\n\nfrom .models import Pet'
        )

    def test_future_import_leads(self):
        source = 'import os\nfrom __future__ import annotations\n'
        assert _sorted_source(source).splitlines()[0] == (
            'from __future__ import annotations'
        )

    def test_plain_imports_precede_from_imports_in_a_section(self):
        source = 'from os import environ\nimport sys\n'
        assert _sorted_source(source) == 'import sys\nfrom os import environ'

    def test_names_are_ordered_constants_classes_then_functions(self):
        source = 'from typing import cast, Any, TYPE_CHECKING\n'
        assert _sorted_source(source) == 'from typing import TYPE_CHECKING, Any, cast'

    def test_recurses_into_type_checking_blocks(self):
        source = (
            'from typing import TYPE_CHECKING\n'
            'if TYPE_CHECKING:\n'
            '    from .models import Pet\n'
            '    import httpx\n'
        )
        assert _sorted_source(source).endswith(
            'if TYPE_CHECKING:\n    import httpx\n\n    from .models import Pet'
        )

    def test_leaves_statements_around_a_run_in_place(self):
        source = "import sys\n__all__ = ('x',)\nimport os\n"
        assert _sorted_source(source) == "import sys\n\n__all__ = ('x',)\nimport os"

    def test_a_class_after_the_imports_gets_two_blank_lines(self):
        source = 'import sys\n\n\nclass Item:\n    x: int\n'
        assert _sorted_source(source) == 'import sys\n\n\nclass Item:\n    x: int'

    def test_no_trailing_blank_line_inside_a_type_checking_block(self):
        source = (
            'from typing import TYPE_CHECKING\n'
            'if TYPE_CHECKING:\n'
            '    import httpx\n'
            '    x = 1\n'
        )
        assert _sorted_source(source).endswith('    import httpx\n    x = 1')

    def test_does_not_mutate_input(self):
        body = ast.parse('from .models import Pet\nimport httpx\n').body
        before = [ast.dump(stmt) for stmt in body]
        sort_import_blocks(body)
        assert [ast.dump(stmt) for stmt in body] == before
