"""Tests for unused-import pruning in generated code.

Codegen collects imports optimistically (every pagination helper, both
DataFrame back-ends, the whole import closure of a model's field types), so
the collected set is always a superset of what a module ends up referencing.
``prune_unused_imports`` reconciles the two at emission time.

The unit tests cover the pruner itself; the end-to-end tests assert the
property that matters -- no generated module ever imports a name it does not
use -- without depending on ruff or any external formatter.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import yaml

from otterapi.codegen.ast_utils import (
    collect_bound_names,
    collect_referenced_names,
    find_unresolved_names,
    prune_unused_imports,
)
from otterapi.codegen.codegen import Codegen
from otterapi.codegen.utils import write_mod
from otterapi.config import DocumentConfig
from otterapi.exceptions import CodeGenerationError

FIXTURES_ROOT = Path(__file__).parent / 'fixtures' / 'golden'


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _prune_source(source: str) -> str:
    """Round-trip *source* through the pruner and return the result."""
    module = ast.parse(source)
    pruned = ast.Module(body=prune_unused_imports(module.body), type_ignores=[])
    ast.fix_missing_locations(pruned)
    return ast.unparse(pruned)


def _bound_import_names(tree: ast.Module) -> list[str]:
    """Every local name bound by an import in *tree* (module level or nested)."""
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module == '__future__':
                continue
            names.extend(
                alias.asname or alias.name for alias in node.names if alias.name != '*'
            )
        elif isinstance(node, ast.Import):
            names.extend(
                alias.asname or alias.name.split('.')[0] for alias in node.names
            )
    return names


def _unused_imports(source: str) -> set[str]:
    """Names imported by *source* but never referenced by it."""
    tree = ast.parse(source)
    referenced = collect_referenced_names(tree.body)
    return {name for name in _bound_import_names(tree) if name not in referenced}


# ---------------------------------------------------------------------------
# collect_referenced_names
# ---------------------------------------------------------------------------


class TestCollectReferencedNames:
    """Tests for collect_referenced_names()."""

    def test_import_statements_do_not_count_as_references(self):
        body = ast.parse('from typing import Any\nimport os\n').body
        assert collect_referenced_names(body) == set()

    def test_plain_name_reference(self):
        body = ast.parse('x = Foo()\n').body
        assert 'Foo' in collect_referenced_names(body)

    def test_attribute_base_is_referenced(self):
        body = ast.parse('x = pd.DataFrame()\n').body
        assert 'pd' in collect_referenced_names(body)

    def test_annotation_reference(self):
        body = ast.parse('def f(a: Pet) -> Order: ...\n').body
        names = collect_referenced_names(body)
        assert {'Pet', 'Order'} <= names

    def test_string_forward_reference_in_return_annotation(self):
        body = ast.parse("def f() -> 'pd.DataFrame': ...\n").body
        assert 'pd' in collect_referenced_names(body)

    def test_nested_string_forward_reference(self):
        body = ast.parse("def f(a: list['pl.DataFrame']): ...\n").body
        assert 'pl' in collect_referenced_names(body)

    def test_unparsable_string_annotation_is_ignored(self):
        body = ast.parse("def f(a: 'not valid python!!'): ...\n").body
        # Must not raise; simply contributes nothing.
        assert collect_referenced_names(body) == set()

    def test_all_entries_count_as_references(self):
        body = ast.parse("__all__ = ('Pet', 'Order')\n").body
        assert {'Pet', 'Order'} <= collect_referenced_names(body)

    def test_docstring_mentions_do_not_count(self):
        body = ast.parse('"""Returns a pd.DataFrame."""\nx = 1\n').body
        assert 'pd' not in collect_referenced_names(body)


# ---------------------------------------------------------------------------
# prune_unused_imports
# ---------------------------------------------------------------------------


class TestPruneUnusedImports:
    """Tests for prune_unused_imports()."""

    def test_removes_unused_name_from_from_import(self):
        result = _prune_source('from typing import Any, Union\n\ndef f(x: Any): ...\n')
        assert 'Union' not in result
        assert 'Any' in result

    def test_drops_statement_when_all_names_unused(self):
        result = _prune_source('from datetime import datetime\n\nx = 1\n')
        assert 'datetime' not in result

    def test_keeps_fully_used_import_untouched(self):
        source = 'from typing import Any\n\ndef f(x: Any): ...\n'
        assert 'Any' in _prune_source(source)

    def test_keeps_future_import(self):
        result = _prune_source('from __future__ import annotations\n\nx = 1\n')
        assert 'from __future__ import annotations' in result

    def test_keeps_star_import(self):
        result = _prune_source('from .pet import *\n\nx = 1\n')
        assert 'from .pet import *' in result

    def test_respects_import_aliases(self):
        result = _prune_source('import pandas as pd\nimport polars as pl\n\nx = pd\n')
        assert 'pandas as pd' in result
        assert 'polars' not in result

    def test_dotted_import_binds_root_package(self):
        result = _prune_source('import os.path\n\nx = os.path.join("a", "b")\n')
        assert 'import os.path' in result

    def test_prunes_inside_type_checking_block(self):
        source = (
            'from typing import TYPE_CHECKING\n'
            'if TYPE_CHECKING:\n'
            '    import pandas as pd\n'
            '    import polars as pl\n'
            "def f() -> 'pd.DataFrame': ...\n"
        )
        result = _prune_source(source)
        assert 'pandas as pd' in result
        assert 'polars' not in result
        assert 'TYPE_CHECKING' in result

    def test_empty_type_checking_block_cascades_away(self):
        """Dropping the block's last import must also drop TYPE_CHECKING itself."""
        source = (
            'from typing import TYPE_CHECKING\n'
            'if TYPE_CHECKING:\n'
            '    import pandas as pd\n'
            'x = 1\n'
        )
        result = _prune_source(source)
        assert 'TYPE_CHECKING' not in result
        assert 'pandas' not in result
        assert result.strip() == 'x = 1'

    def test_all_reexport_keeps_import(self):
        """A model re-exported through __all__ is used, even if never referenced."""
        source = "from .models import Pet\n__all__ = ('Pet',)\n"
        assert 'Pet' in _prune_source(source)

    def test_does_not_mutate_input(self):
        module = ast.parse('from typing import Any, Union\n\ndef f(x: Any): ...\n')
        original = ast.dump(module)
        prune_unused_imports(module.body)
        assert ast.dump(module) == original

    def test_local_import_inside_function_is_left_alone(self):
        """Only module-level / TYPE_CHECKING imports are codegen's concern."""
        source = 'def f():\n    from typing import Any\n    return 1\n'
        assert 'from typing import Any' in _prune_source(source)

    def test_literal_value_does_not_keep_a_same_named_import(self):
        """Literal['Status'] is a value, not a forward reference to Status."""
        source = (
            'from typing import Literal\n'
            'from .models import Status\n'
            "def f(x: Literal['Status']): ...\n"
        )
        result = _prune_source(source)
        assert 'from .models import Status' not in result
        assert 'Literal' in result


# ---------------------------------------------------------------------------
# write_mod integration
# ---------------------------------------------------------------------------


class TestWriteModPruning:
    """write_mod() prunes by default and can be opted out of."""

    def test_pruned_by_default(self, tmp_path: Path):
        body = ast.parse('from typing import Any, Union\n\ndef f(x: Any): ...\n').body
        target = tmp_path / 'mod.py'
        write_mod(body, target, format_code=False)
        assert 'Union' not in target.read_text(encoding='utf-8')

    def test_opt_out(self, tmp_path: Path):
        body = ast.parse('from typing import Any, Union\n\ndef f(x: Any): ...\n').body
        target = tmp_path / 'mod.py'
        write_mod(body, target, format_code=False, prune_imports=False)
        assert 'Union' in target.read_text(encoding='utf-8')


# ---------------------------------------------------------------------------
# find_unresolved_names: the opposite failure -- a forgotten import
# ---------------------------------------------------------------------------


class TestFindUnresolvedNames:
    """Tests for find_unresolved_names()."""

    def test_flags_missing_import(self):
        body = ast.parse('def f(x: Pet) -> Order: ...\n').body
        assert find_unresolved_names(body) == {'Pet', 'Order'}

    def test_imported_names_resolve(self):
        body = ast.parse('from .models import Pet\n\ndef f(x: Pet): ...\n').body
        assert find_unresolved_names(body) == set()

    def test_builtins_resolve(self):
        body = ast.parse('def f(x: int) -> str:\n    return str(len(x))\n').body
        assert find_unresolved_names(body) == set()

    def test_locals_and_parameters_resolve(self):
        source = (
            'def f(client=None, **kwargs):\n'
            '    c = client\n'
            '    rows = [item for item in c]\n'
            '    with open("x") as fh:\n'
            '        return fh, rows, kwargs\n'
        )
        assert find_unresolved_names(ast.parse(source).body) == set()

    def test_annassign_target_is_a_binding_even_with_load_ctx(self):
        """Codegen hand-builds nodes and ast.unparse ignores ctx, so a field
        target may carry Load. It still binds the name."""
        field = ast.AnnAssign(
            target=ast.Name(id='total', ctx=ast.Load()),
            annotation=ast.Name(id='int', ctx=ast.Load()),
            value=None,
            simple=1,
        )
        model = ast.ClassDef(
            name='M', bases=[], keywords=[], body=[field], decorator_list=[]
        )
        assert 'total' in collect_bound_names([model])
        assert find_unresolved_names([model]) == set()

    def test_literal_values_are_not_references(self):
        source = (
            'from typing import Literal\n'
            "def f(status: Literal['pending', 'sold']): ...\n"
        )
        assert find_unresolved_names(ast.parse(source).body) == set()

    def test_annotated_metadata_strings_are_not_references(self):
        source = (
            'from typing import Annotated\n'
            'from pydantic import Field\n'
            "def f(x: Annotated[str, Field(description='Status')]): ...\n"
        )
        assert find_unresolved_names(ast.parse(source).body) == set()

    def test_broken_all_reexport_is_flagged(self):
        body = ast.parse("from .models import Pet\n__all__ = ('Pet', 'Ghost')\n").body
        assert find_unresolved_names(body) == {'Ghost'}

    def test_star_import_stands_the_check_down(self):
        """A star import binds names this module cannot see."""
        body = ast.parse("from .pet import *\n__all__ = ('Anything',)\n").body
        assert find_unresolved_names(body) == set()

    def test_string_forward_reference_resolves_against_type_checking_block(self):
        source = (
            'from typing import TYPE_CHECKING\n'
            'if TYPE_CHECKING:\n'
            '    import pandas as pd\n'
            "def f() -> 'pd.DataFrame': ...\n"
        )
        assert find_unresolved_names(ast.parse(source).body) == set()


class TestWriteModNameValidation:
    """write_mod() refuses to emit a module with unresolvable names."""

    def test_raises_on_missing_import(self, tmp_path: Path):
        body = ast.parse('def f(x: Pet): ...\n').body
        target = tmp_path / 'mod.py'
        with pytest.raises(CodeGenerationError, match='Pet'):
            write_mod(body, target, format_code=False)
        assert not target.exists()

    def test_opt_out_via_validate_code(self, tmp_path: Path):
        body = ast.parse('def f(x: Pet): ...\n').body
        target = tmp_path / 'mod.py'
        write_mod(body, target, format_code=False, validate_code=False)
        assert 'Pet' in target.read_text(encoding='utf-8')

    def test_checks_after_pruning_not_before(self, tmp_path: Path):
        """Pruning runs first, so a name only kept by a pruned import still fails."""
        body = ast.parse(
            'from typing import Any\n\ndef f(x: Any) -> Missing: ...\n'
        ).body
        target = tmp_path / 'mod.py'
        with pytest.raises(CodeGenerationError, match='Missing'):
            write_mod(body, target, format_code=False)


# ---------------------------------------------------------------------------
# End-to-end: no generated module imports a name it does not use
# ---------------------------------------------------------------------------

# (scenario, extra DocumentConfig overrides)
_SCENARIOS: list[tuple[str, dict]] = [
    ('constraints', {}),
    ('discriminator', {}),
    (
        'dataframe',
        {
            'dataframe': {'enabled': True, 'pandas': True, 'polars': False},
        },
    ),
    (
        'paginated',
        {
            'pagination': {'enabled': True, 'auto_detect': True},
        },
    ),
    (
        'paginated',
        {
            'pagination': {'enabled': True, 'auto_detect': True},
            'generate_sync': False,
        },
    ),
    (
        'constraints',
        {
            'module_split': {'enabled': True, 'strategy': 'tag'},
            'reexport_models': True,
        },
    ),
]


@pytest.mark.parametrize(
    ('fixture', 'overrides'),
    _SCENARIOS,
    ids=[f'{name}-{i}' for i, (name, _) in enumerate(_SCENARIOS)],
)
def test_generated_modules_have_no_unused_imports(
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

    offenders: dict[str, set[str]] = {}
    for path in sorted(tmp_path.rglob('*.py')):
        unused = _unused_imports(path.read_text(encoding='utf-8'))
        if unused:
            offenders[path.relative_to(tmp_path).as_posix()] = unused

    assert not offenders, f'generated modules import unused names: {offenders}'
