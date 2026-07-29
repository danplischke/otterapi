"""Tests for the shared code-generation core (``otterapi.codegen.emit``).

Both the non-split writer (:class:`~otterapi.codegen.codegen.Codegen`) and the
split writer (:class:`~otterapi.codegen.splitting.SplitModuleEmitter`) delegate
per-endpoint emission to :mod:`otterapi.codegen.emit`. These tests lock in the
behaviour that the consolidation fixed as a side effect of removing the two
divergent copies:

* split mode now honours ``generate_sync`` / ``generate_async`` (previously it
  emitted both regardless);
* non-split ``pagination`` + ``export`` together no longer raises (the old code
  referenced a non-existent ``PaginationConfig.limit_param``).
"""

from __future__ import annotations

import ast
from pathlib import Path

from otterapi.codegen.codegen import Codegen
from otterapi.config import DocumentConfig

FIXTURES_ROOT = Path(__file__).parent / 'fixtures' / 'golden'
PAGINATED_SPEC = FIXTURES_ROOT / 'paginated' / 'spec.yaml'


def _generate(output_dir: Path, **overrides) -> None:
    config = DocumentConfig.model_validate(
        {
            'source': str(PAGINATED_SPEC),
            'output': str(output_dir),
            'base_url': 'https://example.test',
            **overrides,
        }
    )
    Codegen(config).generate()


def _endpoint_function_names(output_dir: Path) -> set[str]:
    """Module-level function names across every generated endpoint module.

    Skips models/client/runtime files so only endpoint functions remain.
    """
    skip = {'models.py', '_client.py', 'client.py', '__init__.py'}
    names: set[str] = set()
    for path in output_dir.rglob('*.py'):
        if path.name in skip or path.name.startswith('_'):
            continue
        tree = ast.parse(path.read_text(encoding='utf-8'))
        for node in tree.body:  # module level only
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                names.add(node.name)
    return names


class TestGenerateSyncAsyncSelection:
    """``generate_sync`` / ``generate_async`` are honoured in BOTH layouts."""

    def test_nonsplit_async_only(self, tmp_path):
        _generate(tmp_path, generate_sync=False)
        names = _endpoint_function_names(tmp_path)
        assert 'async_list_items' in names
        assert 'list_items' not in names

    def test_nonsplit_sync_only(self, tmp_path):
        _generate(tmp_path, generate_async=False)
        names = _endpoint_function_names(tmp_path)
        assert 'list_items' in names
        assert 'async_list_items' not in names

    def test_split_async_only(self, tmp_path):
        # Regression: split mode used to ignore these flags and emit both.
        _generate(
            tmp_path,
            generate_sync=False,
            module_split={'enabled': True, 'strategy': 'tag'},
        )
        names = _endpoint_function_names(tmp_path)
        assert 'async_list_items' in names
        assert not any(not n.startswith('async_') for n in names if 'list_items' in n)

    def test_split_sync_only(self, tmp_path):
        _generate(
            tmp_path,
            generate_async=False,
            module_split={'enabled': True, 'strategy': 'tag'},
        )
        names = _endpoint_function_names(tmp_path)
        assert 'list_items' in names
        assert 'async_list_items' not in names


class TestPaginationPlusExport:
    """Pagination + export together works in both layouts (was a crash)."""

    def test_nonsplit_pagination_and_export(self, tmp_path):
        # Regression: this combination raised AttributeError on
        # ``PaginationConfig.limit_param`` before the consolidation.
        _generate(
            tmp_path,
            pagination={'enabled': True, 'auto_detect': True},
            export={'enabled': True, 'formats': ['csv', 'jsonl']},
        )
        names = _endpoint_function_names(tmp_path)
        assert 'list_items_export' in names
        assert 'async_list_items_export' in names
        # The paginated iterators the export helpers wrap are present too.
        assert 'list_items_iter' in names

    def test_split_pagination_and_export(self, tmp_path):
        _generate(
            tmp_path,
            pagination={'enabled': True, 'auto_detect': True},
            export={'enabled': True, 'formats': ['csv', 'jsonl']},
            module_split={'enabled': True, 'strategy': 'tag'},
        )
        names = _endpoint_function_names(tmp_path)
        assert 'list_items_export' in names
        assert 'async_list_items_export' in names


class TestCustomLimitParamInExport:
    """A custom pagination ``limit_param`` reaches the export helper (was ignored)."""

    def test_custom_limit_param_used(self, tmp_path):
        _generate(
            tmp_path,
            pagination={
                'enabled': True,
                'auto_detect': True,
                'default_limit_param': 'limit',
                'endpoints': {'list_items': {'style': 'offset', 'limit_param': 'max'}},
            },
            export={'enabled': True, 'formats': ['csv']},
        )
        export_src = (tmp_path / 'endpoints.py').read_text(encoding='utf-8')
        # The paginated export helper forwards the configured limit param name.
        assert "'max'" in export_src or '"max"' in export_src
