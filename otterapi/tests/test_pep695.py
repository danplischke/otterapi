"""Tests for the PEP 695 rewrite applied to emitted runtime helper modules.

Generated code defaults to the pre-3.12 ``TypeVar`` spelling so it keeps
running on Python 3.10/3.11.  Projects that declare ``target_python: "3.12"``
get type parameters instead, which is what linters configured for a modern
Python expect (ruff's UP047 / ``non-pep695-generic-function``).
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from textwrap import dedent

import pytest

from otterapi.codegen._features import (
    ConcurrencyFeature,
    DataFrameFeature,
    ExportFeature,
    PaginationFeature,
)
from otterapi.codegen._pep695 import modernize_type_params
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


class TestModernizeTypeParams:
    def test_rewrites_module_level_generic_function(self):
        source = dedent("""\
            from typing import TypeVar

            T = TypeVar('T')


            def identity(x: T) -> T:
                return x
            """)
        result = modernize_type_params(source)
        assert 'def identity[T](x: T) -> T:' in result
        assert 'TypeVar' not in result

    def test_drops_typevar_but_keeps_sibling_imports(self):
        source = dedent("""\
            from typing import Any, TypeVar

            T = TypeVar('T')


            def first(items: list[T], fallback: Any) -> T:
                return items[0] if items else fallback
            """)
        result = modernize_type_params(source)
        assert 'from typing import Any\n' in result
        assert 'def first[T](' in result

    def test_orders_type_params_by_first_use(self):
        source = dedent("""\
            from typing import TypeVar

            K = TypeVar('K')
            V = TypeVar('V')


            def lookup(table: dict[V, K], key: V) -> K:
                return table[key]
            """)
        assert 'def lookup[V, K](' in modernize_type_params(source)

    def test_type_param_order_follows_source_not_walk_order(self):
        # ``ast.walk`` is breadth-first: the trailing ``T`` sits a level above
        # ``U`` and would be declared first without positional sorting.
        source = dedent("""\
            from typing import TypeVar

            T = TypeVar('T')
            U = TypeVar('U')


            def pick(table: dict[U, T] | T) -> T:
                return table
            """)
        assert 'def pick[U, T](' in modernize_type_params(source)

    def test_async_functions_are_rewritten(self):
        source = dedent("""\
            from typing import TypeVar

            T = TypeVar('T')


            async def fetch(value: T) -> T:
                return value
            """)
        assert 'async def fetch[T](value: T) -> T:' in modernize_type_params(source)

    def test_nested_function_keeps_using_enclosing_type_param(self):
        # PEP 695 puts a function's type parameters in an enclosing scope, so
        # the inner def resolves ``T`` without a parameter list of its own --
        # adding one would shadow it.
        source = dedent("""\
            from typing import TypeVar

            T = TypeVar('T')


            def outer(value: T) -> T:
                def inner(v: T) -> T:
                    return v

                return inner(value)
            """)
        result = modernize_type_params(source)
        assert 'def outer[T](value: T) -> T:' in result
        assert 'def inner(v: T) -> T:' in result

    def test_leaves_non_generic_helpers_alone(self):
        source = dedent("""\
            from typing import Any, TypeVar

            T = TypeVar('T')


            def plain(value: Any) -> Any:
                return value


            def generic(value: T) -> T:
                return value
            """)
        result = modernize_type_params(source)
        assert 'def plain(value: Any) -> Any:' in result
        assert 'def generic[T](value: T) -> T:' in result

    def test_collapses_blank_run_left_by_removed_declaration(self):
        source = dedent("""\
            from typing import TypeVar

            T = TypeVar('T')


            def identity(x: T) -> T:
                return x
            """)
        assert '\n\n\n\n' not in modernize_type_params(source)


class TestModernizeBailsOut:
    """Emitting the legacy form costs a lint warning; emitting broken code
    costs the user their client.  Anything unclear is left untouched."""

    def test_module_without_typevars(self):
        source = 'def plain(x: int) -> int:\n    return x\n'
        assert modernize_type_params(source) == source

    def test_typevar_with_bound(self):
        source = dedent("""\
            from typing import TypeVar

            T = TypeVar('T', bound=int)


            def identity(x: T) -> T:
                return x
            """)
        assert modernize_type_params(source) == source

    def test_typevar_used_by_a_generic_class(self):
        source = dedent("""\
            from typing import Generic, TypeVar

            T = TypeVar('T')


            class Box(Generic[T]):
                value: T
            """)
        assert modernize_type_params(source) == source

    def test_typevar_used_by_a_module_level_alias(self):
        source = dedent("""\
            from typing import TypeVar

            T = TypeVar('T')
            Pair = tuple[T, T]


            def swap(pair: Pair) -> Pair:
                return pair[1], pair[0]
            """)
        assert modernize_type_params(source) == source

    def test_typevar_used_only_by_a_nested_function(self):
        # The outer signature is not generic, so there is nowhere to hang the
        # type parameter -- leave the TypeVar in place.
        source = dedent("""\
            from typing import TypeVar

            T = TypeVar('T')


            def outer() -> None:
                def inner(v: T) -> T:
                    return v
            """)
        assert modernize_type_params(source) == source

    def test_unparsable_source(self):
        source = 'def broken(:\n'
        assert modernize_type_params(source) == source


class TestRuntimeModulesEmission:
    """End-to-end: what actually lands in the user's package."""

    def test_default_target_keeps_typevars(self, tmp_path: Path):
        doc = _doc(pagination=PaginationConfig(enabled=True))
        assert doc.target_python == '3.10'

        pagination = Path(PaginationFeature().write(tmp_path, doc)).read_text('utf-8')
        concurrency = Path(ConcurrencyFeature().write(tmp_path, doc)).read_text('utf-8')

        assert "T = TypeVar('T')" in pagination
        assert 'def paginate_offset(' in pagination
        assert "T = TypeVar('T')" in concurrency
        assert 'def run_sync(' in concurrency

    @pytest.mark.parametrize('target', ['3.12', '3.13', '3.14'])
    def test_modern_target_uses_type_parameters(self, tmp_path: Path, target: str):
        doc = _doc(target_python=target, pagination=PaginationConfig(enabled=True))

        pagination = Path(PaginationFeature().write(tmp_path, doc)).read_text('utf-8')
        concurrency = Path(ConcurrencyFeature().write(tmp_path, doc)).read_text('utf-8')

        assert 'TypeVar' not in pagination
        assert 'TypeVar' not in concurrency
        for name in ('paginate_offset', 'iterate_cursor', 'paginate_page'):
            assert f'def {name}[T](' in pagination
            assert f'async def {name}_async[T](' in pagination
        assert 'def run_sync[T](' in concurrency
        assert 'def run_concurrently[T](' in concurrency

    @pytest.mark.parametrize('target', ['3.10', '3.11'])
    def test_older_targets_keep_typevars(self, tmp_path: Path, target: str):
        doc = _doc(target_python=target, pagination=PaginationConfig(enabled=True))
        pagination = Path(PaginationFeature().write(tmp_path, doc)).read_text('utf-8')
        assert "T = TypeVar('T')" in pagination

    def test_typevar_free_modules_are_untouched(self, tmp_path: Path):
        doc = _doc(
            target_python='3.12',
            dataframe=DataFrameConfig(enabled=True, pandas=True),
            export=ExportConfig(enabled=True),
        )
        emitted = Path(DataFrameFeature().write(tmp_path, doc)).read_text('utf-8')
        baseline = DataFrameFeature().transform_content(
            DataFrameFeature().module_content, doc
        )
        assert emitted == baseline

        export = Path(ExportFeature().write(tmp_path, doc)).read_text('utf-8')
        assert export == ExportFeature().module_content

    @pytest.mark.skipif(
        sys.version_info < (3, 12),
        reason='PEP 695 syntax needs a 3.12+ parser to check',
    )
    def test_emitted_modern_modules_parse(self, tmp_path: Path):
        doc = _doc(target_python='3.12', pagination=PaginationConfig(enabled=True))
        for feature in (PaginationFeature(), ConcurrencyFeature()):
            source = Path(feature.write(tmp_path, doc)).read_text('utf-8')
            ast.parse(source)  # raises SyntaxError on a botched rewrite


class TestExportTypeAliases:
    """``Row``/``PathLike`` stay implicit aliases: the explicit ``TypeAlias``
    annotation asks downstream linters for PEP 695's ``type`` keyword (UP040),
    which generated code cannot use while it supports 3.10/3.11."""

    def test_no_typealias_annotation(self):
        source = ExportFeature().module_content
        tree = ast.parse(source)
        imported = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            for alias in node.names
        }
        referenced = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        assert 'TypeAlias' not in imported | referenced
        assert 'Row = BaseModel | dict' in source
        assert 'PathLike = str | Path | UPath' in source
