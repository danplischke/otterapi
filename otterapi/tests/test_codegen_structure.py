"""Structural tests for the four code-generation scenarios.

These replace the golden-file tests.  Instead of comparing exact text,
they parse the generated AST and assert the structural properties that
actually matter.  Formatting changes never break them.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import yaml

from otterapi.codegen.codegen import Codegen
from otterapi.config import DocumentConfig

FIXTURES_ROOT = Path(__file__).parent / 'fixtures' / 'golden'


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _generate(spec_dir: Path, output_dir: Path) -> dict[str, str]:
    config_path = spec_dir / 'config.yaml'
    overrides = {}
    if config_path.is_file():
        overrides = yaml.safe_load(config_path.read_text(encoding='utf-8')) or {}
    config = DocumentConfig.model_validate(
        {
            'source': str(spec_dir / 'spec.yaml'),
            'output': str(output_dir),
            'base_url': 'https://example.test',
            **overrides,
        }
    )
    Codegen(config).generate()
    return {
        p.relative_to(output_dir).as_posix(): p.read_text(encoding='utf-8')
        for p in sorted(output_dir.rglob('*.py'))
    }


def _classes(source: str) -> set[str]:
    return {
        node.name
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.ClassDef)
    }


def _functions(source: str) -> set[str]:
    return {
        node.name
        for node in ast.walk(ast.parse(source))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _all_exports(source: str) -> set[str]:
    for node in ast.walk(ast.parse(source)):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == '__all__'
            and isinstance(node.value, (ast.Tuple, ast.List))
        ):
            return {
                elt.value
                for elt in node.value.elts
                if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
            }
    return set()


def _base_names(source: str, class_name: str) -> list[str]:
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            result = []
            for base in node.bases:
                if isinstance(base, ast.Name):
                    result.append(base.id)
                elif isinstance(base, ast.Attribute):
                    result.append(base.attr)
            return result
    return []


def _annotated_fields(source: str, class_name: str) -> set[str]:
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return {
                stmt.target.id
                for stmt in node.body
                if isinstance(stmt, ast.AnnAssign)
                and isinstance(stmt.target, ast.Name)
            }
    return set()


# ---------------------------------------------------------------------------
# Per-scenario fixtures  (module-scoped so code is generated once per run)
# ---------------------------------------------------------------------------


@pytest.fixture(scope='module')
def constraints_gen(tmp_path_factory):
    return _generate(
        FIXTURES_ROOT / 'constraints',
        tmp_path_factory.mktemp('constraints'),
    )


@pytest.fixture(scope='module')
def dataframe_gen(tmp_path_factory):
    return _generate(
        FIXTURES_ROOT / 'dataframe',
        tmp_path_factory.mktemp('dataframe'),
    )


@pytest.fixture(scope='module')
def discriminator_gen(tmp_path_factory):
    return _generate(
        FIXTURES_ROOT / 'discriminator',
        tmp_path_factory.mktemp('discriminator'),
    )


@pytest.fixture(scope='module')
def paginated_gen(tmp_path_factory):
    return _generate(
        FIXTURES_ROOT / 'paginated',
        tmp_path_factory.mktemp('paginated'),
    )


# ---------------------------------------------------------------------------
# Constraints
# ---------------------------------------------------------------------------


class TestConstraintsStructure:
    def test_expected_files_emitted(self, constraints_gen):
        for name in (
            'models.py',
            'endpoints.py',
            '__init__.py',
            '_client.py',
            '_retry.py',
            '_concurrency.py',
        ):
            assert name in constraints_gen, f'missing {name}'

    def test_no_optional_modules_emitted(self, constraints_gen):
        assert '_dataframe.py' not in constraints_gen
        assert '_pagination.py' not in constraints_gen

    def test_user_class_defined(self, constraints_gen):
        assert 'User' in _classes(constraints_gen['models.py'])

    def test_user_has_correct_fields(self, constraints_gen):
        fields = _annotated_fields(constraints_gen['models.py'], 'User')
        assert {'id', 'username', 'age', 'tags'} <= fields

    def test_user_inherits_html_repr_mixin(self, constraints_gen):
        assert '_HtmlReprMixin' in _base_names(constraints_gen['models.py'], 'User')

    def test_html_repr_mixin_is_defined(self, constraints_gen):
        assert '_HtmlReprMixin' in _classes(constraints_gen['models.py'])

    def test_endpoint_functions_exist(self, constraints_gen):
        fns = _functions(constraints_gen['endpoints.py'])
        assert 'list_users' in fns
        assert 'async_list_users' in fns

    def test_init_exports_public_api(self, constraints_gen):
        exports = _all_exports(constraints_gen['__init__.py'])
        for name in ('User', 'Client', 'list_users', 'async_list_users', 'BaseAPIError'):
            assert name in exports, f'__init__.__all__ missing {name!r}'

    def test_error_hierarchy_generated(self, constraints_gen):
        classes = _classes(constraints_gen['_client.py'])
        for name in (
            'BaseAPIError',
            'ClientError',
            'ServerError',
            'NotFoundError',
            'RateLimitError',
            'InternalServerError',
        ):
            assert name in classes, f'_client.py missing {name!r}'

    def test_retry_helpers_present(self, constraints_gen):
        fns = _functions(constraints_gen['_retry.py'])
        assert '_backoff_sleep' in fns
        assert '_backoff_sleep_async' in fns

    def test_concurrency_helpers_present(self, constraints_gen):
        fns = _functions(constraints_gen['_concurrency.py'])
        assert 'run_sync' in fns
        assert 'run_concurrently' in fns
        assert 'run_concurrently_async' in fns


# ---------------------------------------------------------------------------
# DataFrame
# ---------------------------------------------------------------------------


class TestDataframeStructure:
    def test_dataframe_module_emitted(self, dataframe_gen):
        assert '_dataframe.py' in dataframe_gen

    def test_pandas_and_polars_endpoint_variants(self, dataframe_gen):
        fns = _functions(dataframe_gen['endpoints.py'])
        assert 'list_rows_df' in fns
        assert 'list_rows_pl' in fns
        assert 'async_list_rows_df' in fns
        assert 'async_list_rows_pl' in fns

    def test_df_variants_exported(self, dataframe_gen):
        exports = _all_exports(dataframe_gen['endpoints.py'])
        assert 'list_rows_df' in exports
        assert 'list_rows_pl' in exports

    def test_row_model_defined(self, dataframe_gen):
        assert 'Row' in _classes(dataframe_gen['models.py'])

    def test_dataframe_module_has_converters(self, dataframe_gen):
        fns = _functions(dataframe_gen['_dataframe.py'])
        assert 'to_pandas' in fns
        assert 'to_polars' in fns


# ---------------------------------------------------------------------------
# Discriminator
# ---------------------------------------------------------------------------


class TestDiscriminatorStructure:
    def test_concrete_models_defined(self, discriminator_gen):
        classes = _classes(discriminator_gen['models.py'])
        assert 'Dog' in classes
        assert 'Cat' in classes

    def test_concrete_models_exported(self, discriminator_gen):
        exports = _all_exports(discriminator_gen['models.py'])
        assert 'Dog' in exports
        assert 'Cat' in exports

    def test_dog_fields(self, discriminator_gen):
        fields = _annotated_fields(discriminator_gen['models.py'], 'Dog')
        assert {'kind', 'bark'} <= fields

    def test_cat_fields(self, discriminator_gen):
        fields = _annotated_fields(discriminator_gen['models.py'], 'Cat')
        assert {'kind', 'meow'} <= fields

    def test_endpoint_functions_exist(self, discriminator_gen):
        fns = _functions(discriminator_gen['endpoints.py'])
        assert 'list_animals' in fns
        assert 'async_list_animals' in fns

    def test_discriminator_field_in_return_type(self, discriminator_gen):
        # The generated endpoint uses Field(discriminator='kind') in its annotation.
        assert "discriminator='kind'" in discriminator_gen['endpoints.py']


# ---------------------------------------------------------------------------
# Paginated
# ---------------------------------------------------------------------------


class TestPaginatedStructure:
    def test_pagination_module_emitted(self, paginated_gen):
        assert '_pagination.py' in paginated_gen

    def test_all_four_endpoint_variants(self, paginated_gen):
        fns = _functions(paginated_gen['endpoints.py'])
        assert 'list_items' in fns
        assert 'async_list_items' in fns
        assert 'list_items_iter' in fns
        assert 'async_list_items_iter' in fns

    def test_all_variants_exported(self, paginated_gen):
        exports = _all_exports(paginated_gen['endpoints.py'])
        for name in (
            'list_items',
            'async_list_items',
            'list_items_iter',
            'async_list_items_iter',
        ):
            assert name in exports, f'endpoints.__all__ missing {name!r}'

    def test_offset_pagination_strategy_used(self, paginated_gen):
        src = paginated_gen['endpoints.py']
        assert 'paginate_offset' in src
        assert 'paginate_offset_async' in src

    def test_item_model_defined(self, paginated_gen):
        assert 'Item' in _classes(paginated_gen['models.py'])

    def test_page_size_from_config(self, paginated_gen):
        # config.yaml sets default_page_size: 50
        assert 'page_size: int = 50' in paginated_gen['endpoints.py']
