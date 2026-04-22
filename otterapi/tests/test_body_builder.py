"""Tests for the BodyStatementBuilder used by endpoint AST builders."""

from __future__ import annotations

import ast

from otterapi.codegen._body_builder import BodyStatementBuilder
from otterapi.codegen.ast_utils import _name


def _unparse(stmts: list[ast.stmt]) -> str:
    module = ast.Module(body=stmts, type_ignores=[])
    return ast.unparse(ast.fix_missing_locations(module))


class TestSinglePrimitives:
    def test_docstring(self):
        stmts = BodyStatementBuilder().add_docstring('Get a user.').build()
        src = _unparse(stmts)
        assert 'Get a user.' in src
        # Should be a top-level expression statement (the standard
        # "first statement is a docstring" Python idiom).
        assert isinstance(stmts[0], ast.Expr)
        assert isinstance(stmts[0].value, ast.Constant)

    def test_client_init_default_names(self):
        stmts = BodyStatementBuilder().add_client_init().build()
        assert _unparse(stmts).strip() == 'c = client or Client()'

    def test_client_init_custom_names(self):
        stmts = (
            BodyStatementBuilder()
            .add_client_init(var='handle', client_arg='cli', client_class='AsyncClient')
            .build()
        )
        assert _unparse(stmts).strip() == 'handle = cli or AsyncClient()'

    def test_method_call_assignment_sync(self):
        stmts = (
            BodyStatementBuilder()
            .add_method_call_assignment(
                target_var='data',
                receiver='c',
                method='_request_json',
                keywords=[ast.keyword(arg='method', value=ast.Constant(value='get'))],
            )
            .build()
        )
        assert _unparse(stmts).strip() == "data = c._request_json(method='get')"

    def test_method_call_assignment_async_wraps_in_await(self):
        stmts = (
            BodyStatementBuilder()
            .add_method_call_assignment(
                target_var='data',
                receiver='c',
                method='_request_json_async',
                is_async=True,
            )
            .build()
        )
        assert _unparse(stmts).strip() == 'data = await c._request_json_async()'

    def test_return_simple(self):
        stmts = BodyStatementBuilder().add_return(_name('data')).build()
        assert _unparse(stmts).strip() == 'return data'

    def test_return_call(self):
        stmts = (
            BodyStatementBuilder()
            .add_return_call('to_pandas', args=[_name('data')])
            .build()
        )
        assert _unparse(stmts).strip() == 'return to_pandas(data)'

    def test_return_call_async(self):
        stmts = (
            BodyStatementBuilder()
            .add_return_call('export_async', args=[_name('rows')], is_async=True)
            .build()
        )
        assert _unparse(stmts).strip() == 'return await export_async(rows)'


class TestComposition:
    def test_full_dataframe_body_shape(self):
        """Mirrors the standalone dataframe body produced by _build_dataframe_body."""
        stmts = (
            BodyStatementBuilder()
            .add_docstring('List users.\n\nReturns:\n    pd.DataFrame')
            .add_client_init()
            .add_method_call_assignment(
                target_var='data',
                receiver='c',
                method='_request_json',
                keywords=[],
            )
            .add_return_call(
                'to_pandas',
                args=[_name('data')],
                keywords=[ast.keyword(arg='path', value=_name('path'))],
            )
            .build()
        )
        src = _unparse(stmts)
        assert 'c = client or Client()' in src
        assert 'data = c._request_json()' in src
        assert 'return to_pandas(data, path=path)' in src

    def test_build_returns_independent_copy(self):
        builder = BodyStatementBuilder().add_return(_name('x'))
        first = builder.build()
        second = builder.build()
        assert first is not second
        assert first == second

    def test_extend_with_external_stmts(self):
        extra = ast.parse('y = 1').body
        stmts = (
            BodyStatementBuilder()
            .add_client_init()
            .extend(extra)
            .add_return(_name('y'))
            .build()
        )
        src = _unparse(stmts)
        assert 'c = client or Client()' in src
        assert 'y = 1' in src
        assert 'return y' in src
