"""Composable AST helpers for building endpoint function bodies.

The endpoint AST builders in :mod:`otterapi.codegen.endpoints` historically
hand-construct body statements with raw ``ast.Assign``, ``ast.BoolOp``, and
``_call`` chains. The same patterns -- "client init", "request + assign",
"return helper(...)" -- recur in every body builder, making bugs easy to
introduce and the source hard to read.

``BodyStatementBuilder`` exposes those patterns as named, fluent methods so
each body type becomes a sequence of intent-revealing calls. It deliberately
stays a thin wrapper over :mod:`ast` -- no DSL, no parsing -- so the output
is identical to the existing hand-rolled code (verified by the golden
fixtures in ``otterapi/tests/fixtures/golden``).
"""

from __future__ import annotations

import ast
from typing import Self

from otterapi.codegen.ast_utils import _assign, _attr, _call, _name


class BodyStatementBuilder:
    """Fluent builder for the recurring shapes that show up in endpoint bodies.

    Example
    -------
    >>> body = (
    ...     BodyStatementBuilder()
    ...     .add_docstring("Get the user.")
    ...     .add_client_init()
    ...     .add_method_call_assignment(
    ...         target_var='data',
    ...         receiver='c',
    ...         method='_request_json',
    ...         keywords=[],
    ...     )
    ...     .add_return_call('to_pandas', args=[_name('data')])
    ...     .build()
    ... )
    """

    def __init__(self) -> None:
        self._stmts: list[ast.stmt] = []

    def add_statement(self, stmt: ast.stmt) -> Self:
        """Append a fully-built statement (escape hatch)."""
        self._stmts.append(stmt)
        return self

    def extend(self, stmts: list[ast.stmt]) -> Self:
        """Append several pre-built statements (escape hatch)."""
        self._stmts.extend(stmts)
        return self

    def add_docstring(self, content: str) -> Self:
        """Append a module/function docstring as ``Expr(Constant(...))``."""
        # Local import keeps the module decoupled from the heavy
        # endpoints.py at import time.
        from otterapi.codegen.endpoints import clean_docstring

        self._stmts.append(ast.Expr(value=ast.Constant(value=clean_docstring(content))))
        return self

    def add_client_init(
        self,
        *,
        var: str = 'c',
        client_arg: str = 'client',
        client_class: str = 'Client',
    ) -> Self:
        """Emit ``<var> = <client_arg> or <client_class>()``."""
        self._stmts.append(
            _assign(
                _name(var),
                ast.BoolOp(
                    op=ast.Or(),
                    values=[_name(client_arg), _call(_name(client_class))],
                ),
            )
        )
        return self

    def add_method_call_assignment(
        self,
        *,
        target_var: str,
        receiver: str,
        method: str,
        keywords: list[ast.keyword] | None = None,
        args: list[ast.expr] | None = None,
        is_async: bool = False,
    ) -> Self:
        """Emit ``<target> = [await] <receiver>.<method>(*args, **keywords)``."""
        call: ast.expr = _call(
            func=_attr(receiver, method),
            args=args or [],
            keywords=keywords or [],
        )
        if is_async:
            call = ast.Await(value=call)
        self._stmts.append(_assign(_name(target_var), call))
        return self

    def add_return(self, value: ast.expr) -> Self:
        """Append ``return <value>``."""
        self._stmts.append(ast.Return(value=value))
        return self

    def add_return_call(
        self,
        func_name: str,
        *,
        args: list[ast.expr] | None = None,
        keywords: list[ast.keyword] | None = None,
        is_async: bool = False,
    ) -> Self:
        """Emit ``return [await] <func_name>(*args, **keywords)``."""
        call: ast.expr = _call(
            func=_name(func_name), args=args or [], keywords=keywords or []
        )
        if is_async:
            call = ast.Await(value=call)
        self._stmts.append(ast.Return(value=call))
        return self

    def build(self) -> list[ast.stmt]:
        """Return the accumulated statements (a copy, so further use is safe)."""
        return list(self._stmts)
