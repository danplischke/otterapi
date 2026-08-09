"""AST utilities and import collection for code generation.

This module provides helper functions for building Python AST nodes
and utilities for collecting and organizing imports during code generation.
"""

import ast
import sys
from collections.abc import Iterable, Sequence
from typing import cast

__all__ = [
    # AST helpers
    '_name',
    '_attr',
    '_subscript',
    '_union_expr',
    '_optional_expr',
    '_argument',
    '_ann_assign',
    '_assign',
    '_import',
    '_call',
    '_func',
    '_async_func',
    '_all',
    # Import collection
    'ImportCollector',
    # Import pruning
    'collect_referenced_names',
    'prune_unused_imports',
    # Shared constants / type aliases
    'MODELS_MODULE',
    'ImportDict',
]

# Relative import path for the generated models module — single source of truth.
MODELS_MODULE = '.models'

# Type alias for import dictionaries used throughout codegen.
ImportDict = dict[str, set[str]]


def _name(name: str) -> ast.Name:
    return ast.Name(id=name, ctx=ast.Load())


def _attr(value: str | ast.expr, attr: str) -> ast.Attribute:
    return ast.Attribute(
        value=_name(value) if isinstance(value, str) else value,
        attr=attr,
        ctx=ast.Load(),
    )


def _subscript(generic: str, inner: ast.expr) -> ast.Subscript:
    return ast.Subscript(value=_name(generic), slice=inner, ctx=ast.Load())


def _union_expr(types: list[ast.expr]) -> ast.expr:
    # A | B | C (using pipe operator instead of Union[A, B, C])
    if not types:
        raise ValueError('_union_expr requires at least one type')
    if len(types) == 1:
        return types[0]
    # Build a chain of BinOp with BitOr: A | B | C
    result = types[0]
    for t in types[1:]:
        result = ast.BinOp(left=result, op=ast.BitOr(), right=t)
    return result


def _optional_expr(inner: ast.expr) -> ast.Subscript:
    return _subscript('Optional', inner)


def _argument(name: str, value: ast.expr | None = None) -> ast.arg:
    return ast.arg(
        arg=name,
        annotation=value,
    )


def _ann_assign(
    target: ast.expr, annotation: ast.expr, value: ast.expr
) -> ast.AnnAssign:
    """Build ``target: annotation = value``."""
    if isinstance(target, ast.Name):
        target = ast.Name(id=target.id, ctx=ast.Store())
    return ast.AnnAssign(
        target=cast('ast.Name | ast.Attribute | ast.Subscript', target),
        annotation=annotation,
        value=value,
        simple=1,
    )


def _assign(target: ast.expr, value: ast.expr) -> ast.Assign:
    # Ensure target has Store context
    if isinstance(target, ast.Name):
        target = ast.Name(id=target.id, ctx=ast.Store())
    elif isinstance(target, ast.Attribute):
        # For attributes, only the outermost needs Store context
        target.ctx = ast.Store()
    return ast.Assign(
        targets=[target],
        value=value,
    )


def _import(module: str, names: list[str]) -> ast.ImportFrom:
    return ast.ImportFrom(
        module=module,
        names=[ast.alias(name=name) for name in names],
        level=0,
    )


def _call(
    func: ast.expr,
    args: list[ast.expr] | None = None,
    keywords: list[ast.keyword] | None = None,
) -> ast.Call:
    return ast.Call(
        func=func,
        args=args or [],
        keywords=keywords or [],
    )


def _func(
    name: str,
    args: list[ast.arg],
    body: list[ast.stmt],
    returns: ast.expr | None = None,
    kwargs: ast.arg | None = None,
    kwonlyargs: Sequence[ast.arg] | None = None,
    kw_defaults: Sequence[ast.expr | None] | None = None,
) -> ast.FunctionDef:
    return ast.FunctionDef(
        name=name,
        args=ast.arguments(
            posonlyargs=[],
            args=args,
            kwarg=kwargs,
            kwonlyargs=list(kwonlyargs) if kwonlyargs else [],
            kw_defaults=list(kw_defaults) if kw_defaults else [],
            defaults=[],
        ),
        body=body,
        decorator_list=[],
        returns=returns,
    )


def _async_func(
    name: str,
    args: list[ast.arg],
    body: list[ast.stmt],
    returns: ast.expr | None = None,
    kwargs: ast.arg | None = None,
    kwonlyargs: Sequence[ast.arg] | None = None,
    kw_defaults: Sequence[ast.expr | None] | None = None,
) -> ast.AsyncFunctionDef:
    return ast.AsyncFunctionDef(
        name=name,
        args=ast.arguments(
            posonlyargs=[],
            args=args,
            kwarg=kwargs,
            kwonlyargs=list(kwonlyargs) if kwonlyargs else [],
            kw_defaults=list(kw_defaults) if kw_defaults else [],
            defaults=[],
        ),
        body=body,
        decorator_list=[],
        returns=returns,
    )


def _all(names: Iterable[str]) -> ast.Assign:
    return _assign(
        target=_name('__all__'),
        value=ast.Tuple(
            elts=[ast.Constant(value=name) for name in names], ctx=ast.Load()
        ),
    )


# =============================================================================
# Import Collection
# =============================================================================


class ImportCollector:
    """Collects and manages imports for generated Python code.

    This class provides a centralized way to collect imports from various
    sources during code generation and convert them to AST import statements.
    It automatically deduplicates imports and sorts them for consistent output.

    Example:
        >>> collector = ImportCollector()
        >>> collector.add_imports({'typing': {'List', 'Dict'}})
        >>> collector.add_imports({'typing': {'Optional'}})
        >>> imports = collector.to_ast()
        >>> # Returns [ImportFrom(module='typing', names=['Dict', 'List', 'Optional'])]
    """

    def __init__(self):
        """Initialize an empty import collector."""
        self._imports: dict[str, set[str]] = {}

    def add_imports(self, imports: dict[str, set[str]]) -> None:
        """Add imports from a dictionary mapping modules to sets of names.

        Args:
            imports: Dictionary mapping module names to sets of imported names.
                    Example: {'typing': {'List', 'Dict'}, 'pydantic': {'BaseModel'}}
        """
        for module, names in imports.items():
            if module not in self._imports:
                self._imports[module] = set()
            self._imports[module].update(names)

    def _get_import_category(self, module: str) -> int:
        """Get the sort category for a module.

        Uses sys.stdlib_module_names to dynamically detect standard library modules.

        Returns:
            0 for standard library, 1 for third-party, 2 for local/relative imports.
        """
        if module.startswith('.'):
            return 2  # Local/relative imports

        # Check if it's a standard library module
        base_module = module.split('.')[0]
        if base_module in sys.stdlib_module_names:
            return 0  # Standard library

        return 1  # Third-party

    def to_ast(self) -> list[ast.ImportFrom]:
        """Convert collected imports to AST ImportFrom statements.

        Imports are sorted according to Python conventions:
        1. Standard library imports
        2. Third-party imports
        3. Local/relative imports

        Within each category, imports are sorted alphabetically by module name.
        Names within each import are also sorted alphabetically.

        Returns:
            List of ast.ImportFrom statements, properly sorted.
        """
        import_stmts = []

        # Sort by (category, module_name) to get proper ordering
        sorted_modules = sorted(
            self._imports.items(),
            key=lambda x: (self._get_import_category(x[0]), x[0]),
        )

        for module, names in sorted_modules:
            # Determine the level for relative imports
            if module.startswith('.'):
                # Count leading dots for relative import level
                level = len(module) - len(module.lstrip('.'))
                import_module = module.lstrip('.') or None
            else:
                level = 0
                import_module = module

            import_stmt = ast.ImportFrom(
                module=import_module,
                names=[ast.alias(name=name, asname=None) for name in sorted(names)],
                level=level,
            )
            import_stmts.append(import_stmt)
        return import_stmts

    def clear(self) -> None:
        """Clear all collected imports."""
        self._imports.clear()


# =============================================================================
# Import Pruning
# =============================================================================
#
# Imports are collected optimistically: a builder registers everything a
# feature *might* reference (every pagination helper, both DataFrame
# back-ends, ...) and a type contributes the imports of its whole subtree
# (``Type.copy_imports_from_sub_types``), because at collection time nobody
# knows which of those names survive into the emitted AST.  That is the right
# trade-off -- under-importing produces a NameError, over-importing only
# produces noise -- but it means the collected set is a *superset* of what the
# module actually uses.
#
# ``prune_unused_imports`` closes the gap by diffing the collected imports
# against the names the assembled module body really references, so the
# generator itself emits an exact import block (no external formatter or
# linter involved).


def _names_in_forward_ref(value: str) -> set[str]:
    """Return the names referenced by a string annotation such as ``'pd.DataFrame'``.

    Non-parsable strings (docstring fragments, media types, URL templates)
    yield an empty set.
    """
    try:
        parsed = ast.parse(value, mode='eval')
    except SyntaxError:
        return set()
    return {node.id for node in ast.walk(parsed) if isinstance(node, ast.Name)}


def _names_in_annotation(annotation: ast.expr | None) -> set[str]:
    """Return every name an annotation references, resolving string forward refs.

    Handles nested forward references such as ``list['pd.DataFrame']`` by
    walking the whole annotation expression.
    """
    if annotation is None:
        return set()

    found: set[str] = set()
    for node in ast.walk(annotation):
        if isinstance(node, ast.Name):
            found.add(node.id)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            found |= _names_in_forward_ref(node.value)
    return found


def _string_elements(node: ast.expr | None) -> set[str]:
    """Return the string constants of a tuple/list literal (used for ``__all__``)."""
    if not isinstance(node, ast.Tuple | ast.List):
        return set()
    return {
        elt.value
        for elt in node.elts
        if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
    }


class _ReferenceCollector(ast.NodeVisitor):
    """Collects every name an AST body *references* (as opposed to *binds*).

    Import statements are skipped: they bind names rather than reference them,
    so counting them would make every import justify itself.
    """

    def __init__(self) -> None:
        self.names: set[str] = set()

    # -- imports bind, they do not reference -----------------------------
    def visit_Import(self, node: ast.Import) -> None:
        return

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        return

    # -- plain references -------------------------------------------------
    def visit_Name(self, node: ast.Name) -> None:
        self.names.add(node.id)

    # -- annotations may be string forward references ---------------------
    def visit_arg(self, node: ast.arg) -> None:
        self.names |= _names_in_annotation(node.annotation)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self.names |= _names_in_annotation(node.annotation)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.names |= _names_in_annotation(node.returns)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.names |= _names_in_annotation(node.returns)
        self.generic_visit(node)

    # -- ``__all__`` re-exports names as strings --------------------------
    def visit_Assign(self, node: ast.Assign) -> None:
        if any(
            isinstance(target, ast.Name) and target.id == '__all__'
            for target in node.targets
        ):
            self.names |= _string_elements(node.value)
        self.generic_visit(node)


def collect_referenced_names(body: Sequence[ast.stmt]) -> set[str]:
    """Return every name referenced by *body*, ignoring import statements.

    A name counts as referenced when it appears as an ``ast.Name``, inside a
    string forward reference in an annotation (``-> 'pd.DataFrame'``), or as a
    string entry of an ``__all__`` tuple (a re-export is a real use).

    Args:
        body: The statements of the module being emitted.

    Returns:
        The set of referenced names.
    """
    collector = _ReferenceCollector()
    for stmt in body:
        collector.visit(stmt)
    return collector.names


def _alias_bound_name(alias: ast.alias, *, dotted: bool) -> str:
    """Return the local name an ``ast.alias`` binds.

    Args:
        alias: The alias node.
        dotted: True for ``import a.b.c`` (binds ``a``), False for
            ``from x import a.b`` -- which cannot happen -- i.e. ``from``-imports.
    """
    if alias.asname:
        return alias.asname
    return alias.name.split('.')[0] if dotted else alias.name


def _filter_import(
    stmt: ast.Import | ast.ImportFrom, used: set[str]
) -> ast.stmt | None:
    """Drop the aliases of *stmt* that bind unused names.

    Returns the trimmed statement, or None when nothing is left. ``__future__``
    imports and star imports are always kept: the first is a compiler
    directive, the second binds names this module cannot see.
    """
    if isinstance(stmt, ast.ImportFrom):
        if stmt.module == '__future__':
            return stmt
        if any(alias.name == '*' for alias in stmt.names):
            return stmt

    dotted = isinstance(stmt, ast.Import)
    kept = [
        alias for alias in stmt.names if _alias_bound_name(alias, dotted=dotted) in used
    ]
    if not kept:
        return None
    if len(kept) == len(stmt.names):
        return stmt

    if isinstance(stmt, ast.Import):
        return ast.Import(names=kept)
    return ast.ImportFrom(module=stmt.module, names=kept, level=stmt.level)


def _is_type_checking_block(stmt: ast.stmt) -> bool:
    """True for ``if TYPE_CHECKING:`` blocks, whose imports are prunable too."""
    return (
        isinstance(stmt, ast.If)
        and isinstance(stmt.test, ast.Name)
        and stmt.test.id == 'TYPE_CHECKING'
        and not stmt.orelse
    )


def _prune_once(body: list[ast.stmt], used: set[str]) -> list[ast.stmt]:
    """Run a single pruning pass over *body* given the *used* name set."""
    pruned: list[ast.stmt] = []
    for stmt in body:
        if isinstance(stmt, ast.Import | ast.ImportFrom):
            kept_stmt = _filter_import(stmt, used)
            if kept_stmt is not None:
                pruned.append(kept_stmt)
        elif _is_type_checking_block(stmt):
            block = cast('ast.If', stmt)
            inner = _prune_once(block.body, used)
            if len(inner) == len(block.body) and all(
                a is b for a, b in zip(inner, block.body, strict=True)
            ):
                pruned.append(block)  # unchanged -- keep identity for the fixed point
            elif inner:
                pruned.append(ast.If(test=block.test, body=inner, orelse=[]))
        else:
            pruned.append(stmt)
    return pruned


def prune_unused_imports(body: list[ast.stmt]) -> list[ast.stmt]:
    """Remove imports whose bound names are never referenced by *body*.

    Only module-level imports and imports nested directly inside an
    ``if TYPE_CHECKING:`` block are considered -- those are the two places
    codegen emits them.  ``from __future__ import ...`` and star imports are
    never touched.

    Pruning runs to a fixed point because removals cascade: dropping the last
    import out of a ``TYPE_CHECKING`` block removes the block, which in turn
    makes ``TYPE_CHECKING`` itself unused.

    Args:
        body: The assembled module body. Not mutated.

    Returns:
        A new list of statements with unused imports removed.
    """
    current = list(body)
    # Each iteration can only shrink the body, so this terminates; the bound
    # is a safety net rather than a real limit.
    for _ in range(len(current) + 1):
        used = collect_referenced_names(current)
        pruned = _prune_once(current, used)
        if len(pruned) == len(current) and all(
            a is b for a, b in zip(pruned, current, strict=True)
        ):
            return pruned
        current = pruned
    return current
