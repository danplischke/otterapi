"""AST utilities and import collection for code generation.

This module provides helper functions for building Python AST nodes
and utilities for collecting and organizing imports during code generation.
"""

import ast
import keyword
import sys
from collections.abc import Iterable

PYTHON_KEYWORDS = set(keyword.kwlist)

__all__ = [
    # AST helpers
    '_name',
    '_attr',
    '_subscript',
    '_union_expr',
    '_optional_expr',
    '_argument',
    '_assign',
    '_import',
    '_call',
    '_func',
    '_async_func',
    '_all',
    # Import collection
    'ImportCollector',
]


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
    kwargs: ast.arg = None,
    kwonlyargs: list[ast.arg] = None,
    kw_defaults: list[ast.expr] = None,
) -> ast.FunctionDef:
    return ast.FunctionDef(
        name=name,
        args=ast.arguments(
            posonlyargs=[],
            args=args,
            kwarg=kwargs,
            kwonlyargs=kwonlyargs or [],
            kw_defaults=kw_defaults or [],
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
    kwargs: ast.arg = None,
    kwonlyargs: list[ast.arg] = None,
    kw_defaults: list[ast.expr] = None,
) -> ast.AsyncFunctionDef:
    return ast.AsyncFunctionDef(
        name=name,
        args=ast.arguments(
            posonlyargs=[],
            args=args,
            kwarg=kwargs,
            kwonlyargs=kwonlyargs or [],
            kw_defaults=kw_defaults or [],
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

    def add_import(self, module: str, name: str) -> None:
        """Add a single import.

        Args:
            module: The module to import from (e.g., 'typing', 'pydantic').
            name: The name to import (e.g., 'List', 'BaseModel').
        """
        if module not in self._imports:
            self._imports[module] = set()
        self._imports[module].add(name)

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

    def to_ast(self, reverse_sort: bool = True) -> list[ast.ImportFrom]:
        """Convert collected imports to AST ImportFrom statements.

        Imports are sorted according to Python conventions:
        1. Standard library imports
        2. Third-party imports
        3. Local/relative imports

        Within each category, imports are sorted alphabetically by module name.
        Names within each import are also sorted alphabetically.

        Args:
            reverse_sort: Deprecated, kept for backward compatibility but ignored.

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

    def has_imports(self) -> bool:
        """Check if any imports have been collected.

        Returns:
            True if imports exist, False otherwise.
        """
        return bool(self._imports)

    def clear(self) -> None:
        """Clear all collected imports."""
        self._imports.clear()

    def get_modules(self) -> set[str]:
        """Get the set of all modules that have been imported.

        Returns:
            Set of module names.
        """
        return set(self._imports.keys())
