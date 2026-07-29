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


def annotation_includes_none(type_ast: 'ast.expr | None') -> bool:
    """True if the annotation admits ``None``.

    Recognises ``X | None`` (BinOp chain), ``Optional[X]`` and
    ``Union[..., None]`` (Subscript forms), and a bare ``None`` constant.
    """
    if type_ast is None:
        return False
    if isinstance(type_ast, ast.Constant) and type_ast.value is None:
        return True
    if isinstance(type_ast, ast.BinOp) and isinstance(type_ast.op, ast.BitOr):
        return annotation_includes_none(type_ast.left) or annotation_includes_none(
            type_ast.right
        )
    if isinstance(type_ast, ast.Subscript) and isinstance(type_ast.value, ast.Name):
        if type_ast.value.id == 'Optional':
            return True
        if type_ast.value.id == 'Union':
            elts = (
                type_ast.slice.elts
                if isinstance(type_ast.slice, ast.Tuple)
                else [type_ast.slice]
            )
            return any(annotation_includes_none(elt) for elt in elts)
    return False


def strip_optional(type_ast: ast.expr) -> ast.expr:
    """Unwrap a single ``| None`` / ``Optional[...]`` layer, else return as-is.

    ``list[X] | None`` -> ``list[X]``, ``Optional[list[X]]`` -> ``list[X]``,
    ``Union[list[X], None]`` -> ``list[X]``. A genuine multi-type union
    (``A | B``) is returned unchanged. Optional list fields are the norm -- a
    not-``required`` array property is typed ``list[X] | None`` -- so list
    detection must see through the wrapper.
    """
    # ``X | None`` (BinOp chain produced by ``_union_expr``): drop a None arm.
    if isinstance(type_ast, ast.BinOp) and isinstance(type_ast.op, ast.BitOr):
        left_none = (
            isinstance(type_ast.left, ast.Constant) and type_ast.left.value is None
        )
        right_none = (
            isinstance(type_ast.right, ast.Constant) and type_ast.right.value is None
        )
        if right_none and not left_none:
            return type_ast.left
        if left_none and not right_none:
            return type_ast.right
        return type_ast
    # ``Optional[X]`` or ``Union[X, None]`` (Subscript forms).
    if isinstance(type_ast, ast.Subscript) and isinstance(type_ast.value, ast.Name):
        if type_ast.value.id == 'Optional':
            return type_ast.slice
        if type_ast.value.id == 'Union' and isinstance(type_ast.slice, ast.Tuple):
            non_none = [
                elt
                for elt in type_ast.slice.elts
                if not (isinstance(elt, ast.Constant) and elt.value is None)
            ]
            if len(non_none) == 1:
                return non_none[0]
    return type_ast


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
