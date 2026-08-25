"""AST utilities and import collection for code generation.

This module provides helper functions for building Python AST nodes
and utilities for collecting and organizing imports during code generation.
"""

import ast
import builtins
import sys
from collections.abc import Iterable, Sequence
from typing import Literal, cast, overload

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
    '_class_def',
    '_function_def',
    # Import collection
    'ImportCollector',
    # Import pruning
    'collect_referenced_names',
    'collect_bound_names',
    'find_unresolved_names',
    'prune_unused_imports',
    # Import ordering
    'IMPORT_SECTION_SEPARATOR',
    'sort_import_blocks',
    # Annotation inspection
    'strip_optional',
    # Shared constants / type aliases
    'MODELS_MODULE',
    'ImportDict',
]

# Relative import path for the generated models module — single source of truth.
MODELS_MODULE = '.models'

# Type alias for import dictionaries used throughout codegen.
ImportDict = dict[str, set[str]]


def _class_def(
    name: str,
    bases: list[ast.expr],
    body: list[ast.stmt],
    keywords: list[ast.keyword] | None = None,
    decorator_list: list[ast.expr] | None = None,
) -> ast.ClassDef:
    """Build an ``ast.ClassDef`` that is valid on every supported interpreter.

    ``type_params`` became a required field of ``ClassDef`` in Python 3.12.
    Passing it unconditionally is a type error when checking against an older
    typeshed, and omitting it makes ``ast.unparse`` fail on 3.12+, so the
    version check has to be here rather than at each of the call sites.
    """
    keywords = keywords or []
    decorator_list = decorator_list or []
    if sys.version_info >= (3, 12):
        return ast.ClassDef(
            name=name,
            bases=bases,
            keywords=keywords,
            body=body,
            decorator_list=decorator_list,
            type_params=[],
        )
    return ast.ClassDef(
        name=name,
        bases=bases,
        keywords=keywords,
        body=body,
        decorator_list=decorator_list,
    )


@overload
def _function_def(
    name: str,
    args: ast.arguments,
    body: list[ast.stmt],
    decorator_list: list[ast.expr] | None = ...,
    returns: ast.expr | None = ...,
    *,
    is_async: Literal[False] = ...,
) -> ast.FunctionDef: ...


@overload
def _function_def(
    name: str,
    args: ast.arguments,
    body: list[ast.stmt],
    decorator_list: list[ast.expr] | None = ...,
    returns: ast.expr | None = ...,
    *,
    is_async: Literal[True],
) -> ast.AsyncFunctionDef: ...


@overload
def _function_def(
    name: str,
    args: ast.arguments,
    body: list[ast.stmt],
    decorator_list: list[ast.expr] | None = ...,
    returns: ast.expr | None = ...,
    *,
    is_async: bool,
) -> ast.FunctionDef | ast.AsyncFunctionDef: ...


def _function_def(
    name: str,
    args: ast.arguments,
    body: list[ast.stmt],
    decorator_list: list[ast.expr] | None = None,
    returns: ast.expr | None = None,
    *,
    is_async: bool = False,
) -> ast.FunctionDef | ast.AsyncFunctionDef:
    """Build a function definition valid on every supported interpreter.

    Same ``type_params`` story as :func:`_class_def`.  Overloaded on
    ``is_async`` so a caller passing the literal gets the exact node type back
    rather than the union.
    """
    decorator_list = decorator_list or []
    node_type = ast.AsyncFunctionDef if is_async else ast.FunctionDef
    if sys.version_info >= (3, 12):
        return node_type(  # type: ignore[no-any-return]
            name=name,
            args=args,
            body=body,
            decorator_list=decorator_list,
            returns=returns,
            type_params=[],
        )
    return node_type(  # type: ignore[no-any-return]
        name=name,
        args=args,
        body=body,
        decorator_list=decorator_list,
        returns=returns,
    )


def strip_optional(annotation: ast.expr | None) -> ast.expr | None:
    """Return *annotation* with a trailing ``| None`` removed.

    Optional model fields are annotated ``list[Thing] | None``, but the shape
    checks that drive DataFrame/export generation and item-type extraction care
    about the ``list[Thing]`` inside.  Returns the annotation unchanged when it
    is not an optional union, and None when the union holds nothing but None.
    """
    if not isinstance(annotation, ast.BinOp) or not isinstance(
        annotation.op, ast.BitOr
    ):
        return annotation

    def _is_none(node: ast.expr) -> bool:
        return isinstance(node, ast.Constant) and node.value is None

    left = None if _is_none(annotation.left) else strip_optional(annotation.left)
    right = None if _is_none(annotation.right) else strip_optional(annotation.right)

    if left is None:
        return right
    if right is None:
        return left
    # A genuine multi-member union (X | Y): nothing to strip.
    return annotation


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
    return _function_def(
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
    return _function_def(
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
        returns=returns,
        is_async=True,
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

    def rebase_relative(self, depth: int) -> None:
        """Re-point single-dot relative imports at a module *depth* levels deep.

        Sibling modules like ``.client`` and ``.models`` are registered by
        builders that do not know where the importing module will land.  A
        module emitted into a subpackage has to reach back up to the package
        root, so ``.client`` becomes ``..client`` one level down, ``...client``
        two levels down, and so on.

        Args:
            depth: The importing module's depth below the package root, where
                1 means the module sits at the root (no rewriting needed).
        """
        if depth <= 1:
            return

        prefix = '.' * depth
        for module in [
            m for m in self._imports if m.startswith('.') and not m.startswith('..')
        ]:
            names = self._imports.pop(module)
            self._imports.setdefault(prefix + module.lstrip('.'), set()).update(names)

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
# Imports are collected optimistically -- a builder registers everything a
# feature *might* reference, and a type contributes its whole subtree's
# imports -- because at collection time nobody knows which names survive into
# the emitted AST. Under-importing is a NameError, over-importing is only
# noise, so the collected set is deliberately a superset. Pruning reconciles
# it against the assembled body once that body exists.


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


def _plain_names(node: ast.expr) -> set[str]:
    """Return the ``ast.Name`` ids in *node*, without interpreting strings."""
    return {sub.id for sub in ast.walk(node) if isinstance(sub, ast.Name)}


def _subscript_base(value: ast.expr) -> str | None:
    """Return the name of a subscript base: ``Literal`` for ``t.Literal[...]``."""
    if isinstance(value, ast.Name):
        return value.id
    if isinstance(value, ast.Attribute):
        return value.attr
    return None


def _names_in_annotation(annotation: ast.expr | None) -> set[str]:
    """Return every name an annotation references, resolving string forward refs.

    A string is a forward reference only where a *type* may appear. Inside
    ``Literal[...]`` and in ``Annotated[...]`` metadata, strings are ordinary
    values -- ``Literal['pending']`` does not reference anything named
    ``pending``, and ``Field(description='Status')`` does not reference
    ``Status`` -- so those positions contribute their plain names only.
    """
    if annotation is None:
        return set()

    if isinstance(annotation, ast.Constant):
        if isinstance(annotation.value, str):
            return _names_in_forward_ref(annotation.value)
        return set()

    if isinstance(annotation, ast.Name):
        return {annotation.id}

    if isinstance(annotation, ast.Subscript):
        found = _names_in_annotation(annotation.value)
        base = _subscript_base(annotation.value)
        if base == 'Literal':
            return found | _plain_names(annotation.slice)
        if base == 'Annotated':
            elts = (
                annotation.slice.elts
                if isinstance(annotation.slice, ast.Tuple)
                else [annotation.slice]
            )
            if elts:
                found |= _names_in_annotation(elts[0])
            for metadata in elts[1:]:
                found |= _plain_names(metadata)
            return found
        return found | _names_in_annotation(annotation.slice)

    if isinstance(annotation, ast.Tuple | ast.List):
        found = set()
        for elt in annotation.elts:
            found |= _names_in_annotation(elt)
        return found

    if isinstance(annotation, ast.BinOp):
        return _names_in_annotation(annotation.left) | _names_in_annotation(
            annotation.right
        )

    return _plain_names(annotation)


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

    def visit_Import(self, node: ast.Import) -> None:
        return

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        return

    def visit_Name(self, node: ast.Name) -> None:
        self.names.add(node.id)

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


def _assignment_target_names(target: ast.expr | None) -> set[str]:
    """Return the names an assignment target binds.

    Recurses through tuple/list/starred unpacking. Attribute and subscript
    targets bind nothing new, so they contribute nothing.
    """
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, ast.Starred):
        return _assignment_target_names(target.value)
    if isinstance(target, ast.Tuple | ast.List):
        names: set[str] = set()
        for elt in target.elts:
            names |= _assignment_target_names(elt)
        return names
    return set()


def collect_bound_names(body: Sequence[ast.stmt]) -> set[str]:
    """Return every name *body* binds, at any scope.

    Bindings are found structurally rather than by ``ast.Name.ctx``: codegen
    builds nodes by hand and ``ast.unparse`` ignores ``ctx``, so an assignment
    target may well carry ``Load``. Trusting ``ctx`` here would report
    perfectly good code as undefined.

    Deliberately scope-blind: a local in one function counts as bound for the
    whole module. That over-approximates, which is the safe direction for
    ``find_unresolved_names`` -- it can let a genuine miss hide behind a
    same-named local, but it can never flag correct code.

    Args:
        body: The statements of the module being emitted.

    Returns:
        The set of bound names.
    """
    bound: set[str] = set()

    for stmt in body:
        for node in ast.walk(stmt):
            bound |= _names_bound_by_target(node)
            bound |= _names_bound_by_declaration(node)

    return bound


def _argument_names(args: ast.arguments) -> set[str]:
    """Every parameter name in an argument list, including *args / **kwargs."""
    names = {a.arg for a in (*args.posonlyargs, *args.args, *args.kwonlyargs)}
    names.update(a.arg for a in (args.vararg, args.kwarg) if a)
    return names


def _names_bound_by_target(node: ast.AST) -> set[str]:
    """Names bound by an assignment-like target on *node*.

    Covers assignments, augmented and walrus assignments, loop and
    comprehension targets, ``with ... as``, and bare ``Store``/``Del`` names.
    """
    if isinstance(node, ast.Assign):
        names: set[str] = set()
        for target in node.targets:
            names |= _assignment_target_names(target)
        return names
    if isinstance(node, ast.AnnAssign | ast.AugAssign | ast.NamedExpr):
        return _assignment_target_names(node.target)
    if isinstance(node, ast.For | ast.AsyncFor | ast.comprehension):
        return _assignment_target_names(node.target)
    if isinstance(node, ast.withitem):
        return _assignment_target_names(node.optional_vars)
    if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store | ast.Del):
        return {node.id}
    return set()


def _names_bound_by_declaration(node: ast.AST) -> set[str]:
    """Names bound by a declaration on *node*.

    Covers def/class names and their parameters, imports, ``except ... as``,
    ``global``/``nonlocal``, and match-statement captures.
    """
    if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
        return {node.name} | _argument_names(node.args)
    if isinstance(node, ast.Lambda):
        return _argument_names(node.args)
    if isinstance(node, ast.ClassDef):
        return {node.name}
    if isinstance(node, ast.Import | ast.ImportFrom):
        dotted = isinstance(node, ast.Import)
        return {
            _alias_bound_name(alias, dotted=dotted)
            for alias in node.names
            if alias.name != '*'
        }
    if isinstance(node, ast.ExceptHandler) and node.name:
        return {node.name}
    if isinstance(node, ast.Global | ast.Nonlocal):
        return set(node.names)
    if isinstance(node, ast.MatchAs | ast.MatchStar) and node.name:
        return {node.name}
    if isinstance(node, ast.MatchMapping) and node.rest:
        return {node.rest}
    return set()


def _has_star_import(body: Sequence[ast.stmt]) -> bool:
    """True if *body* contains ``from x import *``, which binds unknowable names."""
    return any(
        isinstance(stmt, ast.ImportFrom)
        and any(alias.name == '*' for alias in stmt.names)
        for stmt in body
    )


def find_unresolved_names(body: Sequence[ast.stmt]) -> set[str]:
    """Return names *body* references but never binds -- i.e. missing imports.

    This is the counterpart to pruning: pruning only ever removes, so it cannot
    notice that a builder forgot to register an import. That failure mode is
    the damaging one -- it reaches the user as a ``NameError`` in their client
    rather than as noise -- so it is worth failing generation over.

    Returns an empty set for modules containing a star import, whose bindings
    this module cannot see.

    Args:
        body: The statements of the module being emitted.

    Returns:
        The set of unresolved names, empty when everything resolves.
    """
    if _has_star_import(body):
        return set()

    referenced = collect_referenced_names(body)
    bound = collect_bound_names(body)
    return referenced - bound - set(dir(builtins))


def _alias_bound_name(alias: ast.alias, *, dotted: bool) -> str:
    """Return the local name an ``ast.alias`` binds.

    Args:
        alias: The alias node.
        dotted: True for plain ``import a.b.c``, which binds only ``a``.
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
    # Every pass either shrinks the body or stops, so the bound is a backstop.
    for _ in range(len(current) + 1):
        used = collect_referenced_names(current)
        pruned = _prune_once(current, used)
        if len(pruned) == len(current) and all(
            a is b for a, b in zip(pruned, current, strict=True)
        ):
            return pruned
        current = pruned
    return current


# ---------------------------------------------------------------------------
# Import ordering
# ---------------------------------------------------------------------------

# Emitted between two import sections and turned into a blank line once the
# module has been unparsed.  ``ast.unparse`` writes no blank lines at all, and
# isort-style sections have to be separated by one, so the separator has to
# survive as a statement until the source exists.
IMPORT_SECTION_SEPARATOR = '__otterapi_import_section__'

# isort's section order: __future__, standard library, third party, first
# party, then local (relative) imports.  Codegen emits nothing first-party --
# a generated package refers to its own modules relatively -- so that section
# stays empty.
_SECTION_FUTURE = 0
_SECTION_STDLIB = 1
_SECTION_THIRD_PARTY = 2
_SECTION_LOCAL = 4


def _module_root(stmt: ast.Import | ast.ImportFrom) -> str:
    """The top-level package an import statement pulls from."""
    if isinstance(stmt, ast.ImportFrom):
        return (stmt.module or '').split('.')[0]
    return stmt.names[0].name.split('.')[0]


def _import_section(stmt: ast.Import | ast.ImportFrom) -> int:
    """The isort section *stmt* belongs to."""
    if isinstance(stmt, ast.ImportFrom):
        if stmt.level:
            return _SECTION_LOCAL
        if stmt.module == '__future__':
            return _SECTION_FUTURE
    root = _module_root(stmt)
    if root in sys.stdlib_module_names:
        return _SECTION_STDLIB
    return _SECTION_THIRD_PARTY


def _member_rank(name: str) -> int:
    """Rank for isort's ``order_by_type``: constants, classes, then the rest."""
    stripped = name.lstrip('_')
    if not stripped:
        return 2
    if stripped.isupper():
        return 0
    if stripped[0].isupper():
        return 1
    return 2


def _sorted_aliases(aliases: list[ast.alias]) -> list[ast.alias]:
    """Order the names inside one ``from x import a, b`` the way isort does."""
    return sorted(
        aliases, key=lambda alias: (_member_rank(alias.name), alias.name.lower())
    )


def _import_sort_key(stmt: ast.Import | ast.ImportFrom) -> tuple:
    """Sort key placing plain ``import x`` before ``from x import y``."""
    if isinstance(stmt, ast.ImportFrom):
        # Relative imports run furthest-to-closest, so a deeper level sorts
        # first; ``-level`` gets that without a second key.
        return (1, -stmt.level, (stmt.module or '').lower())
    return (0, 0, stmt.names[0].name.lower())


def _sorted_import_run(run: list[ast.stmt]) -> list[ast.stmt]:
    """Sort one contiguous run of imports, separators included."""
    sections: dict[int, list[ast.stmt]] = {}
    for stmt in run:
        imp = cast('ast.Import | ast.ImportFrom', stmt)
        if isinstance(imp, ast.ImportFrom):
            imp = ast.ImportFrom(
                module=imp.module, names=_sorted_aliases(imp.names), level=imp.level
            )
        else:
            imp = ast.Import(names=sorted(imp.names, key=lambda a: a.name.lower()))
        sections.setdefault(_import_section(imp), []).append(imp)

    ordered: list[ast.stmt] = []
    for index, section in enumerate(sorted(sections)):
        if index:
            ordered.append(ast.Expr(value=ast.Constant(value=IMPORT_SECTION_SEPARATOR)))
        ordered.extend(
            sorted(
                sections[section],
                key=lambda s: _import_sort_key(cast('ast.Import | ast.ImportFrom', s)),
            )
        )
    return ordered


def _separator() -> ast.stmt:
    """One blank line, as a statement that survives until unparsing."""
    return ast.Expr(value=ast.Constant(value=IMPORT_SECTION_SEPARATOR))


def _separators_after_imports(next_stmt: ast.stmt) -> int:
    """Separators to emit between a module's imports and what follows.

    isort wants one blank line before a plain statement and two before a
    definition.  ``ast.unparse`` already writes one of its own ahead of a
    class or function, so a single separator covers both cases.
    """
    del next_stmt
    return 1


def sort_import_blocks(
    body: list[ast.stmt], *, top_level: bool = True
) -> list[ast.stmt]:
    """Order every run of imports in *body* the way isort would.

    Codegen collects imports as it discovers the names that need them, so a
    generated module's import block comes out in discovery order -- which
    ruff reports as ``I001`` in the user's own tree, where the generated
    package is linted.  Sorting at emission makes the output clean without
    the user's linter having to fix it, and without otterapi depending on a
    formatter being installed.

    Runs are sorted in place: statements around them, and the relative order
    of anything that is not an import, are left alone.  ``if TYPE_CHECKING:``
    blocks are recursed into, being the other place codegen emits imports.

    Args:
        body: The module body about to be written. Not mutated.
        top_level: Whether *body* is the module body itself, which is the only
            place isort wants blank lines after the imports.

    Returns:
        A new list of statements with each import run ordered, and
        ``IMPORT_SECTION_SEPARATOR`` markers where blank lines belong.
    """
    ordered: list[ast.stmt] = []
    run: list[ast.stmt] = []

    def flush(next_stmt: ast.stmt | None = None) -> None:
        if not run:
            return
        ordered.extend(_sorted_import_run(run))
        run.clear()
        # Without a formatter installed the emitted source is whatever
        # ``ast.unparse`` produced, so the blank lines after the block have to
        # be emitted too -- isort counts them as part of the import block.
        if top_level and next_stmt is not None:
            ordered.extend(
                _separator() for _ in range(_separators_after_imports(next_stmt))
            )

    for stmt in body:
        if isinstance(stmt, ast.Import | ast.ImportFrom):
            run.append(stmt)
            continue
        flush(stmt)
        if _is_type_checking_block(stmt):
            block = cast('ast.If', stmt)
            ordered.append(
                ast.If(
                    test=block.test,
                    body=sort_import_blocks(block.body, top_level=False),
                    orelse=block.orelse,
                )
            )
        else:
            ordered.append(stmt)
    flush()
    return ordered
