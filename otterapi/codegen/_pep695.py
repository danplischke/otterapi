"""Rewrite ``TypeVar``-generic runtime helpers into PEP 695 syntax.

The runtime helpers shipped into a generated package (``_pagination.py``,
``_concurrency.py``, ...) are written against the oldest Python OtterAPI
supports, so they declare their generics the pre-3.12 way::

    T = TypeVar('T')

    def paginate_offset(..., extract_items: Callable[[Any], list[T]]) -> list[T]:

That is the only form that runs on Python 3.11, but projects that target
3.12+ lint it as legacy (ruff's UP047, ``non-pep695-generic-function``).  When
a document sets ``target_python: "3.12"`` (or newer) the same helpers are
emitted with type parameters instead::

    def paginate_offset[T](..., extract_items: Callable[[Any], list[T]]) -> list[T]:

The rewrite is textual on purpose: the templates are copied verbatim into the
user's package, and round-tripping them through ``ast.unparse`` would throw
away their comments, docstring formatting, and blank lines.  ``ast`` is still
what *locates* the edits, so the transform never has to guess where a
signature starts.  Anything it does not fully understand leaves the source
untouched -- emitting the legacy form costs a lint warning, emitting broken
code costs the user their client.
"""

from __future__ import annotations

import ast
import re

__all__ = ['modernize_type_aliases', 'modernize_type_params']

# The Python version that introduced PEP 695 type parameter syntax.
PEP695_MIN_VERSION = (3, 12)


def _module_level_typevars(tree: ast.Module) -> dict[str, int]:
    """Map ``T -> lineno`` for every plain module-level ``T = TypeVar('T')``.

    Declarations carrying a bound, constraints, or variance are not returned:
    they translate to PEP 695 differently enough that the caller is better off
    leaving the module alone.
    """
    typevars: dict[str, int] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        value = node.value
        if not isinstance(target, ast.Name) or not isinstance(value, ast.Call):
            continue
        if not isinstance(value.func, ast.Name) or value.func.id != 'TypeVar':
            continue
        if value.keywords or len(value.args) != 1:
            return {}
        arg = value.args[0]
        if not isinstance(arg, ast.Constant) or arg.value != target.id:
            return {}
        typevars[target.id] = node.lineno
    return typevars


def _names_in(*nodes: ast.AST | None) -> list[str]:
    """Every ``Name`` id referenced by *nodes*, in source order.

    ``ast.walk`` is breadth-first, so ``dict[U, T] | T`` would otherwise report
    the trailing ``T`` before ``U``.  Sorting by position restores the order a
    reader sees, which is the order the type parameters get declared in.
    """
    found: list[tuple[int, int, str]] = []
    for node in nodes:
        if node is None:
            continue
        for child in ast.walk(node):
            if isinstance(child, ast.Name):
                found.append((child.lineno, child.col_offset, child.id))
    return [name for *_, name in sorted(found)]


def _signature_type_params(
    func: ast.FunctionDef | ast.AsyncFunctionDef, typevars: set[str]
) -> list[str]:
    """Type parameters used by *func*'s signature, in order of first use."""
    args = func.args
    annotations = [
        arg.annotation
        for arg in (
            *args.posonlyargs,
            *args.args,
            *args.kwonlyargs,
            *([args.vararg] if args.vararg else []),
            *([args.kwarg] if args.kwarg else []),
        )
    ]
    ordered: list[str] = []
    for name in _names_in(*annotations, func.returns):
        if name in typevars and name not in ordered:
            ordered.append(name)
    return ordered


def _strip_typevar_import(
    line: str, drop: frozenset[str] = frozenset({'TypeVar'})
) -> str | None:
    """Drop ``TypeVar`` (and any other ``drop`` names) from a one-line import.

    Returns the rewritten line, or ``None`` when the whole import goes away
    because it bound nothing else.
    """
    match = re.match(r'^(\s*from\s+[\w.]+\s+import\s+)(.*)$', line)
    if match is None:
        return line
    prefix, names_part = match.groups()
    names = [name.strip() for name in names_part.split(',')]
    kept = [name for name in names if name and name not in drop]
    if not kept:
        return None
    return f'{prefix}{", ".join(kept)}'


def _generic_class_params(node: ast.ClassDef, typevars: set[str]) -> list[str] | None:
    """``[T, E]`` for ``class X(Generic[T, E])``; ``[]`` if not generic; None to bail.

    Only the plain form is understood: ``Generic[...]`` as the one and only
    base, every argument a module-level ``TypeVar``. Anything else (extra
    bases, keywords, ``Generic`` mixed with ``Protocol``) bails out.
    """
    generic = [
        base
        for base in node.bases
        if isinstance(base, ast.Subscript)
        and isinstance(base.value, ast.Name)
        and base.value.id == 'Generic'
    ]
    if not generic:
        # A non-generic class naming a TypeVar anywhere (a base, a method
        # annotation) would lose its declaration: leave the module alone.
        uses = {
            child.id
            for child in ast.walk(node)
            if isinstance(child, ast.Name) and child.id in typevars
        }
        return None if uses else []
    if len(node.bases) != 1 or node.keywords:
        return None
    slice_ = generic[0].slice
    elts = slice_.elts if isinstance(slice_, ast.Tuple) else [slice_]
    params: list[str] = []
    for elt in elts:
        if not isinstance(elt, ast.Name) or elt.id not in typevars:
            return None
        params.append(elt.id)
    return params


class _Bail(Exception):
    """Raised by the per-node rewriters: leave the whole module untouched."""


def _class_edit(
    node: ast.ClassDef, line: str, typevars: set[str]
) -> tuple[str, list[str]] | None:
    """``(new header line, params)`` for a generic class; None if not generic."""
    params = _generic_class_params(node, typevars)
    if params is None:
        raise _Bail
    if not params:
        return None
    pattern = re.compile(
        rf'\bclass\s+{re.escape(node.name)}\s*\(\s*Generic\[[^\]]*\]\s*\)'
    )
    match = pattern.search(line)
    if match is None:
        raise _Bail
    header = f'class {node.name}[{", ".join(params)}]'
    return f'{line[: match.start()]}{header}{line[match.end() :]}', params


def _function_edit(
    node: ast.FunctionDef | ast.AsyncFunctionDef, line: str, typevars: set[str]
) -> tuple[str, list[str]] | None:
    """``(new def line, params)`` for a generic function; None if not generic."""
    params = _signature_type_params(node, typevars)
    if not params:
        return None
    match = re.compile(rf'\bdef\s+{re.escape(node.name)}\s*\(').search(line)
    if match is None:
        # Unexpected layout (e.g. the signature wrapped after ``def``);
        # bail out entirely rather than emit a half-converted module.
        raise _Bail
    insert_at = match.end() - 1
    return f'{line[:insert_at]}[{", ".join(params)}]{line[insert_at:]}', params


def _import_edits(
    tree: ast.Module, lines: list[str], drop: frozenset[str]
) -> dict[int, str | None]:
    """Line edits removing ``drop`` from every one-line ``from ... import``."""
    edits: dict[int, str | None] = {}
    for node in tree.body:
        if not isinstance(node, ast.ImportFrom) or node.lineno != node.end_lineno:
            continue
        if not any(alias.name in drop for alias in node.names):
            continue
        stripped = _strip_typevar_import(lines[node.lineno - 1].rstrip('\n'), drop)
        edits[node.lineno] = None if stripped is None else f'{stripped}\n'
    return edits


def _generic_still_used(tree: ast.Module) -> bool:
    """Whether ``Generic`` is referenced outside class headers and imports."""
    return any(
        isinstance(n, ast.Name) and n.id == 'Generic'
        for stmt in tree.body
        if not isinstance(stmt, ast.ClassDef | ast.ImportFrom)
        for n in ast.walk(stmt)
    )


def modernize_type_params(source: str) -> str:
    """Return *source* with module-level generics using PEP 695 syntax.

    Rewrites ``def f(x: T) -> T`` into ``def f[T](x: T) -> T`` and
    ``class Box(Generic[T])`` into ``class Box[T]``, then removes the
    now-redundant ``TypeVar`` declarations and imports (and ``Generic``, once
    nothing subclasses it).  Functions nested inside another function, and
    methods, are left alone: they keep referring to the type parameter of the
    enclosing scope, which PEP 695 scoping already makes visible.

    *source* is returned unchanged when it declares no simple module-level
    ``TypeVar``, when a ``TypeVar`` is referenced somewhere the rewrite cannot
    follow (a class with other bases, a non-generic class, a type alias, a
    runtime call), or when it does not parse.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source

    typevars = _module_level_typevars(tree)
    if not typevars:
        return source

    lines = source.splitlines(keepends=True)
    # 1-indexed line -> replacement text (None deletes the line).
    edits: dict[int, str | None] = {}
    rewritten: set[str] = set()
    rewrote_generic_class = False

    try:
        for node in tree.body:
            edit: tuple[str, list[str]] | None = None
            if isinstance(node, ast.ClassDef):
                edit = _class_edit(node, lines[node.lineno - 1], set(typevars))
                rewrote_generic_class |= edit is not None
            elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                edit = _function_edit(node, lines[node.lineno - 1], set(typevars))
            if edit is not None:
                edits[node.lineno] = edit[0]
                rewritten.update(edit[1])
    except _Bail:
        return source

    if rewritten != set(typevars):
        # Some TypeVar is used somewhere other than a module-level signature
        # or generic class -- an alias, ``dict[str, T]`` at module scope.
        # Converting only part of the module would break it.
        return source

    for lineno in typevars.values():
        edits[lineno] = None

    drop = {'TypeVar'}
    if rewrote_generic_class and not _generic_still_used(tree):
        drop.add('Generic')
    edits.update(_import_edits(tree, lines, frozenset(drop)))

    result: list[str] = []
    for lineno, line in enumerate(lines, start=1):
        text = edits.get(lineno, line)
        if text is not None:
            result.append(text)
    return _collapse_blank_runs(''.join(result))


def _collapse_blank_runs(source: str) -> str:
    """Squash the 3+ blank-line runs left behind by removed declarations."""
    return re.sub(r'\n{4,}', '\n\n\n', source)


#: Line budget for a regenerated import, matching ruff's default.
_MAX_LINE = 88


def _render_import_without(node: ast.ImportFrom, name: str) -> str | None:
    """Re-render *node* with *name* removed, or None if nothing is left.

    Rebuilding from the AST rather than editing text handles the parenthesized
    multi-line form, which is how a typing import of more than a few names is
    actually written.
    """
    kept = [alias for alias in node.names if alias.name != name]
    if not kept:
        return None

    module = '.' * node.level + (node.module or '')
    rendered = [
        alias.name if alias.asname is None else f'{alias.name} as {alias.asname}'
        for alias in kept
    ]
    single = f'from {module} import {", ".join(rendered)}'
    if len(single) <= _MAX_LINE:
        return single + '\n'

    body = ''.join(f'    {entry},\n' for entry in rendered)
    return f'from {module} import (\n{body})\n'


def modernize_type_aliases(source: str) -> str:
    """Return *source* with ``X: TypeAlias = Y`` rewritten as ``type X = Y``.

    The explicit ``TypeAlias`` annotation is what makes these usable as types
    at all -- without it a checker reads them as plain module variables and
    every signature mentioning them fails.  But on 3.12+ ruff's UP040
    (``non-pep695-type-alias``) wants the ``type`` keyword instead, so a
    project linting its generated client at 3.12 would see a warning per
    alias.  This is the same trade-off ``modernize_type_params`` makes for
    generic functions, applied to aliases.

    Single-line aliases only.  A multi-line one is left alone rather than
    risk emitting a broken module -- a lint warning is cheaper than a client
    that does not import.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source

    lines = source.splitlines(keepends=True)
    edits: dict[int, str | None] = {}
    found = False

    for node in tree.body:
        if not isinstance(node, ast.AnnAssign) or node.value is None:
            continue
        if not isinstance(node.target, ast.Name):
            continue
        annotation = node.annotation
        is_type_alias = (
            isinstance(annotation, ast.Name) and annotation.id == 'TypeAlias'
        ) or (isinstance(annotation, ast.Constant) and annotation.value == 'TypeAlias')
        if not is_type_alias or node.lineno != node.end_lineno:
            continue
        line = lines[node.lineno - 1]
        indent = line[: len(line) - len(line.lstrip())]
        value = ast.unparse(node.value)
        edits[node.lineno] = f'{indent}type {node.target.id} = {value}\n'
        found = True

    if not found:
        return source

    for node in tree.body:
        if not isinstance(node, ast.ImportFrom):
            continue
        if not any(alias.name == 'TypeAlias' for alias in node.names):
            continue
        edits[node.lineno] = _render_import_without(node, 'TypeAlias')
        # Drop the continuation lines of a parenthesized import.
        for lineno in range(node.lineno + 1, (node.end_lineno or node.lineno) + 1):
            edits[lineno] = None

    result: list[str] = []
    for lineno, line in enumerate(lines, start=1):
        text = edits.get(lineno, line)
        if text is not None:
            result.append(text)
    return _collapse_blank_runs(''.join(result))
