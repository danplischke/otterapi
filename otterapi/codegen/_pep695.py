"""Rewrite ``TypeVar``-generic runtime helpers into PEP 695 syntax.

The runtime helpers shipped into a generated package (``_pagination.py``,
``_concurrency.py``, ...) are written against the oldest Python OtterAPI
supports, so they declare their generics the pre-3.12 way::

    T = TypeVar('T')

    def paginate_offset(..., extract_items: Callable[[Any], list[T]]) -> list[T]:

That is the only form that runs on Python 3.10/3.11, but projects that target
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

__all__ = ['modernize_type_params']

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


def _strip_typevar_import(line: str) -> str | None:
    """Drop ``TypeVar`` from a single-line ``from typing import ...``.

    Returns the rewritten line, or ``None`` when the whole import goes away
    because ``TypeVar`` was the only name it bound.
    """
    match = re.match(r'^(\s*from\s+[\w.]+\s+import\s+)(.*)$', line)
    if match is None:
        return line
    prefix, names_part = match.groups()
    names = [name.strip() for name in names_part.split(',')]
    kept = [name for name in names if name and name != 'TypeVar']
    if not kept:
        return None
    return f'{prefix}{", ".join(kept)}'


def modernize_type_params(source: str) -> str:
    """Return *source* with module-level generic functions using PEP 695 syntax.

    Rewrites ``def f(x: T) -> T`` into ``def f[T](x: T) -> T``, then removes the
    now-redundant ``TypeVar`` declarations and imports.  Functions nested inside
    another function are left alone: they keep referring to the type parameter
    of the enclosing function, which PEP 695 scoping already makes visible.

    *source* is returned unchanged when it declares no simple module-level
    ``TypeVar``, when a ``TypeVar`` is referenced somewhere the rewrite cannot
    follow (a generic class, a type alias, a runtime call), or when it does not
    parse.
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

    for node in tree.body:
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        params = _signature_type_params(node, set(typevars))
        if not params:
            continue
        def_line = lines[node.lineno - 1]
        pattern = re.compile(rf'\bdef\s+{re.escape(node.name)}\s*\(')
        match = pattern.search(def_line)
        if match is None:
            # Unexpected layout (e.g. the signature wrapped after ``def``);
            # bail out entirely rather than emit a half-converted module.
            return source
        insert_at = match.end() - 1
        edits[node.lineno] = (
            f'{def_line[:insert_at]}[{", ".join(params)}]{def_line[insert_at:]}'
        )
        rewritten.update(params)

    if rewritten != set(typevars):
        # Some TypeVar is used somewhere other than a module-level function
        # signature -- a generic class, an alias, ``dict[str, T]`` at module
        # scope.  Converting only part of the module would break it.
        return source

    for lineno in typevars.values():
        edits[lineno] = None

    for node in tree.body:
        if not isinstance(node, ast.ImportFrom) or node.lineno != node.end_lineno:
            continue
        if not any(alias.name == 'TypeVar' for alias in node.names):
            continue
        stripped = _strip_typevar_import(lines[node.lineno - 1].rstrip('\n'))
        edits[node.lineno] = None if stripped is None else f'{stripped}\n'

    result: list[str] = []
    for lineno, line in enumerate(lines, start=1):
        text = edits.get(lineno, line)
        if text is not None:
            result.append(text)
    return _collapse_blank_runs(''.join(result))


def _collapse_blank_runs(source: str) -> str:
    """Squash the 3+ blank-line runs left behind by removed declarations."""
    return re.sub(r'\n{4,}', '\n\n\n', source)
