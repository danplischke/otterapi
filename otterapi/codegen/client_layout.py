"""Emit ``Client`` / ``AsyncClient`` classes for ``client_style: 'client'``.

The default (``functions``) layout exposes a standalone function per endpoint and
variant (``get_user``, ``async_get_user``, ``get_user_df`` ...). This module adds
an alternative *public surface*: two client classes whose methods delegate to
those same functions, passing ``client=self``. Sync functions become ``Client``
methods; async functions (``async_*``) become ``AsyncClient`` methods with the
``async_`` prefix stripped, so method names read cleanly::

    client = Client(base_url=...)
    user = client.get_user(user_id=1)            # -> get_user(user_id=1, client=self)

    async with AsyncClient(base_url=...) as api:
        user = await api.get_user(user_id=1)     # -> async_get_user(..., client=self)

Each method is produced by AST-transforming an already-generated function def, so
every variant (core / dataframe / paginated / iterator / export) is wrapped
uniformly with its exact typed signature -- no signature is re-derived. The free
functions remain the implementation the methods call, in ``endpoints.py``.
"""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

from otterapi.codegen.ast_utils import ImportCollector, _all, _attr, _call, _name

if TYPE_CHECKING:
    from otterapi.codegen.emit import TypeResolver

ASYNC_PREFIX = 'async_'

# Non-model annotation vocabulary the wrapped signatures can reference, mapped to
# the import that provides it. ``from __future__ import annotations`` makes these
# lazy, so a miss never breaks import at runtime -- the map keeps generated code
# clean and type-checkable.
_NAME_IMPORTS: dict[str, tuple[str, str]] = {
    'Response': ('httpx', 'Response'),
    'Iterator': ('collections.abc', 'Iterator'),
    'AsyncIterator': ('collections.abc', 'AsyncIterator'),
    'Any': ('typing', 'Any'),
    'Literal': ('typing', 'Literal'),
    'Path': ('pathlib', 'Path'),
    'UPath': ('upath', 'UPath'),
}


def _own_scope_yields(node: ast.AST) -> bool:
    """True if *node*'s own scope contains a ``yield`` (i.e. it is a generator).

    Nested function / lambda scopes are skipped: a ``paginate`` helper whose
    inner ``_fetch_page`` returns must not be mistaken for a generator.
    """
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue
        if isinstance(child, (ast.Yield, ast.YieldFrom)):
            return True
        if _own_scope_yields(child):
            return True
    return False


def _docstring_stmt(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> ast.stmt | None:
    """Return the function's docstring statement, if it has one."""
    if (
        fn.body
        and isinstance(fn.body[0], ast.Expr)
        and isinstance(fn.body[0].value, ast.Constant)
        and isinstance(fn.body[0].value.value, str)
    ):
        return fn.body[0]
    return None


def _method_from_function(
    fn: ast.FunctionDef | ast.AsyncFunctionDef, endpoints_module_alias: str
) -> ast.FunctionDef | ast.AsyncFunctionDef:
    """Transform a generated free function into a delegating client method.

    Drops the ``client`` keyword arg, prepends ``self``, and forwards every
    remaining argument to the original function with ``client=self``.
    """
    is_async_def = isinstance(fn, ast.AsyncFunctionDef)
    is_generator = _own_scope_yields(fn)
    src = fn.args

    method_name = fn.name
    if is_async_def and method_name.startswith(ASYNC_PREFIX):
        method_name = method_name[len(ASYNC_PREFIX) :]

    # Keyword-only args minus ``client`` (kept in lockstep with its default).
    kwonlyargs: list[ast.arg] = []
    kw_defaults: list[ast.expr | None] = []
    for arg, default in zip(src.kwonlyargs, src.kw_defaults):
        if arg.arg == 'client':
            continue
        kwonlyargs.append(arg)
        kw_defaults.append(default)

    new_args = ast.arguments(
        posonlyargs=list(src.posonlyargs),
        args=[ast.arg(arg='self', annotation=None), *src.args],
        vararg=src.vararg,
        kwonlyargs=kwonlyargs,
        kw_defaults=kw_defaults,
        kwarg=src.kwarg,
        defaults=list(src.defaults),
    )

    # Forward positionals positionally, remaining kwonly by keyword, plus
    # client=self, plus any **kwargs.
    call_args = [_name(a.arg) for a in (*src.posonlyargs, *src.args)]
    call_keywords = [ast.keyword(arg=a.arg, value=_name(a.arg)) for a in kwonlyargs]
    call_keywords.append(ast.keyword(arg='client', value=_name('self')))
    if src.kwarg is not None:
        call_keywords.append(ast.keyword(arg=None, value=_name(src.kwarg.arg)))

    call: ast.expr = _call(
        func=_attr(endpoints_module_alias, fn.name),
        args=call_args,
        keywords=call_keywords,
    )

    # A generator wrapper must return the (async) generator object, so it stays a
    # plain ``def`` and is never awaited. A coroutine is awaited; a plain
    # function is returned directly.
    if is_generator:
        return_stmt: ast.stmt = ast.Return(value=call)
        emit_async = False
    elif is_async_def:
        return_stmt = ast.Return(value=ast.Await(value=call))
        emit_async = True
    else:
        return_stmt = ast.Return(value=call)
        emit_async = False

    body: list[ast.stmt] = []
    docstring = _docstring_stmt(fn)
    if docstring is not None:
        body.append(docstring)
    body.append(return_stmt)

    node_cls = ast.AsyncFunctionDef if emit_async else ast.FunctionDef
    return node_cls(
        name=method_name,
        args=new_args,
        body=body,
        decorator_list=[],
        returns=fn.returns,
    )


def _annotation_names(
    methods: list[ast.FunctionDef | ast.AsyncFunctionDef],
) -> tuple[set[str], list[ast.expr]]:
    """Collect referenced names and every annotation AST across the methods.

    Returns ``(names, annotations)`` where *names* is the set of bare
    ``ast.Name`` ids and ``ast.Attribute`` bases (e.g. ``pd`` from
    ``pd.DataFrame``), and *annotations* is the list of annotation expressions
    (for model-name resolution).
    """
    names: set[str] = set()
    annotations: list[ast.expr] = []
    for method in methods:
        arg_iter = [
            *method.args.posonlyargs,
            *method.args.args,
            *method.args.kwonlyargs,
        ]
        if method.args.kwarg is not None:
            arg_iter.append(method.args.kwarg)
        collected = [a.annotation for a in arg_iter if a.annotation is not None]
        if method.returns is not None:
            collected.append(method.returns)
        annotations.extend(collected)
        for ann in collected:
            for node in ast.walk(ann):
                if isinstance(node, ast.Name):
                    names.add(node.id)
                elif isinstance(node, ast.Attribute) and isinstance(
                    node.value, ast.Name
                ):
                    names.add(node.value.id)
    return names, annotations


def _build_imports(
    methods: list[ast.FunctionDef | ast.AsyncFunctionDef],
    resolver: TypeResolver,
) -> tuple[list[ast.ImportFrom], ast.If | None]:
    """Build the import statements the wrapped signatures reference.

    Returns ``(imports, type_checking_block)``. pandas/polars go behind a
    ``TYPE_CHECKING`` guard (they are optional runtime deps used only in
    annotations).
    """
    names, annotations = _annotation_names(methods)

    collector = ImportCollector()
    for ann in annotations:
        collector.add_imports(resolver.collect_model_imports_from_ast(ann))
    for name in names:
        mapping = _NAME_IMPORTS.get(name)
        if mapping is not None:
            module, imported = mapping
            collector.add_imports({module: {imported}})

    type_checking_block: ast.If | None = None
    tc_imports: list[ast.stmt] = []
    if 'pd' in names:
        tc_imports.append(ast.Import(names=[ast.alias(name='pandas', asname='pd')]))
    if 'pl' in names:
        tc_imports.append(ast.Import(names=[ast.alias(name='polars', asname='pl')]))
    if tc_imports:
        collector.add_imports({'typing': {'TYPE_CHECKING'}})
        type_checking_block = ast.If(
            test=_name('TYPE_CHECKING'), body=tc_imports, orelse=[]
        )

    return collector.to_ast(), type_checking_block


def build_client_module_body(
    function_defs: list[ast.FunctionDef | ast.AsyncFunctionDef],
    resolver: TypeResolver,
    *,
    base_client_name: str,
    endpoints_module: str = 'endpoints',
    sync_class_name: str = 'Client',
    async_class_name: str = 'AsyncClient',
) -> tuple[list[ast.stmt], list[str]]:
    """Build the ``_clients.py`` module body and its exported class names.

    Args:
        function_defs: The generated free functions to wrap.
        resolver: Type resolver, for model-name imports.
        base_client_name: Name of the client class (in ``client.py``) both client
            classes subclass, e.g. ``Client`` -- imported aliased so it does not
            clash with the generated sync class.
        endpoints_module: Module (relative to the package) holding the free
            functions.
        sync_class_name / async_class_name: Generated class names.

    Returns:
        ``(module_body, [sync_class_name, async_class_name])``.
    """
    endpoints_alias = '_endpoints'

    sync_methods = [
        _method_from_function(fn, endpoints_alias)
        for fn in function_defs
        if not isinstance(fn, ast.AsyncFunctionDef)
    ]
    async_methods = [
        _method_from_function(fn, endpoints_alias)
        for fn in function_defs
        if isinstance(fn, ast.AsyncFunctionDef)
    ]

    imports, type_checking_block = _build_imports(
        sync_methods + async_methods, resolver
    )

    sync_class = ast.ClassDef(
        name=sync_class_name,
        bases=[_name('_BaseClient')],
        keywords=[],
        body=sync_methods or [ast.Pass()],
        decorator_list=[],
    )
    async_class = ast.ClassDef(
        name=async_class_name,
        bases=[_name('_BaseClient')],
        keywords=[],
        body=async_methods or [ast.Pass()],
        decorator_list=[],
    )

    body: list[ast.stmt] = [
        ast.ImportFrom(
            module='__future__', names=[ast.alias(name='annotations')], level=0
        ),
    ]
    body.extend(imports)
    # Aliased relative imports the ImportCollector can't express.
    body.append(
        ast.ImportFrom(
            module='client',
            names=[ast.alias(name=base_client_name, asname='_BaseClient')],
            level=1,
        )
    )
    body.append(
        ast.ImportFrom(
            module=None,
            names=[ast.alias(name=endpoints_module, asname=endpoints_alias)],
            level=1,
        )
    )
    if type_checking_block is not None:
        body.append(type_checking_block)
    body.append(_all(sorted([async_class_name, sync_class_name])))
    body.append(sync_class)
    body.append(async_class)

    return body, [sync_class_name, async_class_name]
