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

from otterapi.codegen.ast_utils import (
    ImportCollector,
    _all,
    _assign,
    _attr,
    _call,
    _name,
)

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
    fn: ast.FunctionDef | ast.AsyncFunctionDef,
    endpoints_module_alias: str,
    *,
    client_value: ast.expr | None = None,
    method_name: str | None = None,
) -> ast.FunctionDef | ast.AsyncFunctionDef:
    """Transform a generated free function into a delegating client method.

    Drops the ``client`` keyword arg, prepends ``self``, and forwards every
    remaining argument to the original function, passing ``client_value`` (the
    owning client instance) as ``client``.

    Args:
        client_value: Expression bound to ``client`` in the delegated call
            (default ``self``; resource sub-clients pass ``self._client``).
        method_name: Override for the method name (default: the function name
            with any ``async_`` prefix stripped).
    """
    is_async_def = isinstance(fn, ast.AsyncFunctionDef)
    is_generator = _own_scope_yields(fn)
    src = fn.args
    if client_value is None:
        client_value = _name('self')

    if method_name is None:
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
    # client=<client_value>, plus any **kwargs.
    call_args = [_name(a.arg) for a in (*src.posonlyargs, *src.args)]
    call_keywords = [ast.keyword(arg=a.arg, value=_name(a.arg)) for a in kwonlyargs]
    call_keywords.append(ast.keyword(arg='client', value=client_value))
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


# =============================================================================
# Resource-grouped layout (client_style="resource"): client.users.get(...)
# =============================================================================


def _pascal(resource_key: str) -> str:
    """``identity_users`` -> ``IdentityUsers`` for a resource class name."""
    return ''.join(part[:1].upper() + part[1:] for part in resource_key.split('_'))


def _resource_tokens(resource_key: str) -> set[str]:
    """Tokens stripped from a method name for a resource (naive singular/plural).

    ``users`` -> ``{'users', 'user'}`` so ``list_users`` becomes ``list`` and
    ``get_user`` becomes ``get`` under ``client.users``.
    """
    tokens: set[str] = set()
    for comp in resource_key.split('_'):
        tokens.add(comp)
        if comp.endswith('s'):
            tokens.add(comp[:-1])
        else:
            tokens.add(comp + 's')
    return tokens


def _build_resource_name_map(fn_bases: list[str], tokens: set[str]) -> dict[str, str]:
    """Map each function base name to a resource method name.

    Strips the resource tokens; falls back to the full name when stripping would
    produce an empty, invalid, or already-taken identifier. Deterministic
    (sorted) so sync and async siblings resolve to the same method name.
    """
    name_map: dict[str, str] = {}
    taken: set[str] = set()
    for base in sorted(fn_bases):
        candidate = '_'.join(p for p in base.split('_') if p not in tokens)
        if not candidate or not candidate.isidentifier() or candidate in taken:
            candidate = base
        while candidate in taken:  # pathological; keeps names unique
            candidate = candidate + '_'
        name_map[base] = candidate
        taken.add(candidate)
    return name_map


def _base_name(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """Function name with any ``async_`` prefix removed (the resource base name)."""
    if isinstance(fn, ast.AsyncFunctionDef) and fn.name.startswith(ASYNC_PREFIX):
        return fn.name[len(ASYNC_PREFIX) :]
    return fn.name


def _resource_class(class_name: str, methods: list[ast.stmt]) -> ast.ClassDef:
    """A resource sub-client holding ``self._client`` plus delegating methods."""
    init = ast.FunctionDef(
        name='__init__',
        args=ast.arguments(
            posonlyargs=[],
            args=[ast.arg(arg='self'), ast.arg(arg='client')],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
        ),
        body=[_assign(_attr('self', '_client'), _name('client'))],
        decorator_list=[],
        returns=None,
    )
    return ast.ClassDef(
        name=class_name,
        bases=[],
        keywords=[],
        body=[init, *methods],
        decorator_list=[],
    )


def _resource_property(attr_name: str, resource_class_name: str) -> ast.FunctionDef:
    """A ``@property`` on the client returning a fresh resource sub-client."""
    return ast.FunctionDef(
        name=attr_name,
        args=ast.arguments(
            posonlyargs=[],
            args=[ast.arg(arg='self')],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
        ),
        body=[
            ast.Return(value=_call(_name(resource_class_name), args=[_name('self')]))
        ],
        decorator_list=[_name('property')],
        returns=_name(resource_class_name),
    )


def build_resource_client_module_body(
    function_defs: list[ast.FunctionDef | ast.AsyncFunctionDef],
    resolver: TypeResolver,
    resource_of: dict[str, str],
    *,
    base_client_name: str,
    endpoints_module: str = 'endpoints',
    sync_class_name: str = 'Client',
    async_class_name: str = 'AsyncClient',
) -> tuple[list[ast.stmt], list[str]]:
    """Build ``_clients.py`` with resource-grouped sub-clients.

    ``client.users.get(user_id=1)`` instead of ``client.get_user(user_id=1)``:
    functions are grouped into resource sub-clients by ``resource_of`` (produced
    from the ``module_split`` strategy), and method names have the resource token
    stripped.
    """
    endpoints_alias = '_endpoints'
    client_value = _attr('self', '_client')

    # Group function defs by resource, preserving a stable resource order.
    groups: dict[str, list[ast.FunctionDef | ast.AsyncFunctionDef]] = {}
    for fn in function_defs:
        groups.setdefault(resource_of.get(fn.name, 'common'), []).append(fn)

    all_methods: list[ast.FunctionDef | ast.AsyncFunctionDef] = []
    resource_classes: list[ast.stmt] = []
    sync_properties: list[ast.stmt] = []
    async_properties: list[ast.stmt] = []

    for resource_key in sorted(groups):
        group = groups[resource_key]
        tokens = _resource_tokens(resource_key)
        name_map = _build_resource_name_map(
            list({_base_name(fn) for fn in group}), tokens
        )

        def _methods(is_async: bool) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
            built = [
                _method_from_function(
                    fn,
                    endpoints_alias,
                    client_value=client_value,
                    method_name=name_map[_base_name(fn)],
                )
                for fn in group
                if isinstance(fn, ast.AsyncFunctionDef) is is_async
            ]
            return sorted(built, key=lambda m: m.name)

        sync_methods = _methods(is_async=False)
        async_methods = _methods(is_async=True)
        all_methods.extend(sync_methods)
        all_methods.extend(async_methods)

        pascal = _pascal(resource_key)
        sync_cls_name = f'_{pascal}Resource'
        async_cls_name = f'_Async{pascal}Resource'
        resource_classes.append(
            _resource_class(sync_cls_name, sync_methods or [ast.Pass()])
        )
        resource_classes.append(
            _resource_class(async_cls_name, async_methods or [ast.Pass()])
        )
        sync_properties.append(_resource_property(resource_key, sync_cls_name))
        async_properties.append(_resource_property(resource_key, async_cls_name))

    imports, type_checking_block = _build_imports(all_methods, resolver)

    sync_client = ast.ClassDef(
        name=sync_class_name,
        bases=[_name('_BaseClient')],
        keywords=[],
        body=sync_properties or [ast.Pass()],
        decorator_list=[],
    )
    async_client = ast.ClassDef(
        name=async_class_name,
        bases=[_name('_BaseClient')],
        keywords=[],
        body=async_properties or [ast.Pass()],
        decorator_list=[],
    )

    body: list[ast.stmt] = [
        ast.ImportFrom(
            module='__future__', names=[ast.alias(name='annotations')], level=0
        ),
    ]
    body.extend(imports)
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
    body.extend(resource_classes)
    body.append(sync_client)
    body.append(async_client)

    return body, [sync_class_name, async_class_name]
