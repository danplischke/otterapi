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
    MODELS_MODULE,
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
    'Annotated': ('typing', 'Annotated'),
    'Path': ('pathlib', 'Path'),
    'UPath': ('upath', 'UPath'),
    'Field': ('pydantic', 'Field'),
    'Query': ('._query', 'Query'),
    'AsyncQuery': ('._query', 'AsyncQuery'),
}

# Endpoint-variant suffix -> the Query constructor keyword that binds it.
_VARIANT_BINDINGS: tuple[tuple[str, str], ...] = (
    ('_df', 'to_pandas'),
    ('_pl', 'to_polars'),
    ('_iter', 'iterate'),
    ('_export', 'export'),
)

# Public members of the generated base client. Resource accessors / methods on
# ``Client`` / ``AsyncClient`` must not shadow these (e.g. a resource named
# ``close`` would break ``client.close()``).
_CLIENT_RESERVED_MEMBERS: frozenset[str] = frozenset(
    {
        'close',
        'aclose',
        'base_url',
        'headers',
        'timeout',
        'max_retries',
        'backoff_factor',
        'retry_statuses',
    }
)


def _avoid_member_collisions(
    members: list[ast.FunctionDef | ast.AsyncFunctionDef], reserved: frozenset[str]
) -> None:
    """Rename members (in place) that collide with reserved names or each other.

    Appends underscores until the name is free, so a resource accessor or method
    named like a client member does not shadow it. Deterministic, so the sync and
    async classes rename identically.
    """
    taken = set(reserved)
    for node in members:
        name = node.name
        while name in taken:
            name = name + '_'
        node.name = name
        taken.add(name)


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


def _collect_annotation_names(ann: ast.expr, names: set[str]) -> None:
    """Collect ``Name`` ids and ``Attribute`` bases from an annotation.

    Descends into stringized (forward-ref) sub-annotations -- e.g. the DataFrame
    return type is the constant ``"pd.DataFrame"`` and the ``pd`` inside it would
    otherwise be invisible, so its import would be missed.
    """
    for node in ast.walk(ann):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            names.add(node.value.id)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            try:
                inner = ast.parse(node.value, mode='eval').body
            except SyntaxError:
                continue
            _collect_annotation_names(inner, names)


def _method_annotation_names(
    methods: list[ast.FunctionDef | ast.AsyncFunctionDef],
) -> set[str]:
    """Every name referenced by the methods' argument and return annotations."""
    names: set[str] = set()
    for method in methods:
        args = [*method.args.posonlyargs, *method.args.args, *method.args.kwonlyargs]
        if method.args.kwarg is not None:
            args.append(method.args.kwarg)
        for arg in args:
            if arg.annotation is not None:
                _collect_annotation_names(arg.annotation, names)
        if method.returns is not None:
            _collect_annotation_names(method.returns, names)
    return names


def _build_imports(
    methods: list[ast.FunctionDef | ast.AsyncFunctionDef],
    resolver: TypeResolver,
) -> tuple[list[ast.ImportFrom], ast.If | None]:
    """Build the import statements the wrapped signatures reference.

    Returns ``(imports, type_checking_block)``. pandas/polars go behind a
    ``TYPE_CHECKING`` guard (they are optional runtime deps used only in
    annotations).
    """
    names = _method_annotation_names(methods)

    collector = ImportCollector()
    for model_name in names & resolver.model_names():
        collector.add_imports({MODELS_MODULE: {model_name}})
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


def _module_alias(dotted: str) -> str:
    """Import alias for the module a function is called from (``users`` -> ``_ep_users``)."""
    return '_ep_' + dotted.replace('.', '_')


def _module_imports_and_aliases(
    function_defs: list[ast.FunctionDef | ast.AsyncFunctionDef],
    module_of: dict[str, str] | None,
    default_module: str,
) -> tuple[list[ast.ImportFrom], dict[str, str]]:
    """Build imports for the function modules and a per-function call alias.

    Non-split (``module_of is None``): every function lives in ``default_module``,
    imported once as ``_endpoints``. Split (``module_of`` maps function name ->
    dotted module path): each distinct module is imported under its own alias so a
    method can call ``<alias>.<fn>``.
    """
    if module_of is None:
        alias = '_endpoints'
        stmt = ast.ImportFrom(
            module=None,
            names=[ast.alias(name=default_module, asname=alias)],
            level=1,
        )
        return [stmt], {fn.name: alias for fn in function_defs}

    alias_of: dict[str, str] = {}
    modules: dict[str, str] = {}
    for fn in function_defs:
        dotted = module_of.get(fn.name, default_module)
        alias = _module_alias(dotted)
        modules[dotted] = alias
        alias_of[fn.name] = alias

    stmts: list[ast.ImportFrom] = []
    for dotted in sorted(modules):
        parent, _, leaf = dotted.rpartition('.')
        stmts.append(
            ast.ImportFrom(
                module=parent or None,
                names=[ast.alias(name=leaf, asname=modules[dotted])],
                level=1,
            )
        )
    return stmts, alias_of


def build_client_module_body(
    function_defs: list[ast.FunctionDef | ast.AsyncFunctionDef],
    resolver: TypeResolver,
    *,
    base_client_name: str,
    endpoints_module: str = 'endpoints',
    module_of: dict[str, str] | None = None,
    use_query: bool = False,
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
        endpoints_module: Module holding the free functions (non-split default).
        module_of: Optional map of function name -> dotted module (split mode);
            when omitted every function is imported from ``endpoints_module``.
        sync_class_name / async_class_name: Generated class names.

    Returns:
        ``(module_body, [sync_class_name, async_class_name])``.
    """
    module_imports, alias_of = _module_imports_and_aliases(
        function_defs, module_of, endpoints_module
    )

    def flat_methods(
        is_async: bool,
    ) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
        if use_query:
            # Collapse each endpoint's variants into one Query-returning method.
            families = _group_families(function_defs)
            side = 'async' if is_async else 'sync'
            built: list[ast.FunctionDef | ast.AsyncFunctionDef] = []
            for base in sorted(families):
                variants = families[base][side]
                core = variants.get('fetch')
                if core is None:
                    continue
                if len(variants) > 1:
                    built.append(
                        _query_method(
                            core,
                            variants,
                            alias_of[core.name],
                            _name('self'),
                            base,
                            is_async,
                        )
                    )
                else:
                    built.append(
                        _method_from_function(
                            core, alias_of[core.name], method_name=base
                        )
                    )
            return sorted(built, key=lambda m: m.name)
        return [
            _method_from_function(fn, alias_of[fn.name])
            for fn in function_defs
            if isinstance(fn, ast.AsyncFunctionDef) is is_async
        ]

    sync_methods = flat_methods(is_async=False)
    async_methods = flat_methods(is_async=True)

    imports, type_checking_block = _build_imports(
        sync_methods + async_methods, resolver
    )

    # Methods live on Client / AsyncClient, so a method named like a base-client
    # member (close, headers, ...) must be renamed to avoid shadowing it.
    _avoid_member_collisions(sync_methods, _CLIENT_RESERVED_MEMBERS)
    _avoid_member_collisions(async_methods, _CLIENT_RESERVED_MEMBERS)

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
    body.extend(module_imports)
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


def _resource_class_name(path: tuple[str, ...], is_async: bool) -> str:
    """Class name for a resource node, e.g. ``('identity','users')`` ->
    ``_IdentityUsersResource`` (``_AsyncIdentityUsersResource``)."""
    pascal = _pascal('_'.join(path))
    return f'_Async{pascal}Resource' if is_async else f'_{pascal}Resource'


def _child_property(
    child_segment: str, child_class_name: str, *, on_client: bool
) -> ast.FunctionDef:
    """A ``@property`` returning a child resource, threading the real client.

    On the client itself ``self`` is the client; on a sub-resource the stored
    ``self._client`` is passed down.
    """
    client_expr: ast.expr = _name('self') if on_client else _attr('self', '_client')
    return ast.FunctionDef(
        name=child_segment,
        args=ast.arguments(
            posonlyargs=[],
            args=[ast.arg(arg='self')],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
        ),
        body=[ast.Return(value=_call(_name(child_class_name), args=[client_expr]))],
        decorator_list=[_name('property')],
        returns=_name(child_class_name),
    )


def _variant_of(base_name: str) -> tuple[str, str]:
    """Split a function's (async-stripped) name into ``(family, binding)``.

    ``list_users_df`` -> ``('list_users', 'to_pandas')``; a core function with no
    variant suffix -> ``('list_users', 'fetch')``.
    """
    for suffix, binding in _VARIANT_BINDINGS:
        if base_name.endswith(suffix):
            return base_name[: -len(suffix)], binding
    return base_name, 'fetch'


def _group_families(
    fns: list[ast.FunctionDef | ast.AsyncFunctionDef],
) -> dict[str, dict[str, dict[str, ast.FunctionDef | ast.AsyncFunctionDef]]]:
    """Group an endpoint's variant functions into families.

    ``{family_base: {'sync': {binding: fn}, 'async': {binding: fn}}}``.
    """
    families: dict[
        str, dict[str, dict[str, ast.FunctionDef | ast.AsyncFunctionDef]]
    ] = {}
    for fn in fns:
        base, binding = _variant_of(_base_name(fn))
        side = 'async' if isinstance(fn, ast.AsyncFunctionDef) else 'sync'
        families.setdefault(base, {'sync': {}, 'async': {}})[side][binding] = fn
    return families


def _list_item_type(returns: ast.expr | None) -> ast.expr | None:
    """Return ``X`` from a ``list[X]`` return annotation, else None."""
    if (
        isinstance(returns, ast.Subscript)
        and isinstance(returns.value, ast.Name)
        and returns.value.id == 'list'
    ):
        return returns.slice
    return None


def _query_method(
    core_fn: ast.FunctionDef | ast.AsyncFunctionDef,
    variants: dict[str, ast.FunctionDef | ast.AsyncFunctionDef],
    call_alias: str,
    client_value: ast.expr,
    method_name: str,
    is_async: bool,
) -> ast.FunctionDef:
    """Build a method that returns a deferred ``Query`` / ``AsyncQuery``.

    The method carries the core endpoint's typed signature, captures every
    argument into a params dict, and binds the family's variant functions to the
    query's terminals. It is always a plain ``def`` (no I/O happens until a
    terminal is called).
    """
    src = core_fn.args
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

    # {'p': p, ...} plus **kwargs -> the arguments every terminal reuses.
    param_args = [*src.posonlyargs, *src.args, *kwonlyargs]
    keys: list[ast.expr | None] = [ast.Constant(value=a.arg) for a in param_args]
    values: list[ast.expr] = [_name(a.arg) for a in param_args]
    if src.kwarg is not None:
        keys.append(None)
        values.append(_name(src.kwarg.arg))
    params_dict = ast.Dict(keys=keys, values=values)

    query_cls = 'AsyncQuery' if is_async else 'Query'
    binding_order = ('fetch', 'to_pandas', 'to_polars', 'iterate', 'export')
    call_keywords = [
        ast.keyword(arg=binding, value=_attr(call_alias, variants[binding].name))
        for binding in binding_order
        if binding in variants
    ]
    query_call = _call(
        func=_name(query_cls),
        args=[client_value, params_dict],
        keywords=call_keywords,
    )

    item = _list_item_type(core_fn.returns)
    returns = ast.Subscript(
        value=_name(query_cls),
        slice=item if item is not None else _name('Any'),
        ctx=ast.Load(),
    )

    body: list[ast.stmt] = []
    docstring = _docstring_stmt(core_fn)
    if docstring is not None:
        body.append(docstring)
    body.append(ast.Return(value=query_call))

    return ast.FunctionDef(
        name=method_name,
        args=new_args,
        body=body,
        decorator_list=[],
        returns=returns,
    )


def build_resource_client_module_body(
    function_defs: list[ast.FunctionDef | ast.AsyncFunctionDef],
    resolver: TypeResolver,
    resource_path_of: dict[str, tuple[str, ...]],
    *,
    base_client_name: str,
    endpoints_module: str = 'endpoints',
    module_of: dict[str, str] | None = None,
    use_query: bool = False,
    sync_class_name: str = 'Client',
    async_class_name: str = 'AsyncClient',
) -> tuple[list[ast.stmt], list[str]]:
    """Build ``_clients.py`` with (possibly nested) resource sub-clients.

    ``resource_path_of`` maps each function to its resource path tuple; a
    single-segment path gives ``client.users.get(...)``, a two-segment path gives
    ``client.identity.users.get(...)``, and an empty path puts the method directly
    on the client. Method names have the path's tokens stripped. ``module_of``
    (split mode) routes each method to its function's real module.
    """
    module_imports, alias_of = _module_imports_and_aliases(
        function_defs, module_of, endpoints_module
    )

    # Functions living at exactly each resource path.
    functions_at: dict[
        tuple[str, ...], list[ast.FunctionDef | ast.AsyncFunctionDef]
    ] = {}
    for fn in function_defs:
        functions_at.setdefault(resource_path_of.get(fn.name, ()), []).append(fn)

    # Every node incl. ancestors, and each node's child segment names.
    all_paths: set[tuple[str, ...]] = set()
    for path in functions_at:
        for i in range(len(path) + 1):
            all_paths.add(path[:i])
    children_of: dict[tuple[str, ...], set[str]] = {}
    for path in all_paths:
        if path:
            children_of.setdefault(path[:-1], set()).add(path[-1])

    all_methods: list[ast.FunctionDef | ast.AsyncFunctionDef] = []

    def methods_for(
        path: tuple[str, ...], is_async: bool
    ) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
        fns = functions_at.get(path, [])
        tokens: set[str] = set()
        for segment in path:
            tokens |= _resource_tokens(segment)
        # Methods on the client itself delegate with client=self; on a
        # sub-resource with client=self._client.
        client_value: ast.expr = _name('self') if not path else _attr('self', '_client')

        if use_query:
            # One method per endpoint family; a family with variants collapses
            # into a single method returning a deferred Query.
            families = _group_families(fns)
            name_map = _build_resource_name_map(list(families), tokens)
            side = 'async' if is_async else 'sync'
            built: list[ast.FunctionDef | ast.AsyncFunctionDef] = []
            for base, family in families.items():
                variants = family[side]
                core = variants.get('fetch')
                if core is None:  # e.g. generate_sync=False leaves no sync core
                    continue
                if len(variants) > 1:
                    built.append(
                        _query_method(
                            core,
                            variants,
                            alias_of[core.name],
                            client_value,
                            name_map[base],
                            is_async,
                        )
                    )
                else:
                    built.append(
                        _method_from_function(
                            core,
                            alias_of[core.name],
                            client_value=client_value,
                            method_name=name_map[base],
                        )
                    )
            return sorted(built, key=lambda m: m.name)

        name_map = _build_resource_name_map(
            list({_base_name(fn) for fn in fns}), tokens
        )
        built = [
            _method_from_function(
                fn,
                alias_of[fn.name],
                client_value=client_value,
                method_name=name_map[_base_name(fn)],
            )
            for fn in fns
            if isinstance(fn, ast.AsyncFunctionDef) is is_async
        ]
        return sorted(built, key=lambda m: m.name)

    def child_props(
        path: tuple[str, ...], is_async: bool, *, on_client: bool
    ) -> list[ast.stmt]:
        return [
            _child_property(
                child,
                _resource_class_name(path + (child,), is_async),
                on_client=on_client,
            )
            for child in sorted(children_of.get(path, set()))
        ]

    # Sub-resource classes (every non-root node), deepest first so a class is
    # defined before the parent that references it.
    resource_classes: list[ast.stmt] = []
    for path in sorted(all_paths, key=lambda p: (-len(p), p)):
        if not path:
            continue
        for is_async in (False, True):
            methods = methods_for(path, is_async)
            all_methods.extend(methods)
            body = child_props(path, is_async, on_client=False) + methods
            _avoid_member_collisions(body, frozenset({'_client'}))
            resource_classes.append(
                _resource_class(
                    _resource_class_name(path, is_async), body or [ast.Pass()]
                )
            )

    # Root: methods + top-level resource accessors live on Client / AsyncClient,
    # so they must not shadow the base client's own members.
    def client_class(name: str, is_async: bool) -> ast.ClassDef:
        methods = methods_for((), is_async)
        all_methods.extend(methods)
        body = child_props((), is_async, on_client=True) + methods
        _avoid_member_collisions(body, _CLIENT_RESERVED_MEMBERS)
        return ast.ClassDef(
            name=name,
            bases=[_name('_BaseClient')],
            keywords=[],
            body=body or [ast.Pass()],
            decorator_list=[],
        )

    sync_client = client_class(sync_class_name, is_async=False)
    async_client = client_class(async_class_name, is_async=True)

    imports, type_checking_block = _build_imports(all_methods, resolver)

    module_body: list[ast.stmt] = [
        ast.ImportFrom(
            module='__future__', names=[ast.alias(name='annotations')], level=0
        ),
    ]
    module_body.extend(imports)
    module_body.append(
        ast.ImportFrom(
            module='client',
            names=[ast.alias(name=base_client_name, asname='_BaseClient')],
            level=1,
        )
    )
    module_body.extend(module_imports)
    if type_checking_block is not None:
        module_body.append(type_checking_block)
    module_body.append(_all(sorted([async_class_name, sync_class_name])))
    module_body.extend(resource_classes)
    module_body.append(sync_client)
    module_body.append(async_client)

    return module_body, [sync_class_name, async_class_name]
