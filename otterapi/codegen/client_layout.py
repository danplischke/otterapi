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
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from otterapi.codegen.ast_utils import (
    MODELS_MODULE,
    ImportCollector,
    _all,
    _assign,
    _attr,
    _call,
    _class_def,
    _function_def,
    _name,
    strip_optional,
)

if TYPE_CHECKING:
    from otterapi.codegen.emit import TypeResolver
    from otterapi.codegen.types import Endpoint

ASYNC_PREFIX = 'async_'

# Names this module's own wrappers can add to a signature (``Query``), plus a
# fallback vocabulary for the common annotation names. The wrapped signatures'
# real imports come from the endpoint sink (see ``_build_imports``); this map
# is consulted only for names the sink does not provide.
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
_SUFFIX_BINDINGS: dict[str, str] = dict(_VARIANT_BINDINGS)

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
    members: Sequence[ast.FunctionDef | ast.AsyncFunctionDef],
    reserved: frozenset[str],
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
        fn: The generated free function to wrap.
        endpoints_module_alias: Import alias of the module holding ``fn``, so
            the delegated call reads ``<alias>.<fn>(...)``.
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
    call_args: list[ast.expr] = [_name(a.arg) for a in (*src.posonlyargs, *src.args)]
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

    return _function_def(
        method_name, new_args, body, returns=fn.returns, is_async=emit_async
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


def _import_index(sink_imports: Mapping[str, set[str]]) -> dict[str, tuple[str, str]]:
    """Invert ``{module: {name}}`` into ``{name: (module, name)}``.

    Sibling imports are normalized back to a single leading dot: a sink
    assembled for a nested split module has been re-pointed (``..models``),
    but ``_clients.py`` always lives at the package root.
    """
    index: dict[str, tuple[str, str]] = {}
    for module, imported in sink_imports.items():
        if module.startswith('.'):
            module = '.' + module.lstrip('.')
        for name in sorted(imported):
            index.setdefault(name, (module, name))
    return index


def _build_imports(
    methods: list[ast.FunctionDef | ast.AsyncFunctionDef],
    resolver: TypeResolver,
    sink_imports: Mapping[str, set[str]] | None = None,
) -> tuple[list[ast.ImportFrom], ast.If | None]:
    """Build the import statements the wrapped signatures reference.

    Every name a wrapped signature mentions is resolved against the imports the
    endpoint sink collected while emitting those very functions
    (``sink_imports``): that is the source of truth for ``datetime``, ``UUID``,
    ``Decimal`` and whatever else a spec's parameters need. :data:`_NAME_IMPORTS`
    covers what this module adds on top (``Query`` / ``AsyncQuery``) and is the
    fallback for a sink that was not supplied.

    Returns ``(imports, type_checking_block)``. pandas/polars go behind a
    ``TYPE_CHECKING`` guard (they are optional runtime deps used only in
    annotations).
    """
    names = _method_annotation_names(methods)
    provided_by = _import_index(sink_imports or {})

    collector = ImportCollector()
    for model_name in names & resolver.model_names():
        collector.add_imports({MODELS_MODULE: {model_name}})
    for name in names:
        mapping = provided_by.get(name) or _NAME_IMPORTS.get(name)
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
    owner_of: Mapping[str, Endpoint] | None = None,
    sink_imports: Mapping[str, set[str]] | None = None,
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
        owner_of: Function name -> the endpoint it was generated for (the
            sink's ``owners``); groups variant functions by endpoint.
        sink_imports: The endpoint sink's collected imports, resolving every
            name the wrapped signatures reference.
        base_client_name: Name of the client class (in ``client.py``) both client
            classes subclass, e.g. ``Client`` -- imported aliased so it does not
            clash with the generated sync class.
        endpoints_module: Module holding the free functions (non-split default).
        module_of: Optional map of function name -> dotted module (split mode);
            when omitted every function is imported from ``endpoints_module``.
        use_query: Collapse each endpoint family (core request plus its
            pagination / DataFrame / export variants) into one method returning
            a deferred ``Query`` instead of one method per variant.
        sync_class_name: Name of the generated sync client class.
        async_class_name: Name of the generated async client class.

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
            families = _group_families(function_defs, owner_of or {})
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
        sync_methods + async_methods, resolver, sink_imports
    )

    # Methods live on Client / AsyncClient, so a method named like a base-client
    # member (close, headers, ...) must be renamed to avoid shadowing it.
    _avoid_member_collisions(sync_methods, _CLIENT_RESERVED_MEMBERS)
    _avoid_member_collisions(async_methods, _CLIENT_RESERVED_MEMBERS)

    sync_body: list[ast.stmt] = [*sync_methods] or [ast.Pass()]
    async_body: list[ast.stmt] = [*async_methods] or [ast.Pass()]
    sync_class = _class_def(sync_class_name, [_name('_BaseClient')], sync_body)
    async_class = _class_def(async_class_name, [_name('_BaseClient')], async_body)

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


def _resource_class(class_name: str, methods: Sequence[ast.stmt]) -> ast.ClassDef:
    """A resource sub-client holding ``self._client`` plus delegating methods."""
    init = _function_def(
        '__init__',
        ast.arguments(
            posonlyargs=[],
            args=[ast.arg(arg='self'), ast.arg(arg='client')],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
        ),
        [_assign(_attr('self', '_client'), _name('client'))],
    )
    return _class_def(class_name, [], [init, *methods])


def _resource_class_name(path: tuple[str, ...], is_async: bool) -> str:
    """Class name for a resource node.

    ``('identity', 'users')`` -> ``_IdentityUsersResource``
    (``_AsyncIdentityUsersResource`` on the async side).
    """
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
    return _function_def(
        child_segment,
        ast.arguments(
            posonlyargs=[],
            args=[ast.arg(arg='self')],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
        ),
        [ast.Return(value=_call(_name(child_class_name), args=[client_expr]))],
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


def _family_of(
    fn: ast.FunctionDef | ast.AsyncFunctionDef, owner_of: Mapping[str, Endpoint]
) -> tuple[str, str]:
    """``(family, binding)`` for a generated function.

    The family is the owning endpoint's core function name and the binding is
    which variant of it this function is, read off the suffix *relative to that
    core name* -- so an endpoint whose own name ends in ``_export`` or ``_iter``
    is still a core function, not a stray variant of a family that does not
    exist. A function without a recorded owner falls back to the plain suffix
    heuristic.
    """
    base = _base_name(fn)
    owner = owner_of.get(fn.name)
    if owner is None:
        return _variant_of(base)
    core = owner.sync_fn_name
    if base == core:
        return core, 'fetch'
    if base.startswith(core):
        binding = _SUFFIX_BINDINGS.get(base[len(core) :])
        if binding is not None:
            return core, binding
    return base, 'fetch'


def _group_families(
    fns: list[ast.FunctionDef | ast.AsyncFunctionDef],
    owner_of: Mapping[str, Endpoint],
) -> dict[str, dict[str, dict[str, ast.FunctionDef | ast.AsyncFunctionDef]]]:
    """Group an endpoint's variant functions into families.

    ``{family_base: {'sync': {binding: fn}, 'async': {binding: fn}}}``.
    """
    families: dict[
        str, dict[str, dict[str, ast.FunctionDef | ast.AsyncFunctionDef]]
    ] = {}
    for fn in fns:
        base, binding = _family_of(fn, owner_of)
        side = 'async' if isinstance(fn, ast.AsyncFunctionDef) else 'sync'
        families.setdefault(base, {'sync': {}, 'async': {}})[side][binding] = fn
    return families


def _apply_name_overrides(
    name_map: dict[str, str],
    core_of: Mapping[str, str],
    overrides: Mapping[str, str],
) -> dict[str, str]:
    """Rename folded-leaf methods after their segment (see ``_fold_action_leaves``).

    A variant keeps its suffix: the ``_df`` sibling of a folded ``find_by_status``
    becomes ``find_by_status_df``. Names stay unique within the resource.
    """
    if not overrides:
        return name_map
    taken = set(name_map.values())
    for base, current in list(name_map.items()):
        core = core_of.get(base, base)
        override = overrides.get(core)
        if override is None:
            continue
        candidate = override + base[len(core) :]
        taken.discard(current)
        while candidate in taken:
            candidate += '_'
        name_map[base] = candidate
        taken.add(candidate)
    return name_map


def _is_action_segment(segment: str) -> bool:
    """Whether a path segment reads as an RPC-style action, not a collection.

    Collections are plural nouns (``users``, ``invoices``) and keep their own
    sub-client even with a single operation: ``list`` today is ``get`` tomorrow,
    and the shape should not flip when the spec grows. A multi-word segment
    (``find_by_status``, ``upload_image``) or a singular one (``login``,
    ``inventory``) with a single operation is an action, and
    ``client.pet.find_by_status(...)`` reads far better than a one-method
    ``client.pet.find_by_status.find_pets_by_status(...)``.
    """
    return '_' in segment or not segment.endswith('s')


def _fold_action_leaves(
    functions_at: dict[tuple[str, ...], list[ast.FunctionDef | ast.AsyncFunctionDef]],
    owner_of: Mapping[str, Endpoint],
) -> dict[str, str]:
    """Fold single-operation action leaves into their parent resource (in place).

    Returns ``{core function name: method name}``: the folded operation is named
    after its segment on the parent. Deepest paths first, so a leaf folded into
    a parent that then becomes a single-operation action leaf folds again.
    """
    overrides: dict[str, str] = {}
    for path in sorted(functions_at, key=len, reverse=True):
        if not path or path not in functions_at:
            continue
        has_children = any(
            other != path and other[: len(path)] == path for other in functions_at
        )
        fns = functions_at[path]
        owners = {id(owner_of[fn.name]) for fn in fns if fn.name in owner_of}
        if has_children or len(owners) != 1 or not _is_action_segment(path[-1]):
            continue
        core = next(owner_of[fn.name] for fn in fns if fn.name in owner_of).sync_fn_name
        functions_at.setdefault(path[:-1], []).extend(functions_at.pop(path))
        overrides[core] = path[-1]
    return overrides


def _list_item_type(returns: ast.expr | None) -> ast.expr | None:
    """Return ``X`` from a ``list[X]`` (or ``list[X] | None``) return annotation.

    Sees through the optional wrapper an unwrapped, not-required envelope field
    produces, so its ``Query`` keeps the item type instead of degrading to
    ``Query[Any]``.
    """
    if returns is not None:
        returns = strip_optional(returns)
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

    return _function_def(method_name, new_args, body, returns=returns)


def _resource_methods(
    fns: list[ast.FunctionDef | ast.AsyncFunctionDef],
    path: tuple[str, ...],
    is_async: bool,
    alias_of: dict[str, str],
    owner_of: Mapping[str, Endpoint],
    *,
    use_query: bool,
    name_overrides: Mapping[str, str] | None = None,
) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    """Delegating methods for the functions living at one resource ``path``.

    The path's tokens are stripped from the method names (``name_overrides``
    pins the names of operations folded up from an action leaf). Methods on the
    client itself (empty path) delegate with ``client=self``; on a sub-resource
    with ``client=self._client``.
    """
    name_overrides = name_overrides or {}
    tokens: set[str] = set()
    for segment in path:
        tokens |= _resource_tokens(segment)
    client_value: ast.expr = _name('self') if not path else _attr('self', '_client')

    if use_query:
        # One method per endpoint family; a family with variants collapses
        # into a single method returning a deferred Query.
        families = _group_families(fns, owner_of)
        name_map = _apply_name_overrides(
            _build_resource_name_map(list(families), tokens),
            {core: core for core in families},
            name_overrides,
        )
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

    core_of = {_base_name(fn): _family_of(fn, owner_of)[0] for fn in fns}
    name_map = _apply_name_overrides(
        _build_resource_name_map(list(core_of), tokens), core_of, name_overrides
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


def build_resource_client_module_body(
    function_defs: list[ast.FunctionDef | ast.AsyncFunctionDef],
    resolver: TypeResolver,
    resource_path_of: dict[str, tuple[str, ...]],
    *,
    owner_of: Mapping[str, Endpoint] | None = None,
    sink_imports: Mapping[str, set[str]] | None = None,
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
    (split mode) routes each method to its function's real module. ``owner_of``
    and ``sink_imports`` are the endpoint sink's ``owners`` and collected
    imports (see :func:`build_client_module_body`).

    A leaf whose segment is an action rather than a collection and that holds
    a single operation is folded into its parent as a method named after the
    segment (``client.pet.find_by_status(...)``; see ``_is_action_segment``).
    """
    owner_of = owner_of or {}
    module_imports, alias_of = _module_imports_and_aliases(
        function_defs, module_of, endpoints_module
    )

    # Functions living at exactly each resource path.
    functions_at: dict[
        tuple[str, ...], list[ast.FunctionDef | ast.AsyncFunctionDef]
    ] = {}
    for fn in function_defs:
        functions_at.setdefault(resource_path_of.get(fn.name, ()), []).append(fn)
    name_overrides = _fold_action_leaves(functions_at, owner_of)

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
        return _resource_methods(
            functions_at.get(path, []),
            path,
            is_async,
            alias_of,
            owner_of,
            use_query=use_query,
            name_overrides=name_overrides,
        )

    def child_props(
        path: tuple[str, ...], is_async: bool, *, on_client: bool
    ) -> list[ast.FunctionDef]:
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
            members: list[ast.FunctionDef | ast.AsyncFunctionDef] = [
                *child_props(path, is_async, on_client=False),
                *methods,
            ]
            _avoid_member_collisions(members, frozenset({'_client'}))
            resource_classes.append(
                _resource_class(
                    _resource_class_name(path, is_async), members or [ast.Pass()]
                )
            )

    # Root: methods + top-level resource accessors live on Client / AsyncClient,
    # so they must not shadow the base client's own members.
    def client_class(name: str, is_async: bool) -> ast.ClassDef:
        methods = methods_for((), is_async)
        all_methods.extend(methods)
        members: list[ast.FunctionDef | ast.AsyncFunctionDef] = [
            *child_props((), is_async, on_client=True),
            *methods,
        ]
        _avoid_member_collisions(members, _CLIENT_RESERVED_MEMBERS)
        class_body: list[ast.stmt] = [*members] or [ast.Pass()]
        return _class_def(name, [_name('_BaseClient')], class_body)

    sync_client = client_class(sync_class_name, is_async=False)
    async_client = client_class(async_class_name, is_async=True)

    imports, type_checking_block = _build_imports(all_methods, resolver, sink_imports)

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
