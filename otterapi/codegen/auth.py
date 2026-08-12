"""Security-scheme (authentication) code generation.

An OpenAPI document declares how to authenticate under
``components.securitySchemes``, but nothing in the generated client used to
read it: the only mention was a count printed by ``otter validate``.  Callers
were left to hand-write ``headers={'Authorization': f'Bearer {token}'}`` at
every call site, or to subclass the generated client.

This module turns each declared scheme into a constructor parameter on the
generated client plus one line in a generated ``_apply_auth`` method, so
``Client(api_key='...')`` is all it takes.

Supported scheme types:

* ``apiKey`` -- in ``header``, ``query`` or ``cookie``
* ``http`` with ``scheme: bearer`` -- ``Authorization: Bearer <token>``
* ``http`` with ``scheme: basic`` -- takes ``(username, password)``
* ``oauth2`` / ``openIdConnect`` -- treated as a bearer token, since the token
  is what the caller actually holds; obtaining it is out of scope for a
  generated client.

Anything else is skipped with a warning rather than failing generation: an
unrecognised scheme should not stop a client being generated for the rest of
the API.
"""

from __future__ import annotations

import ast
import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from otterapi.codegen.ast_utils import (
    ImportDict,
    _argument,
    _assign,
    _attr,
    _call,
    _name,
    _subscript,
    _union_expr,
)
from otterapi.codegen.utils import sanitize_parameter_field_name, to_snake_case

if TYPE_CHECKING:
    from otterapi.codegen._openapi_adapter import OpenAPIAdapter

logger = logging.getLogger(__name__)

__all__ = [
    'AuthScheme',
    'build_apply_auth_method',
    'build_auth_init_args',
    'build_auth_init_body',
    'collect_auth_schemes',
]

#: Fallback prefix for the environment variables the client reads.
DEFAULT_ENV_PREFIX = 'OTTER'

AuthKind = Literal['header', 'query', 'cookie', 'bearer', 'basic']


@dataclass(frozen=True)
class AuthScheme:
    """One security scheme, reduced to what code generation needs.

    Attributes:
        name: The scheme's key in ``components.securitySchemes``.
        python_name: The constructor parameter / attribute name.
        kind: How the credential is attached to a request.
        wire_name: Header, query or cookie name for ``apiKey`` schemes.
        env_var: Environment variable consulted when the argument is omitted.
    """

    name: str
    python_name: str
    kind: AuthKind
    wire_name: str | None = None
    env_var: str = ''

    @property
    def annotation(self) -> ast.expr:
        """The constructor parameter's type annotation."""
        if self.kind == 'basic':
            # (username, password)
            return _union_expr(
                [
                    _subscript('tuple', ast.Tuple(elts=[_name('str'), _name('str')])),
                    ast.Constant(value=None),
                ]
            )
        return _union_expr([_name('str'), ast.Constant(value=None)])

    @property
    def reads_env(self) -> bool:
        """Whether an omitted argument falls back to an environment variable.

        Basic auth needs two values, so there is no single sensible variable
        to read; it is constructor-only.
        """
        return self.kind != 'basic'


def _env_var_name(prefix: str, scheme_name: str) -> str:
    """Build the environment variable name for *scheme_name*."""
    upper = re.sub(r'[^0-9a-zA-Z]+', '_', to_snake_case(scheme_name)).upper()
    upper = re.sub(r'_+', '_', upper).strip('_')
    return f'{prefix}_{upper}' if prefix else upper


def collect_auth_schemes(
    adapter: OpenAPIAdapter | None,
    env_prefix: str = DEFAULT_ENV_PREFIX,
) -> list[AuthScheme]:
    """Reduce a document's ``securitySchemes`` to generation-ready descriptors.

    Args:
        adapter: The document facade, or None when no document is loaded.
        env_prefix: Prefix for the generated environment-variable names.

    Returns:
        One :class:`AuthScheme` per supported scheme, ordered by spec name so
        the generated signature is stable across runs.
    """
    if adapter is None:
        return []

    components = adapter.document.components
    raw = getattr(components, 'securitySchemes', None) if components else None
    if not raw:
        return []

    schemes: list[AuthScheme] = []
    used_names: set[str] = set()

    for scheme_name in sorted(raw):
        definition = raw[scheme_name]
        # SecurityScheme is a RootModel union; unwrap to the concrete variant.
        definition = getattr(definition, 'root', definition)

        scheme_type = getattr(definition, 'type', None)
        scheme_type = getattr(scheme_type, 'value', scheme_type)

        kind: AuthKind | None = None
        wire_name: str | None = None

        if scheme_type == 'apiKey':
            location = getattr(definition, 'in_', None)
            location = getattr(location, 'value', location)
            if location in ('header', 'query', 'cookie'):
                kind = location
                wire_name = getattr(definition, 'name', None)
        elif scheme_type == 'http':
            http_scheme = (getattr(definition, 'scheme', '') or '').lower()
            if http_scheme == 'bearer':
                kind = 'bearer'
            elif http_scheme == 'basic':
                kind = 'basic'
        elif scheme_type in ('oauth2', 'openIdConnect'):
            # The caller supplies an already-obtained access token.
            kind = 'bearer'

        if kind is None or (kind in ('header', 'query', 'cookie') and not wire_name):
            logger.warning(
                'Skipping unsupported security scheme %r (type=%r); the '
                'generated client will not configure it automatically.',
                scheme_name,
                scheme_type,
            )
            continue

        # sanitize_identifier would PascalCase this into a class-style name;
        # these are constructor parameters, so they stay snake_case.
        python_name = sanitize_parameter_field_name(to_snake_case(scheme_name))
        # Distinct spec names can sanitize to the same identifier.
        candidate = python_name
        suffix = 2
        while candidate in used_names:
            candidate = f'{python_name}_{suffix}'
            suffix += 1
        used_names.add(candidate)

        schemes.append(
            AuthScheme(
                name=scheme_name,
                python_name=candidate,
                kind=kind,
                wire_name=wire_name,
                env_var=_env_var_name(env_prefix, scheme_name),
            )
        )

    return schemes


def build_auth_init_args(
    schemes: list[AuthScheme],
) -> tuple[list[ast.arg], list[ast.expr | None]]:
    """Build the keyword-only ``__init__`` arguments for *schemes*."""
    args = [_argument(s.python_name, s.annotation) for s in schemes]
    defaults: list[ast.expr | None] = [ast.Constant(value=None) for _ in schemes]
    return args, defaults


def build_auth_init_body(
    schemes: list[AuthScheme],
) -> tuple[list[ast.stmt], ImportDict]:
    """Build the ``__init__`` assignments that store each credential.

    Emits ``self.<name> = <name>`` for basic auth, and for everything else
    ``self.<name> = <name> if <name> is not None else os.environ.get('VAR')``
    so a credential can come from the environment instead of the call site.
    """
    if not schemes:
        return [], {}

    body: list[ast.stmt] = []
    imports: ImportDict = {}

    for scheme in schemes:
        if scheme.reads_env:
            imports.setdefault('os', set()).add('environ')
            value: ast.expr = ast.IfExp(
                test=ast.Compare(
                    left=_name(scheme.python_name),
                    ops=[ast.IsNot()],
                    comparators=[ast.Constant(value=None)],
                ),
                body=_name(scheme.python_name),
                orelse=_call(
                    _attr('environ', 'get'),
                    args=[ast.Constant(value=scheme.env_var)],
                ),
            )
        else:
            value = _name(scheme.python_name)

        body.append(_assign(_attr('self', scheme.python_name), value))

    return body, imports


def _credential_stmts(scheme: AuthScheme) -> list[ast.stmt]:
    """Build the statements that attach one scheme's credential to a request."""
    attr = _attr('self', scheme.python_name)

    if scheme.kind == 'bearer':
        # request['headers']['Authorization'] = f'Bearer {self.<name>}'
        return [
            _assign(
                ast.Subscript(
                    value=ast.Subscript(
                        value=_name('request'),
                        slice=ast.Constant(value='headers'),
                        ctx=ast.Load(),
                    ),
                    slice=ast.Constant(value='Authorization'),
                    ctx=ast.Store(),
                ),
                ast.JoinedStr(
                    values=[
                        ast.Constant(value='Bearer '),
                        ast.FormattedValue(value=attr, conversion=-1),
                    ]
                ),
            )
        ]

    if scheme.kind == 'basic':
        # token = b64encode(f'{u}:{p}'.encode()).decode()
        # request['headers']['Authorization'] = f'Basic {token}'
        joined = ast.JoinedStr(
            values=[
                ast.FormattedValue(
                    value=ast.Subscript(
                        value=attr, slice=ast.Constant(value=0), ctx=ast.Load()
                    ),
                    conversion=-1,
                ),
                ast.Constant(value=':'),
                ast.FormattedValue(
                    value=ast.Subscript(
                        value=attr, slice=ast.Constant(value=1), ctx=ast.Load()
                    ),
                    conversion=-1,
                ),
            ]
        )
        encoded = _call(
            _attr(
                _call(_name('b64encode'), args=[_call(_attr(joined, 'encode'))]),
                'decode',
            )
        )
        return [
            _assign(_name('_basic'), encoded),
            _assign(
                ast.Subscript(
                    value=ast.Subscript(
                        value=_name('request'),
                        slice=ast.Constant(value='headers'),
                        ctx=ast.Load(),
                    ),
                    slice=ast.Constant(value='Authorization'),
                    ctx=ast.Store(),
                ),
                ast.JoinedStr(
                    values=[
                        ast.Constant(value='Basic '),
                        ast.FormattedValue(value=_name('_basic'), conversion=-1),
                    ]
                ),
            ),
        ]

    if scheme.kind == 'header':
        return [
            _assign(
                ast.Subscript(
                    value=ast.Subscript(
                        value=_name('request'),
                        slice=ast.Constant(value='headers'),
                        ctx=ast.Load(),
                    ),
                    slice=ast.Constant(value=scheme.wire_name),
                    ctx=ast.Store(),
                ),
                attr,
            )
        ]

    if scheme.kind == 'cookie':
        # request['headers']['Cookie'] = '<name>=<value>', appended to any
        # cookie header already present.
        existing = _call(
            _attr(
                ast.Subscript(
                    value=_name('request'),
                    slice=ast.Constant(value='headers'),
                    ctx=ast.Load(),
                ),
                'get',
            ),
            args=[ast.Constant(value='Cookie'), ast.Constant(value='')],
        )
        cookie = ast.JoinedStr(
            values=[
                ast.Constant(value=f'{scheme.wire_name}='),
                ast.FormattedValue(value=attr, conversion=-1),
            ]
        )
        return [
            _assign(_name('_cookie'), existing),
            _assign(
                ast.Subscript(
                    value=ast.Subscript(
                        value=_name('request'),
                        slice=ast.Constant(value='headers'),
                        ctx=ast.Load(),
                    ),
                    slice=ast.Constant(value='Cookie'),
                    ctx=ast.Store(),
                ),
                ast.IfExp(
                    test=_name('_cookie'),
                    body=ast.JoinedStr(
                        values=[
                            ast.FormattedValue(value=_name('_cookie'), conversion=-1),
                            ast.Constant(value='; '),
                            *cookie.values,
                        ]
                    ),
                    orelse=cookie,
                ),
            ),
        ]

    # query: request['params'] = {**(request['params'] or {}), '<name>': value}
    return [
        _assign(
            ast.Subscript(
                value=_name('request'),
                slice=ast.Constant(value='params'),
                ctx=ast.Store(),
            ),
            ast.Dict(
                keys=[None, ast.Constant(value=scheme.wire_name)],
                values=[
                    ast.BoolOp(
                        op=ast.Or(),
                        values=[
                            ast.Subscript(
                                value=_name('request'),
                                slice=ast.Constant(value='params'),
                                ctx=ast.Load(),
                            ),
                            ast.Dict(keys=[], values=[]),
                        ],
                    ),
                    attr,
                ],
            ),
        )
    ]


def build_apply_auth_method(
    schemes: list[AuthScheme],
) -> tuple[ast.FunctionDef | None, ImportDict]:
    """Build the ``_apply_auth`` method that attaches configured credentials.

    Returns ``(None, {})`` when the document declares no supported scheme, so
    clients for unauthenticated APIs carry no dead method.
    """
    if not schemes:
        return None, {}

    imports: ImportDict = {}
    body: list[ast.stmt] = [
        ast.Expr(
            value=ast.Constant(
                value=(
                    'Attach configured credentials to an outgoing request.\n\n'
                    'Runs before ``_before_request``, so a subclass hook can '
                    'still\noverride or refresh whatever is set here.\n\n'
                    'Only schemes given a value are applied; the rest are '
                    'left off\nentirely, which is what an API expects for an '
                    'optional scheme.\n'
                )
            )
        )
    ]

    for scheme in schemes:
        if scheme.kind == 'basic':
            imports.setdefault('base64', set()).add('b64encode')
        body.append(
            ast.If(
                test=ast.Compare(
                    left=_attr('self', scheme.python_name),
                    ops=[ast.IsNot()],
                    comparators=[ast.Constant(value=None)],
                ),
                body=_credential_stmts(scheme),
                orelse=[],
            )
        )

    return (
        ast.FunctionDef(
            name='_apply_auth',
            args=ast.arguments(
                posonlyargs=[],
                args=[_argument('self'), _argument('request', _name('dict'))],
                kwonlyargs=[],
                kw_defaults=[],
                kwarg=None,
                defaults=[],
            ),
            body=body,
            decorator_list=[],
            returns=ast.Constant(value=None),
        ),
        imports,
    )
