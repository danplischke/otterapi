"""Wire-format serialization for request parameters.

httpx renders query parameters with ``str(value)`` and rejects ``None`` header
values outright.  Neither matches the OpenAPI wire format:

* a ``str``-mixin enum stringifies to ``'PetStatus.AVAILABLE'`` instead of
  ``'available'``;
* a ``datetime`` stringifies with a space separator instead of the ``T`` that
  RFC 3339 (and ``format: date-time``) calls for;
* an unset optional header raises ``TypeError`` instead of being omitted;
* an object-valued query parameter becomes a Python ``repr``;
* a path parameter is interpolated verbatim, so a value containing ``/`` or
  ``?`` escapes its URL segment.

This module centralises the conversion so every generated call site -- query,
header, and path -- puts the same bytes on the wire, including the
``style``/``explode`` rules from the OpenAPI parameter object.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, time
from decimal import Decimal
from enum import Enum
from typing import Any
from urllib.parse import quote

__all__ = [
    'format_path_param',
    'serialize_headers',
    'serialize_query_params',
]

# ``(style, explode)`` exactly as the pair appears on an OpenAPI parameter.
# Callers pass the *resolved* pair; the spec's per-style defaults for
# ``explode`` are applied at generation time, not here.
DEFAULT_QUERY_STYLE = ('form', True)
DEFAULT_PATH_STYLE = ('simple', False)

# Separators for the non-exploded array styles.
_ARRAY_DELIMITERS = {
    'form': ',',
    'simple': ',',
    'spaceDelimited': ' ',
    'pipeDelimited': '|',
}


def _scalar(value: Any) -> str:
    """Render a single scalar in its OpenAPI wire form."""
    # Enum first: a ``class Status(str, Enum)`` member is also a ``str``, and
    # its ``str()`` is the member repr rather than the wire value.
    if isinstance(value, Enum):
        value = value.value
    if isinstance(value, bool):
        # OpenAPI booleans are lower-case; ``str(True)`` is not.
        return 'true' if value else 'false'
    if isinstance(value, str):
        return value
    if isinstance(value, (date, time)):
        # Covers datetime too (a date subclass). ``str()`` would use a space
        # separator instead of the RFC 3339 ``T``.
        return value.isoformat()
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).decode()
    if isinstance(value, Decimal):
        return str(value)
    return str(value)


def _quote(value: str) -> str:
    """Percent-encode a path segment so it cannot escape into path or query."""
    return quote(value, safe='')


def _as_mapping(value: Any) -> dict[str, Any] | None:
    """Return a plain dict for anything object-shaped, else ``None``."""
    if isinstance(value, Mapping):
        return dict(value)
    dump = getattr(value, 'model_dump', None)
    if callable(dump):
        return dict(dump(mode='json', by_alias=True, exclude_none=True))
    dump = getattr(value, 'dict', None)  # Pydantic v1
    if callable(dump):
        return dict(dump())
    return None


def _is_sequence(value: Any) -> bool:
    """True for list-shaped values, excluding the string types."""
    if isinstance(value, (str, bytes, bytearray, Mapping)):
        return False
    return isinstance(value, (Sequence, set, frozenset))


def _object_pairs(
    name: str, obj: Mapping[str, Any], style: str, explode: bool
) -> list[tuple[str, str]]:
    """Expand an object-valued query parameter per ``style``/``explode``."""
    items = [(str(k), v) for k, v in obj.items() if v is not None]
    if style == 'deepObject':
        return [(f'{name}[{k}]', _scalar(v)) for k, v in items]
    if explode:
        # form + explode: each property becomes its own query parameter.
        return [(k, _scalar(v)) for k, v in items]
    flat: list[str] = []
    for key, val in items:
        flat.extend((key, _scalar(val)))
    return [(name, ','.join(flat))]


def _array_pairs(
    name: str, values: Sequence[Any], style: str, explode: bool
) -> list[tuple[str, str]]:
    """Expand an array-valued query parameter per ``style``/``explode``."""
    if explode and style != 'deepObject':
        return [(name, _scalar(v)) for v in values if v is not None]
    delimiter = _ARRAY_DELIMITERS.get(style, ',')
    return [(name, delimiter.join(_scalar(v) for v in values if v is not None))]


def _query_pairs(
    name: str, value: Any, style: str, explode: bool
) -> list[tuple[str, str]]:
    """Expand one query parameter into the pairs that go on the wire."""
    if isinstance(value, Enum) or isinstance(value, (str, bytes, bytearray)):
        return [(name, _scalar(value))]
    mapping = _as_mapping(value)
    if mapping is not None:
        return _object_pairs(name, mapping, style, explode)
    if _is_sequence(value):
        return _array_pairs(name, list(value), style, explode)
    return [(name, _scalar(value))]


def serialize_query_params(
    params: Any,
    styles: Any = None,
) -> list[tuple[str, str]] | None:
    """Flatten query parameters into httpx-ready ``(key, value)`` pairs.

    ``None`` values are dropped (an unset optional parameter is an absent
    parameter, not an empty one).  A list of pairs rather than a dict is
    returned so that exploded arrays and objects can repeat a key.

    Args:
        params: Raw parameter mapping from the endpoint call site, or None.
            Typed loosely because it arrives as a value out of the
            heterogeneous request dict that ``_before_request`` may rewrite.
        styles: Optional ``{name: (style, explode)}`` overrides.  Parameters
            absent from this mapping use the OpenAPI default of
            ``('form', True)``.

    Returns:
        The pairs to hand to httpx, or ``None`` when nothing is left to send.
    """
    if not params:
        return None
    styles = styles or {}
    pairs: list[tuple[str, str]] = []
    for name, value in params.items():
        if value is None:
            continue
        style, explode = styles.get(name, DEFAULT_QUERY_STYLE)
        pairs.extend(_query_pairs(name, value, style, explode))
    return pairs or None


def serialize_headers(headers: Any) -> dict[str, str]:
    """Drop unset headers and render the rest as wire strings.

    httpx requires ``str``/``bytes`` header values, so an unset optional header
    parameter would otherwise raise ``TypeError`` on the default call path.

    Args:
        headers: Raw header mapping, or None.  Typed loosely for the same
            reason as :func:`serialize_query_params`.
    """
    if not headers:
        return {}
    rendered: dict[str, str] = {}
    for name, value in headers.items():
        if value is None:
            continue
        if _is_sequence(value):
            rendered[name] = ','.join(_scalar(v) for v in value if v is not None)
        else:
            rendered[name] = _scalar(value)
    return rendered


def format_path_param(
    value: Any,
    style: str = 'simple',
    explode: bool = False,
    name: str | None = None,
) -> str:
    """Render one path parameter, percent-encoded so it stays in its segment.

    Args:
        value: The caller-supplied parameter value.
        style: OpenAPI path style -- ``simple`` (default), ``label``, or
            ``matrix``.
        explode: The resolved ``explode`` flag for the parameter.
        name: The parameter's wire name.  Only ``matrix`` style needs it.

    Returns:
        The encoded string to interpolate into the request path.
    """
    if isinstance(value, Enum) or isinstance(value, (str, bytes, bytearray)):
        rendered = _quote(_scalar(value))
    else:
        mapping = _as_mapping(value)
        if mapping is not None:
            items = [(str(k), v) for k, v in mapping.items() if v is not None]
            if explode:
                rendered = ','.join(
                    f'{_quote(k)}={_quote(_scalar(v))}' for k, v in items
                )
            else:
                flat: list[str] = []
                for key, val in items:
                    flat.extend((_quote(key), _quote(_scalar(val))))
                rendered = ','.join(flat)
        elif _is_sequence(value):
            rendered = ','.join(_quote(_scalar(v)) for v in value if v is not None)
        else:
            rendered = _quote(_scalar(value))

    if style == 'label':
        return f'.{rendered}'
    if style == 'matrix':
        return f';{name}={rendered}' if name else f';{rendered}'
    return rendered
