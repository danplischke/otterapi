"""Request/response logging for OtterAPI generated clients.

Every request the client sends is logged through the standard :mod:`logging`
module under the ``<package>._client`` logger:

* ``INFO``  -- method, URL, status code and elapsed time;
* ``DEBUG`` -- additionally the request and response headers and bodies.

Credentials never reach the log: the ``Authorization`` / cookie / API-key
headers, the query parameters and header names of the document's auth
schemes, and body fields that look like secrets (``password``,
``client_secret``, ``access_token`` ...) are replaced with ``[REDACTED]``.

Nothing is printed unless logging is configured -- either by the application
(``logging.basicConfig(level=logging.INFO)``) or by setting the client's
environment variable (e.g. ``PETSTORE_LOG=debug``), which attaches a stderr
handler to this logger only.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Iterable, Mapping
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

logger = logging.getLogger(f'{__package__}._client' if __package__ else '_client')

REDACTED = '[REDACTED]'

#: Header names (lower case) always redacted.
SENSITIVE_HEADERS = frozenset(
    {
        'authorization',
        'proxy-authorization',
        'cookie',
        'set-cookie',
        'x-api-key',
        'api-key',
        'x-auth-token',
    }
)

#: Body field / query parameter names (lower case) treated as secrets. Kept
#: explicit: a substring match on "token" would also hide pagination cursors
#: such as ``next_page_token``, which are exactly what one debugs.
SENSITIVE_KEYS = frozenset(
    {
        'password',
        'passwd',
        'secret',
        'client_secret',
        'token',
        'access_token',
        'refresh_token',
        'id_token',
        'api_key',
        'apikey',
        'api-key',
        'private_key',
        'authorization',
        'credentials',
    }
)

#: Suffixes that mark a name as a secret (``db_password``, ``webhook_secret``).
SENSITIVE_SUFFIXES = ('_password', '_secret', '-secret', '_api_key')

#: Longest body (in characters) logged at DEBUG before it is truncated.
MAX_BODY_CHARS = 1000


def configure_from_env(env_var: str) -> None:
    """Turn on logging when ``env_var`` is ``debug`` or ``info``.

    Off by default, so importing a generated client never prints anything.
    The handler is attached to this client's logger only, and only once.
    """
    value = os.environ.get(env_var, '').strip().lower()
    level = {'debug': logging.DEBUG, 'info': logging.INFO}.get(value)
    if level is None:
        return
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter('%(asctime)s %(name)s %(levelname)s %(message)s')
        )
        logger.addHandler(handler)


def _is_sensitive(name: str, extra: frozenset[str]) -> bool:
    lowered = name.lower()
    if lowered in SENSITIVE_HEADERS or lowered in SENSITIVE_KEYS or lowered in extra:
        return True
    return lowered.endswith(SENSITIVE_SUFFIXES)


def redact_headers(
    headers: Mapping[str, Any] | Iterable[tuple[str, Any]] | None,
    extra: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    """A copy of ``headers`` with credential headers replaced."""
    if headers is None:
        return {}
    items = headers.items() if isinstance(headers, Mapping) else headers
    return {
        str(name): (REDACTED if _is_sensitive(str(name), extra) else value)
        for name, value in items
    }


def redact_url(url: str, extra: frozenset[str] = frozenset()) -> str:
    """``url`` with the values of secret query parameters replaced."""
    parts = urlsplit(url)
    if not parts.query:
        return url
    query = [
        (key, REDACTED if _is_sensitive(key, extra) else value)
        for key, value in parse_qsl(parts.query, keep_blank_values=True)
    ]
    return urlunsplit(parts._replace(query=urlencode(query, safe='[]')))


def redact_body(value: Any, extra: frozenset[str] = frozenset()) -> Any:
    """A copy of a JSON-like body with secret-looking fields replaced."""
    if isinstance(value, Mapping):
        return {
            key: (
                REDACTED
                if isinstance(key, str) and _is_sensitive(key, extra)
                else redact_body(item, extra)
            )
            for key, item in value.items()
        }
    if isinstance(value, list | tuple):
        return [redact_body(item, extra) for item in value]
    return value


def _render(value: Any) -> str:
    try:
        text = json.dumps(value, default=str)
    except (TypeError, ValueError):
        text = repr(value)
    if len(text) > MAX_BODY_CHARS:
        return f'{text[:MAX_BODY_CHARS]}... ({len(text)} chars)'
    return text


def _response_body(response: Any, extra: frozenset[str]) -> str:
    try:
        return _render(redact_body(response.json(), extra))
    except Exception:  # not JSON (or not read): log the size only
        try:
            return f'<{len(response.content)} bytes>'
        except Exception:
            return '<unread stream>'


def log_request(request: Mapping[str, Any], extra: frozenset[str]) -> float:
    """Log an outgoing request at DEBUG; return its start time."""
    if logger.isEnabledFor(logging.DEBUG):
        body = next(
            (request[key] for key in ('json', 'data') if request.get(key) is not None),
            None,
        )
        params = request.get('params')
        logger.debug(
            'Request: %s %s%s headers=%s%s',
            str(request.get('method', '')).upper(),
            redact_url(str(request.get('url', '')), extra),
            f' params={redact_headers(params, extra)}' if params else '',
            redact_headers(request.get('headers'), extra),
            f' body={_render(redact_body(body, extra))}' if body is not None else '',
        )
    return time.perf_counter()


def log_response(response: Any, started: float, extra: frozenset[str]) -> None:
    """Log a received response: a summary at INFO, the details at DEBUG."""
    if not logger.isEnabledFor(logging.INFO):
        return
    elapsed_ms = (time.perf_counter() - started) * 1000
    request = getattr(response, 'request', None)
    method = getattr(request, 'method', '')
    url = redact_url(str(getattr(request, 'url', '')), extra)
    logger.info('%s %s -> %s (%.0f ms)', method, url, response.status_code, elapsed_ms)
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            'Response: %s headers=%s body=%s',
            response.status_code,
            redact_headers(response.headers.items(), extra),
            _response_body(response, extra),
        )
