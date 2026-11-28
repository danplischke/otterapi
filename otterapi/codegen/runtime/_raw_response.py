"""Raw-response access for OtterAPI generated clients.

Every endpoint with a parsed body has a ``<name>_with_response`` variant that
returns one of these instead of the parsed model, so the status code and the
headers (rate limits, ``ETag``, request IDs, ``Location``) of that one call
are reachable::

    raw = get_user_with_response(user_id=1)
    raw.status_code, raw.headers['x-ratelimit-remaining']
    user = raw.parse()      # exactly what get_user(user_id=1) returns

Error responses still raise the usual ``APIError`` subclasses; the response
is on the exception as ``err.response``.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, Generic, TypeVar

T = TypeVar('T')

_UNPARSED: Any = object()


class RawResponse(Generic[T]):
    """An HTTP response plus a typed, lazily-run ``parse()``.

    Attributes:
        http_response: The underlying HTTP response object.
    """

    def __init__(self, http_response: Any, parse: Callable[[], T]) -> None:
        self.http_response = http_response
        self._parse = parse
        self._parsed: Any = _UNPARSED

    @property
    def status_code(self) -> int:
        """The HTTP status code."""
        return self.http_response.status_code

    @property
    def headers(self) -> Any:
        """The response headers (case-insensitive mapping)."""
        return self.http_response.headers

    def parse(self) -> T:
        """Parse the body into what the plain endpoint function returns."""
        if self._parsed is _UNPARSED:
            self._parsed = self._parse()
        return self._parsed

    def __repr__(self) -> str:
        return f'{type(self).__name__}(status_code={self.status_code})'


class AsyncRawResponse(Generic[T]):
    """An HTTP response plus an awaitable, typed ``parse()`` (see RawResponse)."""

    def __init__(self, http_response: Any, parse: Callable[[], Awaitable[T]]) -> None:
        self.http_response = http_response
        self._parse = parse
        self._parsed: Any = _UNPARSED

    @property
    def status_code(self) -> int:
        """The HTTP status code."""
        return self.http_response.status_code

    @property
    def headers(self) -> Any:
        """The response headers (case-insensitive mapping)."""
        return self.http_response.headers

    async def parse(self) -> T:
        """Parse the body into what the plain endpoint function returns."""
        if self._parsed is _UNPARSED:
            self._parsed = await self._parse()
        return self._parsed

    def __repr__(self) -> str:
        return f'{type(self).__name__}(status_code={self.status_code})'
