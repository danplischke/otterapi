"""Deferred query objects for list endpoints (client_style='client' / 'resource').

A list endpoint's method returns a :class:`Query` (or :class:`AsyncQuery`)
instead of a bare list. The endpoint arguments are captured up front; a terminal
method then executes:

    q = client.users.list(status="active")
    rows = q.all()            # list[User]
    df   = q.to_pandas()      # pandas DataFrame
    for u in q.iter(): ...    # streamed, page by page
    q.export("users.csv")     # write to a file

Absent terminals (a feature not enabled at generation time) raise a clear error.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generic, TypeVar

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Iterator

    import pandas as pd
    import polars as pl

T = TypeVar('T')

# The export wrapper adds these keyword-only arguments to an endpoint's
# signature and aliases a same-named endpoint input as ``<name>_``; replayed
# arguments must be renamed the same way (see codegen/export.py). The
# paginator's own knobs (``offset`` / ``page_size`` / ``max_items``) are not
# aliased: the wrapper mirrors them from ``_iter`` under their real names.
_EXPORT_RESERVED = frozenset({'output_path', 'format', 'batch_size'})


def _export_params(params: dict[str, Any]) -> dict[str, Any]:
    return {(k + '_' if k in _EXPORT_RESERVED else k): v for k, v in params.items()}


def _require(name: str, fn: Callable | None) -> Callable:
    if fn is None:
        raise RuntimeError(
            f"'{name}' is not available for this endpoint; enable the "
            'corresponding feature (dataframe / pagination / export) when '
            'generating the client.'
        )
    return fn


class Query(Generic[T]):
    """A deferred synchronous list-endpoint result.

    Construction captures the client and the call arguments; each terminal
    method reuses them. Endpoint methods return these; a hand-written facade
    over the free functions may build one the same way.
    """

    def __init__(
        self,
        client: Any,
        params: dict[str, Any],
        *,
        fetch: Callable[..., list[T]],
        to_pandas: Callable[..., Any] | None = None,
        to_polars: Callable[..., Any] | None = None,
        iterate: Callable[..., Iterator[T]] | None = None,
        export: Callable[..., int] | None = None,
    ) -> None:
        self._client = client
        self._params = params
        self._fetch = fetch
        self._to_pandas = to_pandas
        self._to_polars = to_polars
        self._iterate = iterate
        self._export = export

    def all(self) -> list[T]:
        """Fetch and return the full list."""
        return self._fetch(client=self._client, **self._params)

    def __iter__(self) -> Iterator[T]:
        return iter(self.all())

    def to_pandas(self) -> pd.DataFrame:
        """Fetch and return a pandas ``DataFrame``."""
        return _require('to_pandas', self._to_pandas)(
            client=self._client, **self._params
        )

    def to_polars(self) -> pl.DataFrame:
        """Fetch and return a polars ``DataFrame``."""
        return _require('to_polars', self._to_polars)(
            client=self._client, **self._params
        )

    def iter(self) -> Iterator[T]:
        """Stream items one at a time, fetching pages on demand."""
        return _require('iter', self._iterate)(client=self._client, **self._params)

    def export(
        self, output_path: Any, *, format: str = 'csv', **format_kwargs: Any
    ) -> int:
        """Stream results to a file. Returns the number of rows written."""
        return _require('export', self._export)(
            output_path,
            client=self._client,
            format=format,
            **_export_params(self._params),
            **format_kwargs,
        )


class AsyncQuery(Generic[T]):
    """A deferred asynchronous list-endpoint result (see :class:`Query`)."""

    def __init__(
        self,
        client: Any,
        params: dict[str, Any],
        *,
        fetch: Callable[..., Any],
        to_pandas: Callable[..., Any] | None = None,
        to_polars: Callable[..., Any] | None = None,
        iterate: Callable[..., AsyncIterator[T]] | None = None,
        export: Callable[..., Any] | None = None,
    ) -> None:
        self._client = client
        self._params = params
        self._fetch = fetch
        self._to_pandas = to_pandas
        self._to_polars = to_polars
        self._iterate = iterate
        self._export = export

    async def all(self) -> list[T]:
        """Fetch and return the full list."""
        return await self._fetch(client=self._client, **self._params)

    def __aiter__(self) -> AsyncIterator[T]:
        # Mirrors ``Query.__iter__``: without a pagination terminal, plain
        # ``async for`` iterates the materialized list.
        if self._iterate is None:
            return self._iter_all()
        return self.iter()

    async def _iter_all(self) -> AsyncIterator[T]:
        for item in await self.all():
            yield item

    async def to_pandas(self) -> pd.DataFrame:
        """Fetch and return a pandas ``DataFrame``."""
        return await _require('to_pandas', self._to_pandas)(
            client=self._client, **self._params
        )

    async def to_polars(self) -> pl.DataFrame:
        """Fetch and return a polars ``DataFrame``."""
        return await _require('to_polars', self._to_polars)(
            client=self._client, **self._params
        )

    def iter(self) -> AsyncIterator[T]:
        """Stream items one at a time (consume with ``async for``)."""
        return _require('iter', self._iterate)(client=self._client, **self._params)

    async def export(
        self, output_path: Any, *, format: str = 'csv', **format_kwargs: Any
    ) -> int:
        """Stream results to a file. Returns the number of rows written."""
        return await _require('export', self._export)(
            output_path,
            client=self._client,
            format=format,
            **_export_params(self._params),
            **format_kwargs,
        )
