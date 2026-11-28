"""Pagination utilities for OtterAPI generated clients."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Generator, Iterator
from contextlib import contextmanager
from functools import partial
from typing import Any, Generic, TypeVar

T = TypeVar('T')
E = TypeVar('E')

logger = logging.getLogger(__name__)


def extract_path(data: dict | list, path: str | None) -> Any:
    """Extract nested data using dot notation path.

    Args:
        data: The response data (dict or list).
        path: Dot notation path (e.g., "data.users").

    Returns:
        The extracted data at the specified path.

    Raises:
        KeyError: If the path does not exist in the data.

    Examples:
        >>> extract_path({"data": {"users": [1, 2, 3]}}, "data.users")
        [1, 2, 3]
        >>> extract_path([1, 2, 3], None)
        [1, 2, 3]
    """
    if path is None:
        return data

    current = data
    for key in path.split('.'):
        if isinstance(current, dict):
            if key not in current:
                raise KeyError(
                    f"Key '{key}' not found in response. "
                    f'Available keys: {list(current.keys())}. Full path: {path}'
                )
            current = current[key]
        elif isinstance(current, list) and key.isdigit():
            current = current[int(key)]
        elif current is not None and hasattr(current, key):
            # A parsed response model: walk its attributes.
            current = getattr(current, key)
        else:
            raise KeyError(
                f"Cannot access '{key}' on {type(current).__name__}. Full path: {path}"
            )

    return current


def merge_body_fields(body: Any, fields: dict[str, Any]) -> dict[str, Any]:
    """Overlay pagination fields onto a dumped JSON request body.

    ``None`` values are skipped, so the first request keeps whatever position
    the caller put in the body themselves (e.g. a saved ``after``) unless a
    start position was passed explicitly.
    """
    if body is None:
        merged: dict[str, Any] = {}
    elif isinstance(body, dict):
        merged = dict(body)
    else:
        raise TypeError(
            'Body pagination needs a JSON object request body, got '
            f'{type(body).__name__}'
        )
    for key, value in fields.items():
        if value is not None:
            merged[key] = value
    return merged


def last_item_value(items: list[Any], path: str) -> Any:
    """``path`` on the last of ``items`` (None for an empty page).

    For APIs whose pages carry no next cursor and instead expect the last
    row's own cursor back as the next request's ``after``.
    """
    if not items:
        return None
    return extract_path(items[-1], path)


def _advance_cursor(
    page: Any,
    current: str | None,
    get_next_cursor: Callable[[Any], str | None],
    get_total: Callable[[Any], int | None] | None,
    fetched: int,
) -> str | None:
    """The cursor for the next request, or None to stop.

    Stops once ``total`` items have been fetched (saving the request for an
    empty last page), when there is no next cursor, and when the server hands
    back the cursor it was just sent -- a server ignoring the cursor would
    otherwise be paged forever.
    """
    if get_total is not None:
        total = get_total(page)
        if total is not None and fetched >= total:
            return None
    next_cursor = get_next_cursor(page)
    if not next_cursor:
        return None
    if next_cursor == current:
        logger.warning(
            'Stopping pagination: the server returned the same cursor (%r) '
            'it was sent, so the next page would repeat this one.',
            next_cursor,
        )
        return None
    return next_cursor


# ---------------------------------------------------------------------------
# Progress helpers
# ---------------------------------------------------------------------------


@contextmanager
def _progress_context(
    progress: bool | str,
    desc: str = '',
    total: int | None = None,
) -> Generator[Any, None, None]:
    """Context manager that yields a tqdm bar (or None when disabled/unavailable)."""
    if not progress:
        yield None
        return
    label = progress if isinstance(progress, str) else desc
    try:
        from tqdm.auto import tqdm  # type: ignore[import-untyped]

        bar = tqdm(desc=label, total=total, unit=' items')
        try:
            yield bar
        finally:
            bar.close()
    except ImportError:
        import sys

        if label:
            print(f'{label}...', file=sys.stderr, flush=True)
        yield None


# ---------------------------------------------------------------------------
# Page objects
# ---------------------------------------------------------------------------
#
# Every style is implemented once, as a page fetcher: it requests one page and
# decides where the next one starts. The item-level helpers further down are
# thin drivers over the fetchers, so the stop rules live in one place.


class _PageBase(Generic[T, E]):
    """Fields shared by :class:`Page` and :class:`AsyncPage`.

    Attributes:
        items: The items on this page.
        data: The parsed response envelope, for fields beyond the items.
        total: The total item count, when the response reports one.
        total_pages: The total page count (page-number pagination only).
        next_cursor / next_offset / next_page: Where the next page starts
            (pass it back as the start position to resume), or None.
    """

    def __init__(
        self,
        items: list[T],
        data: E,
        *,
        total: int | None = None,
        total_pages: int | None = None,
        next_cursor: str | None = None,
        next_offset: int | None = None,
        next_page: int | None = None,
        fetch_next: Callable[[], Any] | None = None,
    ) -> None:
        self.items = items
        self.data = data
        self.total = total
        self.total_pages = total_pages
        self.next_cursor = next_cursor
        self.next_offset = next_offset
        self.next_page = next_page
        self._fetch_next = fetch_next

    def has_next_page(self) -> bool:
        """Whether another page follows this one."""
        return self._fetch_next is not None

    def __iter__(self) -> Iterator[T]:
        return iter(self.items)

    def __len__(self) -> int:
        return len(self.items)

    def __repr__(self) -> str:
        return (
            f'{type(self).__name__}(items={len(self.items)}, total={self.total}, '
            f'has_next_page={self.has_next_page()})'
        )


class Page(_PageBase[T, E]):
    """One page of a paginated endpoint (see :class:`_PageBase` for fields)."""

    def get_next_page(self) -> Page[T, E] | None:
        """Fetch the next page, or return None on the last page."""
        return self._fetch_next() if self._fetch_next is not None else None

    def iter_pages(self, *, max_pages: int | None = None) -> Iterator[Page[T, E]]:
        """Yield this page and every page after it, fetching on demand."""
        page: Page[T, E] | None = self
        count = 0
        while page is not None:
            yield page
            count += 1
            if max_pages is not None and count >= max_pages:
                return
            page = page.get_next_page()


class AsyncPage(_PageBase[T, E]):
    """One page of a paginated endpoint, fetched asynchronously."""

    async def get_next_page(self) -> AsyncPage[T, E] | None:
        """Fetch the next page, or return None on the last page."""
        return await self._fetch_next() if self._fetch_next is not None else None

    async def iter_pages(
        self, *, max_pages: int | None = None
    ) -> AsyncIterator[AsyncPage[T, E]]:
        """Yield this page and every page after it (``async for``)."""
        page: AsyncPage[T, E] | None = self
        count = 0
        while page is not None:
            yield page
            count += 1
            if max_pages is not None and count >= max_pages:
                return
            page = await page.get_next_page()


def _maybe(getter: Callable[[Any], Any] | None, data: Any) -> Any:
    return getter(data) if getter is not None else None


def _offset_step(
    data: Any,
    extract_items: Callable[[Any], Any],
    get_total: Callable[[Any], int | None] | None,
    offset: int,
    limit: int,
) -> tuple[list[Any], int | None, int | None]:
    """``(items, total, next offset or None)`` for one offset page."""
    items = list(extract_items(data) or [])
    total = _maybe(get_total, data)
    following: int | None = offset + len(items)
    if len(items) < max(limit, 1) or (total is not None and following >= total):
        following = None
    return items, total, following


def _request_limit(page_size: int, budget: int | None) -> int:
    return page_size if budget is None else min(page_size, budget)


def fetch_offset_page(
    fetch_page: Callable[[int, int], Any],
    extract_items: Callable[[Any], list[T]],
    get_total: Callable[[Any], int | None] | None = None,
    *,
    offset: int = 0,
    page_size: int = 100,
    _budget: int | None = None,
) -> Page[T, Any]:
    """Fetch one offset-paginated page.

    ``_budget`` (used by the item-level helpers' ``max_items``) caps how many
    more items to request, so the last request asks only for what is left.
    """
    limit = _request_limit(page_size, _budget)
    data = fetch_page(offset, limit)
    items, total, following = _offset_step(
        data, extract_items, get_total, offset, limit
    )
    budget = None if _budget is None else _budget - len(items)
    if budget is not None and budget <= 0:
        following = None
    fetch_next = None
    if following is not None:
        fetch_next = partial(
            fetch_offset_page,
            fetch_page,
            extract_items,
            get_total,
            offset=following,
            page_size=page_size,
            _budget=budget,
        )
    return Page(items, data, total=total, next_offset=following, fetch_next=fetch_next)


async def fetch_offset_page_async(
    fetch_page: Callable[[int, int], Awaitable[Any]],
    extract_items: Callable[[Any], list[T]],
    get_total: Callable[[Any], int | None] | None = None,
    *,
    offset: int = 0,
    page_size: int = 100,
    _budget: int | None = None,
) -> AsyncPage[T, Any]:
    """Async version of :func:`fetch_offset_page`."""
    limit = _request_limit(page_size, _budget)
    data = await fetch_page(offset, limit)
    items, total, following = _offset_step(
        data, extract_items, get_total, offset, limit
    )
    budget = None if _budget is None else _budget - len(items)
    if budget is not None and budget <= 0:
        following = None
    fetch_next = None
    if following is not None:
        fetch_next = partial(
            fetch_offset_page_async,
            fetch_page,
            extract_items,
            get_total,
            offset=following,
            page_size=page_size,
            _budget=budget,
        )
    return AsyncPage(
        items, data, total=total, next_offset=following, fetch_next=fetch_next
    )


def _cursor_step(
    data: Any,
    extract_items: Callable[[Any], Any],
    get_next_cursor: Callable[[Any], str | None],
    get_total: Callable[[Any], int | None] | None,
    cursor: str | None,
    fetched_before: int,
) -> tuple[list[Any], int, str | None]:
    """``(items, fetched so far, next cursor or None)`` for one cursor page."""
    items = list(extract_items(data) or [])
    fetched = fetched_before + len(items)
    following = (
        _advance_cursor(data, cursor, get_next_cursor, get_total, fetched)
        if items
        else None
    )
    return items, fetched, following


def fetch_cursor_page(
    fetch_page: Callable[[str | None, int], Any],
    extract_items: Callable[[Any], list[T]],
    get_next_cursor: Callable[[Any], str | None],
    *,
    get_total: Callable[[Any], int | None] | None = None,
    cursor: str | None = None,
    page_size: int = 100,
    _fetched: int = 0,
) -> Page[T, Any]:
    """Fetch one cursor-paginated page."""
    data = fetch_page(cursor, page_size)
    items, fetched, following = _cursor_step(
        data, extract_items, get_next_cursor, get_total, cursor, _fetched
    )
    fetch_next = None
    if following is not None:
        fetch_next = partial(
            fetch_cursor_page,
            fetch_page,
            extract_items,
            get_next_cursor,
            get_total=get_total,
            cursor=following,
            page_size=page_size,
            _fetched=fetched,
        )
    return Page(
        items,
        data,
        total=_maybe(get_total, data),
        next_cursor=following,
        fetch_next=fetch_next,
    )


async def fetch_cursor_page_async(
    fetch_page: Callable[[str | None, int], Awaitable[Any]],
    extract_items: Callable[[Any], list[T]],
    get_next_cursor: Callable[[Any], str | None],
    *,
    get_total: Callable[[Any], int | None] | None = None,
    cursor: str | None = None,
    page_size: int = 100,
    _fetched: int = 0,
) -> AsyncPage[T, Any]:
    """Async version of :func:`fetch_cursor_page`."""
    data = await fetch_page(cursor, page_size)
    items, fetched, following = _cursor_step(
        data, extract_items, get_next_cursor, get_total, cursor, _fetched
    )
    fetch_next = None
    if following is not None:
        fetch_next = partial(
            fetch_cursor_page_async,
            fetch_page,
            extract_items,
            get_next_cursor,
            get_total=get_total,
            cursor=following,
            page_size=page_size,
            _fetched=fetched,
        )
    return AsyncPage(
        items,
        data,
        total=_maybe(get_total, data),
        next_cursor=following,
        fetch_next=fetch_next,
    )


def _numbered_step(
    data: Any,
    extract_items: Callable[[Any], Any],
    get_total_pages: Callable[[Any], int | None] | None,
    page: int,
    page_size: int,
) -> tuple[list[Any], int | None, int | None]:
    """``(items, total pages, next page number or None)`` for one page."""
    items = list(extract_items(data) or [])
    total_pages = _maybe(get_total_pages, data)
    following: int | None = page + 1
    if len(items) < max(page_size, 1) or (
        total_pages is not None and page >= total_pages
    ):
        following = None
    return items, total_pages, following


def fetch_numbered_page(
    fetch_page: Callable[[int, int], Any],
    extract_items: Callable[[Any], list[T]],
    get_total_pages: Callable[[Any], int | None] | None = None,
    *,
    get_total: Callable[[Any], int | None] | None = None,
    page: int = 1,
    page_size: int = 100,
) -> Page[T, Any]:
    """Fetch one page of a page-number-paginated endpoint."""
    data = fetch_page(page, page_size)
    items, total_pages, following = _numbered_step(
        data, extract_items, get_total_pages, page, page_size
    )
    fetch_next = None
    if following is not None:
        fetch_next = partial(
            fetch_numbered_page,
            fetch_page,
            extract_items,
            get_total_pages,
            get_total=get_total,
            page=following,
            page_size=page_size,
        )
    return Page(
        items,
        data,
        total=_maybe(get_total, data),
        total_pages=total_pages,
        next_page=following,
        fetch_next=fetch_next,
    )


async def fetch_numbered_page_async(
    fetch_page: Callable[[int, int], Awaitable[Any]],
    extract_items: Callable[[Any], list[T]],
    get_total_pages: Callable[[Any], int | None] | None = None,
    *,
    get_total: Callable[[Any], int | None] | None = None,
    page: int = 1,
    page_size: int = 100,
) -> AsyncPage[T, Any]:
    """Async version of :func:`fetch_numbered_page`."""
    data = await fetch_page(page, page_size)
    items, total_pages, following = _numbered_step(
        data, extract_items, get_total_pages, page, page_size
    )
    fetch_next = None
    if following is not None:
        fetch_next = partial(
            fetch_numbered_page_async,
            fetch_page,
            extract_items,
            get_total_pages,
            get_total=get_total,
            page=following,
            page_size=page_size,
        )
    return AsyncPage(
        items,
        data,
        total=_maybe(get_total, data),
        total_pages=total_pages,
        next_page=following,
        fetch_next=fetch_next,
    )


# ---------------------------------------------------------------------------
# Item-level helpers: drivers over the page fetchers
# ---------------------------------------------------------------------------
#
# ``iterate_*`` stream items; ``paginate_*`` collect them. ``max_items`` stops
# without requesting another page, and ``max_pages`` caps the requests.


def _iter_items(
    first_page: Callable[[], Page[T, Any]],
    max_items: int | None,
    progress: bool | str,
    max_pages: int | None = None,
) -> Iterator[T]:
    if max_items is not None and max_items <= 0:
        return
    yielded = 0
    with _progress_context(progress, desc='Fetching') as bar:
        for page in first_page().iter_pages(max_pages=max_pages):
            for item in page.items:
                yield item
                yielded += 1
                if bar is not None:
                    bar.update(1)
                if max_items is not None and yielded >= max_items:
                    return


async def _aiter_items(
    first_page: Callable[[], Awaitable[AsyncPage[T, Any]]],
    max_items: int | None,
    progress: bool | str,
    max_pages: int | None = None,
) -> AsyncIterator[T]:
    if max_items is not None and max_items <= 0:
        return
    yielded = 0
    with _progress_context(progress, desc='Fetching') as bar:
        page = await first_page()
        async for current in page.iter_pages(max_pages=max_pages):
            for item in current.items:
                yield item
                yielded += 1
                if bar is not None:
                    bar.update(1)
                if max_items is not None and yielded >= max_items:
                    return


async def _collect(items: AsyncIterator[T]) -> list[T]:
    return [item async for item in items]


def iterate_offset(
    fetch_page: Callable[[int, int], Any],
    extract_items: Callable[[Any], list[T]],
    get_total: Callable[[Any], int | None] | None = None,
    *,
    start_offset: int = 0,
    page_size: int = 100,
    max_items: int | None = None,
    progress: bool | str = True,
) -> Iterator[T]:
    """Stream the items of an offset-paginated endpoint.

    ``fetch_page(offset, limit)`` requests one page and ``extract_items``
    reads the items from it; ``get_total`` (optional) reads the total count,
    which saves the request for an empty last page.
    """
    return _iter_items(
        partial(
            fetch_offset_page,
            fetch_page,
            extract_items,
            get_total,
            offset=start_offset,
            page_size=page_size,
            _budget=max_items,
        ),
        max_items,
        progress,
    )


def iterate_offset_async(
    fetch_page: Callable[[int, int], Awaitable[Any]],
    extract_items: Callable[[Any], list[T]],
    get_total: Callable[[Any], int | None] | None = None,
    *,
    start_offset: int = 0,
    page_size: int = 100,
    max_items: int | None = None,
    progress: bool | str = True,
) -> AsyncIterator[T]:
    """Async version of :func:`iterate_offset` (consume with ``async for``)."""
    return _aiter_items(
        partial(
            fetch_offset_page_async,
            fetch_page,
            extract_items,
            get_total,
            offset=start_offset,
            page_size=page_size,
            _budget=max_items,
        ),
        max_items,
        progress,
    )


def paginate_offset(
    fetch_page: Callable[[int, int], Any],
    extract_items: Callable[[Any], list[T]],
    get_total: Callable[[Any], int | None] | None = None,
    *,
    start_offset: int = 0,
    page_size: int = 100,
    max_items: int | None = None,
    progress: bool | str = True,
) -> list[T]:
    """Collect every item of an offset-paginated endpoint (see :func:`iterate_offset`)."""
    return list(
        iterate_offset(
            fetch_page,
            extract_items,
            get_total,
            start_offset=start_offset,
            page_size=page_size,
            max_items=max_items,
            progress=progress,
        )
    )


async def paginate_offset_async(
    fetch_page: Callable[[int, int], Awaitable[Any]],
    extract_items: Callable[[Any], list[T]],
    get_total: Callable[[Any], int | None] | None = None,
    *,
    start_offset: int = 0,
    page_size: int = 100,
    max_items: int | None = None,
    progress: bool | str = True,
) -> list[T]:
    """Async version of :func:`paginate_offset`."""
    return await _collect(
        iterate_offset_async(
            fetch_page,
            extract_items,
            get_total,
            start_offset=start_offset,
            page_size=page_size,
            max_items=max_items,
            progress=progress,
        )
    )


def iterate_cursor(
    fetch_page: Callable[[str | None, int], Any],
    extract_items: Callable[[Any], list[T]],
    get_next_cursor: Callable[[Any], str | None],
    *,
    get_total: Callable[[Any], int | None] | None = None,
    start_cursor: str | None = None,
    page_size: int = 100,
    max_items: int | None = None,
    progress: bool | str = True,
) -> Iterator[T]:
    """Stream the items of a cursor-paginated endpoint.

    ``fetch_page(cursor, limit)`` requests one page and ``get_next_cursor``
    reads the next page's cursor from it. Paging stops on an empty page, once
    ``get_total`` items were fetched, or when the cursor stops advancing.
    """
    return _iter_items(
        partial(
            fetch_cursor_page,
            fetch_page,
            extract_items,
            get_next_cursor,
            get_total=get_total,
            cursor=start_cursor,
            page_size=page_size,
        ),
        max_items,
        progress,
    )


def iterate_cursor_async(
    fetch_page: Callable[[str | None, int], Awaitable[Any]],
    extract_items: Callable[[Any], list[T]],
    get_next_cursor: Callable[[Any], str | None],
    *,
    get_total: Callable[[Any], int | None] | None = None,
    start_cursor: str | None = None,
    page_size: int = 100,
    max_items: int | None = None,
    progress: bool | str = True,
) -> AsyncIterator[T]:
    """Async version of :func:`iterate_cursor` (consume with ``async for``)."""
    return _aiter_items(
        partial(
            fetch_cursor_page_async,
            fetch_page,
            extract_items,
            get_next_cursor,
            get_total=get_total,
            cursor=start_cursor,
            page_size=page_size,
        ),
        max_items,
        progress,
    )


def paginate_cursor(
    fetch_page: Callable[[str | None, int], Any],
    extract_items: Callable[[Any], list[T]],
    get_next_cursor: Callable[[Any], str | None],
    *,
    get_total: Callable[[Any], int | None] | None = None,
    start_cursor: str | None = None,
    page_size: int = 100,
    max_items: int | None = None,
    progress: bool | str = True,
) -> list[T]:
    """Collect every item of a cursor-paginated endpoint (see :func:`iterate_cursor`)."""
    return list(
        iterate_cursor(
            fetch_page,
            extract_items,
            get_next_cursor,
            get_total=get_total,
            start_cursor=start_cursor,
            page_size=page_size,
            max_items=max_items,
            progress=progress,
        )
    )


async def paginate_cursor_async(
    fetch_page: Callable[[str | None, int], Awaitable[Any]],
    extract_items: Callable[[Any], list[T]],
    get_next_cursor: Callable[[Any], str | None],
    *,
    get_total: Callable[[Any], int | None] | None = None,
    start_cursor: str | None = None,
    page_size: int = 100,
    max_items: int | None = None,
    progress: bool | str = True,
) -> list[T]:
    """Async version of :func:`paginate_cursor`."""
    return await _collect(
        iterate_cursor_async(
            fetch_page,
            extract_items,
            get_next_cursor,
            get_total=get_total,
            start_cursor=start_cursor,
            page_size=page_size,
            max_items=max_items,
            progress=progress,
        )
    )


def iterate_page(
    fetch_page: Callable[[int, int], Any],
    extract_items: Callable[[Any], list[T]],
    get_total_pages: Callable[[Any], int | None] | None = None,
    *,
    start_page: int = 1,
    page_size: int = 100,
    max_items: int | None = None,
    max_pages: int | None = None,
    progress: bool | str = True,
) -> Iterator[T]:
    """Stream the items of a page-number-paginated endpoint.

    ``fetch_page(page, per_page)`` requests one page; ``get_total_pages``
    (optional) reads the page count.
    """
    return _iter_items(
        partial(
            fetch_numbered_page,
            fetch_page,
            extract_items,
            get_total_pages,
            page=start_page,
            page_size=page_size,
        ),
        max_items,
        progress,
        max_pages,
    )


def iterate_page_async(
    fetch_page: Callable[[int, int], Awaitable[Any]],
    extract_items: Callable[[Any], list[T]],
    get_total_pages: Callable[[Any], int | None] | None = None,
    *,
    start_page: int = 1,
    page_size: int = 100,
    max_items: int | None = None,
    max_pages: int | None = None,
    progress: bool | str = True,
) -> AsyncIterator[T]:
    """Async version of :func:`iterate_page` (consume with ``async for``)."""
    return _aiter_items(
        partial(
            fetch_numbered_page_async,
            fetch_page,
            extract_items,
            get_total_pages,
            page=start_page,
            page_size=page_size,
        ),
        max_items,
        progress,
        max_pages,
    )


def paginate_page(
    fetch_page: Callable[[int, int], Any],
    extract_items: Callable[[Any], list[T]],
    get_total_pages: Callable[[Any], int | None] | None = None,
    *,
    start_page: int = 1,
    page_size: int = 100,
    max_items: int | None = None,
    max_pages: int | None = None,
    progress: bool | str = True,
) -> list[T]:
    """Collect every item of a page-number-paginated endpoint (see :func:`iterate_page`)."""
    return list(
        iterate_page(
            fetch_page,
            extract_items,
            get_total_pages,
            start_page=start_page,
            page_size=page_size,
            max_items=max_items,
            max_pages=max_pages,
            progress=progress,
        )
    )


async def paginate_page_async(
    fetch_page: Callable[[int, int], Awaitable[Any]],
    extract_items: Callable[[Any], list[T]],
    get_total_pages: Callable[[Any], int | None] | None = None,
    *,
    start_page: int = 1,
    page_size: int = 100,
    max_items: int | None = None,
    max_pages: int | None = None,
    progress: bool | str = True,
) -> list[T]:
    """Async version of :func:`paginate_page`."""
    return await _collect(
        iterate_page_async(
            fetch_page,
            extract_items,
            get_total_pages,
            start_page=start_page,
            page_size=page_size,
            max_items=max_items,
            max_pages=max_pages,
            progress=progress,
        )
    )
