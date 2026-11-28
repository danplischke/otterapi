"""Pagination utilities for OtterAPI generated clients."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Generator, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

T = TypeVar('T')

logger = logging.getLogger(__name__)


@dataclass
class OffsetPaginationConfig:
    """Configuration for offset-based pagination."""

    offset_param: str = 'offset'
    limit_param: str = 'limit'
    data_path: str | None = None
    total_path: str | None = None


@dataclass
class CursorPaginationConfig:
    """Configuration for cursor-based pagination."""

    cursor_param: str = 'cursor'
    limit_param: str = 'limit'
    data_path: str | None = None
    next_cursor_path: str | None = None


@dataclass
class PagePaginationConfig:
    """Configuration for page-based pagination."""

    page_param: str = 'page'
    per_page_param: str = 'per_page'
    data_path: str | None = None
    total_pages_path: str | None = None


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
# Offset pagination
# ---------------------------------------------------------------------------


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
    """Generic offset-based pagination that returns all items.

    Args:
        fetch_page: Function that fetches a page given (offset, limit).
        extract_items: Function that extracts items from a page response.
        get_total: Optional function to get total count from response.
        start_offset: Starting offset (default: 0).
        page_size: Items per page (default: 100).
        max_items: Maximum items to return (default: unlimited).
        progress: Show a progress bar. Pass ``True`` or a description string.
            Requires ``tqdm``; falls back to a stderr message if unavailable.

    Returns:
        List of all items.
    """
    all_items: list[T] = []
    current_offset = start_offset

    with _progress_context(progress, desc='Fetching') as bar:
        while True:
            if max_items is not None and len(all_items) >= max_items:
                break

            request_limit = page_size
            if max_items is not None:
                remaining = max_items - len(all_items)
                request_limit = min(page_size, remaining)

            page = fetch_page(current_offset, request_limit)
            items = extract_items(page)

            if not items:
                break

            all_items.extend(items)
            if bar is not None:
                bar.update(len(items))

            if len(items) < request_limit:
                break

            if get_total is not None:
                total = get_total(page)
                if total is not None and current_offset + len(items) >= total:
                    break

            current_offset += len(items)

    if max_items is not None and len(all_items) > max_items:
        return all_items[:max_items]

    return all_items


async def paginate_offset_async(
    fetch_page: Callable[[int, int], Any],  # Returns Awaitable[Any]
    extract_items: Callable[[Any], list[T]],
    get_total: Callable[[Any], int | None] | None = None,
    *,
    start_offset: int = 0,
    page_size: int = 100,
    max_items: int | None = None,
    progress: bool | str = True,
) -> list[T]:
    """Async version of paginate_offset."""
    all_items: list[T] = []
    current_offset = start_offset
    bar: Any = None
    if progress:
        try:
            from tqdm.auto import tqdm  # type: ignore[import-untyped]

            label = progress if isinstance(progress, str) else 'Fetching'
            bar = tqdm(desc=label, unit=' items')
        except ImportError:
            pass

    try:
        while True:
            if max_items is not None and len(all_items) >= max_items:
                break

            request_limit = page_size
            if max_items is not None:
                remaining = max_items - len(all_items)
                request_limit = min(page_size, remaining)

            page = await fetch_page(current_offset, request_limit)
            items = extract_items(page)

            if not items:
                break

            all_items.extend(items)
            if bar is not None:
                bar.update(len(items))

            if len(items) < request_limit:
                break

            if get_total is not None:
                total = get_total(page)
                if total is not None and current_offset + len(items) >= total:
                    break

            current_offset += len(items)
    finally:
        if bar is not None:
            bar.close()

    if max_items is not None and len(all_items) > max_items:
        return all_items[:max_items]

    return all_items


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
    """Generic offset-based pagination iterator (streaming).

    Yields items one at a time for memory-efficient processing.

    Args:
        fetch_page: Function that fetches a page given (offset, limit).
        extract_items: Function that extracts items from a page response.
        get_total: Optional function to get total count from response.
        start_offset: Starting offset (default: 0).
        page_size: Items per page (default: 100).
        max_items: Maximum items to yield (default: unlimited).
        progress: Show a progress bar. Pass ``True`` or a description string.

    Yields:
        Items one at a time.
    """
    current_offset = start_offset
    items_yielded = 0

    with _progress_context(progress, desc='Fetching') as bar:
        while True:
            if max_items is not None and items_yielded >= max_items:
                return

            request_limit = page_size
            if max_items is not None:
                remaining = max_items - items_yielded
                request_limit = min(page_size, remaining)

            page = fetch_page(current_offset, request_limit)
            items = extract_items(page)

            if not items:
                return

            for item in items:
                yield item
                items_yielded += 1
                if bar is not None:
                    bar.update(1)

                if max_items is not None and items_yielded >= max_items:
                    return

            if len(items) < request_limit:
                return

            if get_total is not None:
                total = get_total(page)
                if total is not None and current_offset + len(items) >= total:
                    return

            current_offset += len(items)


async def iterate_offset_async(
    fetch_page: Callable[[int, int], Any],  # Returns Awaitable[Any]
    extract_items: Callable[[Any], list[T]],
    get_total: Callable[[Any], int | None] | None = None,
    *,
    start_offset: int = 0,
    page_size: int = 100,
    max_items: int | None = None,
    progress: bool | str = True,
) -> AsyncIterator[T]:
    """Async version of iterate_offset."""
    current_offset = start_offset
    items_yielded = 0
    bar: Any = None
    if progress:
        try:
            from tqdm.auto import tqdm  # type: ignore[import-untyped]

            label = progress if isinstance(progress, str) else 'Fetching'
            bar = tqdm(desc=label, unit=' items')
        except ImportError:
            pass

    try:
        while True:
            if max_items is not None and items_yielded >= max_items:
                return

            request_limit = page_size
            if max_items is not None:
                remaining = max_items - items_yielded
                request_limit = min(page_size, remaining)

            page = await fetch_page(current_offset, request_limit)
            items = extract_items(page)

            if not items:
                return

            for item in items:
                yield item
                items_yielded += 1
                if bar is not None:
                    bar.update(1)

                if max_items is not None and items_yielded >= max_items:
                    return

            if len(items) < request_limit:
                return

            if get_total is not None:
                total = get_total(page)
                if total is not None and current_offset + len(items) >= total:
                    return

            current_offset += len(items)
    finally:
        if bar is not None:
            bar.close()


# ---------------------------------------------------------------------------
# Cursor pagination
# ---------------------------------------------------------------------------


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
    """Generic cursor-based pagination that returns all items.

    Args:
        fetch_page: Function that fetches a page given (cursor, limit).
        extract_items: Function that extracts items from a page response.
        get_next_cursor: Function to get next cursor from response.
        get_total: Optional function reading the total item count; once that
            many items have been fetched, no further page is requested.
        start_cursor: Starting cursor (default: None for first page).
        page_size: Items per page (default: 100).
        max_items: Maximum items to return (default: unlimited).
        progress: Show a progress bar. Pass ``True`` or a description string.

    Returns:
        List of all items.
    """
    all_items: list[T] = []
    current_cursor = start_cursor

    with _progress_context(progress, desc='Fetching') as bar:
        while True:
            if max_items is not None and len(all_items) >= max_items:
                break

            page = fetch_page(current_cursor, page_size)
            items = extract_items(page)

            if not items:
                break

            all_items.extend(items)
            if bar is not None:
                bar.update(len(items))

            if max_items is not None and len(all_items) >= max_items:
                break

            current_cursor = _advance_cursor(
                page, current_cursor, get_next_cursor, get_total, len(all_items)
            )
            if not current_cursor:
                break

    if max_items is not None and len(all_items) > max_items:
        return all_items[:max_items]

    return all_items


async def paginate_cursor_async(
    fetch_page: Callable[[str | None, int], Any],  # Returns Awaitable[Any]
    extract_items: Callable[[Any], list[T]],
    get_next_cursor: Callable[[Any], str | None],
    *,
    get_total: Callable[[Any], int | None] | None = None,
    start_cursor: str | None = None,
    page_size: int = 100,
    max_items: int | None = None,
    progress: bool | str = True,
) -> list[T]:
    """Async version of paginate_cursor."""
    all_items: list[T] = []
    current_cursor = start_cursor
    bar: Any = None
    if progress:
        try:
            from tqdm.auto import tqdm  # type: ignore[import-untyped]

            label = progress if isinstance(progress, str) else 'Fetching'
            bar = tqdm(desc=label, unit=' items')
        except ImportError:
            pass

    try:
        while True:
            if max_items is not None and len(all_items) >= max_items:
                break

            page = await fetch_page(current_cursor, page_size)
            items = extract_items(page)

            if not items:
                break

            all_items.extend(items)
            if bar is not None:
                bar.update(len(items))

            if max_items is not None and len(all_items) >= max_items:
                break

            current_cursor = _advance_cursor(
                page, current_cursor, get_next_cursor, get_total, len(all_items)
            )
            if not current_cursor:
                break
    finally:
        if bar is not None:
            bar.close()

    if max_items is not None and len(all_items) > max_items:
        return all_items[:max_items]

    return all_items


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
    """Generic cursor-based pagination iterator (streaming).

    Args:
        fetch_page: Function that fetches a page given (cursor, limit).
        extract_items: Function that extracts items from a page response.
        get_next_cursor: Function to get next cursor from response.
        get_total: Optional function reading the total item count; once that
            many items have been fetched, no further page is requested.
        start_cursor: Starting cursor (default: None for first page).
        page_size: Items per page (default: 100).
        max_items: Maximum items to yield (default: unlimited).
        progress: Show a progress bar. Pass ``True`` or a description string.

    Yields:
        Items one at a time.
    """
    current_cursor = start_cursor
    items_yielded = 0

    with _progress_context(progress, desc='Fetching') as bar:
        while True:
            if max_items is not None and items_yielded >= max_items:
                return

            page = fetch_page(current_cursor, page_size)
            items = extract_items(page)

            if not items:
                return

            for item in items:
                yield item
                items_yielded += 1
                if bar is not None:
                    bar.update(1)

                if max_items is not None and items_yielded >= max_items:
                    return

            current_cursor = _advance_cursor(
                page, current_cursor, get_next_cursor, get_total, items_yielded
            )
            if not current_cursor:
                return


async def iterate_cursor_async(
    fetch_page: Callable[[str | None, int], Any],  # Returns Awaitable[Any]
    extract_items: Callable[[Any], list[T]],
    get_next_cursor: Callable[[Any], str | None],
    *,
    get_total: Callable[[Any], int | None] | None = None,
    start_cursor: str | None = None,
    page_size: int = 100,
    max_items: int | None = None,
    progress: bool | str = True,
) -> AsyncIterator[T]:
    """Async version of iterate_cursor."""
    current_cursor = start_cursor
    items_yielded = 0
    bar: Any = None
    if progress:
        try:
            from tqdm.auto import tqdm  # type: ignore[import-untyped]

            label = progress if isinstance(progress, str) else 'Fetching'
            bar = tqdm(desc=label, unit=' items')
        except ImportError:
            pass

    try:
        while True:
            if max_items is not None and items_yielded >= max_items:
                return

            page = await fetch_page(current_cursor, page_size)
            items = extract_items(page)

            if not items:
                return

            for item in items:
                yield item
                items_yielded += 1
                if bar is not None:
                    bar.update(1)

                if max_items is not None and items_yielded >= max_items:
                    return

            current_cursor = _advance_cursor(
                page, current_cursor, get_next_cursor, get_total, items_yielded
            )
            if not current_cursor:
                return
    finally:
        if bar is not None:
            bar.close()


# ---------------------------------------------------------------------------
# Page-number pagination
# ---------------------------------------------------------------------------


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
    """Generic page-based pagination iterator (streaming).

    Yields items one at a time for memory-efficient processing.

    Args:
        fetch_page: Function that fetches a page given (page, per_page).
        extract_items: Function that extracts items from a page response.
        get_total_pages: Optional function to get total pages from response.
        start_page: Starting page number (default: 1).
        page_size: Items per page (default: 100).
        max_items: Maximum items to yield (default: unlimited).
        max_pages: Maximum pages to fetch (default: unlimited).
        progress: Show a progress bar. Pass ``True`` or a description string.

    Yields:
        Items one at a time.
    """
    current_page = start_page
    items_yielded = 0
    pages_fetched = 0

    with _progress_context(progress, desc='Fetching') as bar:
        while True:
            if max_items is not None and items_yielded >= max_items:
                return

            if max_pages is not None and pages_fetched >= max_pages:
                return

            page = fetch_page(current_page, page_size)
            items = extract_items(page)
            pages_fetched += 1

            if not items:
                return

            for item in items:
                yield item
                items_yielded += 1
                if bar is not None:
                    bar.update(1)

                if max_items is not None and items_yielded >= max_items:
                    return

            if len(items) < page_size:
                return

            if get_total_pages is not None:
                total_pages = get_total_pages(page)
                if total_pages is not None and current_page >= total_pages:
                    return

            current_page += 1


async def iterate_page_async(
    fetch_page: Callable[[int, int], Any],  # Returns Awaitable[Any]
    extract_items: Callable[[Any], list[T]],
    get_total_pages: Callable[[Any], int | None] | None = None,
    *,
    start_page: int = 1,
    page_size: int = 100,
    max_items: int | None = None,
    max_pages: int | None = None,
    progress: bool | str = True,
) -> AsyncIterator[T]:
    """Async version of iterate_page."""
    current_page = start_page
    items_yielded = 0
    pages_fetched = 0
    bar: Any = None
    if progress:
        try:
            from tqdm.auto import tqdm  # type: ignore[import-untyped]

            label = progress if isinstance(progress, str) else 'Fetching'
            bar = tqdm(desc=label, unit=' items')
        except ImportError:
            pass

    try:
        while True:
            if max_items is not None and items_yielded >= max_items:
                return

            if max_pages is not None and pages_fetched >= max_pages:
                return

            page = await fetch_page(current_page, page_size)
            items = extract_items(page)
            pages_fetched += 1

            if not items:
                return

            for item in items:
                yield item
                items_yielded += 1
                if bar is not None:
                    bar.update(1)

                if max_items is not None and items_yielded >= max_items:
                    return

            if len(items) < page_size:
                return

            if get_total_pages is not None:
                total_pages = get_total_pages(page)
                if total_pages is not None and current_page >= total_pages:
                    return

            current_page += 1
    finally:
        if bar is not None:
            bar.close()


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
    """Generic page-based pagination that returns all items.

    Args:
        fetch_page: Function that fetches a page given (page, per_page).
        extract_items: Function that extracts items from a page response.
        get_total_pages: Optional function to get total pages from response.
        start_page: Starting page number (default: 1).
        page_size: Items per page (default: 100).
        max_items: Maximum items to return (default: unlimited).
        max_pages: Maximum pages to fetch (default: unlimited).
        progress: Show a progress bar. Pass ``True`` or a description string.

    Returns:
        List of all items.
    """
    all_items: list[T] = []
    current_page = start_page
    pages_fetched = 0

    with _progress_context(progress, desc='Fetching') as bar:
        while True:
            if max_items is not None and len(all_items) >= max_items:
                break

            if max_pages is not None and pages_fetched >= max_pages:
                break

            page = fetch_page(current_page, page_size)
            items = extract_items(page)
            pages_fetched += 1

            if not items:
                break

            all_items.extend(items)
            if bar is not None:
                bar.update(len(items))

            if len(items) < page_size:
                break

            if get_total_pages is not None:
                total_pages = get_total_pages(page)
                if total_pages is not None and current_page >= total_pages:
                    break

            current_page += 1

    if max_items is not None and len(all_items) > max_items:
        return all_items[:max_items]

    return all_items


async def paginate_page_async(
    fetch_page: Callable[[int, int], Any],  # Returns Awaitable[Any]
    extract_items: Callable[[Any], list[T]],
    get_total_pages: Callable[[Any], int | None] | None = None,
    *,
    start_page: int = 1,
    page_size: int = 100,
    max_items: int | None = None,
    max_pages: int | None = None,
    progress: bool | str = True,
) -> list[T]:
    """Async version of paginate_page."""
    all_items: list[T] = []
    current_page = start_page
    pages_fetched = 0
    bar: Any = None
    if progress:
        try:
            from tqdm.auto import tqdm  # type: ignore[import-untyped]

            label = progress if isinstance(progress, str) else 'Fetching'
            bar = tqdm(desc=label, unit=' items')
        except ImportError:
            pass

    try:
        while True:
            if max_items is not None and len(all_items) >= max_items:
                break

            if max_pages is not None and pages_fetched >= max_pages:
                break

            page = await fetch_page(current_page, page_size)
            items = extract_items(page)
            pages_fetched += 1

            if not items:
                break

            all_items.extend(items)
            if bar is not None:
                bar.update(len(items))

            if len(items) < page_size:
                break

            if get_total_pages is not None:
                total_pages = get_total_pages(page)
                if total_pages is not None and current_page >= total_pages:
                    break

            current_page += 1
    finally:
        if bar is not None:
            bar.close()

    if max_items is not None and len(all_items) > max_items:
        return all_items[:max_items]

    return all_items


# ---------------------------------------------------------------------------
# Page objects
# ---------------------------------------------------------------------------
#
# The item-level helpers above discard everything but the items. A page keeps
# the rest: the total, where the next page starts (to resume or checkpoint a
# long iteration) and the validated envelope for any other field. The
# next-page decisions mirror the item-level helpers exactly.

E = TypeVar('E')


class Page(Generic[T, E]):
    """One page of a paginated endpoint.

    Attributes:
        items: The items on this page.
        data: The parsed response envelope, for fields beyond the items.
        total: The total item count, when the response reports one.
        total_pages: The total page count (page-number pagination only).
        next_cursor: The cursor of the next page (cursor pagination), or None.
        next_offset: The offset of the next page (offset pagination), or None.
        next_page: The number of the next page (page pagination), or None.

    Pass ``next_cursor`` / ``next_offset`` / ``next_page`` back as the start
    position of the endpoint's ``_page`` function to resume later.
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
        fetch_next: Callable[[], Page[T, E]] | None = None,
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

    def get_next_page(self) -> Page[T, E] | None:
        """Fetch the next page, or return None on the last page."""
        if self._fetch_next is None:
            return None
        return self._fetch_next()

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

    def __iter__(self) -> Iterator[T]:
        return iter(self.items)

    def __len__(self) -> int:
        return len(self.items)

    def __repr__(self) -> str:
        return (
            f'{type(self).__name__}(items={len(self.items)}, total={self.total}, '
            f'has_next_page={self.has_next_page()})'
        )


class AsyncPage(Generic[T, E]):
    """One page of a paginated endpoint, fetched asynchronously (see :class:`Page`)."""

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
        fetch_next: Callable[[], Awaitable[AsyncPage[T, E]]] | None = None,
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

    async def get_next_page(self) -> AsyncPage[T, E] | None:
        """Fetch the next page, or return None on the last page."""
        if self._fetch_next is None:
            return None
        return await self._fetch_next()

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

    def __iter__(self) -> Iterator[T]:
        return iter(self.items)

    def __len__(self) -> int:
        return len(self.items)

    def __repr__(self) -> str:
        return (
            f'{type(self).__name__}(items={len(self.items)}, total={self.total}, '
            f'has_next_page={self.has_next_page()})'
        )


def _page_items(extract_items: Callable[[Any], Any], data: Any) -> list[Any]:
    return list(extract_items(data) or [])


def _maybe(getter: Callable[[Any], Any] | None, data: Any) -> Any:
    return getter(data) if getter is not None else None


def _next_offset(
    offset: int, items: list[Any], page_size: int, total: int | None
) -> int | None:
    """Offset of the page after ``items``, or None (as in ``paginate_offset``)."""
    if not items or len(items) < page_size:
        return None
    following = offset + len(items)
    if total is not None and following >= total:
        return None
    return following


def _next_page_number(
    page: int, items: list[Any], page_size: int, total_pages: int | None
) -> int | None:
    """Number of the page after ``items``, or None (as in ``paginate_page``)."""
    if not items or len(items) < page_size:
        return None
    if total_pages is not None and page >= total_pages:
        return None
    return page + 1


def fetch_offset_page(
    fetch_page: Callable[[int, int], Any],
    extract_items: Callable[[Any], list[T]],
    get_total: Callable[[Any], int | None] | None = None,
    *,
    offset: int = 0,
    page_size: int = 100,
) -> Page[T, Any]:
    """Fetch one offset-paginated page."""
    data = fetch_page(offset, page_size)
    items = _page_items(extract_items, data)
    total = _maybe(get_total, data)
    following = _next_offset(offset, items, page_size, total)
    fetch_next: Callable[[], Page[T, Any]] | None = None
    if following is not None:
        start = following

        def _fetch_next() -> Page[T, Any]:
            return fetch_offset_page(
                fetch_page,
                extract_items,
                get_total,
                offset=start,
                page_size=page_size,
            )

        fetch_next = _fetch_next

    return Page(items, data, total=total, next_offset=following, fetch_next=fetch_next)


async def fetch_offset_page_async(
    fetch_page: Callable[[int, int], Awaitable[Any]],
    extract_items: Callable[[Any], list[T]],
    get_total: Callable[[Any], int | None] | None = None,
    *,
    offset: int = 0,
    page_size: int = 100,
) -> AsyncPage[T, Any]:
    """Async version of :func:`fetch_offset_page`."""
    data = await fetch_page(offset, page_size)
    items = _page_items(extract_items, data)
    total = _maybe(get_total, data)
    following = _next_offset(offset, items, page_size, total)
    fetch_next: Callable[[], Awaitable[AsyncPage[T, Any]]] | None = None
    if following is not None:
        start = following

        async def _fetch_next() -> AsyncPage[T, Any]:
            return await fetch_offset_page_async(
                fetch_page,
                extract_items,
                get_total,
                offset=start,
                page_size=page_size,
            )

        fetch_next = _fetch_next

    return AsyncPage(
        items, data, total=total, next_offset=following, fetch_next=fetch_next
    )


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
    items = _page_items(extract_items, data)
    fetched = _fetched + len(items)
    following = (
        _advance_cursor(data, cursor, get_next_cursor, get_total, fetched)
        if items
        else None
    )
    fetch_next: Callable[[], Page[T, Any]] | None = None
    if following is not None:
        start = following

        def _fetch_next() -> Page[T, Any]:
            return fetch_cursor_page(
                fetch_page,
                extract_items,
                get_next_cursor,
                get_total=get_total,
                cursor=start,
                page_size=page_size,
                _fetched=fetched,
            )

        fetch_next = _fetch_next

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
    items = _page_items(extract_items, data)
    fetched = _fetched + len(items)
    following = (
        _advance_cursor(data, cursor, get_next_cursor, get_total, fetched)
        if items
        else None
    )
    fetch_next: Callable[[], Awaitable[AsyncPage[T, Any]]] | None = None
    if following is not None:
        start = following

        async def _fetch_next() -> AsyncPage[T, Any]:
            return await fetch_cursor_page_async(
                fetch_page,
                extract_items,
                get_next_cursor,
                get_total=get_total,
                cursor=start,
                page_size=page_size,
                _fetched=fetched,
            )

        fetch_next = _fetch_next

    return AsyncPage(
        items,
        data,
        total=_maybe(get_total, data),
        next_cursor=following,
        fetch_next=fetch_next,
    )


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
    items = _page_items(extract_items, data)
    total_pages = _maybe(get_total_pages, data)
    following = _next_page_number(page, items, page_size, total_pages)
    fetch_next: Callable[[], Page[T, Any]] | None = None
    if following is not None:
        start = following

        def _fetch_next() -> Page[T, Any]:
            return fetch_numbered_page(
                fetch_page,
                extract_items,
                get_total_pages,
                get_total=get_total,
                page=start,
                page_size=page_size,
            )

        fetch_next = _fetch_next

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
    items = _page_items(extract_items, data)
    total_pages = _maybe(get_total_pages, data)
    following = _next_page_number(page, items, page_size, total_pages)
    fetch_next: Callable[[], Awaitable[AsyncPage[T, Any]]] | None = None
    if following is not None:
        start = following

        async def _fetch_next() -> AsyncPage[T, Any]:
            return await fetch_numbered_page_async(
                fetch_page,
                extract_items,
                get_total_pages,
                get_total=get_total,
                page=start,
                page_size=page_size,
            )

        fetch_next = _fetch_next

    return AsyncPage(
        items,
        data,
        total=_maybe(get_total, data),
        total_pages=total_pages,
        next_page=following,
        fetch_next=fetch_next,
    )
