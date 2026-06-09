"""Pagination utilities for OtterAPI generated clients."""

from collections.abc import AsyncIterator, Callable, Generator, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, TypeVar

T = TypeVar('T')
PageT = TypeVar('PageT')


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
                    f"Available keys: {list(current.keys())}. Full path: {path}"
                )
            current = current[key]
        elif isinstance(current, list) and key.isdigit():
            current = current[int(key)]
        else:
            raise KeyError(
                f"Cannot access '{key}' on {type(current).__name__}. "
                f"Full path: {path}"
            )

    return current


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
    fetch_page: Callable[[int, int], PageT],
    extract_items: Callable[[PageT], list[T]],
    get_total: Callable[[PageT], int | None] | None = None,
    *,
    start_offset: int = 0,
    page_size: int = 100,
    max_items: int | None = None,
    progress: bool | str = False,
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
    fetch_page: Callable[[int, int], Any],  # Returns Awaitable[PageT]
    extract_items: Callable[[PageT], list[T]],
    get_total: Callable[[PageT], int | None] | None = None,
    *,
    start_offset: int = 0,
    page_size: int = 100,
    max_items: int | None = None,
    progress: bool | str = False,
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
    fetch_page: Callable[[int, int], PageT],
    extract_items: Callable[[PageT], list[T]],
    get_total: Callable[[PageT], int | None] | None = None,
    *,
    start_offset: int = 0,
    page_size: int = 100,
    max_items: int | None = None,
    progress: bool | str = False,
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
    fetch_page: Callable[[int, int], Any],  # Returns Awaitable[PageT]
    extract_items: Callable[[PageT], list[T]],
    get_total: Callable[[PageT], int | None] | None = None,
    *,
    start_offset: int = 0,
    page_size: int = 100,
    max_items: int | None = None,
    progress: bool | str = False,
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
    fetch_page: Callable[[str | None, int], PageT],
    extract_items: Callable[[PageT], list[T]],
    get_next_cursor: Callable[[PageT], str | None],
    *,
    start_cursor: str | None = None,
    page_size: int = 100,
    max_items: int | None = None,
    progress: bool | str = False,
) -> list[T]:
    """Generic cursor-based pagination that returns all items.

    Args:
        fetch_page: Function that fetches a page given (cursor, limit).
        extract_items: Function that extracts items from a page response.
        get_next_cursor: Function to get next cursor from response.
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

            current_cursor = get_next_cursor(page)
            if not current_cursor:
                break

    if max_items is not None and len(all_items) > max_items:
        return all_items[:max_items]

    return all_items


async def paginate_cursor_async(
    fetch_page: Callable[[str | None, int], Any],  # Returns Awaitable[PageT]
    extract_items: Callable[[PageT], list[T]],
    get_next_cursor: Callable[[PageT], str | None],
    *,
    start_cursor: str | None = None,
    page_size: int = 100,
    max_items: int | None = None,
    progress: bool | str = False,
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

            current_cursor = get_next_cursor(page)
            if not current_cursor:
                break
    finally:
        if bar is not None:
            bar.close()

    if max_items is not None and len(all_items) > max_items:
        return all_items[:max_items]

    return all_items


def iterate_cursor(
    fetch_page: Callable[[str | None, int], PageT],
    extract_items: Callable[[PageT], list[T]],
    get_next_cursor: Callable[[PageT], str | None],
    *,
    start_cursor: str | None = None,
    page_size: int = 100,
    max_items: int | None = None,
    progress: bool | str = False,
) -> Iterator[T]:
    """Generic cursor-based pagination iterator (streaming).

    Args:
        fetch_page: Function that fetches a page given (cursor, limit).
        extract_items: Function that extracts items from a page response.
        get_next_cursor: Function to get next cursor from response.
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

            current_cursor = get_next_cursor(page)
            if not current_cursor:
                return


async def iterate_cursor_async(
    fetch_page: Callable[[str | None, int], Any],  # Returns Awaitable[PageT]
    extract_items: Callable[[PageT], list[T]],
    get_next_cursor: Callable[[PageT], str | None],
    *,
    start_cursor: str | None = None,
    page_size: int = 100,
    max_items: int | None = None,
    progress: bool | str = False,
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

            current_cursor = get_next_cursor(page)
            if not current_cursor:
                return
    finally:
        if bar is not None:
            bar.close()


# ---------------------------------------------------------------------------
# Page-number pagination
# ---------------------------------------------------------------------------


def iterate_page(
    fetch_page: Callable[[int, int], PageT],
    extract_items: Callable[[PageT], list[T]],
    get_total_pages: Callable[[PageT], int | None] | None = None,
    *,
    start_page: int = 1,
    page_size: int = 100,
    max_items: int | None = None,
    max_pages: int | None = None,
    progress: bool | str = False,
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
    fetch_page: Callable[[int, int], Any],  # Returns Awaitable[PageT]
    extract_items: Callable[[PageT], list[T]],
    get_total_pages: Callable[[PageT], int | None] | None = None,
    *,
    start_page: int = 1,
    page_size: int = 100,
    max_items: int | None = None,
    max_pages: int | None = None,
    progress: bool | str = False,
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
    fetch_page: Callable[[int, int], PageT],
    extract_items: Callable[[PageT], list[T]],
    get_total_pages: Callable[[PageT], int | None] | None = None,
    *,
    start_page: int = 1,
    page_size: int = 100,
    max_items: int | None = None,
    max_pages: int | None = None,
    progress: bool | str = False,
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
    fetch_page: Callable[[int, int], Any],  # Returns Awaitable[PageT]
    extract_items: Callable[[PageT], list[T]],
    get_total_pages: Callable[[PageT], int | None] | None = None,
    *,
    start_page: int = 1,
    page_size: int = 100,
    max_items: int | None = None,
    max_pages: int | None = None,
    progress: bool | str = False,
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
