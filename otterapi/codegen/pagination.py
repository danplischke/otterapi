"""Pagination utilities for OtterAPI code generation.

This module provides utilities for:
- Generating the _pagination.py utility file for runtime pagination
- Determining pagination configuration for endpoints
- Building paginated endpoint functions
"""

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from upath import UPath

if TYPE_CHECKING:
    from otterapi.codegen.types import Type
    from otterapi.config import PaginationConfig

__all__ = [
    'PAGINATION_MODULE_CONTENT',
    'PaginationMethodConfig',
    'generate_pagination_module',
    'get_pagination_imports',
    'get_pagination_type_checking_imports',
    'get_pagination_config_for_endpoint',
    'endpoint_is_paginated',
]


# =============================================================================
# Pagination Module Content
# =============================================================================

PAGINATION_MODULE_CONTENT = '''\
"""Pagination utilities for OtterAPI generated clients."""

from collections.abc import AsyncIterator, Callable, Iterator
from dataclasses import dataclass
from typing import Any, TypeVar

T = TypeVar("T")
PageT = TypeVar("PageT")


@dataclass
class OffsetPaginationConfig:
    """Configuration for offset-based pagination."""

    offset_param: str = "offset"
    limit_param: str = "limit"
    data_path: str | None = None
    total_path: str | None = None


@dataclass
class CursorPaginationConfig:
    """Configuration for cursor-based pagination."""

    cursor_param: str = "cursor"
    limit_param: str = "limit"
    data_path: str | None = None
    next_cursor_path: str | None = None


@dataclass
class PagePaginationConfig:
    """Configuration for page-based pagination."""

    page_param: str = "page"
    per_page_param: str = "per_page"
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
    for key in path.split("."):
        if isinstance(current, dict):
            if key not in current:
                raise KeyError(
                    f"Key \\'{key}\\' not found in response. "
                    f"Available keys: {list(current.keys())}. Full path: {path}"
                )
            current = current[key]
        elif isinstance(current, list) and key.isdigit():
            current = current[int(key)]
        else:
            raise KeyError(
                f"Cannot access \\'{key}\\' on {type(current).__name__}. "
                f"Full path: {path}"
            )

    return current


def paginate_offset(
    fetch_page: Callable[[int, int], PageT],
    extract_items: Callable[[PageT], list[T]],
    get_total: Callable[[PageT], int | None] | None = None,
    *,
    start_offset: int = 0,
    page_size: int = 100,
    max_items: int | None = None,
) -> list[T]:
    """Generic offset-based pagination that returns all items.

    Args:
        fetch_page: Function that fetches a page given (offset, limit).
        extract_items: Function that extracts items from a page response.
        get_total: Optional function to get total count from response.
        start_offset: Starting offset (default: 0).
        page_size: Items per page (default: 100).
        max_items: Maximum items to return (default: unlimited).

    Returns:
        List of all items.
    """
    all_items: list[T] = []
    current_offset = start_offset

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
) -> list[T]:
    """Async version of paginate_offset."""
    all_items: list[T] = []
    current_offset = start_offset

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


def iterate_offset(
    fetch_page: Callable[[int, int], PageT],
    extract_items: Callable[[PageT], list[T]],
    get_total: Callable[[PageT], int | None] | None = None,
    *,
    start_offset: int = 0,
    page_size: int = 100,
    max_items: int | None = None,
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

    Yields:
        Items one at a time.
    """
    current_offset = start_offset
    items_yielded = 0

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
) -> AsyncIterator[T]:
    """Async version of iterate_offset."""
    current_offset = start_offset
    items_yielded = 0

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

            if max_items is not None and items_yielded >= max_items:
                return

        if len(items) < request_limit:
            return

        if get_total is not None:
            total = get_total(page)
            if total is not None and current_offset + len(items) >= total:
                return

        current_offset += len(items)


def paginate_cursor(
    fetch_page: Callable[[str | None, int], PageT],
    extract_items: Callable[[PageT], list[T]],
    get_next_cursor: Callable[[PageT], str | None],
    *,
    start_cursor: str | None = None,
    page_size: int = 100,
    max_items: int | None = None,
) -> list[T]:
    """Generic cursor-based pagination that returns all items.

    Args:
        fetch_page: Function that fetches a page given (cursor, limit).
        extract_items: Function that extracts items from a page response.
        get_next_cursor: Function to get next cursor from response.
        start_cursor: Starting cursor (default: None for first page).
        page_size: Items per page (default: 100).
        max_items: Maximum items to return (default: unlimited).

    Returns:
        List of all items.
    """
    all_items: list[T] = []
    current_cursor = start_cursor

    while True:
        if max_items is not None and len(all_items) >= max_items:
            break

        page = fetch_page(current_cursor, page_size)
        items = extract_items(page)

        if not items:
            break

        all_items.extend(items)

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
) -> list[T]:
    """Async version of paginate_cursor."""
    all_items: list[T] = []
    current_cursor = start_cursor

    while True:
        if max_items is not None and len(all_items) >= max_items:
            break

        page = await fetch_page(current_cursor, page_size)
        items = extract_items(page)

        if not items:
            break

        all_items.extend(items)

        if max_items is not None and len(all_items) >= max_items:
            break

        current_cursor = get_next_cursor(page)
        if not current_cursor:
            break

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
) -> Iterator[T]:
    """Generic cursor-based pagination iterator (streaming).

    Args:
        fetch_page: Function that fetches a page given (cursor, limit).
        extract_items: Function that extracts items from a page response.
        get_next_cursor: Function to get next cursor from response.
        start_cursor: Starting cursor (default: None for first page).
        page_size: Items per page (default: 100).
        max_items: Maximum items to yield (default: unlimited).

    Yields:
        Items one at a time.
    """
    current_cursor = start_cursor
    items_yielded = 0

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
) -> AsyncIterator[T]:
    """Async version of iterate_cursor."""
    current_cursor = start_cursor
    items_yielded = 0

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

            if max_items is not None and items_yielded >= max_items:
                return

        current_cursor = get_next_cursor(page)
        if not current_cursor:
            return


def iterate_page(
    fetch_page: Callable[[int, int], PageT],
    extract_items: Callable[[PageT], list[T]],
    get_total_pages: Callable[[PageT], int | None] | None = None,
    *,
    start_page: int = 1,
    page_size: int = 100,
    max_items: int | None = None,
    max_pages: int | None = None,
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

    Yields:
        Items one at a time.
    """
    current_page = start_page
    items_yielded = 0
    pages_fetched = 0

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
) -> AsyncIterator[T]:
    """Async version of iterate_page."""
    current_page = start_page
    items_yielded = 0
    pages_fetched = 0

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

            if max_items is not None and items_yielded >= max_items:
                return

        if len(items) < page_size:
            return

        if get_total_pages is not None:
            total_pages = get_total_pages(page)
            if total_pages is not None and current_page >= total_pages:
                return

        current_page += 1


def paginate_page(
    fetch_page: Callable[[int, int], PageT],
    extract_items: Callable[[PageT], list[T]],
    get_total_pages: Callable[[PageT], int | None] | None = None,
    *,
    start_page: int = 1,
    page_size: int = 100,
    max_items: int | None = None,
    max_pages: int | None = None,
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

    Returns:
        List of all items.
    """
    all_items: list[T] = []
    current_page = start_page
    pages_fetched = 0

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
) -> list[T]:
    """Async version of paginate_page."""
    all_items: list[T] = []
    current_page = start_page
    pages_fetched = 0

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
'''


# =============================================================================
# Pagination Module Generation
# =============================================================================


def generate_pagination_module(output_dir: Path | UPath) -> Path | UPath:
    """Generate the _pagination.py utility module.

    Args:
        output_dir: The output directory where the module should be written.

    Returns:
        The path to the generated file.
    """
    output_dir = UPath(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    file_path = output_dir / '_pagination.py'
    file_path.write_text(PAGINATION_MODULE_CONTENT)

    return file_path


def get_pagination_imports() -> dict[str, set[str]]:
    """Get the imports needed for pagination method generation.

    Returns:
        A dictionary mapping module names to sets of imported names.
    """
    return {
        'typing': {'TYPE_CHECKING'},
        'collections.abc': {'Iterator', 'AsyncIterator'},
    }


def get_pagination_type_checking_imports() -> list[str]:
    """Get the TYPE_CHECKING imports for pagination.

    Returns:
        List of import statements to include inside TYPE_CHECKING block.
    """
    return []


# =============================================================================
# Pagination Configuration
# =============================================================================


@dataclass
class PaginationMethodConfig:
    """Configuration for pagination method generation.

    Attributes:
        style: The pagination style to use.
        offset_param: Name of offset parameter (for offset style).
        limit_param: Name of limit parameter.
        cursor_param: Name of cursor parameter (for cursor style).
        page_param: Name of page parameter (for page style).
        per_page_param: Name of per_page parameter (for page style).
        data_path: JSON path to items array in response.
        total_path: JSON path to total count in response.
        next_cursor_path: JSON path to next cursor in response.
        total_pages_path: JSON path to total pages in response.
        default_page_size: Default page size.
        max_page_size: Maximum page size.
    """

    style: str = 'offset'
    offset_param: str = 'offset'
    limit_param: str = 'limit'
    cursor_param: str = 'cursor'
    page_param: str = 'page'
    per_page_param: str = 'per_page'
    data_path: str | None = None
    total_path: str | None = None
    next_cursor_path: str | None = None
    total_pages_path: str | None = None
    default_page_size: int = 100
    max_page_size: int | None = None


def endpoint_is_paginated(
    endpoint_name: str,
    pagination_config: 'PaginationConfig',
    endpoint_parameters: list | None = None,
) -> bool:
    """Check if an endpoint is configured for pagination.

    Args:
        endpoint_name: The name of the endpoint function.
        pagination_config: The global pagination configuration.
        endpoint_parameters: Optional list of endpoint parameters for auto-detection.

    Returns:
        True if the endpoint should have pagination methods generated.
    """
    if not pagination_config.enabled:
        return False

    should_generate, _ = pagination_config.should_generate_for_endpoint(
        endpoint_name, endpoint_parameters
    )
    return should_generate


def get_pagination_config_for_endpoint(
    endpoint_name: str,
    pagination_config: 'PaginationConfig',
    endpoint_parameters: list | None = None,
) -> PaginationMethodConfig | None:
    """Get the pagination method configuration for an endpoint.

    Args:
        endpoint_name: The name of the endpoint function.
        pagination_config: The global pagination configuration.
        endpoint_parameters: Optional list of endpoint parameters for auto-detection.

    Returns:
        PaginationMethodConfig if pagination is configured, None otherwise.
    """
    if not pagination_config.enabled:
        return None

    should_generate, resolved = pagination_config.should_generate_for_endpoint(
        endpoint_name, endpoint_parameters
    )

    if not should_generate or resolved is None:
        return None

    return PaginationMethodConfig(
        style=resolved.style.value,
        offset_param=resolved.offset_param,
        limit_param=resolved.limit_param,
        cursor_param=resolved.cursor_param,
        page_param=resolved.page_param,
        per_page_param=resolved.per_page_param,
        data_path=resolved.data_path,
        total_path=resolved.total_path,
        next_cursor_path=resolved.next_cursor_path,
        total_pages_path=resolved.total_pages_path,
        default_page_size=resolved.default_page_size,
        max_page_size=resolved.max_page_size,
    )


# =============================================================================
# Pagination Function Building Utilities
# =============================================================================


def get_item_type_from_list_type(response_type: 'Type | None') -> ast.expr | None:
    """Extract the item type from a list response type.

    For example, if response_type is list[User], returns the AST for User.

    Args:
        response_type: The response Type object.

    Returns:
        The AST expression for the item type, or None if not a list type.
    """
    if not response_type or not response_type.annotation_ast:
        return None

    ann = response_type.annotation_ast
    if isinstance(ann, ast.Subscript):
        if isinstance(ann.value, ast.Name) and ann.value.id == 'list':
            return ann.slice

    return None


def build_extract_items_lambda(
    data_path: str | None, attr_name: str | None
) -> ast.expr:
    """Build a lambda expression for extracting items from a page response.

    Args:
        data_path: Optional JSON path to items (e.g., "data.users").
        attr_name: Optional attribute name on response model (e.g., "users").

    Returns:
        AST for a lambda expression like `lambda page: page.users` or
        `lambda page: extract_path(page, "data.users")`.
    """
    page_arg = ast.arg(arg='page', annotation=None)

    if attr_name:
        # lambda page: page.attr_name
        body = ast.Attribute(
            value=ast.Name(id='page', ctx=ast.Load()),
            attr=attr_name,
            ctx=ast.Load(),
        )
    elif data_path:
        # lambda page: extract_path(page, "data_path")
        body = ast.Call(
            func=ast.Name(id='extract_path', ctx=ast.Load()),
            args=[
                ast.Name(id='page', ctx=ast.Load()),
                ast.Constant(value=data_path),
            ],
            keywords=[],
        )
    else:
        # lambda page: page (assume response is the list itself)
        body = ast.Name(id='page', ctx=ast.Load())

    return ast.Lambda(
        args=ast.arguments(
            posonlyargs=[],
            args=[page_arg],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
        ),
        body=body,
    )


def build_get_total_lambda(
    total_path: str | None, attr_name: str | None
) -> ast.expr | None:
    """Build a lambda expression for getting total count from a page response.

    Args:
        total_path: Optional JSON path to total (e.g., "meta.total").
        attr_name: Optional attribute name on response model (e.g., "total").

    Returns:
        AST for a lambda expression or None if no total is available.
    """
    if not total_path and not attr_name:
        return None

    page_arg = ast.arg(arg='page', annotation=None)

    if attr_name:
        # lambda page: page.attr_name
        body = ast.Attribute(
            value=ast.Name(id='page', ctx=ast.Load()),
            attr=attr_name,
            ctx=ast.Load(),
        )
    else:
        # lambda page: extract_path(page, "total_path")
        body = ast.Call(
            func=ast.Name(id='extract_path', ctx=ast.Load()),
            args=[
                ast.Name(id='page', ctx=ast.Load()),
                ast.Constant(value=total_path),
            ],
            keywords=[],
        )

    return ast.Lambda(
        args=ast.arguments(
            posonlyargs=[],
            args=[page_arg],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
        ),
        body=body,
    )


def build_get_next_cursor_lambda(
    next_cursor_path: str | None, attr_name: str | None
) -> ast.expr | None:
    """Build a lambda expression for getting next cursor from a page response.

    Args:
        next_cursor_path: Optional JSON path to next cursor.
        attr_name: Optional attribute name on response model.

    Returns:
        AST for a lambda expression or None if no cursor path is available.
    """
    if not next_cursor_path and not attr_name:
        return None

    page_arg = ast.arg(arg='page', annotation=None)

    if attr_name:
        body = ast.Attribute(
            value=ast.Name(id='page', ctx=ast.Load()),
            attr=attr_name,
            ctx=ast.Load(),
        )
    else:
        body = ast.Call(
            func=ast.Name(id='extract_path', ctx=ast.Load()),
            args=[
                ast.Name(id='page', ctx=ast.Load()),
                ast.Constant(value=next_cursor_path),
            ],
            keywords=[],
        )

    return ast.Lambda(
        args=ast.arguments(
            posonlyargs=[],
            args=[page_arg],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
        ),
        body=body,
    )
