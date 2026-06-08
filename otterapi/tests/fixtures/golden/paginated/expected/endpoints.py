from collections.abc import AsyncIterator, Iterator
from typing import Any, Union
from ._pagination import (
    extract_path,
    iterate_cursor,
    iterate_cursor_async,
    iterate_offset,
    iterate_offset_async,
    iterate_page,
    iterate_page_async,
    paginate_cursor,
    paginate_cursor_async,
    paginate_offset,
    paginate_offset_async,
    paginate_page,
    paginate_page_async,
)
from .client import Client
from .models import Item

__all__ = ('async_list_items', 'async_list_items_iter', 'list_items', 'list_items_iter')


def list_items(
    *,
    offset: int | None = None,
    page_size: int = 50,
    max_items: int | None = None,
    client: Client | None = None,
    **kwargs: Any,
) -> list[Item]:
    c = client or Client()

    def fetch_page(off: int, limit: int):
        response = c._request(
            method='get',
            path='/items',
            params={'offset': off, 'limit': limit},
            **kwargs,
        )
        return c._parse_response(response, list[Item])

    return paginate_offset(
        fetch_page=fetch_page,
        extract_items=lambda page: page,
        start_offset=offset or 0,
        page_size=page_size,
        max_items=max_items,
    )


async def async_list_items(
    *,
    offset: int | None = None,
    page_size: int = 50,
    max_items: int | None = None,
    client: Client | None = None,
    **kwargs: Any,
) -> list[Item]:
    c = client or Client()

    async def fetch_page(off: int, limit: int):
        response = await c._request_async(
            method='get',
            path='/items',
            params={'offset': off, 'limit': limit},
            **kwargs,
        )
        return await c._parse_response_async(response, list[Item])

    return await paginate_offset_async(
        fetch_page=fetch_page,
        extract_items=lambda page: page,
        start_offset=offset or 0,
        page_size=page_size,
        max_items=max_items,
    )


def list_items_iter(
    *,
    offset: int | None = None,
    page_size: int = 50,
    max_items: int | None = None,
    client: Client | None = None,
    **kwargs: Any,
) -> Iterator[Item]:
    c = client or Client()

    def fetch_page(off: int, limit: int):
        response = c._request(
            method='get',
            path='/items',
            params={'offset': off, 'limit': limit},
            **kwargs,
        )
        return c._parse_response(response, list[Item])

    yield from iterate_offset(
        fetch_page=fetch_page,
        extract_items=lambda page: page,
        start_offset=offset or 0,
        page_size=page_size,
        max_items=max_items,
    )


async def async_list_items_iter(
    *,
    offset: int | None = None,
    page_size: int = 50,
    max_items: int | None = None,
    client: Client | None = None,
    **kwargs: Any,
) -> AsyncIterator[Item]:
    c = client or Client()

    async def fetch_page(off: int, limit: int):
        response = await c._request_async(
            method='get',
            path='/items',
            params={'offset': off, 'limit': limit},
            **kwargs,
        )
        return await c._parse_response_async(response, list[Item])

    async for item in iterate_offset_async(
        fetch_page=fetch_page,
        extract_items=lambda page: page,
        start_offset=offset or 0,
        page_size=page_size,
        max_items=max_items,
    ):
        yield item
