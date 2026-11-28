"""Item-level pagination helpers (runtime ``_pagination.py``), called directly."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from otterapi.codegen.runtime import _pagination as rt

ROWS = list(range(23))


class Offsets:
    def __init__(self, total: int | None = None) -> None:
        self.calls: list[tuple[int, int]] = []
        self.total = total

    def __call__(self, offset: int, limit: int) -> dict:
        self.calls.append((offset, limit))
        return {'items': ROWS[offset : offset + limit], 'total': self.total}


def items(page: dict) -> list[int]:
    return page['items']


def total(page: dict) -> int | None:
    return page['total']


def test_offset_trims_the_last_request_to_max_items() -> None:
    fetch = Offsets()
    got = rt.paginate_offset(fetch, items, page_size=10, max_items=15, progress=False)
    assert got == ROWS[:15]
    assert fetch.calls == [(0, 10), (10, 5)]


def test_offset_stops_on_a_short_page_and_on_total() -> None:
    fetch = Offsets()
    assert rt.paginate_offset(fetch, items, page_size=10, progress=False) == ROWS
    assert fetch.calls == [(0, 10), (10, 10), (20, 10)]
    fetch = Offsets(total=20)
    rt.paginate_offset(fetch, items, total, page_size=10, progress=False)
    assert fetch.calls == [(0, 10), (10, 10)]  # no request past the total


def test_max_items_zero_sends_nothing() -> None:
    fetch = Offsets()
    assert rt.paginate_offset(fetch, items, max_items=0, progress=False) == []
    assert fetch.calls == []


def test_iterate_is_lazy() -> None:
    fetch = Offsets()
    stream = rt.iterate_offset(fetch, items, page_size=10, progress=False)
    assert fetch.calls == []
    assert next(stream) == 0
    assert fetch.calls == [(0, 10)]


def test_page_style_max_pages_and_total_pages() -> None:
    calls: list[int] = []

    def fetch(page: int, per: int) -> dict:
        calls.append(page)
        start = (page - 1) * per
        return {'items': ROWS[start : start + per], 'pages': 3}

    got = rt.paginate_page(fetch, items, page_size=5, max_pages=2, progress=False)
    assert got == ROWS[:10] and calls == [1, 2]
    calls.clear()
    rt.paginate_page(fetch, items, lambda p: p['pages'], page_size=5, progress=False)
    assert calls == [1, 2, 3]


def test_cursor_max_items_does_not_fetch_another_page() -> None:
    calls: list[str | None] = []

    def fetch(cursor: str | None, limit: int) -> dict:
        calls.append(cursor)
        start = int(cursor or 0)
        return {'items': ROWS[start : start + limit], 'next': str(start + limit)}

    got = rt.paginate_cursor(
        fetch, items, lambda p: p['next'], page_size=10, max_items=10, progress=False
    )
    assert got == ROWS[:10] and calls == [None]


@pytest.mark.parametrize('style', ['offset', 'cursor', 'page'])
def test_async_matches_sync(style: str) -> None:
    def fetch(start: Any, size: int) -> dict:
        if style == 'page':
            begin = (start - 1) * size
        else:
            begin = int(start or 0)
        rows = ROWS[begin : begin + size]
        return {'items': rows, 'next': str(begin + size) if rows else None}

    async def afetch(start: Any, size: int) -> dict:
        return fetch(start, size)

    extra: list[Any] = [lambda p: p['next']] if style == 'cursor' else []
    sync = getattr(rt, f'paginate_{style}')(
        fetch, items, *extra, page_size=7, max_items=20, progress=False
    )
    collected = asyncio.run(
        getattr(rt, f'paginate_{style}_async')(
            afetch, items, *extra, page_size=7, max_items=20, progress=False
        )
    )
    assert sync == collected == ROWS[:20]
