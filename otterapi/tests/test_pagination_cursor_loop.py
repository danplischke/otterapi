"""A server that keeps handing back a cursor it already returned.

Without a guard the loop never reaches an empty page: it runs until the process
is killed while the result list grows.  The helpers stop with an error instead
of returning what they have as though it were everything.
"""

from __future__ import annotations

import asyncio

import pytest

from otterapi.codegen.runtime._pagination import (
    iterate_cursor,
    iterate_cursor_async,
    paginate_cursor,
    paginate_cursor_async,
)


class FakeServer:
    """Serves pages of ``[cursor-name]`` items following a fixed cursor chain."""

    def __init__(self, chain: list[str | None]) -> None:
        # chain[i] is the cursor returned by the i-th page.
        self.chain = chain
        self.fetches = 0

    def fetch(self, cursor, limit):
        self.fetches += 1
        if self.fetches > 50:
            raise AssertionError('pagination did not stop')
        index = self.fetches - 1
        return {
            'items': [f'item-{index}'],
            'next': self.chain[min(index, len(self.chain) - 1)],
        }

    async def fetch_async(self, cursor, limit):
        return self.fetch(cursor, limit)


def _extract(page):
    return page['items']


def _next(page):
    return page['next']


class TestSyncHelpers:
    def test_a_well_behaved_chain_is_unaffected(self):
        server = FakeServer(['b', 'c', None])
        items = paginate_cursor(
            server.fetch, _extract, _next, page_size=1, progress=False
        )
        assert items == ['item-0', 'item-1', 'item-2']

    def test_the_same_cursor_forever_raises(self):
        server = FakeServer(['stuck'])
        with pytest.raises(RuntimeError, match="'stuck'"):
            paginate_cursor(server.fetch, _extract, _next, page_size=1, progress=False)
        assert server.fetches == 2

    def test_a_cycle_raises(self):
        server = FakeServer(['b', 'c', 'b'])
        with pytest.raises(RuntimeError, match='not advancing'):
            paginate_cursor(server.fetch, _extract, _next, page_size=1, progress=False)

    def test_returning_the_cursor_the_caller_started_from_raises(self):
        server = FakeServer(['start'])
        with pytest.raises(RuntimeError):
            paginate_cursor(
                server.fetch,
                _extract,
                _next,
                start_cursor='start',
                page_size=1,
                progress=False,
            )

    def test_iterator_raises_after_yielding_the_pages_it_got(self):
        server = FakeServer(['b', 'b'])
        seen = []
        with pytest.raises(RuntimeError):
            for item in iterate_cursor(
                server.fetch, _extract, _next, page_size=1, progress=False
            ):
                seen.append(item)
        assert seen == ['item-0', 'item-1']

    def test_max_items_still_ends_the_loop_before_the_guard_matters(self):
        server = FakeServer(['stuck'])
        items = paginate_cursor(
            server.fetch,
            _extract,
            _next,
            page_size=1,
            max_items=1,
            progress=False,
        )
        assert items == ['item-0']

    def test_an_empty_cursor_still_means_the_end(self):
        for last in (None, ''):
            server = FakeServer(['b', last])
            items = paginate_cursor(
                server.fetch, _extract, _next, page_size=1, progress=False
            )
            assert items == ['item-0', 'item-1']


class TestAsyncHelpers:
    def test_paginate_async_raises_on_a_repeated_cursor(self):
        server = FakeServer(['stuck'])

        async def run():
            return await paginate_cursor_async(
                server.fetch_async, _extract, _next, page_size=1, progress=False
            )

        with pytest.raises(RuntimeError, match='not advancing'):
            asyncio.run(run())

    def test_iterate_async_raises_on_a_repeated_cursor(self):
        server = FakeServer(['b', 'b'])
        seen = []

        async def run():
            async for item in iterate_cursor_async(
                server.fetch_async, _extract, _next, page_size=1, progress=False
            ):
                seen.append(item)

        with pytest.raises(RuntimeError):
            asyncio.run(run())
        assert seen == ['item-0', 'item-1']

    def test_a_well_behaved_chain_is_unaffected(self):
        server = FakeServer(['b', None])

        async def run():
            return await paginate_cursor_async(
                server.fetch_async, _extract, _next, page_size=1, progress=False
            )

        assert asyncio.run(run()) == ['item-0', 'item-1']
