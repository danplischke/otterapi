"""Tests for _concurrency.py runtime helpers."""

from __future__ import annotations

import asyncio

import pytest

from otterapi.codegen.runtime._concurrency import (
    run_concurrently,
    run_concurrently_async,
    run_sync,
)


class TestRunSync:
    def test_returns_value(self):
        async def coro():
            return 42

        assert run_sync(coro()) == 42

    def test_propagates_exception(self):
        async def failing():
            raise ValueError('boom')

        with pytest.raises(ValueError, match='boom'):
            run_sync(failing())

    def test_works_from_running_loop(self):
        """run_sync dispatches to a thread when an event loop is already running."""

        async def inner():
            return 'nested'

        async def outer():
            return run_sync(inner())

        assert asyncio.run(outer()) == 'nested'

    def test_result_ordering_preserved(self):
        async def echo(x):
            return x

        assert run_sync(echo('hello')) == 'hello'


class TestRunConcurrently:
    def test_returns_results_in_input_order(self):
        async def identity(x):
            return x

        results = run_concurrently([identity(i) for i in range(5)])
        assert results == list(range(5))

    def test_respects_concurrency_limit(self):
        active = 0
        peak = 0

        async def task(i):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0)
            active -= 1
            return i

        run_concurrently([task(i) for i in range(20)], concurrency=5)
        assert peak <= 5

    def test_empty_iterable_returns_empty_list(self):
        assert run_concurrently([]) == []

    def test_accepts_generator_expression(self):
        async def identity(x):
            return x

        results = run_concurrently(identity(i) for i in range(3))
        assert results == [0, 1, 2]

    def test_single_coroutine(self):
        async def answer():
            return 99

        assert run_concurrently([answer()]) == [99]

    def test_default_concurrency_is_10(self):
        active = 0
        peak = 0

        async def task(i):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0)
            active -= 1
            return i

        run_concurrently([task(i) for i in range(30)])
        assert peak <= 10


class TestRunConcurrentlyAsync:
    @pytest.mark.asyncio
    async def test_returns_results_in_order(self):
        async def identity(x):
            return x

        results = await run_concurrently_async([identity(i) for i in range(5)])
        assert results == list(range(5))

    @pytest.mark.asyncio
    async def test_empty_list(self):
        assert await run_concurrently_async([]) == []

    @pytest.mark.asyncio
    async def test_respects_concurrency_limit(self):
        active = 0
        peak = 0

        async def task(i):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0)
            active -= 1
            return i

        await run_concurrently_async([task(i) for i in range(20)], concurrency=4)
        assert peak <= 4
