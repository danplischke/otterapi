"""Concurrency utilities for OtterAPI generated clients.

These helpers let you run multiple async endpoint calls in parallel and
call async functions from synchronous contexts (including Jupyter notebooks,
where ``asyncio.run()`` fails because an event loop is already running).

Example — parallel fan-out over any mix of endpoints::

    from myapi import run_concurrently, async_get_gene, async_get_gwas

    results = run_concurrently([
        async_get_gene(symbol="BRCA1"),
        async_get_gwas(gene="BRCA1", p_threshold=5e-8),
        async_get_expression(gene="BRCA1", tissue="breast"),
    ], concurrency=3)
    gene, gwas, expr = results

Example — bulk over a list of targets::

    genes = run_concurrently(
        [async_get_gene(symbol=g) for g in gene_list],
        concurrency=10,
    )

Example — single async call from Jupyter or a script::

    gene = run_sync(async_get_gene(symbol="BRCA1"))
"""

from __future__ import annotations

import asyncio
import concurrent.futures
from collections.abc import Coroutine, Iterable
from typing import TypeVar

T = TypeVar('T')


def run_sync(coro: Coroutine[object, object, T]) -> T:
    """Run a coroutine synchronously.

    Works in both plain scripts and Jupyter notebooks.  In a notebook the
    kernel already runs an event loop, so ``asyncio.run()`` would raise
    ``RuntimeError: This event loop is already running``.  This helper
    detects that case and executes the coroutine in a separate thread with
    its own event loop instead.

    Args:
        coro: The coroutine to execute.

    Returns:
        Whatever the coroutine returns.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    # Running loop (Jupyter) — dispatch to a worker thread with its own loop.
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


async def _bounded_gather(
    coros: list[Coroutine[object, object, T]],
    concurrency: int,
) -> list[T]:
    sem = asyncio.Semaphore(concurrency)

    async def _bounded(c: Coroutine[object, object, T]) -> T:
        async with sem:
            return await c

    return list(await asyncio.gather(*(_bounded(c) for c in coros)))


def run_concurrently(
    coros: Iterable[Coroutine[object, object, T]],
    *,
    concurrency: int = 10,
) -> list[T]:
    """Run coroutines concurrently, returning results in input order.

    Works from both sync scripts and Jupyter notebooks (see :func:`run_sync`).

    At most ``concurrency`` coroutines execute simultaneously.  Results are
    returned in the same order as the input iterable regardless of completion
    order.

    Args:
        coros: Iterable of coroutines (e.g. ``[async_fn(x) for x in items]``).
        concurrency: Maximum number of concurrent coroutines. Default: 10.

    Returns:
        List of results in input order.
    """
    return run_sync(_bounded_gather(list(coros), concurrency))


async def run_concurrently_async(
    coros: Iterable[Coroutine[object, object, T]],
    *,
    concurrency: int = 10,
) -> list[T]:
    """Async variant of :func:`run_concurrently` for use inside async contexts.

    Args:
        coros: Iterable of coroutines.
        concurrency: Maximum number of concurrent coroutines. Default: 10.

    Returns:
        List of results in input order.
    """
    return await _bounded_gather(list(coros), concurrency)
