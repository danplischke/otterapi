"""Retry and backoff helpers for OtterAPI generated clients."""

from __future__ import annotations

import asyncio
import time
from typing import Any


def _backoff_sleep(attempt: int, factor: float, response: Any) -> None:
    """Sleep before a retry, honouring the Retry-After header when present."""
    if response is not None:
        retry_after = response.headers.get('Retry-After')
        if retry_after is not None:
            try:
                time.sleep(float(retry_after))
                return
            except ValueError:
                pass
    time.sleep(min(factor * (2**attempt), 60.0))


async def _backoff_sleep_async(attempt: int, factor: float, response: Any) -> None:
    """Async variant of :func:`_backoff_sleep`."""
    if response is not None:
        retry_after = response.headers.get('Retry-After')
        if retry_after is not None:
            try:
                await asyncio.sleep(float(retry_after))
                return
            except ValueError:
                pass
    await asyncio.sleep(min(factor * (2**attempt), 60.0))
