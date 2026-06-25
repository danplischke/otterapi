"""Retry and backoff helpers for OtterAPI generated clients."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


def _backoff_sleep(attempt: int, factor: float, response: Any) -> None:
    """Sleep before a retry, honouring the Retry-After header when present."""
    status = getattr(response, 'status_code', None) if response is not None else None
    if response is not None:
        retry_after = response.headers.get('Retry-After')
        if retry_after is not None:
            try:
                delay = float(retry_after)
                logger.warning(
                    'Retry-After header present; sleeping %.1fs before retry attempt %d '
                    '(status=%s)',
                    delay,
                    attempt + 1,
                    status,
                )
                time.sleep(delay)
                return
            except ValueError:
                logger.debug(
                    'Unrecognised Retry-After value %r; falling back to exponential backoff',
                    retry_after,
                )
    delay = min(factor * (2**attempt), 60.0)
    logger.debug(
        'Retry attempt %d: sleeping %.1fs (status=%s)',
        attempt + 1,
        delay,
        status,
    )
    time.sleep(delay)


async def _backoff_sleep_async(attempt: int, factor: float, response: Any) -> None:
    """Async variant of :func:`_backoff_sleep`."""
    status = getattr(response, 'status_code', None) if response is not None else None
    if response is not None:
        retry_after = response.headers.get('Retry-After')
        if retry_after is not None:
            try:
                delay = float(retry_after)
                logger.warning(
                    'Retry-After header present; sleeping %.1fs before retry attempt %d '
                    '(status=%s)',
                    delay,
                    attempt + 1,
                    status,
                )
                await asyncio.sleep(delay)
                return
            except ValueError:
                logger.debug(
                    'Unrecognised Retry-After value %r; falling back to exponential backoff',
                    retry_after,
                )
    delay = min(factor * (2**attempt), 60.0)
    logger.debug(
        'Retry attempt %d: sleeping %.1fs (status=%s)',
        attempt + 1,
        delay,
        status,
    )
    await asyncio.sleep(delay)
