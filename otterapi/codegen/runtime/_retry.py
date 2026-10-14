"""Retry and backoff helpers for OtterAPI generated clients."""

from __future__ import annotations

import asyncio
import logging
import math
import time
from collections.abc import Iterable
from typing import Any

import httpx

logger = logging.getLogger(__name__)

#: Longest single wait between attempts, in seconds.  It bounds the exponential
#: schedule and a server-supplied ``Retry-After`` alike: a misconfigured or
#: hostile server must not be able to park the caller for hours.
MAX_RETRY_DELAY = 60.0

#: Methods that are safe to repeat after the server may already have acted on
#: the first request (RFC 9110 section 9.2.2).  POST and PATCH are not: a
#: timeout or a 5xx does not say whether the payment was created or the row
#: updated, and repeating the call can do it twice.
IDEMPOTENT_METHODS = frozenset({'DELETE', 'GET', 'HEAD', 'OPTIONS', 'PUT', 'TRACE'})

# A server that answers 429 or 503 is declining to process the request, so
# repeating it is safe whatever the method.
_NOT_PROCESSED_STATUSES = frozenset({429, 503})

# Failures raised before a single byte of the request reached the server.
_NEVER_SENT = (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout)


def _is_replayable(method: str, retry_methods: Iterable[str]) -> bool:
    """Whether *method* may be repeated after the server could have acted."""
    return method.upper() in {name.upper() for name in retry_methods}


def _should_retry_response(
    method: str,
    status_code: int,
    retry_statuses: Iterable[int],
    retry_methods: Iterable[str],
) -> bool:
    """Decide whether an error response is worth another attempt.

    The status must be one the client was told to retry.  Beyond that, only a
    replayable method (``retry_methods``) is retried on any such status; every
    other method is retried only on 429 and 503, where the server says it did
    not process the request.
    """
    if status_code not in retry_statuses:
        return False
    return status_code in _NOT_PROCESSED_STATUSES or _is_replayable(
        method, retry_methods
    )


def _should_retry_error(
    method: str, error: BaseException, retry_methods: Iterable[str]
) -> bool:
    """Decide whether a transport failure is worth another attempt.

    A failure that happened before the request was sent is always safe to
    retry.  Any later one (a read timeout, a dropped connection) may have
    followed the server acting on the request, so it is retried only for a
    replayable method.
    """
    return isinstance(error, _NEVER_SENT) or _is_replayable(method, retry_methods)


def _backoff_delay(
    attempt: int, factor: float, response: Any, max_delay: float
) -> tuple[float, float | None]:
    """Work out how long to wait before the next attempt.

    Returns ``(seconds, requested)``: the wait to use, which is never negative
    and never above *max_delay*, and the ``Retry-After`` value the server asked
    for (``None`` when it sent none or sent something unusable).
    """
    if response is not None:
        retry_after = response.headers.get('Retry-After')
        if retry_after is not None:
            try:
                requested = float(retry_after)
            except ValueError:
                requested = math.nan
            # ``nan`` and ``inf`` parse as floats but cannot be slept on.
            if math.isfinite(requested) and requested >= 0:
                return min(requested, max_delay), requested
            logger.debug(
                'Unrecognised Retry-After value %r; falling back to exponential backoff',
                retry_after,
            )
    # The exponent is clamped so a very large attempt count cannot overflow.
    delay = factor * (2 ** min(attempt, 30))
    return max(0.0, min(delay, max_delay)), None


def _plan_backoff(
    attempt: int, factor: float, response: Any, max_delay: float
) -> float:
    """Pick the wait before the next attempt and log the decision."""
    status = getattr(response, 'status_code', None) if response is not None else None
    delay, requested = _backoff_delay(attempt, factor, response, max_delay)
    if requested is None:
        logger.debug(
            'Retry attempt %d: sleeping %.1fs (status=%s)',
            attempt + 1,
            delay,
            status,
        )
    elif requested > delay:
        logger.warning(
            'Retry-After asks for %.0fs; waiting %.1fs before retry attempt %d '
            '(status=%s)',
            requested,
            delay,
            attempt + 1,
            status,
        )
    else:
        logger.warning(
            'Retry-After header present; sleeping %.1fs before retry attempt %d '
            '(status=%s)',
            delay,
            attempt + 1,
            status,
        )
    return delay


def _backoff_sleep(
    attempt: int,
    factor: float,
    response: Any,
    max_delay: float = MAX_RETRY_DELAY,
) -> None:
    """Sleep before a retry, honouring the Retry-After header when present."""
    time.sleep(_plan_backoff(attempt, factor, response, max_delay))


async def _backoff_sleep_async(
    attempt: int,
    factor: float,
    response: Any,
    max_delay: float = MAX_RETRY_DELAY,
) -> None:
    """Async variant of :func:`_backoff_sleep`."""
    await asyncio.sleep(_plan_backoff(attempt, factor, response, max_delay))
