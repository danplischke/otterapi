"""Tests for _retry.py runtime helpers."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from otterapi.codegen.runtime._retry import _backoff_sleep, _backoff_sleep_async


def _resp(retry_after=None):
    r = MagicMock()
    r.headers.get.return_value = retry_after
    return r


class TestBackoffSleep:
    def test_exponential_attempt_0(self):
        with patch('time.sleep') as mock_sleep:
            _backoff_sleep(0, 0.5, None)
            mock_sleep.assert_called_once_with(0.5)

    def test_exponential_attempt_1(self):
        with patch('time.sleep') as mock_sleep:
            _backoff_sleep(1, 0.5, None)
            mock_sleep.assert_called_once_with(1.0)

    def test_exponential_attempt_2(self):
        with patch('time.sleep') as mock_sleep:
            _backoff_sleep(2, 0.5, None)
            mock_sleep.assert_called_once_with(2.0)

    def test_capped_at_60_seconds(self):
        with patch('time.sleep') as mock_sleep:
            _backoff_sleep(100, 0.5, None)
            mock_sleep.assert_called_once_with(60.0)

    def test_retry_after_header_overrides_backoff(self):
        with patch('time.sleep') as mock_sleep:
            _backoff_sleep(0, 0.5, _resp(retry_after='7'))
            mock_sleep.assert_called_once_with(7.0)

    def test_float_retry_after_parsed(self):
        with patch('time.sleep') as mock_sleep:
            _backoff_sleep(0, 0.5, _resp(retry_after='2.5'))
            mock_sleep.assert_called_once_with(2.5)

    def test_invalid_retry_after_falls_back_to_exponential(self):
        with patch('time.sleep') as mock_sleep:
            _backoff_sleep(0, 0.5, _resp(retry_after='not-a-number'))
            mock_sleep.assert_called_once_with(0.5)

    def test_no_retry_after_uses_backoff(self):
        with patch('time.sleep') as mock_sleep:
            _backoff_sleep(0, 0.5, _resp(retry_after=None))
            mock_sleep.assert_called_once_with(0.5)

    def test_backoff_factor_scales_sleep(self):
        with patch('time.sleep') as mock_sleep:
            _backoff_sleep(1, 2.0, None)
            mock_sleep.assert_called_once_with(4.0)


class TestBackoffSleepAsync:
    @pytest.mark.asyncio
    async def test_exponential_attempt_0(self):
        with patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
            await _backoff_sleep_async(0, 0.5, None)
            mock_sleep.assert_called_once_with(0.5)

    @pytest.mark.asyncio
    async def test_capped_at_60_seconds(self):
        with patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
            await _backoff_sleep_async(100, 0.5, None)
            mock_sleep.assert_called_once_with(60.0)

    @pytest.mark.asyncio
    async def test_retry_after_header(self):
        with patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
            await _backoff_sleep_async(0, 0.5, _resp(retry_after='3'))
            mock_sleep.assert_called_once_with(3.0)

    @pytest.mark.asyncio
    async def test_invalid_retry_after_falls_back(self):
        with patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
            await _backoff_sleep_async(0, 0.5, _resp(retry_after='bad'))
            mock_sleep.assert_called_once_with(0.5)
