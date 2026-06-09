"""Tests for retry loop and lifecycle methods in the generated base client.

Uses the same constraints golden-fixture spec as test_per_status_errors.py
to generate a real client once, then exercises retry and context-manager behaviour.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest

from otterapi.codegen.codegen import Codegen
from otterapi.config import DocumentConfig

SPEC = Path(__file__).parent / 'fixtures' / 'golden' / 'constraints' / 'spec.yaml'


def _generate(target: Path) -> None:
    config = DocumentConfig(
        source=str(SPEC),
        output=str(target),
        base_url='https://example.test',
    )
    Codegen(config).generate()


@pytest.fixture(scope='module')
def retry_pkg(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp('retry_lifecycle')
    pkg = 'retry_lifecycle_pkg'
    sys.path.insert(0, str(tmp_path))
    _generate(tmp_path / pkg)
    for stale in [m for m in list(sys.modules) if m == pkg or m.startswith(pkg + '.')]:
        sys.modules.pop(stale, None)
    try:
        yield importlib.import_module(pkg)
    finally:
        try:
            sys.path.remove(str(tmp_path))
        except ValueError:
            pass


class TestRetryBehavior:
    def _make_handler(self, responses):
        """Returns a transport handler that cycles through a list of (status, body) pairs."""
        it = iter(responses)

        def handler(request: httpx.Request) -> httpx.Response:
            status, body = next(it)
            return httpx.Response(status, json={'detail': body})

        return handler

    def test_no_retry_on_max_retries_0(self, retry_pkg):
        Client = retry_pkg.Client
        calls = 0

        def handler(request):
            nonlocal calls
            calls += 1
            return httpx.Response(503, json={'detail': 'unavailable'})

        with httpx.Client(transport=httpx.MockTransport(handler)) as http:
            client = Client(http_client=http, max_retries=0)
            with patch.object(retry_pkg._client, '_backoff_sleep'):
                with pytest.raises(Exception):
                    retry_pkg.list_users(client=client)
        assert calls == 1

    def test_retries_on_retriable_status(self, retry_pkg):
        Client = retry_pkg.Client
        calls = 0

        def handler(request):
            nonlocal calls
            calls += 1
            if calls < 3:
                return httpx.Response(503, json={'detail': 'unavailable'})
            return httpx.Response(200, json=[])

        with httpx.Client(transport=httpx.MockTransport(handler)) as http:
            client = Client(http_client=http, max_retries=3)
            with patch.object(retry_pkg._client, '_backoff_sleep'):
                retry_pkg.list_users(client=client)
        assert calls == 3

    def test_raises_after_exhausting_retries(self, retry_pkg):
        Client = retry_pkg.Client
        calls = 0

        def handler(request):
            nonlocal calls
            calls += 1
            return httpx.Response(503, json={'detail': 'always down'})

        with httpx.Client(transport=httpx.MockTransport(handler)) as http:
            client = Client(http_client=http, max_retries=2)
            with patch.object(retry_pkg._client, '_backoff_sleep'):
                with pytest.raises(Exception):
                    retry_pkg.list_users(client=client)
        # 1 initial attempt + 2 retries = 3 total
        assert calls == 3

    def test_no_retry_on_non_retriable_status(self, retry_pkg):
        Client = retry_pkg.Client
        calls = 0

        def handler(request):
            nonlocal calls
            calls += 1
            return httpx.Response(404, json={'detail': 'not found'})

        with httpx.Client(transport=httpx.MockTransport(handler)) as http:
            client = Client(http_client=http, max_retries=3)
            with pytest.raises(Exception):
                retry_pkg.list_users(client=client)
        assert calls == 1

    def test_custom_retry_statuses(self, retry_pkg):
        Client = retry_pkg.Client
        calls = 0

        def handler(request):
            nonlocal calls
            calls += 1
            return httpx.Response(404, json={'detail': 'nope'})

        with httpx.Client(transport=httpx.MockTransport(handler)) as http:
            client = Client(http_client=http, max_retries=2, retry_statuses=frozenset({404}))
            with patch.object(retry_pkg._client, '_backoff_sleep'):
                with pytest.raises(Exception):
                    retry_pkg.list_users(client=client)
        assert calls == 3


class TestLifecycle:
    def test_sync_context_manager_closes_client(self, retry_pkg):
        Client = retry_pkg.Client
        client = Client()
        assert not client._sync_client.is_closed
        with client:
            pass
        assert client._sync_client.is_closed

    def test_close_no_op_for_external_client(self, retry_pkg):
        Client = retry_pkg.Client
        http = httpx.Client()
        client = Client(http_client=http)
        client.close()
        assert not http.is_closed
        http.close()

    def test_context_manager_returns_self(self, retry_pkg):
        Client = retry_pkg.Client
        client = Client()
        with client as c:
            assert c is client

    @pytest.mark.asyncio
    async def test_async_context_manager(self, retry_pkg):
        Client = retry_pkg.Client
        async with Client() as client:
            assert client is not None

    @pytest.mark.asyncio
    async def test_aclose_closes_external_async_client(self, retry_pkg):
        Client = retry_pkg.Client
        async_http = httpx.AsyncClient()
        client = Client(async_http_client=async_http)
        await client.aclose()
        assert async_http.is_closed
