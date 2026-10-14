"""Tests for which failures a generated client retries, and for how long it waits.

A timeout or a 5xx does not say whether the server acted on the request, so
repeating a POST can create the payment twice.  The client therefore repeats
non-idempotent methods only when the request provably was not processed (it
never left the machine, or the server answered 429/503), and the caller can opt
specific methods back in.  ``Retry-After`` is honoured but bounded: the server
decides how long the caller waits, so the server must not be able to pick
"forever".
"""

from __future__ import annotations

import importlib
import sys
from typing import Any
from unittest.mock import patch

import httpx
import pytest
import yaml

from otterapi.codegen.codegen import Codegen
from otterapi.codegen.runtime import _retry
from otterapi.codegen.runtime._retry import (
    IDEMPOTENT_METHODS,
    MAX_RETRY_DELAY,
    _backoff_delay,
    _should_retry_error,
    _should_retry_response,
)
from otterapi.config import DocumentConfig

_ID = {
    'in': 'path',
    'name': 'item_id',
    'required': True,
    'schema': {'type': 'string'},
}
_OK = {'200': {'description': 'ok'}}

SPEC: dict[str, Any] = {
    'openapi': '3.0.0',
    'info': {'title': 'Retry API', 'version': '1.0.0'},
    'servers': [{'url': 'https://api.example.test'}],
    'paths': {
        '/items': {
            'get': {'operationId': 'listItems', 'responses': _OK},
            'post': {'operationId': 'createItem', 'responses': _OK},
        },
        '/items/{item_id}': {
            'put': {
                'operationId': 'replaceItem',
                'parameters': [_ID],
                'responses': _OK,
            },
            'patch': {
                'operationId': 'patchItem',
                'parameters': [_ID],
                'responses': _OK,
            },
            'delete': {
                'operationId': 'deleteItem',
                'parameters': [_ID],
                'responses': _OK,
            },
        },
    },
}


@pytest.fixture(scope='module')
def pkg(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp('retry_policy')
    spec_file = tmp_path / 'openapi.yaml'
    spec_file.write_text(yaml.safe_dump(SPEC), encoding='utf-8')
    Codegen(
        DocumentConfig(source=str(spec_file), output=str(tmp_path / 'policy_pkg'))
    ).generate()
    sys.path.insert(0, str(tmp_path))
    try:
        for name in [m for m in sys.modules if m.split('.')[0] == 'policy_pkg']:
            sys.modules.pop(name, None)
        yield importlib.import_module('policy_pkg')
    finally:
        sys.path.remove(str(tmp_path))


class Script:
    """A transport that plays back a fixed list of outcomes, then succeeds.

    An outcome is an HTTP status, ``(status, headers)``, or an exception class
    to raise.
    """

    def __init__(self, *outcomes: Any) -> None:
        self.outcomes = list(outcomes)
        self.calls = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.calls += 1
        outcome = self.outcomes.pop(0) if self.outcomes else 200
        if isinstance(outcome, type) and issubclass(outcome, Exception):
            raise outcome('scripted failure', request=request)
        if isinstance(outcome, tuple):
            status, headers = outcome
            return httpx.Response(status, headers=headers, json={})
        return httpx.Response(outcome, json={})

    def sync_client(self, pkg: Any, **kwargs: Any) -> Any:
        kwargs.setdefault('backoff_factor', 0)
        return pkg.Client(
            http_client=httpx.Client(transport=httpx.MockTransport(self)), **kwargs
        )

    def async_client(self, pkg: Any, **kwargs: Any) -> Any:
        kwargs.setdefault('backoff_factor', 0)
        return pkg.Client(
            async_http_client=httpx.AsyncClient(transport=httpx.MockTransport(self)),
            **kwargs,
        )


def _attempts(script: Script, call) -> tuple[int, str]:
    """Run *call*; return how many requests were sent and how it ended."""
    try:
        call()
    except Exception as exc:
        return script.calls, type(exc).__name__
    return script.calls, 'ok'


class TestDecisionHelpers:
    @pytest.mark.parametrize('method', sorted(IDEMPOTENT_METHODS))
    @pytest.mark.parametrize('status', [429, 500, 502, 503, 504])
    def test_idempotent_methods_retry_every_retryable_status(self, method, status):
        assert _should_retry_response(
            method, status, {429, 500, 502, 503, 504}, IDEMPOTENT_METHODS
        )

    @pytest.mark.parametrize('method', ['POST', 'PATCH', 'post'])
    @pytest.mark.parametrize(
        ('status', 'expected'),
        [(429, True), (503, True), (500, False), (502, False), (504, False)],
    )
    def test_other_methods_retry_only_when_nothing_was_processed(
        self, method, status, expected
    ):
        assert (
            _should_retry_response(
                method, status, {429, 500, 502, 503, 504}, IDEMPOTENT_METHODS
            )
            is expected
        )

    def test_a_status_outside_retry_statuses_is_never_retried(self):
        assert not _should_retry_response('GET', 404, {429, 503}, IDEMPOTENT_METHODS)
        assert not _should_retry_response('POST', 503, {429}, IDEMPOTENT_METHODS)

    def test_opting_a_method_in_retries_it_on_every_retryable_status(self):
        assert _should_retry_response('POST', 504, {504}, {'POST'})
        assert _should_retry_response('POST', 504, {504}, {'post'})

    @pytest.mark.parametrize(
        'error', [httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout]
    )
    def test_failures_before_the_request_was_sent_retry_for_any_method(self, error):
        assert _should_retry_error('POST', error('x'), IDEMPOTENT_METHODS)

    @pytest.mark.parametrize(
        'error',
        [
            httpx.ReadTimeout,
            httpx.ReadError,
            httpx.WriteError,
            httpx.WriteTimeout,
            httpx.RemoteProtocolError,
        ],
    )
    def test_failures_after_sending_retry_only_replayable_methods(self, error):
        assert not _should_retry_error('POST', error('x'), IDEMPOTENT_METHODS)
        assert not _should_retry_error('PATCH', error('x'), IDEMPOTENT_METHODS)
        assert _should_retry_error('GET', error('x'), IDEMPOTENT_METHODS)
        assert _should_retry_error('POST', error('x'), {'POST'})


class TestBackoffDelay:
    @pytest.mark.parametrize(
        ('header', 'expected'),
        [('7', 7.0), ('2.5', 2.5), ('0', 0.0), (' 3 ', 3.0)],
    )
    def test_retry_after_is_honoured(self, header, expected):
        response = httpx.Response(503, headers={'Retry-After': header})
        assert _backoff_delay(0, 0.5, response, MAX_RETRY_DELAY)[0] == expected

    def test_a_huge_retry_after_is_capped(self):
        response = httpx.Response(503, headers={'Retry-After': '86400'})
        assert _backoff_delay(0, 0.5, response, MAX_RETRY_DELAY) == (60.0, 86400.0)

    def test_the_cap_is_adjustable(self):
        response = httpx.Response(429, headers={'Retry-After': '120'})
        assert _backoff_delay(0, 0.5, response, 300.0)[0] == 120.0
        assert _backoff_delay(0, 0.5, response, 30.0)[0] == 30.0

    @pytest.mark.parametrize(
        'header', ['-1', 'nan', 'inf', '-inf', '1e999', '', 'soon']
    )
    def test_unusable_retry_after_falls_back_to_exponential_backoff(self, header):
        response = httpx.Response(503, headers={'Retry-After': header})
        assert _backoff_delay(2, 0.5, response, MAX_RETRY_DELAY) == (2.0, None)

    def test_a_huge_attempt_number_cannot_overflow(self):
        assert _backoff_delay(5000, 0.5, None, MAX_RETRY_DELAY) == (60.0, None)

    def test_a_negative_factor_never_produces_a_negative_wait(self):
        assert _backoff_delay(1, -1.0, None, MAX_RETRY_DELAY)[0] == 0.0

    def test_sleep_receives_the_capped_value(self):
        response = httpx.Response(503, headers={'Retry-After': '86400'})
        with patch('time.sleep') as sleep:
            _retry._backoff_sleep(0, 0.5, response)
        sleep.assert_called_once_with(60.0)

    @pytest.mark.asyncio
    async def test_async_sleep_receives_the_capped_value(self):
        response = httpx.Response(503, headers={'Retry-After': 'inf'})
        with patch('asyncio.sleep') as sleep:
            await _retry._backoff_sleep_async(1, 0.5, response)
        sleep.assert_called_once_with(1.0)


class TestGeneratedClientRetries:
    def test_get_is_retried_after_a_read_timeout(self, pkg):
        script = Script(httpx.ReadTimeout)
        client = script.sync_client(pkg)
        assert _attempts(script, lambda: pkg.list_items(client=client)) == (2, 'ok')

    def test_post_is_not_repeated_after_a_read_timeout(self, pkg):
        script = Script(httpx.ReadTimeout)
        client = script.sync_client(pkg)
        assert _attempts(script, lambda: pkg.create_item(client=client)) == (
            1,
            'ReadTimeout',
        )

    @pytest.mark.parametrize('status', [500, 502, 504])
    def test_post_is_not_repeated_after_a_server_error(self, pkg, status):
        script = Script(status)
        client = script.sync_client(pkg)
        attempts, outcome = _attempts(script, lambda: pkg.create_item(client=client))
        assert attempts == 1
        assert outcome.endswith('Error')

    @pytest.mark.parametrize('status', [429, 503])
    def test_post_is_repeated_when_the_server_declined_it(self, pkg, status):
        script = Script(status)
        client = script.sync_client(pkg)
        assert _attempts(script, lambda: pkg.create_item(client=client)) == (2, 'ok')

    @pytest.mark.parametrize('error', [httpx.ConnectError, httpx.ConnectTimeout])
    def test_post_is_repeated_when_the_request_never_left(self, pkg, error):
        script = Script(error)
        client = script.sync_client(pkg)
        assert _attempts(script, lambda: pkg.create_item(client=client)) == (2, 'ok')

    def test_patch_follows_the_same_rule_as_post(self, pkg):
        script = Script(httpx.ReadTimeout)
        client = script.sync_client(pkg)
        assert _attempts(script, lambda: pkg.patch_item('1', client=client)) == (
            1,
            'ReadTimeout',
        )

    @pytest.mark.parametrize('operation', ['replace_item', 'delete_item'])
    def test_put_and_delete_are_retried(self, pkg, operation):
        script = Script(504)
        client = script.sync_client(pkg)
        call = getattr(pkg, operation)
        assert _attempts(script, lambda: call('1', client=client)) == (2, 'ok')

    def test_a_method_can_be_opted_in(self, pkg):
        script = Script(httpx.ReadTimeout, 504)
        client = script.sync_client(pkg, retry_methods={'post'})
        assert _attempts(script, lambda: pkg.create_item(client=client)) == (3, 'ok')

    def test_opting_in_does_not_change_the_default_for_other_clients(self, pkg):
        script = Script(504)
        client = script.sync_client(pkg)
        assert _attempts(script, lambda: pkg.create_item(client=client))[0] == 1

    def test_retries_are_still_bounded(self, pkg):
        script = Script(*[503] * 10)
        client = script.sync_client(pkg, max_retries=2)
        attempts, outcome = _attempts(script, lambda: pkg.create_item(client=client))
        assert (attempts, outcome) == (3, 'ServiceUnavailableError')

    def test_the_default_retry_methods_are_the_idempotent_ones(self, pkg):
        client = pkg.Client()
        assert client.retry_methods == IDEMPOTENT_METHODS

    @pytest.mark.asyncio
    async def test_async_post_is_not_repeated_after_a_read_timeout(self, pkg):
        script = Script(httpx.ReadTimeout)
        client = script.async_client(pkg)
        with pytest.raises(httpx.ReadTimeout):
            await pkg.async_create_item(client=client)
        assert script.calls == 1

    @pytest.mark.asyncio
    async def test_async_post_is_repeated_after_a_503(self, pkg):
        script = Script(503)
        client = script.async_client(pkg)
        await pkg.async_create_item(client=client)
        assert script.calls == 2

    def test_a_huge_retry_after_does_not_stall_the_client(self, pkg):
        script = Script((503, {'Retry-After': '86400'}))
        client = script.sync_client(pkg, backoff_factor=0.5)
        with patch('time.sleep') as sleep:
            pkg.list_items(client=client)
        sleep.assert_called_once_with(60.0)

    def test_the_wait_cap_is_configurable(self, pkg):
        script = Script((429, {'Retry-After': '120'}))
        client = script.sync_client(pkg, max_retry_delay=300.0)
        with patch('time.sleep') as sleep:
            pkg.list_items(client=client)
        sleep.assert_called_once_with(120.0)
