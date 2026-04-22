"""Tests for the per-status-code exception hierarchy in generated clients.

We rely on the existing smoke fixture to generate a client, then poke at
its ``_client`` module to verify the new subclass hierarchy + dispatch.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

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


@pytest.fixture
def generated_pkg(tmp_path):
    pkg = 'per_status_pkg'
    sys.path.insert(0, str(tmp_path))
    _generate(tmp_path / pkg)
    try:
        for stale in [
            m for m in list(sys.modules) if m == pkg or m.startswith(pkg + '.')
        ]:
            sys.modules.pop(stale, None)
        yield importlib.import_module(pkg)
    finally:
        try:
            sys.path.remove(str(tmp_path))
        except ValueError:
            pass


@pytest.fixture
def errors(generated_pkg):
    return generated_pkg._client


class TestHierarchyShape:
    def test_all_specific_subclasses_emitted(self, errors):
        for name in (
            'ClientError',
            'ServerError',
            'BadRequestError',
            'UnauthorizedError',
            'ForbiddenError',
            'NotFoundError',
            'ConflictError',
            'UnprocessableEntityError',
            'RateLimitError',
            'InternalServerError',
            'BadGatewayError',
            'ServiceUnavailableError',
            'GatewayTimeoutError',
        ):
            assert hasattr(errors, name), f'missing exception class: {name}'

    def test_4xx_subclasses_inherit_from_client_error(self, errors):
        for name in (
            'BadRequestError',
            'UnauthorizedError',
            'ForbiddenError',
            'NotFoundError',
            'ConflictError',
            'UnprocessableEntityError',
            'RateLimitError',
        ):
            assert issubclass(getattr(errors, name), errors.ClientError)
            assert issubclass(getattr(errors, name), errors.BaseAPIError)

    def test_5xx_subclasses_inherit_from_server_error(self, errors):
        for name in (
            'InternalServerError',
            'BadGatewayError',
            'ServiceUnavailableError',
            'GatewayTimeoutError',
        ):
            assert issubclass(getattr(errors, name), errors.ServerError)
            assert issubclass(getattr(errors, name), errors.BaseAPIError)


class TestDispatch:
    def _raise_for(self, generated_pkg, status: int, body: str = ''):
        Client = generated_pkg.Client
        list_users = generated_pkg.list_users

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(status, json={'detail': body or 'oops'})

        with httpx.Client(transport=httpx.MockTransport(handler)) as http:
            client = Client(http_client=http)
            list_users(client=client)

    @pytest.mark.parametrize(
        'status,expected',
        [
            (400, 'BadRequestError'),
            (401, 'UnauthorizedError'),
            (403, 'ForbiddenError'),
            (404, 'NotFoundError'),
            (409, 'ConflictError'),
            (422, 'UnprocessableEntityError'),
            (429, 'RateLimitError'),
            (500, 'InternalServerError'),
            (502, 'BadGatewayError'),
            (503, 'ServiceUnavailableError'),
            (504, 'GatewayTimeoutError'),
        ],
    )
    def test_specific_status_raises_specific_class(
        self, generated_pkg, errors, status, expected
    ):
        cls = getattr(errors, expected)
        with pytest.raises(cls) as exc:
            self._raise_for(generated_pkg, status)
        assert exc.value.status_code == status

    def test_unmapped_4xx_falls_through_to_client_error(self, generated_pkg, errors):
        # 418 isn't in the registry; should still be a ClientError, not bare BaseAPIError.
        with pytest.raises(errors.ClientError) as exc:
            self._raise_for(generated_pkg, 418)
        assert exc.value.status_code == 418
        # And NOT a NotFoundError (specific subclass mismatch).
        assert not isinstance(exc.value, errors.NotFoundError)

    def test_unmapped_5xx_falls_through_to_server_error(self, generated_pkg, errors):
        with pytest.raises(errors.ServerError) as exc:
            self._raise_for(generated_pkg, 599)
        assert exc.value.status_code == 599

    def test_catch_all_via_base_class_still_works(self, generated_pkg, errors):
        # Backwards-compatibility: existing code that catches BaseAPIError
        # must still catch every status.
        with pytest.raises(errors.BaseAPIError):
            self._raise_for(generated_pkg, 404)
        with pytest.raises(errors.BaseAPIError):
            self._raise_for(generated_pkg, 500)


class TestRegistryHelpers:
    def test_resolver_helper_is_exported(self, errors):
        assert callable(errors._resolve_error_class)

    def test_resolver_returns_default_for_2xx(self, errors):
        # Successful statuses don't raise, but if someone calls the
        # resolver directly with one, ``default`` should win.
        assert (
            errors._resolve_error_class(200, errors.BaseAPIError) is errors.BaseAPIError
        )

    def test_status_map_keys_match_subclasses(self, errors):
        for status, cls in errors._STATUS_ERROR_MAP.items():
            assert isinstance(status, int)
            assert issubclass(cls, errors.BaseAPIError)
