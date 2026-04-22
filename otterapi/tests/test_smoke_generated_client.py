"""End-to-end smoke test: generate a client, import it, hit a mock backend.

This catches the "everything compiles individually but the whole pipeline is
broken" class of regressions that golden-file tests miss (silent semantic
breakage, missing imports, runtime AttributeErrors).
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


def _generate_into(target: Path) -> None:
    config = DocumentConfig(
        source=str(SPEC),
        output=str(target),
        base_url='https://example.test',
    )
    Codegen(config).generate()


def _import_generated(parent_dir: Path, package_name: str):
    """Add ``parent_dir`` to ``sys.path`` and import ``package_name`` fresh."""
    sys.path.insert(0, str(parent_dir))
    try:
        # Drop any cached submodules from earlier test runs so we always pick
        # up the freshly generated source.
        stale = [
            mod
            for mod in list(sys.modules)
            if mod == package_name or mod.startswith(package_name + '.')
        ]
        for mod in stale:
            sys.modules.pop(mod, None)
        return importlib.import_module(package_name)
    finally:
        # Remove our injected entry on the way out so other tests stay clean.
        try:
            sys.path.remove(str(parent_dir))
        except ValueError:
            pass


@pytest.fixture
def generated_client_module(tmp_path: Path):
    package_name = 'smoke_client_pkg'
    parent = tmp_path
    _generate_into(parent / package_name)
    return _import_generated(parent, package_name)


def _mock_handler(request: httpx.Request) -> httpx.Response:
    if request.url.path == '/users' and request.method == 'GET':
        return httpx.Response(
            200,
            json=[
                {'id': 1, 'username': 'alice', 'age': 30, 'tags': ['admin']},
                {'id': 2, 'username': 'bob_2', 'age': 28, 'tags': []},
            ],
        )
    return httpx.Response(404, json={'detail': 'unknown route'})


class TestSyncGeneratedClient:
    def test_list_users_returns_typed_models(self, generated_client_module):
        Client = generated_client_module.Client
        list_users = generated_client_module.list_users
        User = generated_client_module.User

        with httpx.Client(transport=httpx.MockTransport(_mock_handler)) as http:
            client = Client(http_client=http)
            users = list_users(client=client)

        assert isinstance(users, list)
        assert len(users) == 2
        assert all(isinstance(u, User) for u in users)
        assert users[0].id == 1
        assert users[0].username == 'alice'
        assert users[0].age == 30
        assert users[0].tags == ['admin']

    def test_constraints_enforced_at_validation(self, generated_client_module):
        # Wave 1.1 sanity: invalid usernames should be rejected by Pydantic
        # via the constraints we forwarded into Field().
        from pydantic import ValidationError

        User = generated_client_module.User
        with pytest.raises(ValidationError):
            User(id=1, username='Bad-Name!', age=20)
        with pytest.raises(ValidationError):
            User(id=1, username='ok', age=200)
        with pytest.raises(ValidationError):
            User(id=0, username='ok', age=20)

    def test_extra_fields_rejected(self, generated_client_module):
        # Wave 1.2: additionalProperties: false → extra='forbid'.
        from pydantic import ValidationError

        User = generated_client_module.User
        with pytest.raises(ValidationError):
            User.model_validate(
                {'id': 1, 'username': 'ok', 'age': 20, 'unexpected': 'nope'}
            )

    def test_api_error_raised_on_4xx(self, generated_client_module):
        Client = generated_client_module.Client
        list_users = generated_client_module.list_users
        # Wave 3.14: error classes are re-exported from the package root --
        # users no longer need to reach into the underscore-prefixed
        # ``_client`` submodule.
        BaseAPIError = generated_client_module.BaseAPIError

        def fail_handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(404, json={'detail': 'no users for you'})

        with httpx.Client(transport=httpx.MockTransport(fail_handler)) as http:
            client = Client(http_client=http)
            with pytest.raises(BaseAPIError) as exc_info:
                list_users(client=client)
        assert exc_info.value.status_code == 404
        assert 'no users for you' in str(exc_info.value)


class TestAsyncGeneratedClient:
    @pytest.mark.asyncio
    async def test_async_list_users_returns_typed_models(self, generated_client_module):
        Client = generated_client_module.Client
        async_list_users = generated_client_module.async_list_users
        User = generated_client_module.User

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(_mock_handler)
        ) as http:
            client = Client(async_http_client=http)
            users = await async_list_users(client=client)

        assert isinstance(users, list)
        assert len(users) == 2
        assert all(isinstance(u, User) for u in users)
        assert users[1].username == 'bob_2'


class TestGeneratedPackageStructure:
    def test_package_exposes_expected_symbols(self, generated_client_module):
        # The __init__.py re-exports the client + endpoint helpers + the
        # full per-status error hierarchy (Wave 3.14).
        for name in (
            'Client',
            'User',
            'list_users',
            'async_list_users',
            'BaseAPIError',
            'ClientError',
            'ServerError',
            'NotFoundError',
            'RateLimitError',
            'InternalServerError',
        ):
            assert hasattr(generated_client_module, name), f'missing: {name}'

    def test_models_carry_constraint_metadata(self, generated_client_module):
        User = generated_client_module.User
        # Pydantic stores Field constraints on model_fields.
        username_field = User.model_fields['username']
        meta = {type(m).__name__: m for m in username_field.metadata}
        # Each constraint becomes a separate metadata entry on the Pydantic field.
        joined = ' '.join(repr(v) for v in meta.values())
        assert 'min_length=3' in joined
        assert 'max_length=20' in joined
        assert (
            "pattern='^[a-z][a-z0-9_]*$'" in joined or "'^[a-z][a-z0-9_]*$'" in joined
        )
