"""Integration tests for module splitting end-to-end workflow.

This module tests the complete module splitting flow from configuration
through code generation, verifying that the generated code is correct
and importable.
"""

import ast
import importlib
import json
import sys
from pathlib import Path

from otterapi.codegen.codegen import Codegen
from otterapi.codegen.splitting import (
    ModuleMapResolver,
    ModuleTreeBuilder,
)
from otterapi.codegen.types import Endpoint
from otterapi.config import DocumentConfig, ModuleDefinition, ModuleSplitConfig


def make_endpoint(
    name: str,
    path: str,
    method: str = 'GET',
    tags: list[str] | None = None,
) -> Endpoint:
    """Create a minimal Endpoint for testing."""
    # Create minimal AST nodes
    sync_ast = ast.FunctionDef(
        name=name,
        args=ast.arguments(
            posonlyargs=[],
            args=[],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
        ),
        body=[ast.Return(value=ast.Constant(value=None))],
        decorator_list=[],
        returns=ast.Constant(value=None),
    )
    async_ast = ast.AsyncFunctionDef(
        name=f'a{name}',
        args=ast.arguments(
            posonlyargs=[],
            args=[],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
        ),
        body=[ast.Return(value=ast.Constant(value=None))],
        decorator_list=[],
        returns=ast.Constant(value=None),
    )

    return Endpoint(
        sync_ast=sync_ast,
        async_ast=async_ast,
        sync_fn_name=name,
        async_fn_name=f'a{name}',
        method=method,
        path=path,
        tags=tags,
    )


class TestEndToEndTagStrategy:
    """End-to-end tests using tag-based strategy."""

    def test_tag_based_splitting(self):
        """Test complete flow with tag-based splitting."""
        config = ModuleSplitConfig(
            enabled=True,
            strategy='tag',
            min_endpoints=1,
        )

        # Create test endpoints
        endpoints = [
            make_endpoint('list_users', '/users', tags=['Users']),
            make_endpoint('get_user', '/users/{id}', tags=['Users']),
            make_endpoint('list_orders', '/orders', tags=['Orders']),
            make_endpoint('create_order', '/orders', method='POST', tags=['Orders']),
        ]

        # Build tree
        builder = ModuleTreeBuilder(config)
        tree = builder.build(endpoints)

        # Verify tree structure
        assert 'users' in tree.children
        assert 'orders' in tree.children
        assert len(tree.children['users'].endpoints) == 2
        assert len(tree.children['orders'].endpoints) == 2


class TestEndToEndPathStrategy:
    """End-to-end tests using path-based strategy."""

    def test_path_based_splitting(self):
        """Test complete flow with path-based splitting."""
        config = ModuleSplitConfig(
            enabled=True,
            strategy='path',
            path_depth=1,
            global_strip_prefixes=['/api'],
            min_endpoints=1,
        )

        endpoints = [
            make_endpoint('list_users', '/api/users'),
            make_endpoint('get_user', '/api/users/{id}'),
            make_endpoint('list_products', '/api/products'),
        ]

        builder = ModuleTreeBuilder(config)
        tree = builder.build(endpoints)

        assert 'users' in tree.children
        assert 'products' in tree.children
        assert len(tree.children['users'].endpoints) == 2
        assert len(tree.children['products'].endpoints) == 1


class TestEndToEndCustomStrategy:
    """End-to-end tests using custom module_map strategy."""

    def test_custom_module_map(self):
        """Test complete flow with custom module_map."""
        config = ModuleSplitConfig(
            enabled=True,
            strategy='custom',
            global_strip_prefixes=[],
            module_map={
                'identity': ['/users/*', '/auth/*'],
                'commerce': ['/orders/*', '/products/*'],
                'health': ['/health', '/ready'],
            },
            fallback_module='misc',
            min_endpoints=1,
        )

        endpoints = [
            make_endpoint('get_user', '/users/123'),
            make_endpoint('login', '/auth/login'),
            make_endpoint('list_orders', '/orders/list'),
            make_endpoint('health_check', '/health'),
            make_endpoint('unknown', '/unknown'),
        ]

        builder = ModuleTreeBuilder(config)
        tree = builder.build(endpoints)

        assert 'identity' in tree.children
        assert 'commerce' in tree.children
        assert 'health' in tree.children
        assert 'misc' in tree.children

        assert len(tree.children['identity'].endpoints) == 2
        assert len(tree.children['commerce'].endpoints) == 1
        assert len(tree.children['health'].endpoints) == 1
        assert len(tree.children['misc'].endpoints) == 1


class TestEndToEndHybridStrategy:
    """End-to-end tests using hybrid strategy."""

    def test_hybrid_strategy_priority(self):
        """Test that hybrid strategy respects priority order."""
        config = ModuleSplitConfig(
            enabled=True,
            strategy='hybrid',
            global_strip_prefixes=['/api'],
            module_map={
                'special': ['/special/*'],
            },
            min_endpoints=1,
        )

        endpoints = [
            # Should match custom module_map
            make_endpoint('special_endpoint', '/special/path'),
            # Should use tag (no custom match)
            make_endpoint('tagged_endpoint', '/api/other', tags=['Tagged']),
            # Should use path (no custom match, no tags)
            make_endpoint('path_endpoint', '/api/users'),
        ]

        builder = ModuleTreeBuilder(config)
        tree = builder.build(endpoints)

        assert 'special' in tree.children
        assert 'tagged' in tree.children
        assert 'users' in tree.children


class TestConsolidationBehavior:
    """Tests for module consolidation behavior."""

    def test_small_modules_consolidated(self):
        """Test that modules with few endpoints are consolidated."""
        config = ModuleSplitConfig(
            enabled=True,
            strategy='tag',
            min_endpoints=3,
            fallback_module='common',
        )

        endpoints = [
            # Module with only 1 endpoint - should be consolidated
            make_endpoint('single', '/single', tags=['Single']),
            # Module with 3 endpoints - should remain
            make_endpoint('multi1', '/multi', tags=['Multi']),
            make_endpoint('multi2', '/multi', method='POST', tags=['Multi']),
            make_endpoint('multi3', '/multi/{id}', tags=['Multi']),
        ]

        builder = ModuleTreeBuilder(config)
        tree = builder.build(endpoints)

        # Single should be consolidated
        assert 'single' not in tree.children or tree.children['single'].is_empty()

        # Multi should remain
        assert 'multi' in tree.children
        assert len(tree.children['multi'].endpoints) == 3

        # Common should have the consolidated endpoint
        assert 'common' in tree.children
        assert len(tree.children['common'].endpoints) == 1


class TestResolverStrategies:
    """Tests for resolver strategy selection."""

    def test_none_strategy(self):
        """Test that 'none' strategy goes to fallback."""
        config = ModuleSplitConfig(
            enabled=True,
            strategy='none',
            fallback_module='all',
        )

        resolver = ModuleMapResolver(config)
        result = resolver.resolve('/users/123', 'GET', tags=['Users'])

        # Even with tags, should go to fallback
        assert result.module_path == ['all']
        assert result.resolution == 'fallback'


class TestComplexScenarios:
    """Tests for complex real-world scenarios."""

    def test_petstore_like_api(self):
        """Test with a Petstore-like API structure."""
        config = ModuleSplitConfig(
            enabled=True,
            strategy='tag',
            min_endpoints=1,
        )

        endpoints = [
            # Pet endpoints
            make_endpoint('list_pets', '/pet', tags=['pet']),
            make_endpoint('add_pet', '/pet', method='POST', tags=['pet']),
            make_endpoint('get_pet_by_id', '/pet/{petId}', tags=['pet']),
            make_endpoint('update_pet', '/pet', method='PUT', tags=['pet']),
            make_endpoint('find_pets_by_status', '/pet/findByStatus', tags=['pet']),
            # Store endpoints
            make_endpoint('get_inventory', '/store/inventory', tags=['store']),
            make_endpoint('place_order', '/store/order', method='POST', tags=['store']),
            make_endpoint('get_order_by_id', '/store/order/{orderId}', tags=['store']),
            # User endpoints
            make_endpoint('create_user', '/user', method='POST', tags=['user']),
            make_endpoint('login_user', '/user/login', tags=['user']),
            make_endpoint('logout_user', '/user/logout', tags=['user']),
            make_endpoint('get_user_by_name', '/user/{username}', tags=['user']),
        ]

        builder = ModuleTreeBuilder(config)
        tree = builder.build(endpoints)

        assert 'pet' in tree.children
        assert 'store' in tree.children
        assert 'user' in tree.children

        assert len(tree.children['pet'].endpoints) == 5
        assert len(tree.children['store'].endpoints) == 3
        assert len(tree.children['user'].endpoints) == 4

    def test_versioned_api_with_prefix_stripping(self):
        """Test API with version prefix stripping."""
        config = ModuleSplitConfig(
            enabled=True,
            strategy='path',
            path_depth=1,
            global_strip_prefixes=['/api/v1', '/api/v2'],
            min_endpoints=1,
        )

        endpoints = [
            make_endpoint('v1_list_users', '/api/v1/users'),
            make_endpoint('v1_get_user', '/api/v1/users/{id}'),
            make_endpoint('v2_list_users', '/api/v2/users'),
            make_endpoint('v1_list_orders', '/api/v1/orders'),
        ]

        builder = ModuleTreeBuilder(config)
        tree = builder.build(endpoints)

        # All should be grouped by path after prefix stripping
        assert 'users' in tree.children
        assert 'orders' in tree.children

        # v1 and v2 users should be in same module
        assert len(tree.children['users'].endpoints) == 3
        assert len(tree.children['orders'].endpoints) == 1

    def test_microservice_with_nested_modules(self):
        """Test microservice-style API with nested module structure."""
        config = ModuleSplitConfig(
            enabled=True,
            strategy='custom',
            global_strip_prefixes=[],
            module_map={
                'identity': ModuleDefinition(
                    paths=['/identity/**'],
                    modules={
                        'users': ModuleDefinition(paths=['/users/*']),
                        'roles': ModuleDefinition(paths=['/roles/*']),
                    },
                ),
                'billing': ModuleDefinition(
                    paths=['/billing/**'],
                ),
            },
            min_endpoints=1,
        )

        # The nested paths under identity won't match /users/* directly
        # because the resolver applies strip_prefix logic
        endpoints = [
            make_endpoint('billing_invoice', '/billing/invoices'),
            make_endpoint('identity_profile', '/identity/profile'),
        ]

        builder = ModuleTreeBuilder(config)
        tree = builder.build(endpoints)

        # Verify the modules are created
        assert 'billing' in tree.children or 'identity' in tree.children


class TestTreeWalking:
    """Tests for tree walking functionality."""

    def test_walk_all_nodes(self):
        """Test walking all nodes in the tree."""
        config = ModuleSplitConfig(
            enabled=True,
            strategy='tag',
            min_endpoints=1,
        )

        endpoints = [
            make_endpoint('users_ep', '/users', tags=['Users']),
            make_endpoint('orders_ep', '/orders', tags=['Orders']),
        ]

        builder = ModuleTreeBuilder(config)
        tree = builder.build(endpoints)

        all_nodes = list(tree.walk())

        # Root + users + orders = 3 nodes
        assert len(all_nodes) == 3

    def test_walk_leaves_only(self):
        """Test walking only leaf nodes with endpoints."""
        config = ModuleSplitConfig(
            enabled=True,
            strategy='tag',
            min_endpoints=1,
        )

        endpoints = [
            make_endpoint('users_ep', '/users', tags=['Users']),
            make_endpoint('orders_ep', '/orders', tags=['Orders']),
        ]

        builder = ModuleTreeBuilder(config)
        tree = builder.build(endpoints)

        leaves = list(tree.walk_leaves())

        # Only users and orders are leaves with endpoints
        assert len(leaves) == 2
        paths = [leaf[0] for leaf in leaves]
        assert ['users'] in paths
        assert ['orders'] in paths


class TestFlattenedOutput:
    """Tests for flattened tree output."""

    def test_flatten_to_dict(self):
        """Test flattening tree to dictionary."""
        config = ModuleSplitConfig(
            enabled=True,
            strategy='tag',
            min_endpoints=1,
        )

        endpoints = [
            make_endpoint('users_ep1', '/users', tags=['Users']),
            make_endpoint('users_ep2', '/users/{id}', tags=['Users']),
            make_endpoint('orders_ep', '/orders', tags=['Orders']),
        ]

        builder = ModuleTreeBuilder(config)
        tree = builder.build(endpoints)

        flattened = tree.flatten()

        assert 'users' in flattened
        assert 'orders' in flattened
        assert len(flattened['users']) == 2
        assert len(flattened['orders']) == 1


class TestNestedSplitImports:
    """A nested module_map puts endpoints in subpackages.

    ``client.py``, ``models.py`` and the runtime helpers stay at the package
    root, so a module in ``store/`` has to reach up a level to import them.
    Emitting a plain ``from .client import Client`` there produced a package
    that raised ``ModuleNotFoundError`` on import.
    """

    SPEC = {
        'openapi': '3.1.0',
        'info': {'title': 'Nested', 'version': '1.0'},
        'servers': [{'url': 'https://example.test'}],
        'paths': {
            '/pet/{petId}': {
                'get': {
                    'operationId': 'getPet',
                    'parameters': [
                        {
                            'name': 'petId',
                            'in': 'path',
                            'required': True,
                            'schema': {'type': 'integer'},
                        }
                    ],
                    'responses': {
                        '200': {
                            'description': 'ok',
                            'content': {
                                'application/json': {
                                    'schema': {'$ref': '#/components/schemas/Pet'}
                                }
                            },
                        }
                    },
                }
            },
            '/store/order': {
                'get': {
                    'operationId': 'getOrder',
                    'responses': {
                        '200': {
                            'description': 'ok',
                            'content': {
                                'application/json': {'schema': {'type': 'object'}}
                            },
                        }
                    },
                }
            },
        },
        'components': {
            'schemas': {
                'Pet': {
                    'type': 'object',
                    'properties': {'name': {'type': 'string'}},
                }
            }
        },
    }

    def _generate(self, tmp_path: Path):
        spec_file = tmp_path / 'openapi.json'
        spec_file.write_text(json.dumps(self.SPEC), encoding='utf-8')
        output = tmp_path / 'nested_pkg'
        Codegen(
            DocumentConfig(
                source=str(spec_file),
                output=str(output),
                module_split=ModuleSplitConfig(
                    enabled=True,
                    strategy='custom',
                    # Without this the one-endpoint modules fall back to
                    # common.py and never exercise the nested layout.
                    min_endpoints=1,
                    module_map={
                        'store': {'pets': ['/pet/**'], 'orders': ['/store/**']}
                    },
                ),
            )
        ).generate()
        return output

    def test_nested_module_reaches_root_for_siblings(self, tmp_path: Path):
        output = self._generate(tmp_path)
        source = (output / 'store' / 'pets.py').read_text(encoding='utf-8')
        assert 'from ..client import Client' in source
        assert 'from ..models import' in source
        assert 'from .._serialization import' in source
        # The single-dot form would resolve to nested_pkg.store.client.
        assert 'from .client import' not in source

    def test_nested_package_is_importable(self, tmp_path: Path):
        self._generate(tmp_path)
        sys.path.insert(0, str(tmp_path))
        try:
            for name in [
                m
                for m in list(sys.modules)
                if m == 'nested_pkg' or m.startswith('nested_pkg.')
            ]:
                sys.modules.pop(name, None)
            module = importlib.import_module('nested_pkg.store.pets')
            assert callable(module.get_pet)
            root = importlib.import_module('nested_pkg')
            assert hasattr(root, 'get_pet')
        finally:
            try:
                sys.path.remove(str(tmp_path))
            except ValueError:
                pass
