"""Tests for the module tree builder.

This module tests the ModuleTree dataclass and ModuleTreeBuilder class
used for organizing endpoints into a hierarchical structure.
"""

import ast

from otterapi.codegen.splitting import (
    ModuleTree,
    ModuleTreeBuilder,
    build_module_tree,
)
from otterapi.codegen.types import Endpoint
from otterapi.config import ModuleSplitConfig


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
        body=[ast.Pass()],
        decorator_list=[],
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
        body=[ast.Pass()],
        decorator_list=[],
    )

    return Endpoint(
        sync_ast=sync_ast,
        async_ast=async_ast,
        sync_fn_name=name,
        async_fn_name=f'a{name}',
        name=name,
        method=method,
        path=path,
        tags=tags,
    )


class TestModuleTree:
    """Tests for the ModuleTree dataclass."""

    def test_empty_tree(self):
        """Test creating an empty tree."""
        tree = ModuleTree(name='root')
        assert tree.name == 'root'
        assert tree.endpoints == []
        assert tree.children == {}
        assert tree.definition is None
        assert tree.description is None

    def test_add_endpoint_to_root(self):
        """Test adding an endpoint to the root."""
        tree = ModuleTree(name='root')
        endpoint = make_endpoint('get_users', '/users')

        tree.add_endpoint([], endpoint)

        assert len(tree.endpoints) == 1
        assert tree.endpoints[0] == endpoint

    def test_add_endpoint_single_level(self):
        """Test adding an endpoint to a single-level path."""
        tree = ModuleTree(name='root')
        endpoint = make_endpoint('get_users', '/users')

        tree.add_endpoint(['users'], endpoint)

        assert len(tree.endpoints) == 0
        assert 'users' in tree.children
        assert len(tree.children['users'].endpoints) == 1

    def test_add_endpoint_nested_path(self):
        """Test adding an endpoint to a nested path."""
        tree = ModuleTree(name='root')
        endpoint = make_endpoint('get_user', '/api/v1/users/{id}')

        tree.add_endpoint(['api', 'v1', 'users'], endpoint)

        assert 'api' in tree.children
        assert 'v1' in tree.children['api'].children
        assert 'users' in tree.children['api'].children['v1'].children
        users_node = tree.children['api'].children['v1'].children['users']
        assert len(users_node.endpoints) == 1

    def test_add_multiple_endpoints_same_path(self):
        """Test adding multiple endpoints to the same path."""
        tree = ModuleTree(name='root')
        ep1 = make_endpoint('get_users', '/users', 'GET')
        ep2 = make_endpoint('create_user', '/users', 'POST')

        tree.add_endpoint(['users'], ep1)
        tree.add_endpoint(['users'], ep2)

        assert len(tree.children['users'].endpoints) == 2

    def test_get_node_exists(self):
        """Test getting an existing node."""
        tree = ModuleTree(name='root')
        endpoint = make_endpoint('get_users', '/users')
        tree.add_endpoint(['api', 'users'], endpoint)

        node = tree.get_node(['api', 'users'])

        assert node is not None
        assert len(node.endpoints) == 1

    def test_get_node_not_exists(self):
        """Test getting a non-existent node."""
        tree = ModuleTree(name='root')

        node = tree.get_node(['nonexistent'])

        assert node is None

    def test_get_node_empty_path(self):
        """Test getting root node with empty path."""
        tree = ModuleTree(name='root')

        node = tree.get_node([])

        assert node == tree

    def test_walk_empty_tree(self):
        """Test walking an empty tree."""
        tree = ModuleTree(name='root')

        paths = list(tree.walk())

        assert len(paths) == 1
        assert paths[0] == ([], tree)

    def test_walk_with_children(self):
        """Test walking a tree with children."""
        tree = ModuleTree(name='root')
        tree.add_endpoint(['users'], make_endpoint('get_users', '/users'))
        tree.add_endpoint(['orders'], make_endpoint('get_orders', '/orders'))

        paths = list(tree.walk())

        # Root + users + orders
        assert len(paths) == 3
        path_names = ['/'.join(p[0]) for p in paths]
        assert '' in path_names  # root
        assert 'orders' in path_names
        assert 'users' in path_names

    def test_walk_nested(self):
        """Test walking a nested tree."""
        tree = ModuleTree(name='root')
        tree.add_endpoint(['api', 'v1', 'users'], make_endpoint('get_users', '/users'))

        paths = list(tree.walk())

        # Root + api + api/v1 + api/v1/users
        assert len(paths) == 4

    def test_walk_leaves_empty_tree(self):
        """Test walk_leaves on empty tree."""
        tree = ModuleTree(name='root')

        leaves = list(tree.walk_leaves())

        assert len(leaves) == 0

    def test_walk_leaves_single_level(self):
        """Test walk_leaves with endpoints at leaves."""
        tree = ModuleTree(name='root')
        tree.add_endpoint(['users'], make_endpoint('get_users', '/users'))
        tree.add_endpoint(['orders'], make_endpoint('get_orders', '/orders'))

        leaves = list(tree.walk_leaves())

        assert len(leaves) == 2
        leaf_names = ['/'.join(p[0]) for p in leaves]
        assert 'users' in leaf_names
        assert 'orders' in leaf_names

    def test_walk_leaves_nested(self):
        """Test walk_leaves with nested endpoints."""
        tree = ModuleTree(name='root')
        tree.add_endpoint(['api', 'users'], make_endpoint('get_users', '/users'))

        leaves = list(tree.walk_leaves())

        # Only api/users has endpoints
        assert len(leaves) == 1
        assert leaves[0][0] == ['api', 'users']

    def test_count_endpoints_empty(self):
        """Test counting endpoints in empty tree."""
        tree = ModuleTree(name='root')

        assert tree.count_endpoints() == 0

    def test_count_endpoints_root_only(self):
        """Test counting endpoints at root."""
        tree = ModuleTree(name='root')
        tree.add_endpoint([], make_endpoint('get_root', '/'))

        assert tree.count_endpoints() == 1

    def test_count_endpoints_nested(self):
        """Test counting endpoints across tree."""
        tree = ModuleTree(name='root')
        tree.add_endpoint(['users'], make_endpoint('get_users', '/users'))
        tree.add_endpoint(['users'], make_endpoint('create_user', '/users'))
        tree.add_endpoint(['orders'], make_endpoint('get_orders', '/orders'))

        assert tree.count_endpoints() == 3

    def test_is_empty_true(self):
        """Test is_empty on empty tree."""
        tree = ModuleTree(name='root')

        assert tree.is_empty() is True

    def test_is_empty_false(self):
        """Test is_empty on tree with endpoints."""
        tree = ModuleTree(name='root')
        tree.add_endpoint(['users'], make_endpoint('get_users', '/users'))

        assert tree.is_empty() is False

    def test_flatten_empty(self):
        """Test flattening empty tree."""
        tree = ModuleTree(name='root')

        result = tree.flatten()

        assert result == {}

    def test_flatten_single_module(self):
        """Test flattening tree with single module."""
        tree = ModuleTree(name='root')
        tree.add_endpoint(['users'], make_endpoint('get_users', '/users'))

        result = tree.flatten()

        assert 'users' in result
        assert len(result['users']) == 1

    def test_flatten_multiple_modules(self):
        """Test flattening tree with multiple modules."""
        tree = ModuleTree(name='root')
        tree.add_endpoint(['users'], make_endpoint('get_users', '/users'))
        tree.add_endpoint(['orders'], make_endpoint('get_orders', '/orders'))

        result = tree.flatten()

        assert len(result) == 2
        assert 'users' in result
        assert 'orders' in result

    def test_flatten_nested(self):
        """Test flattening nested tree."""
        tree = ModuleTree(name='root')
        tree.add_endpoint(['api', 'v1', 'users'], make_endpoint('get_users', '/users'))

        result = tree.flatten()

        assert 'api.v1.users' in result


class TestModuleTreeBuilder:
    """Tests for the ModuleTreeBuilder class."""

    def test_builder_initialization(self):
        """Test builder initialization."""
        config = ModuleSplitConfig(enabled=True)
        builder = ModuleTreeBuilder(config)

        assert builder.config == config
        assert builder.resolver is not None

    def test_build_empty_endpoints(self):
        """Test building tree from empty endpoints list."""
        config = ModuleSplitConfig(enabled=True)
        builder = ModuleTreeBuilder(config)

        tree = builder.build([])

        assert tree.name == '__root__'
        assert tree.is_empty()

    def test_build_single_endpoint_tag_strategy(self):
        """Test building tree with single endpoint using tag strategy."""
        config = ModuleSplitConfig(
            enabled=True,
            strategy='tag',
            min_endpoints=1,  # Prevent consolidation
        )
        builder = ModuleTreeBuilder(config)
        endpoint = make_endpoint('get_users', '/users', tags=['Users'])

        tree = builder.build([endpoint])

        assert 'users' in tree.children
        assert len(tree.children['users'].endpoints) == 1

    def test_build_single_endpoint_path_strategy(self):
        """Test building tree with single endpoint using path strategy."""
        config = ModuleSplitConfig(
            enabled=True,
            strategy='path',
            global_strip_prefixes=[],
            min_endpoints=1,  # Prevent consolidation
        )
        builder = ModuleTreeBuilder(config)
        endpoint = make_endpoint('get_users', '/users/123')

        tree = builder.build([endpoint])

        assert 'users' in tree.children
        assert len(tree.children['users'].endpoints) == 1

    def test_build_multiple_endpoints_same_module(self):
        """Test building tree with multiple endpoints in same module."""
        config = ModuleSplitConfig(
            enabled=True,
            strategy='tag',
            min_endpoints=1,  # Prevent consolidation
        )
        builder = ModuleTreeBuilder(config)
        endpoints = [
            make_endpoint('get_users', '/users', tags=['Users']),
            make_endpoint('create_user', '/users', method='POST', tags=['Users']),
            make_endpoint('get_user', '/users/{id}', tags=['Users']),
        ]

        tree = builder.build(endpoints)

        assert 'users' in tree.children
        assert len(tree.children['users'].endpoints) == 3

    def test_build_multiple_modules(self):
        """Test building tree with multiple modules."""
        config = ModuleSplitConfig(
            enabled=True,
            strategy='tag',
            min_endpoints=1,  # Prevent consolidation
        )
        builder = ModuleTreeBuilder(config)
        endpoints = [
            make_endpoint('get_users', '/users', tags=['Users']),
            make_endpoint('get_orders', '/orders', tags=['Orders']),
            make_endpoint('get_products', '/products', tags=['Products']),
        ]

        tree = builder.build(endpoints)

        assert len(tree.children) == 3
        assert 'users' in tree.children
        assert 'orders' in tree.children
        assert 'products' in tree.children

    def test_build_with_custom_module_map(self):
        """Test building tree with custom module_map."""
        config = ModuleSplitConfig(
            enabled=True,
            strategy='custom',
            global_strip_prefixes=[],
            module_map={
                'users': ['/users/*', '/users'],
                'orders': ['/orders/*'],
            },
            min_endpoints=1,  # Prevent consolidation
        )
        builder = ModuleTreeBuilder(config)
        endpoints = [
            make_endpoint('get_users', '/users'),
            make_endpoint('get_user', '/users/123'),
            make_endpoint('get_orders', '/orders/456'),
        ]

        tree = builder.build(endpoints)

        assert 'users' in tree.children
        assert 'orders' in tree.children
        assert len(tree.children['users'].endpoints) == 2
        assert len(tree.children['orders'].endpoints) == 1

    def test_build_with_fallback(self):
        """Test that unmatched endpoints go to fallback module."""
        config = ModuleSplitConfig(
            enabled=True,
            strategy='custom',
            global_strip_prefixes=[],
            module_map={'users': ['/users/*']},
            fallback_module='misc',
            min_endpoints=1,  # Prevent consolidation
        )
        builder = ModuleTreeBuilder(config)
        endpoints = [
            make_endpoint('get_user', '/users/123'),
            make_endpoint('get_health', '/health'),
        ]

        tree = builder.build(endpoints)

        assert 'users' in tree.children
        assert 'misc' in tree.children


class TestModuleConsolidation:
    """Tests for small module consolidation."""

    def test_consolidate_small_modules(self):
        """Test that small modules are consolidated to fallback."""
        config = ModuleSplitConfig(
            enabled=True,
            strategy='tag',
            min_endpoints=3,
            fallback_module='common',
        )
        builder = ModuleTreeBuilder(config)
        endpoints = [
            # Users has only 1 endpoint - should be consolidated
            make_endpoint('get_users', '/users', tags=['Users']),
            # Orders has 3 endpoints - should remain
            make_endpoint('get_orders', '/orders', tags=['Orders']),
            make_endpoint('create_order', '/orders', method='POST', tags=['Orders']),
            make_endpoint('get_order', '/orders/{id}', tags=['Orders']),
        ]

        tree = builder.build(endpoints)

        # Users should be consolidated into common
        assert 'users' not in tree.children or tree.children['users'].is_empty()
        assert 'orders' in tree.children
        assert len(tree.children['orders'].endpoints) == 3
        assert 'common' in tree.children
        assert len(tree.children['common'].endpoints) == 1

    def test_no_consolidation_when_min_endpoints_is_1(self):
        """Test that consolidation is disabled when min_endpoints is 1."""
        config = ModuleSplitConfig(
            enabled=True,
            strategy='tag',
            min_endpoints=1,
        )
        builder = ModuleTreeBuilder(config)
        endpoints = [
            make_endpoint('get_users', '/users', tags=['Users']),
        ]

        tree = builder.build(endpoints)

        assert 'users' in tree.children
        assert len(tree.children['users'].endpoints) == 1


class TestBuildModuleTreeFunction:
    """Tests for the convenience build_module_tree function."""

    def test_build_module_tree_function(self):
        """Test the convenience function."""
        config = ModuleSplitConfig(
            enabled=True,
            strategy='tag',
            min_endpoints=1,  # Prevent consolidation
        )
        endpoints = [
            make_endpoint('get_users', '/users', tags=['Users']),
        ]

        tree = build_module_tree(endpoints, config)

        assert isinstance(tree, ModuleTree)
        assert 'users' in tree.children


class TestTreeWithDefinitions:
    """Tests for tree nodes with ModuleDefinitions attached."""

    def test_definition_stored_on_node(self):
        """Test that ModuleDefinition is stored on matching nodes."""
        from otterapi.config import ModuleDefinition

        config = ModuleSplitConfig(
            enabled=True,
            strategy='custom',
            global_strip_prefixes=[],
            module_map={
                'users': ModuleDefinition(
                    paths=['/users/*'],
                    description='User management endpoints',
                ),
            },
            min_endpoints=1,  # Prevent consolidation
        )
        builder = ModuleTreeBuilder(config)
        endpoints = [
            make_endpoint('get_user', '/users/123'),
        ]

        tree = builder.build(endpoints)

        users_node = tree.children.get('users')
        assert users_node is not None
        assert users_node.definition is not None
        assert users_node.description == 'User management endpoints'
