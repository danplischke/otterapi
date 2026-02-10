"""Tests for pagination configuration and code generation."""

import ast
import tempfile
from pathlib import Path

import pytest

from otterapi.codegen.endpoints import (
    build_standalone_paginated_fn,
    build_standalone_paginated_iter_fn,
)
from otterapi.codegen.pagination import (
    PAGINATION_MODULE_CONTENT,
    PaginationMethodConfig,
    endpoint_is_paginated,
    generate_pagination_module,
    get_pagination_config_for_endpoint,
)
from otterapi.config import (
    EndpointPaginationConfig,
    PaginationConfig,
    PaginationStyle,
    ResolvedPaginationConfig,
)


class TestEndpointPaginationConfig:
    """Tests for EndpointPaginationConfig model."""

    def test_default_values(self):
        """Test that default values are None (inherit from parent)."""
        config = EndpointPaginationConfig()
        assert config.enabled is None
        assert config.style is None
        assert config.offset_param is None
        assert config.limit_param is None
        assert config.cursor_param is None
        assert config.page_param is None
        assert config.per_page_param is None
        assert config.data_path is None
        assert config.total_path is None
        assert config.next_cursor_path is None
        assert config.total_pages_path is None
        assert config.default_page_size is None
        assert config.max_page_size is None

    def test_explicit_values(self):
        """Test setting explicit values."""
        config = EndpointPaginationConfig(
            enabled=True,
            style='offset',
            offset_param='skip',
            limit_param='take',
            data_path='data.items',
            total_path='meta.total',
            default_page_size=50,
        )
        assert config.enabled is True
        assert config.style == PaginationStyle.OFFSET
        assert config.offset_param == 'skip'
        assert config.limit_param == 'take'
        assert config.data_path == 'data.items'
        assert config.total_path == 'meta.total'
        assert config.default_page_size == 50

    def test_style_normalization(self):
        """Test that style strings are normalized to enum."""
        config = EndpointPaginationConfig(style='CURSOR')
        assert config.style == PaginationStyle.CURSOR

        config = EndpointPaginationConfig(style='page')
        assert config.style == PaginationStyle.PAGE

    def test_rejects_extra_fields(self):
        """Test that extra fields are rejected."""
        with pytest.raises(Exception):
            EndpointPaginationConfig(unknown_field='value')


class TestPaginationConfig:
    """Tests for PaginationConfig model."""

    def test_default_values(self):
        """Test default configuration values."""
        config = PaginationConfig()
        assert config.enabled is False
        assert config.default_style == PaginationStyle.OFFSET
        assert config.default_page_size == 100
        assert config.default_data_path is None
        assert config.default_offset_param == 'offset'
        assert config.default_limit_param == 'limit'
        assert config.default_cursor_param == 'cursor'
        assert config.default_page_param == 'page'
        assert config.default_per_page_param == 'per_page'
        assert config.endpoints == {}

    def test_enabled_with_defaults(self):
        """Test enabled config uses correct defaults."""
        config = PaginationConfig(enabled=True)
        assert config.enabled is True
        assert config.default_style == PaginationStyle.OFFSET
        assert config.default_page_size == 100

    def test_with_endpoints(self):
        """Test configuration with endpoint overrides."""
        config = PaginationConfig(
            enabled=True,
            endpoints={
                'list_users': EndpointPaginationConfig(
                    style='offset',
                    data_path='users',
                ),
                'list_items': EndpointPaginationConfig(
                    style='cursor',
                    cursor_param='after',
                    next_cursor_path='meta.next',
                ),
            },
        )
        assert 'list_users' in config.endpoints
        assert config.endpoints['list_users'].data_path == 'users'
        assert config.endpoints['list_items'].style == PaginationStyle.CURSOR

    def test_style_normalization(self):
        """Test that default_style strings are normalized."""
        config = PaginationConfig(default_style='cursor')
        assert config.default_style == PaginationStyle.CURSOR


class TestShouldGenerateForEndpoint:
    """Tests for PaginationConfig.should_generate_for_endpoint method."""

    def test_disabled_returns_false(self):
        """Test that disabled config returns False."""
        config = PaginationConfig(enabled=False)
        should_generate, resolved = config.should_generate_for_endpoint('list_users')
        assert should_generate is False
        assert resolved is None

    def test_no_endpoint_config_returns_false(self):
        """Test that endpoints not configured return False."""
        config = PaginationConfig(enabled=True)
        should_generate, resolved = config.should_generate_for_endpoint('list_users')
        assert should_generate is False
        assert resolved is None

    def test_endpoint_explicitly_disabled(self):
        """Test that explicitly disabled endpoints return False."""
        config = PaginationConfig(
            enabled=True,
            endpoints={
                'list_users': EndpointPaginationConfig(enabled=False),
            },
        )
        should_generate, resolved = config.should_generate_for_endpoint('list_users')
        assert should_generate is False
        assert resolved is None

    def test_endpoint_configured_offset(self):
        """Test configured endpoint with offset pagination."""
        config = PaginationConfig(
            enabled=True,
            endpoints={
                'list_users': EndpointPaginationConfig(
                    style='offset',
                    data_path='users',
                    total_path='total',
                ),
            },
        )
        should_generate, resolved = config.should_generate_for_endpoint('list_users')
        assert should_generate is True
        assert resolved is not None
        assert resolved.style == PaginationStyle.OFFSET
        assert resolved.data_path == 'users'
        assert resolved.total_path == 'total'
        assert resolved.offset_param == 'offset'  # default
        assert resolved.limit_param == 'limit'  # default

    def test_endpoint_configured_cursor(self):
        """Test configured endpoint with cursor pagination."""
        config = PaginationConfig(
            enabled=True,
            endpoints={
                'list_items': EndpointPaginationConfig(
                    style='cursor',
                    cursor_param='after',
                    next_cursor_path='meta.next_cursor',
                    data_path='items',
                ),
            },
        )
        should_generate, resolved = config.should_generate_for_endpoint('list_items')
        assert should_generate is True
        assert resolved is not None
        assert resolved.style == PaginationStyle.CURSOR
        assert resolved.cursor_param == 'after'
        assert resolved.next_cursor_path == 'meta.next_cursor'
        assert resolved.data_path == 'items'

    def test_endpoint_configured_page(self):
        """Test configured endpoint with page pagination."""
        config = PaginationConfig(
            enabled=True,
            endpoints={
                'list_orders': EndpointPaginationConfig(
                    style='page',
                    page_param='p',
                    per_page_param='size',
                    total_pages_path='pagination.total_pages',
                ),
            },
        )
        should_generate, resolved = config.should_generate_for_endpoint('list_orders')
        assert should_generate is True
        assert resolved is not None
        assert resolved.style == PaginationStyle.PAGE
        assert resolved.page_param == 'p'
        assert resolved.per_page_param == 'size'
        assert resolved.total_pages_path == 'pagination.total_pages'

    def test_inherits_defaults(self):
        """Test that endpoint config inherits defaults."""
        config = PaginationConfig(
            enabled=True,
            default_style=PaginationStyle.OFFSET,
            default_page_size=50,
            default_data_path='data',
            default_offset_param='skip',
            default_limit_param='take',
            endpoints={
                'list_users': EndpointPaginationConfig(),  # minimal config
            },
        )
        should_generate, resolved = config.should_generate_for_endpoint('list_users')
        assert should_generate is True
        assert resolved is not None
        assert resolved.style == PaginationStyle.OFFSET
        assert resolved.default_page_size == 50
        assert resolved.data_path == 'data'
        assert resolved.offset_param == 'skip'
        assert resolved.limit_param == 'take'

    def test_endpoint_overrides_defaults(self):
        """Test that endpoint config overrides defaults."""
        config = PaginationConfig(
            enabled=True,
            default_page_size=100,
            default_data_path='data',
            endpoints={
                'list_users': EndpointPaginationConfig(
                    default_page_size=25,
                    data_path='users',
                ),
            },
        )
        should_generate, resolved = config.should_generate_for_endpoint('list_users')
        assert should_generate is True
        assert resolved is not None
        assert resolved.default_page_size == 25
        assert resolved.data_path == 'users'


class TestPaginationMethodConfig:
    """Tests for PaginationMethodConfig dataclass."""

    def test_default_values(self):
        """Test default values."""
        config = PaginationMethodConfig()
        assert config.style == 'offset'
        assert config.offset_param == 'offset'
        assert config.limit_param == 'limit'
        assert config.cursor_param == 'cursor'
        assert config.page_param == 'page'
        assert config.per_page_param == 'per_page'
        assert config.data_path is None
        assert config.total_path is None
        assert config.next_cursor_path is None
        assert config.total_pages_path is None
        assert config.default_page_size == 100
        assert config.max_page_size is None

    def test_custom_values(self):
        """Test setting custom values."""
        config = PaginationMethodConfig(
            style='cursor',
            cursor_param='after',
            data_path='items',
            next_cursor_path='meta.next',
            default_page_size=50,
        )
        assert config.style == 'cursor'
        assert config.cursor_param == 'after'
        assert config.data_path == 'items'
        assert config.next_cursor_path == 'meta.next'
        assert config.default_page_size == 50


class TestEndpointIsPaginated:
    """Tests for endpoint_is_paginated function."""

    def test_disabled_config(self):
        """Test with disabled pagination config."""
        config = PaginationConfig(enabled=False)
        assert endpoint_is_paginated('list_users', config) is False

    def test_unconfigured_endpoint(self):
        """Test with endpoint not in config."""
        config = PaginationConfig(enabled=True)
        assert endpoint_is_paginated('list_users', config) is False

    def test_configured_endpoint(self):
        """Test with configured endpoint."""
        config = PaginationConfig(
            enabled=True,
            endpoints={
                'list_users': EndpointPaginationConfig(style='offset'),
            },
        )
        assert endpoint_is_paginated('list_users', config) is True
        assert endpoint_is_paginated('get_user', config) is False


class TestGetPaginationConfigForEndpoint:
    """Tests for get_pagination_config_for_endpoint function."""

    def test_disabled_config(self):
        """Test with disabled pagination config."""
        config = PaginationConfig(enabled=False)
        result = get_pagination_config_for_endpoint('list_users', config)
        assert result is None

    def test_unconfigured_endpoint(self):
        """Test with endpoint not in config."""
        config = PaginationConfig(enabled=True)
        result = get_pagination_config_for_endpoint('list_users', config)
        assert result is None

    def test_configured_endpoint(self):
        """Test with configured endpoint."""
        config = PaginationConfig(
            enabled=True,
            endpoints={
                'list_users': EndpointPaginationConfig(
                    style='offset',
                    data_path='users',
                    total_path='total',
                    default_page_size=50,
                ),
            },
        )
        result = get_pagination_config_for_endpoint('list_users', config)
        assert result is not None
        assert isinstance(result, PaginationMethodConfig)
        assert result.style == 'offset'
        assert result.data_path == 'users'
        assert result.total_path == 'total'
        assert result.default_page_size == 50


class TestResolvedPaginationConfig:
    """Tests for ResolvedPaginationConfig model."""

    def test_all_fields(self):
        """Test creating a resolved config with all fields."""
        config = ResolvedPaginationConfig(
            style=PaginationStyle.OFFSET,
            offset_param='offset',
            limit_param='limit',
            cursor_param='cursor',
            page_param='page',
            per_page_param='per_page',
            data_path='data.items',
            total_path='meta.total',
            next_cursor_path=None,
            total_pages_path=None,
            default_page_size=100,
            max_page_size=1000,
        )
        assert config.style == PaginationStyle.OFFSET
        assert config.offset_param == 'offset'
        assert config.data_path == 'data.items'
        assert config.default_page_size == 100
        assert config.max_page_size == 1000

    def test_rejects_extra_fields(self):
        """Test that extra fields are rejected."""
        with pytest.raises(Exception):
            ResolvedPaginationConfig(
                style=PaginationStyle.OFFSET,
                offset_param='offset',
                limit_param='limit',
                cursor_param='cursor',
                page_param='page',
                per_page_param='per_page',
                data_path=None,
                total_path=None,
                next_cursor_path=None,
                total_pages_path=None,
                default_page_size=100,
                max_page_size=None,
                unknown_field='value',
            )


class TestPaginationStyleEnum:
    """Tests for PaginationStyle enum."""

    def test_values(self):
        """Test enum values."""
        assert PaginationStyle.OFFSET.value == 'offset'
        assert PaginationStyle.CURSOR.value == 'cursor'
        assert PaginationStyle.PAGE.value == 'page'
        assert PaginationStyle.LINK.value == 'link'

    def test_from_string(self):
        """Test creating from string."""
        assert PaginationStyle('offset') == PaginationStyle.OFFSET
        assert PaginationStyle('cursor') == PaginationStyle.CURSOR
        assert PaginationStyle('page') == PaginationStyle.PAGE


class TestPaginationModuleGeneration:
    """Tests for pagination module generation."""

    def test_generate_pagination_module(self):
        """Test that the pagination module is generated correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = generate_pagination_module(Path(tmpdir))
            assert output_path.exists()
            content = output_path.read_text()
            assert 'paginate_offset' in content
            assert 'paginate_cursor' in content
            assert 'paginate_page' in content
            assert 'iterate_offset' in content
            assert 'iterate_cursor' in content
            assert 'extract_path' in content

    def test_pagination_module_content_is_valid_python(self):
        """Test that the pagination module content is valid Python."""
        # Should not raise SyntaxError
        ast.parse(PAGINATION_MODULE_CONTENT)


class TestPaginatedFunctionGeneration:
    """Tests for paginated function code generation."""

    def test_build_offset_paginated_function(self):
        """Test generating an offset-paginated function."""
        fn_ast, imports = build_standalone_paginated_fn(
            fn_name='list_users',
            method='get',
            path='/users',
            parameters=None,
            request_body_info=None,
            response_type=None,
            pagination_style='offset',
            pagination_config={
                'offset_param': 'offset',
                'limit_param': 'limit',
                'data_path': 'users',
                'total_path': 'total',
                'default_page_size': 100,
            },
            item_type_ast=ast.Name(id='User', ctx=ast.Load()),
            docs='List all users.',
            is_async=False,
        )
        assert fn_ast.name == 'list_users'
        assert isinstance(fn_ast, ast.FunctionDef)

        # Check that the function has the expected parameters
        arg_names = [arg.arg for arg in fn_ast.args.kwonlyargs]
        assert 'offset' in arg_names
        assert 'page_size' in arg_names
        assert 'max_items' in arg_names

    def test_build_cursor_paginated_function(self):
        """Test generating a cursor-paginated function."""
        fn_ast, imports = build_standalone_paginated_fn(
            fn_name='list_items',
            method='get',
            path='/items',
            parameters=None,
            request_body_info=None,
            response_type=None,
            pagination_style='cursor',
            pagination_config={
                'cursor_param': 'after',
                'limit_param': 'limit',
                'data_path': 'items',
                'next_cursor_path': 'meta.next_cursor',
                'default_page_size': 50,
            },
            item_type_ast=ast.Name(id='Item', ctx=ast.Load()),
            docs='List all items.',
            is_async=False,
        )
        assert fn_ast.name == 'list_items'
        assert isinstance(fn_ast, ast.FunctionDef)

        arg_names = [arg.arg for arg in fn_ast.args.kwonlyargs]
        assert 'cursor' in arg_names
        assert 'page_size' in arg_names
        assert 'max_items' in arg_names

    def test_build_async_paginated_function(self):
        """Test generating an async paginated function."""
        fn_ast, imports = build_standalone_paginated_fn(
            fn_name='async_list_users',
            method='get',
            path='/users',
            parameters=None,
            request_body_info=None,
            response_type=None,
            pagination_style='offset',
            pagination_config={
                'offset_param': 'offset',
                'limit_param': 'limit',
                'data_path': 'users',
                'default_page_size': 100,
            },
            item_type_ast=ast.Name(id='User', ctx=ast.Load()),
            docs='List all users async.',
            is_async=True,
        )
        assert fn_ast.name == 'async_list_users'
        assert isinstance(fn_ast, ast.AsyncFunctionDef)

    def test_build_paginated_iter_function(self):
        """Test generating a paginated iterator function."""
        fn_ast, imports = build_standalone_paginated_iter_fn(
            fn_name='list_users_iter',
            method='get',
            path='/users',
            parameters=None,
            request_body_info=None,
            response_type=None,
            pagination_style='offset',
            pagination_config={
                'offset_param': 'offset',
                'limit_param': 'limit',
                'data_path': 'users',
                'default_page_size': 100,
            },
            item_type_ast=ast.Name(id='User', ctx=ast.Load()),
            docs='Iterate over users.',
            is_async=False,
        )
        assert fn_ast.name == 'list_users_iter'
        assert isinstance(fn_ast, ast.FunctionDef)

        # Check that Iterator is in imports
        assert 'collections.abc' in imports
        assert 'Iterator' in imports['collections.abc']

    def test_build_async_paginated_iter_function(self):
        """Test generating an async paginated iterator function."""
        fn_ast, imports = build_standalone_paginated_iter_fn(
            fn_name='async_list_users_iter',
            method='get',
            path='/users',
            parameters=None,
            request_body_info=None,
            response_type=None,
            pagination_style='cursor',
            pagination_config={
                'cursor_param': 'after',
                'limit_param': 'limit',
                'data_path': 'users',
                'next_cursor_path': 'next',
                'default_page_size': 100,
            },
            item_type_ast=ast.Name(id='User', ctx=ast.Load()),
            docs='Iterate over users async.',
            is_async=True,
        )
        assert fn_ast.name == 'async_list_users_iter'
        assert isinstance(fn_ast, ast.AsyncFunctionDef)

        # Check that AsyncIterator is in imports
        assert 'collections.abc' in imports
        assert 'AsyncIterator' in imports['collections.abc']

    def test_generated_code_is_valid_python(self):
        """Test that generated paginated function code is valid Python."""
        fn_ast, _ = build_standalone_paginated_fn(
            fn_name='list_users',
            method='get',
            path='/users',
            parameters=None,
            request_body_info=None,
            response_type=None,
            pagination_style='offset',
            pagination_config={
                'offset_param': 'offset',
                'limit_param': 'limit',
                'data_path': 'users',
                'default_page_size': 100,
            },
            item_type_ast=None,
            docs='List users.',
            is_async=False,
        )

        # Wrap in a module and compile to verify it's valid Python
        module = ast.Module(body=[fn_ast], type_ignores=[])
        ast.fix_missing_locations(module)
        # Should not raise SyntaxError
        compile(module, '<test>', 'exec')


class TestPaginationCodegenIntegration:
    """Integration tests for pagination code generation with Codegen."""

    def test_codegen_with_offset_pagination(self):
        """Test code generation with offset-based pagination."""
        import json

        from otterapi.codegen.codegen import Codegen
        from otterapi.config import DocumentConfig

        openapi_spec = {
            'openapi': '3.0.0',
            'info': {'title': 'Test API', 'version': '1.0.0'},
            'servers': [{'url': 'https://api.example.com'}],
            'paths': {
                '/users': {
                    'get': {
                        'operationId': 'listUsers',
                        'parameters': [
                            {
                                'name': 'offset',
                                'in': 'query',
                                'schema': {'type': 'integer'},
                            },
                            {
                                'name': 'limit',
                                'in': 'query',
                                'schema': {'type': 'integer'},
                            },
                        ],
                        'responses': {
                            '200': {
                                'description': 'List of users',
                                'content': {
                                    'application/json': {
                                        'schema': {
                                            'type': 'object',
                                            'properties': {
                                                'users': {
                                                    'type': 'array',
                                                    'items': {
                                                        '$ref': '#/components/schemas/User'
                                                    },
                                                },
                                                'total': {'type': 'integer'},
                                            },
                                        }
                                    }
                                },
                            }
                        },
                    }
                }
            },
            'components': {
                'schemas': {
                    'User': {
                        'type': 'object',
                        'properties': {
                            'id': {'type': 'integer'},
                            'name': {'type': 'string'},
                        },
                    }
                }
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            spec_path = Path(tmpdir) / 'openapi.json'
            spec_path.write_text(json.dumps(openapi_spec))
            output_path = Path(tmpdir) / 'client'

            config = DocumentConfig(
                source=str(spec_path),
                output=str(output_path),
                pagination=PaginationConfig(
                    enabled=True,
                    endpoints={
                        'list_users': EndpointPaginationConfig(
                            style='offset',
                            data_path='users',
                            total_path='total',
                        )
                    },
                ),
            )

            codegen = Codegen(config)
            codegen.generate()

            # Verify pagination module was generated
            assert (output_path / '_pagination.py').exists()

            # Verify endpoints.py was generated and is valid Python
            endpoints_file = output_path / 'endpoints.py'
            assert endpoints_file.exists()
            content = endpoints_file.read_text()
            ast.parse(content)  # Should not raise

            # Verify pagination functions are present
            assert 'list_users_iter' in content
            assert 'async_list_users_iter' in content
            assert 'paginate_offset' in content

            # Verify no duplicate function definitions
            assert content.count('def list_users(') == 1
            assert content.count('async def async_list_users(') == 1

    def test_codegen_with_cursor_pagination(self):
        """Test code generation with cursor-based pagination."""
        import json

        from otterapi.codegen.codegen import Codegen
        from otterapi.config import DocumentConfig

        openapi_spec = {
            'openapi': '3.0.0',
            'info': {'title': 'Test API', 'version': '1.0.0'},
            'servers': [{'url': 'https://api.example.com'}],
            'paths': {
                '/items': {
                    'get': {
                        'operationId': 'listItems',
                        'parameters': [
                            {
                                'name': 'cursor',
                                'in': 'query',
                                'schema': {'type': 'string'},
                            },
                            {
                                'name': 'limit',
                                'in': 'query',
                                'schema': {'type': 'integer'},
                            },
                        ],
                        'responses': {
                            '200': {
                                'description': 'List of items',
                                'content': {
                                    'application/json': {
                                        'schema': {
                                            'type': 'object',
                                            'properties': {
                                                'items': {
                                                    'type': 'array',
                                                    'items': {
                                                        '$ref': '#/components/schemas/Item'
                                                    },
                                                },
                                                'next_cursor': {'type': 'string'},
                                            },
                                        }
                                    }
                                },
                            }
                        },
                    }
                }
            },
            'components': {
                'schemas': {
                    'Item': {
                        'type': 'object',
                        'properties': {
                            'id': {'type': 'integer'},
                            'name': {'type': 'string'},
                        },
                    }
                }
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            spec_path = Path(tmpdir) / 'openapi.json'
            spec_path.write_text(json.dumps(openapi_spec))
            output_path = Path(tmpdir) / 'client'

            config = DocumentConfig(
                source=str(spec_path),
                output=str(output_path),
                pagination=PaginationConfig(
                    enabled=True,
                    endpoints={
                        'list_items': EndpointPaginationConfig(
                            style='cursor',
                            cursor_param='cursor',
                            data_path='items',
                            next_cursor_path='next_cursor',
                        )
                    },
                ),
            )

            codegen = Codegen(config)
            codegen.generate()

            # Verify endpoints.py was generated and is valid Python
            endpoints_file = output_path / 'endpoints.py'
            assert endpoints_file.exists()
            content = endpoints_file.read_text()
            ast.parse(content)  # Should not raise

            # Verify cursor-based pagination is used
            assert 'cursor: str | None' in content
            assert 'paginate_cursor' in content

    def test_codegen_without_pagination(self):
        """Test that code generation works normally without pagination config."""
        import json

        from otterapi.codegen.codegen import Codegen
        from otterapi.config import DocumentConfig

        openapi_spec = {
            'openapi': '3.0.0',
            'info': {'title': 'Test API', 'version': '1.0.0'},
            'servers': [{'url': 'https://api.example.com'}],
            'paths': {
                '/users': {
                    'get': {
                        'operationId': 'listUsers',
                        'responses': {
                            '200': {
                                'description': 'List of users',
                                'content': {
                                    'application/json': {
                                        'schema': {
                                            'type': 'array',
                                            'items': {
                                                '$ref': '#/components/schemas/User'
                                            },
                                        }
                                    }
                                },
                            }
                        },
                    }
                }
            },
            'components': {
                'schemas': {
                    'User': {
                        'type': 'object',
                        'properties': {
                            'id': {'type': 'integer'},
                            'name': {'type': 'string'},
                        },
                    }
                }
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            spec_path = Path(tmpdir) / 'openapi.json'
            spec_path.write_text(json.dumps(openapi_spec))
            output_path = Path(tmpdir) / 'client'

            config = DocumentConfig(
                source=str(spec_path),
                output=str(output_path),
            )

            codegen = Codegen(config)
            codegen.generate()

            # Verify pagination module was NOT generated
            assert not (output_path / '_pagination.py').exists()

            # Verify endpoints.py was generated normally
            endpoints_file = output_path / 'endpoints.py'
            assert endpoints_file.exists()
            content = endpoints_file.read_text()
            ast.parse(content)  # Should not raise

            # Verify no pagination imports
            assert 'paginate_offset' not in content
            assert 'paginate_cursor' not in content


class TestPaginationAutoDetect:
    """Tests for pagination auto-detection feature."""

    def test_auto_detect_default_enabled(self):
        """Test that auto_detect is enabled by default."""
        config = PaginationConfig(enabled=True)
        assert config.auto_detect is True

    def test_auto_detect_disabled(self):
        """Test that auto_detect can be disabled."""
        config = PaginationConfig(enabled=True, auto_detect=False)
        assert config.auto_detect is False

    def test_auto_detect_offset_pagination(self):
        """Test auto-detection of offset pagination parameters."""
        from dataclasses import dataclass

        @dataclass
        class MockParam:
            name: str

        config = PaginationConfig(enabled=True, auto_detect=True)
        params = [
            MockParam(name='offset'),
            MockParam(name='limit'),
            MockParam(name='query'),
        ]

        should_generate, resolved = config.should_generate_for_endpoint(
            'list_users', endpoint_parameters=params
        )
        assert should_generate is True
        assert resolved is not None
        assert resolved.style == PaginationStyle.OFFSET
        assert resolved.offset_param == 'offset'
        assert resolved.limit_param == 'limit'

    def test_auto_detect_cursor_pagination(self):
        """Test auto-detection of cursor pagination parameters."""
        from dataclasses import dataclass

        @dataclass
        class MockParam:
            name: str

        config = PaginationConfig(enabled=True, auto_detect=True)
        params = [
            MockParam(name='cursor'),
            MockParam(name='limit'),
            MockParam(name='query'),
        ]

        should_generate, resolved = config.should_generate_for_endpoint(
            'list_items', endpoint_parameters=params
        )
        assert should_generate is True
        assert resolved is not None
        assert resolved.style == PaginationStyle.CURSOR
        assert resolved.cursor_param == 'cursor'
        assert resolved.limit_param == 'limit'

    def test_auto_detect_page_pagination(self):
        """Test auto-detection of page pagination parameters."""
        from dataclasses import dataclass

        @dataclass
        class MockParam:
            name: str

        config = PaginationConfig(enabled=True, auto_detect=True)
        params = [
            MockParam(name='page'),
            MockParam(name='per_page'),
            MockParam(name='query'),
        ]

        should_generate, resolved = config.should_generate_for_endpoint(
            'list_orders', endpoint_parameters=params
        )
        assert should_generate is True
        assert resolved is not None
        assert resolved.style == PaginationStyle.PAGE
        assert resolved.page_param == 'page'
        assert resolved.per_page_param == 'per_page'

    def test_auto_detect_no_pagination_params(self):
        """Test that endpoints without pagination params are not detected."""
        from dataclasses import dataclass

        @dataclass
        class MockParam:
            name: str

        config = PaginationConfig(enabled=True, auto_detect=True)
        params = [MockParam(name='query'), MockParam(name='filter')]

        should_generate, resolved = config.should_generate_for_endpoint(
            'search_users', endpoint_parameters=params
        )
        assert should_generate is False
        assert resolved is None

    def test_auto_detect_custom_param_names(self):
        """Test auto-detection with custom parameter names."""
        from dataclasses import dataclass

        @dataclass
        class MockParam:
            name: str

        config = PaginationConfig(
            enabled=True,
            auto_detect=True,
            default_offset_param='skip',
            default_limit_param='take',
        )
        params = [
            MockParam(name='skip'),
            MockParam(name='take'),
            MockParam(name='query'),
        ]

        should_generate, resolved = config.should_generate_for_endpoint(
            'list_users', endpoint_parameters=params
        )
        assert should_generate is True
        assert resolved is not None
        assert resolved.style == PaginationStyle.OFFSET
        assert resolved.offset_param == 'skip'
        assert resolved.limit_param == 'take'

    def test_auto_detect_disabled_no_detection(self):
        """Test that auto_detect=False prevents auto-detection."""
        from dataclasses import dataclass

        @dataclass
        class MockParam:
            name: str

        config = PaginationConfig(enabled=True, auto_detect=False)
        params = [MockParam(name='offset'), MockParam(name='limit')]

        should_generate, resolved = config.should_generate_for_endpoint(
            'list_users', endpoint_parameters=params
        )
        assert should_generate is False
        assert resolved is None

    def test_auto_detect_no_params_passed(self):
        """Test that auto-detection is skipped when no params passed."""
        config = PaginationConfig(enabled=True, auto_detect=True)

        should_generate, resolved = config.should_generate_for_endpoint(
            'list_users', endpoint_parameters=None
        )
        assert should_generate is False
        assert resolved is None

    def test_explicit_config_takes_precedence(self):
        """Test that explicit endpoint config takes precedence over auto-detect."""
        from dataclasses import dataclass

        @dataclass
        class MockParam:
            name: str

        config = PaginationConfig(
            enabled=True,
            auto_detect=True,
            endpoints={
                'list_users': EndpointPaginationConfig(
                    style='cursor',
                    data_path='users',
                ),
            },
        )
        # Even though params suggest offset, explicit config says cursor
        params = [MockParam(name='offset'), MockParam(name='limit')]

        should_generate, resolved = config.should_generate_for_endpoint(
            'list_users', endpoint_parameters=params
        )
        assert should_generate is True
        assert resolved is not None
        assert resolved.style == PaginationStyle.CURSOR
        assert resolved.data_path == 'users'

    def test_explicit_disabled_prevents_auto_detect(self):
        """Test that explicitly disabled endpoint prevents auto-detect."""
        from dataclasses import dataclass

        @dataclass
        class MockParam:
            name: str

        config = PaginationConfig(
            enabled=True,
            auto_detect=True,
            endpoints={
                'list_users': EndpointPaginationConfig(enabled=False),
            },
        )
        params = [MockParam(name='offset'), MockParam(name='limit')]

        should_generate, resolved = config.should_generate_for_endpoint(
            'list_users', endpoint_parameters=params
        )
        assert should_generate is False
        assert resolved is None
