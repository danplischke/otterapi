"""Tests for the module map resolver.

This module tests the ModuleMapResolver class and ResolvedModule dataclass
used for matching endpoint paths to target modules.
"""

from otterapi.codegen.splitting.resolver import ModuleMapResolver, ResolvedModule
from otterapi.config import ModuleDefinition, ModuleSplitConfig


class TestResolvedModule:
    """Tests for the ResolvedModule dataclass."""

    def test_module_name_single(self):
        """Test module_name property with single component."""
        resolved = ResolvedModule(module_path=['users'])
        assert resolved.module_name == 'users'

    def test_module_name_multiple(self):
        """Test module_name property with multiple components."""
        resolved = ResolvedModule(module_path=['api', 'v1', 'users'])
        assert resolved.module_name == 'api.v1.users'

    def test_module_name_empty(self):
        """Test module_name property with empty path."""
        resolved = ResolvedModule(module_path=[])
        assert resolved.module_name == ''

    def test_file_path_single(self):
        """Test file_path property with single component."""
        resolved = ResolvedModule(module_path=['users'])
        assert resolved.file_path == 'users.py'

    def test_file_path_nested(self):
        """Test file_path property with nested path."""
        resolved = ResolvedModule(module_path=['api', 'v1', 'users'])
        assert resolved.file_path == 'api/v1/users.py'

    def test_flat_file_path(self):
        """Test flat_file_path property."""
        resolved = ResolvedModule(module_path=['api', 'v1', 'users'])
        assert resolved.flat_file_path == 'api_v1_users.py'

    def test_resolution_types(self):
        """Test different resolution types."""
        custom = ResolvedModule(module_path=['users'], resolution='custom')
        tag = ResolvedModule(module_path=['users'], resolution='tag')
        path = ResolvedModule(module_path=['users'], resolution='path')
        fallback = ResolvedModule(module_path=['common'], resolution='fallback')

        assert custom.resolution == 'custom'
        assert tag.resolution == 'tag'
        assert path.resolution == 'path'
        assert fallback.resolution == 'fallback'


class TestModuleMapResolverBasic:
    """Basic tests for ModuleMapResolver."""

    def test_resolver_initialization(self):
        """Test resolver initialization."""
        config = ModuleSplitConfig(enabled=True)
        resolver = ModuleMapResolver(config)
        assert resolver.config == config

    def test_fallback_resolution(self):
        """Test that unmatched paths go to fallback module."""
        config = ModuleSplitConfig(
            enabled=True,
            strategy='custom',
            fallback_module='misc',
            module_map={},
        )
        resolver = ModuleMapResolver(config)
        result = resolver.resolve('/unknown/path', 'GET')

        assert result.module_path == ['misc']
        assert result.resolution == 'fallback'

    def test_custom_fallback_name(self):
        """Test custom fallback module name."""
        config = ModuleSplitConfig(
            enabled=True,
            strategy='custom',  # Use custom strategy with no module_map to force fallback
            fallback_module='other',
            module_map={},
        )
        resolver = ModuleMapResolver(config)
        result = resolver.resolve('/unmatched', 'GET')

        assert result.module_path == ['other']


class TestGlobalPrefixStripping:
    """Tests for global prefix stripping."""

    def test_strip_api_prefix(self):
        """Test stripping /api prefix."""
        config = ModuleSplitConfig(
            enabled=True,
            strategy='path',
            global_strip_prefixes=['/api'],
        )
        resolver = ModuleMapResolver(config)
        result = resolver.resolve('/api/users', 'GET')

        assert result.module_path == ['users']
        assert result.stripped_path == '/users'

    def test_strip_versioned_prefix(self):
        """Test stripping versioned API prefix."""
        config = ModuleSplitConfig(
            enabled=True,
            strategy='path',
            global_strip_prefixes=['/api/v1', '/api/v2'],
        )
        resolver = ModuleMapResolver(config)

        result_v1 = resolver.resolve('/api/v1/users', 'GET')
        assert result_v1.stripped_path == '/users'

        result_v2 = resolver.resolve('/api/v2/users', 'GET')
        assert result_v2.stripped_path == '/users'

    def test_no_matching_prefix(self):
        """Test path without matching prefix."""
        config = ModuleSplitConfig(
            enabled=True,
            strategy='path',
            global_strip_prefixes=['/api'],
        )
        resolver = ModuleMapResolver(config)
        result = resolver.resolve('/users', 'GET')

        assert result.stripped_path == '/users'

    def test_prefix_order_matters(self):
        """Test that longer prefixes should be first for correct matching."""
        config = ModuleSplitConfig(
            enabled=True,
            strategy='path',
            global_strip_prefixes=['/api/v1', '/api'],
        )
        resolver = ModuleMapResolver(config)
        result = resolver.resolve('/api/v1/users', 'GET')

        # Should strip /api/v1, not just /api
        assert result.stripped_path == '/users'


class TestCustomModuleMap:
    """Tests for custom module_map resolution."""

    def test_simple_pattern_match(self):
        """Test simple pattern matching."""
        config = ModuleSplitConfig(
            enabled=True,
            strategy='custom',
            global_strip_prefixes=[],
            module_map={'users': ['/users', '/users/*']},
        )
        resolver = ModuleMapResolver(config)

        result = resolver.resolve('/users', 'GET')
        assert result.module_path == ['users']
        assert result.resolution == 'custom'

        result = resolver.resolve('/users/123', 'GET')
        assert result.module_path == ['users']

    def test_wildcard_pattern(self):
        """Test wildcard * pattern matching."""
        config = ModuleSplitConfig(
            enabled=True,
            strategy='custom',
            global_strip_prefixes=[],
            module_map={'users': ['/users/*']},
        )
        resolver = ModuleMapResolver(config)

        # Should match
        result = resolver.resolve('/users/123', 'GET')
        assert result.module_path == ['users']

        # Should not match (no wildcard for root)
        result = resolver.resolve('/users', 'GET')
        assert result.resolution == 'fallback'

    def test_double_wildcard_pattern(self):
        """Test recursive ** pattern matching."""
        config = ModuleSplitConfig(
            enabled=True,
            strategy='custom',
            global_strip_prefixes=[],
            module_map={'users': ['/users/**']},
        )
        resolver = ModuleMapResolver(config)

        result = resolver.resolve('/users/123', 'GET')
        assert result.module_path == ['users']

        result = resolver.resolve('/users/123/profile', 'GET')
        assert result.module_path == ['users']

        result = resolver.resolve('/users/123/settings/privacy', 'GET')
        assert result.module_path == ['users']

    def test_multiple_modules(self):
        """Test multiple modules in module_map."""
        config = ModuleSplitConfig(
            enabled=True,
            strategy='custom',
            global_strip_prefixes=[],
            module_map={
                'users': ['/users/*'],
                'orders': ['/orders/*'],
                'health': ['/health', '/ready'],
            },
        )
        resolver = ModuleMapResolver(config)

        assert resolver.resolve('/users/123', 'GET').module_path == ['users']
        assert resolver.resolve('/orders/456', 'GET').module_path == ['orders']
        assert resolver.resolve('/health', 'GET').module_path == ['health']
        assert resolver.resolve('/ready', 'GET').module_path == ['health']

    def test_nested_modules(self):
        """Test nested module definitions."""
        config = ModuleSplitConfig(
            enabled=True,
            strategy='custom',
            global_strip_prefixes=[],
            module_map={
                'identity': ModuleDefinition(
                    paths=['/identity/**'],
                    modules={
                        'users': ModuleDefinition(paths=['/users/*']),
                        'auth': ModuleDefinition(paths=['/auth/*', '/login']),
                    },
                ),
            },
        )
        resolver = ModuleMapResolver(config)

        # Match the parent
        result = resolver.resolve('/identity/settings', 'GET')
        assert result.module_path == ['identity']

    def test_module_strip_prefix(self):
        """Test per-module strip_prefix."""
        config = ModuleSplitConfig(
            enabled=True,
            strategy='custom',
            global_strip_prefixes=[],
            module_map={
                'v2': ModuleDefinition(
                    paths=['/v2/**'],
                    strip_prefix='/v2',
                    modules={
                        'users': ModuleDefinition(paths=['/users/*']),
                    },
                ),
            },
        )
        resolver = ModuleMapResolver(config)

        result = resolver.resolve('/v2/users/123', 'GET')
        assert result.module_path == ['v2', 'users']


class TestTagBasedResolution:
    """Tests for tag-based resolution."""

    def test_tag_resolution(self):
        """Test resolution based on OpenAPI tags."""
        config = ModuleSplitConfig(
            enabled=True,
            strategy='tag',
        )
        resolver = ModuleMapResolver(config)

        result = resolver.resolve('/users/123', 'GET', tags=['Users'])
        assert result.module_path == ['users']
        assert result.resolution == 'tag'

    def test_first_tag_used(self):
        """Test that only the first tag is used."""
        config = ModuleSplitConfig(
            enabled=True,
            strategy='tag',
        )
        resolver = ModuleMapResolver(config)

        result = resolver.resolve('/users/123', 'GET', tags=['Users', 'Admin'])
        assert result.module_path == ['users']

    def test_no_tags_fallback(self):
        """Test fallback when no tags provided."""
        config = ModuleSplitConfig(
            enabled=True,
            strategy='tag',
            fallback_module='common',
        )
        resolver = ModuleMapResolver(config)

        result = resolver.resolve('/users/123', 'GET', tags=None)
        assert result.module_path == ['common']
        assert result.resolution == 'fallback'

    def test_empty_tags_fallback(self):
        """Test fallback when tags list is empty."""
        config = ModuleSplitConfig(
            enabled=True,
            strategy='tag',
        )
        resolver = ModuleMapResolver(config)

        result = resolver.resolve('/users/123', 'GET', tags=[])
        assert result.resolution == 'fallback'

    def test_tag_sanitization(self):
        """Test that tags are sanitized to valid Python identifiers."""
        config = ModuleSplitConfig(
            enabled=True,
            strategy='tag',
        )
        resolver = ModuleMapResolver(config)

        result = resolver.resolve('/test', 'GET', tags=['User Management'])
        assert result.module_path == ['user_management']

        result = resolver.resolve('/test', 'GET', tags=['123-invalid'])
        assert result.module_path == ['_123_invalid']


class TestPathBasedResolution:
    """Tests for path-based resolution."""

    def test_path_resolution_depth_1(self):
        """Test path resolution with depth 1."""
        config = ModuleSplitConfig(
            enabled=True,
            strategy='path',
            path_depth=1,
            global_strip_prefixes=[],
        )
        resolver = ModuleMapResolver(config)

        result = resolver.resolve('/users/123', 'GET')
        assert result.module_path == ['users']
        assert result.resolution == 'path'

    def test_path_resolution_depth_2(self):
        """Test path resolution with depth 2."""
        config = ModuleSplitConfig(
            enabled=True,
            strategy='path',
            path_depth=2,
            global_strip_prefixes=[],
        )
        resolver = ModuleMapResolver(config)

        result = resolver.resolve('/api/users/123', 'GET')
        assert result.module_path == ['api_users']

    def test_path_ignores_parameters(self):
        """Test that path parameters are ignored."""
        config = ModuleSplitConfig(
            enabled=True,
            strategy='path',
            path_depth=2,
            global_strip_prefixes=[],
        )
        resolver = ModuleMapResolver(config)

        result = resolver.resolve('/users/{id}/profile', 'GET')
        assert result.module_path == ['users_profile']

    def test_path_sanitization(self):
        """Test that path segments are sanitized."""
        config = ModuleSplitConfig(
            enabled=True,
            strategy='path',
            global_strip_prefixes=[],
        )
        resolver = ModuleMapResolver(config)

        result = resolver.resolve('/user-management/list', 'GET')
        assert result.module_path == ['user_management']

    def test_root_path_fallback(self):
        """Test fallback for root path."""
        config = ModuleSplitConfig(
            enabled=True,
            strategy='path',
        )
        resolver = ModuleMapResolver(config)

        result = resolver.resolve('/', 'GET')
        assert result.resolution == 'fallback'


class TestHybridStrategy:
    """Tests for hybrid strategy combining custom, tag, and path."""

    def test_hybrid_custom_first(self):
        """Test that custom module_map takes priority in hybrid mode."""
        config = ModuleSplitConfig(
            enabled=True,
            strategy='hybrid',
            global_strip_prefixes=[],
            module_map={'users': ['/users/*']},
        )
        resolver = ModuleMapResolver(config)

        # Custom match should take priority over tags
        result = resolver.resolve('/users/123', 'GET', tags=['Other'])
        assert result.module_path == ['users']
        assert result.resolution == 'custom'

    def test_hybrid_tag_second(self):
        """Test that tags are used when no custom match."""
        config = ModuleSplitConfig(
            enabled=True,
            strategy='hybrid',
            global_strip_prefixes=[],
            module_map={'users': ['/users/*']},
        )
        resolver = ModuleMapResolver(config)

        # No custom match, should use tag
        result = resolver.resolve('/orders/123', 'GET', tags=['Orders'])
        assert result.module_path == ['orders']
        assert result.resolution == 'tag'

    def test_hybrid_path_third(self):
        """Test that path is used when no custom or tag match."""
        config = ModuleSplitConfig(
            enabled=True,
            strategy='hybrid',
            global_strip_prefixes=[],
            module_map={'users': ['/users/*']},
        )
        resolver = ModuleMapResolver(config)

        # No custom match, no tags, should use path
        result = resolver.resolve('/orders/123', 'GET', tags=None)
        assert result.module_path == ['orders']
        assert result.resolution == 'path'

    def test_hybrid_fallback_last(self):
        """Test fallback when nothing else matches."""
        config = ModuleSplitConfig(
            enabled=True,
            strategy='hybrid',
            global_strip_prefixes=[],
            module_map={},
            fallback_module='misc',
        )
        resolver = ModuleMapResolver(config)

        result = resolver.resolve('/', 'GET', tags=None)
        assert result.module_path == ['misc']
        assert result.resolution == 'fallback'


class TestPatternMatching:
    """Tests for glob pattern matching edge cases."""

    def test_exact_match(self):
        """Test exact path matching."""
        config = ModuleSplitConfig(
            enabled=True,
            strategy='custom',
            global_strip_prefixes=[],
            module_map={'health': ['/health']},
        )
        resolver = ModuleMapResolver(config)

        assert resolver.resolve('/health', 'GET').module_path == ['health']
        assert resolver.resolve('/health/', 'GET').module_path == ['health']
        assert resolver.resolve('/healthy', 'GET').resolution == 'fallback'

    def test_trailing_slash_handling(self):
        """Test that trailing slashes are handled correctly."""
        config = ModuleSplitConfig(
            enabled=True,
            strategy='custom',
            global_strip_prefixes=[],
            module_map={'users': ['/users/*']},
        )
        resolver = ModuleMapResolver(config)

        result1 = resolver.resolve('/users/123', 'GET')
        result2 = resolver.resolve('/users/123/', 'GET')

        assert result1.module_path == result2.module_path

    def test_question_mark_pattern(self):
        """Test ? pattern for single character matching."""
        config = ModuleSplitConfig(
            enabled=True,
            strategy='custom',
            global_strip_prefixes=[],
            module_map={'version': ['/v?/users']},
        )
        resolver = ModuleMapResolver(config)

        assert resolver.resolve('/v1/users', 'GET').module_path == ['version']
        assert resolver.resolve('/v2/users', 'GET').module_path == ['version']
        # v10 has two characters, shouldn't match
        assert resolver.resolve('/v10/users', 'GET').resolution == 'fallback'
