"""Tests for module splitting configuration models.

This module tests the configuration models used for module splitting:
- SplitStrategy enum
- ModuleDefinition model
- ModuleSplitConfig model
- Configuration normalization
"""

import pytest
from pydantic import ValidationError

from otterapi.config import (
    ModuleDefinition,
    ModuleSplitConfig,
    SplitStrategy,
    _normalize_module_map,
)


class TestSplitStrategy:
    """Tests for the SplitStrategy enum."""

    def test_enum_values(self):
        """Test that all expected strategy values exist."""
        assert SplitStrategy.NONE.value == 'none'
        assert SplitStrategy.PATH.value == 'path'
        assert SplitStrategy.TAG.value == 'tag'
        assert SplitStrategy.HYBRID.value == 'hybrid'
        assert SplitStrategy.CUSTOM.value == 'custom'

    def test_enum_from_string(self):
        """Test creating enum from string values."""
        assert SplitStrategy('none') == SplitStrategy.NONE
        assert SplitStrategy('path') == SplitStrategy.PATH
        assert SplitStrategy('tag') == SplitStrategy.TAG
        assert SplitStrategy('hybrid') == SplitStrategy.HYBRID
        assert SplitStrategy('custom') == SplitStrategy.CUSTOM

    def test_invalid_strategy(self):
        """Test that invalid strategy values raise an error."""
        with pytest.raises(ValueError):
            SplitStrategy('invalid')


class TestModuleDefinition:
    """Tests for the ModuleDefinition model."""

    def test_empty_definition(self):
        """Test creating an empty ModuleDefinition."""
        definition = ModuleDefinition()
        assert definition.paths == []
        assert definition.modules == {}
        assert definition.strip_prefix is None
        assert definition.package_prefix is None
        assert definition.file_name is None
        assert definition.description is None

    def test_with_paths(self):
        """Test ModuleDefinition with paths."""
        definition = ModuleDefinition(paths=['/users/*', '/user/*'])
        assert definition.paths == ['/users/*', '/user/*']

    def test_with_nested_modules(self):
        """Test ModuleDefinition with nested modules."""
        child = ModuleDefinition(paths=['/child/*'])
        definition = ModuleDefinition(modules={'child': child})
        assert 'child' in definition.modules
        assert definition.modules['child'].paths == ['/child/*']

    def test_with_strip_prefix(self):
        """Test ModuleDefinition with strip_prefix."""
        definition = ModuleDefinition(
            paths=['/users/*'],
            strip_prefix='/api/v1',
        )
        assert definition.strip_prefix == '/api/v1'

    def test_with_description(self):
        """Test ModuleDefinition with description."""
        definition = ModuleDefinition(
            paths=['/users/*'],
            description='User management endpoints',
        )
        assert definition.description == 'User management endpoints'

    def test_full_definition(self):
        """Test ModuleDefinition with all fields."""
        definition = ModuleDefinition(
            paths=['/users/*'],
            modules={'admin': ModuleDefinition(paths=['/admin/*'])},
            strip_prefix='/api',
            package_prefix='api.users',
            file_name='user_endpoints.py',
            description='User endpoints',
        )
        assert definition.paths == ['/users/*']
        assert 'admin' in definition.modules
        assert definition.strip_prefix == '/api'
        assert definition.package_prefix == 'api.users'
        assert definition.file_name == 'user_endpoints.py'
        assert definition.description == 'User endpoints'

    def test_extra_fields_forbidden(self):
        """Test that extra fields are not allowed."""
        with pytest.raises(ValidationError):
            ModuleDefinition(paths=['/users/*'], extra_field='not allowed')


class TestModuleSplitConfig:
    """Tests for the ModuleSplitConfig model."""

    def test_default_config(self):
        """Test default ModuleSplitConfig values."""
        config = ModuleSplitConfig()
        assert config.enabled is False
        assert config.strategy == SplitStrategy.HYBRID
        assert '/api' in config.global_strip_prefixes
        assert config.path_depth == 1
        assert config.min_endpoints == 2
        assert config.fallback_module == 'common'
        assert config.module_map == {}
        assert config.flat_structure is False
        assert config.split_models is False
        assert config.shared_models_module == '_models'

    def test_enabled_config(self):
        """Test enabled ModuleSplitConfig."""
        config = ModuleSplitConfig(enabled=True, strategy='tag')
        assert config.enabled is True
        assert config.strategy == SplitStrategy.TAG

    def test_strategy_normalization(self):
        """Test that string strategies are converted to enum."""
        config = ModuleSplitConfig(strategy='path')
        assert config.strategy == SplitStrategy.PATH

        config = ModuleSplitConfig(strategy='CUSTOM')
        assert config.strategy == SplitStrategy.CUSTOM

    def test_path_depth_validation(self):
        """Test path_depth validation bounds."""
        # Valid values
        ModuleSplitConfig(path_depth=1)
        ModuleSplitConfig(path_depth=5)

        # Invalid values
        with pytest.raises(ValidationError):
            ModuleSplitConfig(path_depth=0)
        with pytest.raises(ValidationError):
            ModuleSplitConfig(path_depth=6)

    def test_min_endpoints_validation(self):
        """Test min_endpoints validation."""
        ModuleSplitConfig(min_endpoints=1)
        ModuleSplitConfig(min_endpoints=100)

        with pytest.raises(ValidationError):
            ModuleSplitConfig(min_endpoints=0)

    def test_custom_fallback_module(self):
        """Test custom fallback module name."""
        config = ModuleSplitConfig(fallback_module='misc')
        assert config.fallback_module == 'misc'

    def test_flat_structure(self):
        """Test flat structure option."""
        config = ModuleSplitConfig(flat_structure=True)
        assert config.flat_structure is True


class TestModuleMapNormalization:
    """Tests for module_map normalization."""

    def test_normalize_string_pattern(self):
        """Test normalizing a single string pattern."""
        module_map = {'users': '/users/*'}
        normalized = _normalize_module_map(module_map)

        assert 'users' in normalized
        assert isinstance(normalized['users'], ModuleDefinition)
        assert normalized['users'].paths == ['/users/*']

    def test_normalize_list_patterns(self):
        """Test normalizing a list of patterns."""
        module_map = {'users': ['/users/*', '/user/*']}
        normalized = _normalize_module_map(module_map)

        assert 'users' in normalized
        assert normalized['users'].paths == ['/users/*', '/user/*']

    def test_normalize_module_definition(self):
        """Test that ModuleDefinition passes through."""
        definition = ModuleDefinition(paths=['/users/*'], description='Users')
        module_map = {'users': definition}
        normalized = _normalize_module_map(module_map)

        assert normalized['users'] == definition

    def test_normalize_dict_as_definition(self):
        """Test normalizing a dict that should become ModuleDefinition."""
        module_map = {
            'users': {
                'paths': ['/users/*'],
                'description': 'User endpoints',
            }
        }
        normalized = _normalize_module_map(module_map)

        assert normalized['users'].paths == ['/users/*']
        assert normalized['users'].description == 'User endpoints'

    def test_normalize_nested_shorthand(self):
        """Test normalizing nested module shorthand."""
        module_map = {
            'identity': {
                'users': ['/users/*'],
                'auth': ['/auth/*', '/login'],
            }
        }
        normalized = _normalize_module_map(module_map)

        assert 'identity' in normalized
        assert 'users' in normalized['identity'].modules
        assert 'auth' in normalized['identity'].modules
        assert normalized['identity'].modules['users'].paths == ['/users/*']
        assert normalized['identity'].modules['auth'].paths == ['/auth/*', '/login']

    def test_normalize_deeply_nested(self):
        """Test normalizing deeply nested modules."""
        module_map = {
            'api': {
                'v1': {
                    'users': ['/users/*'],
                    'admin': {
                        'roles': ['/roles/*'],
                    },
                }
            }
        }
        normalized = _normalize_module_map(module_map)

        assert 'api' in normalized
        api_def = normalized['api']
        assert 'v1' in api_def.modules
        v1_def = api_def.modules['v1']
        assert 'users' in v1_def.modules
        assert 'admin' in v1_def.modules
        admin_def = v1_def.modules['admin']
        assert 'roles' in admin_def.modules
        assert admin_def.modules['roles'].paths == ['/roles/*']

    def test_config_normalizes_module_map(self):
        """Test that ModuleSplitConfig normalizes module_map on creation."""
        config = ModuleSplitConfig(
            enabled=True,
            module_map={
                'users': ['/users/*'],
                'identity': {
                    'auth': ['/auth/*'],
                },
            },
        )

        assert isinstance(config.module_map['users'], ModuleDefinition)
        assert config.module_map['users'].paths == ['/users/*']
        assert isinstance(config.module_map['identity'], ModuleDefinition)
        assert 'auth' in config.module_map['identity'].modules


class TestDocumentConfigIntegration:
    """Tests for ModuleSplitConfig integration with DocumentConfig."""

    def test_default_module_split(self):
        """Test that DocumentConfig has default ModuleSplitConfig."""
        from otterapi.config import DocumentConfig

        config = DocumentConfig(
            source='https://example.com/openapi.json',
            output='./client',
        )
        assert isinstance(config.module_split, ModuleSplitConfig)
        assert config.module_split.enabled is False

    def test_custom_module_split(self):
        """Test DocumentConfig with custom module split config."""
        from otterapi.config import DocumentConfig

        config = DocumentConfig(
            source='https://example.com/openapi.json',
            output='./client',
            module_split=ModuleSplitConfig(
                enabled=True,
                strategy='tag',
                fallback_module='misc',
            ),
        )
        assert config.module_split.enabled is True
        assert config.module_split.strategy == SplitStrategy.TAG
        assert config.module_split.fallback_module == 'misc'

    def test_module_split_from_dict(self):
        """Test DocumentConfig with module_split as dict."""
        from otterapi.config import DocumentConfig

        config = DocumentConfig(
            source='https://example.com/openapi.json',
            output='./client',
            module_split={
                'enabled': True,
                'strategy': 'custom',
                'module_map': {
                    'users': ['/users/*'],
                },
            },
        )
        assert config.module_split.enabled is True
        assert config.module_split.strategy == SplitStrategy.CUSTOM
        assert 'users' in config.module_split.module_map
