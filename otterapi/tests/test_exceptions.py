"""Test suite for OtterAPI exceptions and configuration.

This module provides comprehensive tests for the custom exception hierarchy
and configuration loading functionality.
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from otterapi.config import (
    CodegenConfig,
    DocumentConfig,
    _expand_env_vars,
    _expand_env_vars_recursive,
    create_default_config,
    get_config,
    load_json,
    load_yaml,
)
from otterapi.exceptions import (
    CodeGenerationError,
    ConfigurationError,
    EndpointGenerationError,
    OtterAPIError,
    OutputError,
    SchemaError,
    SchemaLoadError,
    SchemaReferenceError,
    SchemaValidationError,
    TypeGenerationError,
    UnsupportedFeatureError,
)


class TestOtterAPIError:
    """Tests for the base OtterAPIError exception."""

    def test_basic_message(self):
        """Test that the error stores the message."""
        error = OtterAPIError('Something went wrong')
        assert error.message == 'Something went wrong'
        assert str(error) == 'Something went wrong'

    def test_inheritance(self):
        """Test that OtterAPIError inherits from Exception."""
        error = OtterAPIError('Test')
        assert isinstance(error, Exception)

    def test_can_be_caught_as_exception(self):
        """Test that the error can be caught as a generic Exception."""
        with pytest.raises(Exception):
            raise OtterAPIError('Test error')


class TestSchemaErrors:
    """Tests for schema-related exceptions."""

    def test_schema_error_inheritance(self):
        """Test that SchemaError inherits from OtterAPIError."""
        error = SchemaError('Schema issue')
        assert isinstance(error, OtterAPIError)

    def test_schema_load_error_with_source(self):
        """Test SchemaLoadError with just a source."""
        error = SchemaLoadError('https://api.example.com/openapi.json')
        assert error.source == 'https://api.example.com/openapi.json'
        assert error.cause is None
        assert 'https://api.example.com/openapi.json' in str(error)

    def test_schema_load_error_with_cause(self):
        """Test SchemaLoadError with a cause exception."""
        cause = ConnectionError('Network unavailable')
        error = SchemaLoadError('https://api.example.com/openapi.json', cause=cause)
        assert error.source == 'https://api.example.com/openapi.json'
        assert error.cause == cause
        assert 'Network unavailable' in str(error)

    def test_schema_validation_error_with_errors(self):
        """Test SchemaValidationError with validation errors."""
        errors = ['Missing info.title', 'Invalid paths format']
        error = SchemaValidationError('./api.yaml', errors=errors)
        assert error.source == './api.yaml'
        assert error.errors == errors
        assert 'Missing info.title' in str(error)

    def test_schema_validation_error_without_errors(self):
        """Test SchemaValidationError without detailed errors."""
        error = SchemaValidationError('./api.yaml')
        assert error.source == './api.yaml'
        assert error.errors == []

    def test_schema_reference_error(self):
        """Test SchemaReferenceError."""
        error = SchemaReferenceError(
            '#/components/schemas/Pet', reason='Schema not found'
        )
        assert error.reference == '#/components/schemas/Pet'
        assert error.reason == 'Schema not found'
        assert '#/components/schemas/Pet' in str(error)
        assert 'Schema not found' in str(error)


class TestCodeGenerationErrors:
    """Tests for code generation exceptions."""

    def test_code_generation_error_basic(self):
        """Test basic CodeGenerationError."""
        error = CodeGenerationError('Generation failed')
        assert 'Generation failed' in str(error)

    def test_code_generation_error_with_context(self):
        """Test CodeGenerationError with context."""
        error = CodeGenerationError('Generation failed', context='Pet model')
        assert error.context == 'Pet model'
        assert 'Pet model' in str(error)

    def test_code_generation_error_with_cause(self):
        """Test CodeGenerationError with a cause."""
        cause = TypeError('Invalid type')
        error = CodeGenerationError('Generation failed', cause=cause)
        assert error.cause == cause
        assert 'Invalid type' in str(error)

    def test_type_generation_error(self):
        """Test TypeGenerationError."""
        error = TypeGenerationError('Pet', schema_path='#/components/schemas/Pet')
        assert error.type_name == 'Pet'
        assert error.schema_path == '#/components/schemas/Pet'
        assert 'Pet' in str(error)

    def test_endpoint_generation_error(self):
        """Test EndpointGenerationError."""
        error = EndpointGenerationError('listPets', method='GET', path='/pets')
        assert error.operation_id == 'listPets'
        assert error.method == 'GET'
        assert error.path == '/pets'
        assert 'listPets' in str(error)
        assert 'GET /pets' in str(error)


class TestConfigurationError:
    """Tests for ConfigurationError."""

    def test_basic_configuration_error(self):
        """Test basic ConfigurationError."""
        error = ConfigurationError('Invalid configuration')
        assert 'Invalid configuration' in str(error)

    def test_configuration_error_with_path(self):
        """Test ConfigurationError with config path."""
        error = ConfigurationError('Invalid value', config_path='./otter.yml')
        assert error.config_path == './otter.yml'
        assert './otter.yml' in str(error)

    def test_configuration_error_with_field(self):
        """Test ConfigurationError with field name."""
        error = ConfigurationError('Invalid value', field='documents[0].source')
        assert error.field == 'documents[0].source'
        assert 'documents[0].source' in str(error)


class TestOutputError:
    """Tests for OutputError."""

    def test_output_error_basic(self):
        """Test basic OutputError."""
        error = OutputError('./output/models.py')
        assert error.output_path == './output/models.py'
        assert './output/models.py' in str(error)

    def test_output_error_with_cause(self):
        """Test OutputError with cause."""
        cause = PermissionError('Access denied')
        error = OutputError('./output/models.py', cause=cause)
        assert error.cause == cause
        assert 'Access denied' in str(error)


class TestUnsupportedFeatureError:
    """Tests for UnsupportedFeatureError."""

    def test_unsupported_feature_basic(self):
        """Test basic UnsupportedFeatureError."""
        error = UnsupportedFeatureError('XML request bodies')
        assert error.feature == 'XML request bodies'
        assert 'XML request bodies' in str(error)

    def test_unsupported_feature_with_suggestion(self):
        """Test UnsupportedFeatureError with suggestion."""
        error = UnsupportedFeatureError(
            'External $ref references',
            suggestion='Use a bundler tool to inline external references',
        )
        assert error.suggestion is not None
        assert 'bundler' in str(error)


class TestEnvironmentVariableExpansion:
    """Tests for environment variable expansion in configuration."""

    def test_expand_simple_env_var(self):
        """Test expanding a simple environment variable."""
        with patch.dict(os.environ, {'TEST_VAR': 'test_value'}):
            result = _expand_env_vars('${TEST_VAR}')
            assert result == 'test_value'

    def test_expand_env_var_with_default(self):
        """Test expanding an env var with default when not set."""
        # Ensure the variable is not set
        os.environ.pop('UNSET_VAR', None)
        result = _expand_env_vars('${UNSET_VAR:-default_value}')
        assert result == 'default_value'

    def test_expand_env_var_with_default_when_set(self):
        """Test that set env var overrides default."""
        with patch.dict(os.environ, {'SET_VAR': 'actual_value'}):
            result = _expand_env_vars('${SET_VAR:-default_value}')
            assert result == 'actual_value'

    def test_expand_multiple_env_vars(self):
        """Test expanding multiple env vars in one string."""
        with patch.dict(os.environ, {'HOST': 'localhost', 'PORT': '8080'}):
            result = _expand_env_vars('http://${HOST}:${PORT}/api')
            assert result == 'http://localhost:8080/api'

    def test_expand_env_vars_recursive(self):
        """Test recursive expansion in nested structures."""
        with patch.dict(os.environ, {'API_URL': 'https://api.example.com'}):
            data = {
                'documents': [
                    {'source': '${API_URL}/openapi.json', 'output': './client'}
                ]
            }
            result = _expand_env_vars_recursive(data)
            assert (
                result['documents'][0]['source']
                == 'https://api.example.com/openapi.json'
            )


class TestDocumentConfig:
    """Tests for DocumentConfig validation."""

    def test_valid_document_config(self):
        """Test creating a valid DocumentConfig."""
        config = DocumentConfig(source='./api.yaml', output='./client')
        assert config.source == './api.yaml'
        assert config.output == './client'
        assert config.models_file == 'models.py'
        assert config.endpoints_file == 'endpoints.py'

    def test_source_cannot_be_empty(self):
        """Test that source cannot be empty."""
        with pytest.raises(ValueError):
            DocumentConfig(source='', output='./client')

    def test_source_cannot_be_whitespace(self):
        """Test that source cannot be just whitespace."""
        with pytest.raises(ValueError):
            DocumentConfig(source='   ', output='./client')

    def test_output_cannot_be_empty(self):
        """Test that output cannot be empty."""
        with pytest.raises(ValueError):
            DocumentConfig(source='./api.yaml', output='')

    def test_models_file_must_end_with_py(self):
        """Test that models_file must end with .py."""
        with pytest.raises(ValueError):
            DocumentConfig(
                source='./api.yaml', output='./client', models_file='models.txt'
            )

    def test_endpoints_file_must_end_with_py(self):
        """Test that endpoints_file must end with .py."""
        with pytest.raises(ValueError):
            DocumentConfig(
                source='./api.yaml', output='./client', endpoints_file='endpoints'
            )

    def test_custom_filenames(self):
        """Test custom model and endpoint filenames."""
        config = DocumentConfig(
            source='./api.yaml',
            output='./client',
            models_file='api_models.py',
            endpoints_file='api_endpoints.py',
        )
        assert config.models_file == 'api_models.py'
        assert config.endpoints_file == 'api_endpoints.py'

    def test_optional_fields_defaults(self):
        """Test that optional fields have correct defaults."""
        config = DocumentConfig(source='./api.yaml', output='./client')
        assert config.base_url is None
        assert config.models_import_path is None
        assert config.generate_async is True
        assert config.generate_sync is True
        assert config.client_class_name is None


class TestCodegenConfig:
    """Tests for CodegenConfig validation."""

    def test_valid_codegen_config(self):
        """Test creating a valid CodegenConfig."""
        config = CodegenConfig(
            documents=[DocumentConfig(source='./api.yaml', output='./client')]
        )
        assert len(config.documents) == 1
        assert config.generate_endpoints is True

    def test_documents_cannot_be_empty(self):
        """Test that documents list cannot be empty."""
        with pytest.raises(ValueError):
            CodegenConfig(documents=[])

    def test_multiple_documents(self):
        """Test config with multiple documents."""
        config = CodegenConfig(
            documents=[
                DocumentConfig(source='./api1.yaml', output='./client1'),
                DocumentConfig(source='./api2.yaml', output='./client2'),
            ]
        )
        assert len(config.documents) == 2


class TestConfigLoading:
    """Tests for configuration file loading."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_load_yaml_file(self, temp_dir):
        """Test loading a YAML configuration file."""
        config_file = temp_dir / 'otter.yaml'
        config_data = {'documents': [{'source': './api.yaml', 'output': './client'}]}
        config_file.write_text(yaml.dump(config_data))

        result = load_yaml(config_file)
        assert result['documents'][0]['source'] == './api.yaml'

    def test_load_json_file(self, temp_dir):
        """Test loading a JSON configuration file."""
        config_file = temp_dir / 'otter.json'
        config_data = {'documents': [{'source': './api.json', 'output': './client'}]}
        config_file.write_text(json.dumps(config_data))

        result = load_json(config_file)
        assert result['documents'][0]['source'] == './api.json'

    def test_load_yaml_file_not_found(self, temp_dir):
        """Test that loading non-existent YAML file raises error."""
        with pytest.raises(FileNotFoundError):
            load_yaml(temp_dir / 'nonexistent.yaml')

    def test_load_json_file_not_found(self, temp_dir):
        """Test that loading non-existent JSON file raises error."""
        with pytest.raises(FileNotFoundError):
            load_json(temp_dir / 'nonexistent.json')

    def test_get_config_from_yaml(self, temp_dir):
        """Test get_config with a YAML file."""
        config_file = temp_dir / 'otter.yaml'
        config_data = {'documents': [{'source': './api.yaml', 'output': './client'}]}
        config_file.write_text(yaml.dump(config_data))

        config = get_config(str(config_file))
        assert len(config.documents) == 1
        assert config.documents[0].source == './api.yaml'

    def test_get_config_from_json(self, temp_dir):
        """Test get_config with a JSON file."""
        config_file = temp_dir / 'otter.json'
        config_data = {'documents': [{'source': './api.json', 'output': './client'}]}
        config_file.write_text(json.dumps(config_data))

        config = get_config(str(config_file))
        assert len(config.documents) == 1

    def test_get_config_auto_discovery_yaml(self, temp_dir):
        """Test that get_config discovers otter.yaml in cwd."""
        config_file = temp_dir / 'otter.yaml'
        config_data = {'documents': [{'source': './api.yaml', 'output': './client'}]}
        config_file.write_text(yaml.dump(config_data))

        # Change to temp dir
        original_cwd = os.getcwd()
        try:
            os.chdir(temp_dir)
            config = get_config()
            assert len(config.documents) == 1
        finally:
            os.chdir(original_cwd)

    def test_get_config_auto_discovery_yml(self, temp_dir):
        """Test that get_config discovers otter.yml in cwd."""
        config_file = temp_dir / 'otter.yml'
        config_data = {'documents': [{'source': './api.yaml', 'output': './client'}]}
        config_file.write_text(yaml.dump(config_data))

        original_cwd = os.getcwd()
        try:
            os.chdir(temp_dir)
            config = get_config()
            assert len(config.documents) == 1
        finally:
            os.chdir(original_cwd)

    def test_get_config_from_env_vars(self, temp_dir):
        """Test get_config from environment variables."""
        original_cwd = os.getcwd()
        try:
            os.chdir(temp_dir)
            with patch.dict(
                os.environ,
                {
                    'OTTER_SOURCE': 'https://api.example.com/openapi.json',
                    'OTTER_OUTPUT': './generated',
                },
            ):
                config = get_config()
                assert len(config.documents) == 1
                assert (
                    config.documents[0].source == 'https://api.example.com/openapi.json'
                )
                assert config.documents[0].output == './generated'
        finally:
            os.chdir(original_cwd)

    def test_get_config_not_found(self, temp_dir):
        """Test that get_config raises error when no config found."""
        original_cwd = os.getcwd()
        try:
            os.chdir(temp_dir)
            # Clear any env vars that might provide config
            env_to_clear = ['OTTER_SOURCE', 'OTTER_OUTPUT']
            with patch.dict(os.environ, {k: '' for k in env_to_clear}, clear=False):
                for k in env_to_clear:
                    os.environ.pop(k, None)
                with pytest.raises(FileNotFoundError):
                    get_config()
        finally:
            os.chdir(original_cwd)

    def test_env_var_expansion_in_yaml(self, temp_dir):
        """Test that env vars are expanded when loading YAML."""
        config_file = temp_dir / 'otter.yaml'
        config_content = """
documents:
  - source: ${API_URL}/openapi.json
    output: ./client
"""
        config_file.write_text(config_content)

        with patch.dict(os.environ, {'API_URL': 'https://api.example.com'}):
            config = get_config(str(config_file))
            assert config.documents[0].source == 'https://api.example.com/openapi.json'


class TestCreateDefaultConfig:
    """Tests for create_default_config function."""

    def test_creates_valid_config(self):
        """Test that create_default_config returns valid config."""
        config_dict = create_default_config()

        # Should be a valid config structure
        assert 'documents' in config_dict
        assert len(config_dict['documents']) > 0

        # Should have required fields
        doc = config_dict['documents'][0]
        assert 'source' in doc
        assert 'output' in doc

    def test_default_config_validates(self):
        """Test that default config passes validation."""
        config_dict = create_default_config()

        # Should not raise
        config = CodegenConfig.model_validate(config_dict)
        assert len(config.documents) == 1
