"""Test configuration for otterapi package."""

import os
import tempfile
from unittest.mock import patch

import pytest

from otterapi.config import CodegenConfig, DocumentConfig, get_config


class TestDocumentConfig:
    """Test DocumentConfig model."""

    def test_valid_document_config(self):
        """Test creating a valid DocumentConfig."""
        config = DocumentConfig(
            source='https://api.example.com/openapi.json', output='./generated'
        )
        assert config.source == 'https://api.example.com/openapi.json'
        assert config.output == './generated'
        assert config.models_file is None
        assert config.models_import_path is None
        assert config.endpoints_file is None

    def test_document_config_with_optional_fields(self):
        """Test DocumentConfig with all optional fields."""
        config = DocumentConfig(
            source='./openapi.yaml',
            output='./output',
            models_file='custom_models.py',
            models_import_path='myapp.models',
            endpoints_file='custom_endpoints.py',
        )
        assert config.source == './openapi.yaml'
        assert config.output == './output'
        assert config.models_file == 'custom_models.py'
        assert config.models_import_path == 'myapp.models'
        assert config.endpoints_file == 'custom_endpoints.py'

    def test_document_config_validation(self):
        """Test DocumentConfig validation."""
        with pytest.raises(ValueError):
            DocumentConfig()  # missing required fields


class TestCodegenConfig:
    """Test CodegenConfig model."""

    def test_valid_codegen_config(self):
        """Test creating a valid CodegenConfig."""
        doc_config = DocumentConfig(
            source='https://api.example.com/openapi.json', output='./generated'
        )
        config = CodegenConfig(documents=[doc_config])

        assert len(config.documents) == 1
        assert config.documents[0].source == 'https://api.example.com/openapi.json'
        assert config.generate_endpoints is True

    def test_codegen_config_multiple_documents(self):
        """Test CodegenConfig with multiple documents."""
        doc1 = DocumentConfig(source='api1.json', output='./gen1')
        doc2 = DocumentConfig(source='api2.json', output='./gen2')

        config = CodegenConfig(documents=[doc1, doc2], generate_endpoints=False)

        assert len(config.documents) == 2
        assert config.generate_endpoints is False

    def test_codegen_config_validation(self):
        """Test CodegenConfig validation."""
        with pytest.raises(ValueError):
            CodegenConfig()  # missing required documents field


class TestGetConfig:
    """Test get_config function."""

    def test_get_config_with_yaml_file(self):
        """Test loading config from YAML file."""
        yaml_content = """
documents:
  - source: "https://api.example.com/openapi.json"
    output: "./generated"
generate_endpoints: true
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            f.flush()

            try:
                config = get_config(f.name)
                assert len(config.documents) == 1
                assert (
                    config.documents[0].source == 'https://api.example.com/openapi.json'
                )
                assert config.generate_endpoints is True
            finally:
                os.unlink(f.name)

    def test_get_config_with_json_file(self):
        """Test loading config from JSON file."""
        json_content = """
{
  "documents": [
    {
      "source": "https://api.example.com/openapi.json",
      "output": "./generated"
    }
  ],
  "generate_endpoints": false
}
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write(json_content)
            f.flush()

            try:
                with patch(
                    'otterapi.config.CodegenConfig.model_validate_yaml'
                ) as mock_yaml:
                    mock_yaml.return_value = CodegenConfig(
                        documents=[
                            DocumentConfig(
                                source='https://api.example.com/openapi.json',
                                output='./generated',
                            )
                        ],
                        generate_endpoints=False,
                    )
                    config = get_config(f.name)
                    assert len(config.documents) == 1
                    assert config.generate_endpoints is False
            finally:
                os.unlink(f.name)

    @patch('os.getcwd')
    @patch('pathlib.Path.exists')
    @patch('pathlib.Path.read_text')
    @patch('yaml.load')
    def test_get_config_default_yaml(
        self, mock_yaml_load, mock_read_text, mock_exists, mock_getcwd
    ):
        """Test loading config from default otterapi.yaml file."""
        mock_getcwd.return_value = '/test/dir'
        mock_exists.return_value = True
        mock_read_text.return_value = 'yaml content'
        mock_yaml_load.return_value = {
            'documents': [{'source': 'api.json', 'output': './out'}],
            'generate_endpoints': True,
        }

        config = get_config()

        assert len(config.documents) == 1
        mock_exists.assert_called()
        mock_yaml_load.assert_called_once()

    @patch('os.getcwd')
    @patch('pathlib.Path.exists')
    @patch('pathlib.Path.read_text')
    @patch('tomllib.loads')
    def test_get_config_from_pyproject_toml(
        self, mock_toml_loads, mock_read_text, mock_exists, mock_getcwd
    ):
        """Test loading config from pyproject.toml file."""
        mock_getcwd.return_value = '/test/dir'

        def exists_side_effect(path):
            if 'otterapi.yaml' in str(path) or 'otterapi.yml' in str(path):
                return False
            elif 'pyproject.toml' in str(path):
                return True
            return False

        mock_exists.side_effect = exists_side_effect
        mock_read_text.return_value = 'toml content'
        mock_toml_loads.return_value = {
            'tool': {
                'otterapi': {
                    'documents': [{'source': 'api.json', 'output': './out'}],
                    'generate_endpoints': True,
                }
            }
        }

        config = get_config()

        assert len(config.documents) == 1
        mock_toml_loads.assert_called_once()

    @patch('os.getcwd')
    @patch('pathlib.Path.exists')
    def test_get_config_file_not_found(self, mock_exists, mock_getcwd):
        """Test FileNotFoundError when no config file exists."""
        mock_getcwd.return_value = '/test/dir'
        mock_exists.return_value = False

        with pytest.raises(FileNotFoundError, match='config not found'):
            get_config()

    def test_get_config_invalid_path(self):
        """Test get_config with invalid file path."""
        with pytest.raises(FileNotFoundError):
            get_config('/nonexistent/path/config.yaml')


@pytest.fixture
def sample_document_config():
    """Fixture providing a sample DocumentConfig."""
    return DocumentConfig(
        source='https://api.example.com/openapi.json',
        output='./generated',
        models_file='models.py',
        endpoints_file='endpoints.py',
    )


@pytest.fixture
def sample_codegen_config(sample_document_config):
    """Fixture providing a sample CodegenConfig."""
    return CodegenConfig(documents=[sample_document_config], generate_endpoints=True)


def test_config_serialization_roundtrip(sample_codegen_config):
    """Test that config can be serialized and deserialized."""
    # Convert to dict and back
    config_dict = sample_codegen_config.model_dump()
    restored_config = CodegenConfig.model_validate(config_dict)

    assert (
        restored_config.documents[0].source == sample_codegen_config.documents[0].source
    )
    assert (
        restored_config.documents[0].output == sample_codegen_config.documents[0].output
    )
    assert (
        restored_config.generate_endpoints == sample_codegen_config.generate_endpoints
    )
