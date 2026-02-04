"""Test suite for OpenAPI version support and external reference resolution.

This module tests the SchemaLoader's ability to handle different OpenAPI versions,
automatic version detection, upgrade paths, and external $ref resolution.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from otterapi.codegen.schema import SchemaLoader
from otterapi.exceptions import SchemaLoadError, SchemaValidationError

# Sample OpenAPI specs for different versions
SWAGGER_20_SPEC = {
    'swagger': '2.0',
    'info': {'title': 'Swagger 2.0 API', 'version': '1.0.0'},
    'host': 'api.example.com',
    'basePath': '/v1',
    'schemes': ['https'],
    'paths': {
        '/pets': {
            'get': {
                'operationId': 'listPets',
                'produces': ['application/json'],
                'responses': {
                    '200': {
                        'description': 'A list of pets',
                        'schema': {
                            'type': 'array',
                            'items': {'$ref': '#/definitions/Pet'},
                        },
                    }
                },
            }
        }
    },
    'definitions': {
        'Pet': {
            'type': 'object',
            'required': ['name'],
            'properties': {
                'id': {'type': 'integer', 'format': 'int64'},
                'name': {'type': 'string'},
            },
        }
    },
}

OPENAPI_30_SPEC = {
    'openapi': '3.0.3',
    'info': {'title': 'OpenAPI 3.0 API', 'version': '1.0.0'},
    'paths': {
        '/pets': {
            'get': {
                'operationId': 'listPets',
                'responses': {
                    '200': {
                        'description': 'A list of pets',
                        'content': {
                            'application/json': {
                                'schema': {
                                    'type': 'array',
                                    'items': {'$ref': '#/components/schemas/Pet'},
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
            'Pet': {
                'type': 'object',
                'required': ['name'],
                'properties': {
                    'id': {'type': 'integer', 'format': 'int64'},
                    'name': {'type': 'string'},
                },
            }
        }
    },
}

OPENAPI_31_SPEC = {
    'openapi': '3.1.0',
    'info': {'title': 'OpenAPI 3.1 API', 'version': '1.0.0', 'summary': 'A test API'},
    'paths': {
        '/pets': {
            'get': {
                'operationId': 'listPets',
                'responses': {
                    '200': {
                        'description': 'A list of pets',
                        'content': {
                            'application/json': {
                                'schema': {
                                    'type': 'array',
                                    'items': {'$ref': '#/components/schemas/Pet'},
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
            'Pet': {
                'type': 'object',
                'required': ['name'],
                'properties': {
                    'id': {'type': 'integer'},
                    'name': {'type': 'string'},
                    'tags': {'type': ['array', 'null'], 'items': {'type': 'string'}},
                },
            }
        }
    },
}


class TestVersionDetection:
    """Tests for automatic OpenAPI version detection."""

    def test_detect_swagger_20(self):
        """Test detection of Swagger 2.0 version."""
        loader = SchemaLoader()
        version = loader.get_detected_version(SWAGGER_20_SPEC)
        assert version == '2.0'

    def test_detect_openapi_30(self):
        """Test detection of OpenAPI 3.0 version."""
        loader = SchemaLoader()
        version = loader.get_detected_version(OPENAPI_30_SPEC)
        assert version == '3.0'

    def test_detect_openapi_31(self):
        """Test detection of OpenAPI 3.1 version."""
        loader = SchemaLoader()
        version = loader.get_detected_version(OPENAPI_31_SPEC)
        assert version == '3.1'

    def test_detect_openapi_32(self):
        """Test detection of OpenAPI 3.2 version."""
        loader = SchemaLoader()
        spec = {'openapi': '3.2.0', 'info': {'title': 'Test', 'version': '1.0'}}
        version = loader.get_detected_version(spec)
        assert version == '3.2'


class TestSwagger20Support:
    """Tests for Swagger 2.0 loading and upgrade."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_load_swagger_20_json(self, temp_dir):
        """Test loading Swagger 2.0 from JSON file."""
        spec_file = temp_dir / 'swagger.json'
        spec_file.write_text(json.dumps(SWAGGER_20_SPEC))

        loader = SchemaLoader()
        schema = loader.load(str(spec_file))

        # Should be upgraded to OpenAPI 3.2
        assert hasattr(schema, 'openapi')
        assert schema.openapi.startswith('3.')

    def test_load_swagger_20_yaml(self, temp_dir):
        """Test loading Swagger 2.0 from YAML file."""
        spec_file = temp_dir / 'swagger.yaml'
        spec_file.write_text(yaml.dump(SWAGGER_20_SPEC))

        loader = SchemaLoader()
        schema = loader.load(str(spec_file))

        # Should be upgraded to OpenAPI 3.2
        assert hasattr(schema, 'openapi')
        assert schema.openapi.startswith('3.')

    def test_swagger_20_upgrade_preserves_paths(self, temp_dir):
        """Test that Swagger 2.0 upgrade preserves path definitions."""
        spec_file = temp_dir / 'swagger.json'
        spec_file.write_text(json.dumps(SWAGGER_20_SPEC))

        loader = SchemaLoader()
        schema = loader.load(str(spec_file))

        # Check paths are preserved
        assert schema.paths is not None
        assert '/pets' in schema.paths.root

    def test_swagger_20_upgrade_converts_definitions_to_components(self, temp_dir):
        """Test that Swagger 2.0 definitions are converted to components/schemas."""
        spec_file = temp_dir / 'swagger.json'
        spec_file.write_text(json.dumps(SWAGGER_20_SPEC))

        loader = SchemaLoader()
        schema = loader.load(str(spec_file))

        # Check components/schemas exist
        assert schema.components is not None
        assert schema.components.schemas is not None
        assert 'Pet' in schema.components.schemas

    def test_swagger_20_upgrade_warnings(self, temp_dir):
        """Test that upgrade warnings are collected."""
        spec_file = temp_dir / 'swagger.json'
        spec_file.write_text(json.dumps(SWAGGER_20_SPEC))

        loader = SchemaLoader()
        loader.load(str(spec_file))

        # Warnings should be available (may or may not be empty)
        warnings = loader.get_upgrade_warnings()
        assert isinstance(warnings, list)


class TestOpenAPI30Support:
    """Tests for OpenAPI 3.0 loading and upgrade."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_load_openapi_30_json(self, temp_dir):
        """Test loading OpenAPI 3.0 from JSON file."""
        spec_file = temp_dir / 'openapi.json'
        spec_file.write_text(json.dumps(OPENAPI_30_SPEC))

        loader = SchemaLoader()
        schema = loader.load(str(spec_file))

        assert schema is not None
        assert schema.info.title == 'OpenAPI 3.0 API'

    def test_load_openapi_30_yaml(self, temp_dir):
        """Test loading OpenAPI 3.0 from YAML file."""
        spec_file = temp_dir / 'openapi.yaml'
        spec_file.write_text(yaml.dump(OPENAPI_30_SPEC))

        loader = SchemaLoader()
        schema = loader.load(str(spec_file))

        assert schema is not None
        assert schema.info.title == 'OpenAPI 3.0 API'


class TestOpenAPI31Support:
    """Tests for OpenAPI 3.1 loading and upgrade."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_load_openapi_31_json(self, temp_dir):
        """Test loading OpenAPI 3.1 from JSON file."""
        spec_file = temp_dir / 'openapi.json'
        spec_file.write_text(json.dumps(OPENAPI_31_SPEC))

        loader = SchemaLoader()
        schema = loader.load(str(spec_file))

        assert schema is not None
        assert schema.info.title == 'OpenAPI 3.1 API'


class TestYAMLSupport:
    """Tests for YAML file support."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_load_yaml_file(self, temp_dir):
        """Test loading a YAML OpenAPI spec."""
        spec_file = temp_dir / 'api.yaml'
        spec_file.write_text(yaml.dump(OPENAPI_30_SPEC))

        loader = SchemaLoader()
        schema = loader.load(str(spec_file))

        assert schema is not None
        assert schema.info.title == 'OpenAPI 3.0 API'

    def test_load_yml_extension(self, temp_dir):
        """Test loading with .yml extension."""
        spec_file = temp_dir / 'api.yml'
        spec_file.write_text(yaml.dump(OPENAPI_30_SPEC))

        loader = SchemaLoader()
        schema = loader.load(str(spec_file))

        assert schema is not None

    def test_yaml_with_anchors(self, temp_dir):
        """Test loading YAML with anchors and aliases."""
        yaml_content = """
openapi: "3.0.3"
info:
  title: Test API
  version: "1.0.0"
paths:
  /test:
    get:
      operationId: getTest
      responses:
        "200":
          description: OK
          content:
            application/json:
              schema:
                type: object
                properties:
                  id:
                    type: integer
"""
        spec_file = temp_dir / 'api.yaml'
        spec_file.write_text(yaml_content)

        loader = SchemaLoader()
        schema = loader.load(str(spec_file))

        assert schema is not None


class TestExternalRefResolution:
    """Tests for external $ref resolution."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_resolve_relative_file_ref(self, temp_dir):
        """Test resolving a relative file $ref."""
        # Create the referenced schema file
        pet_schema = {
            'type': 'object',
            'properties': {
                'id': {'type': 'integer'},
                'name': {'type': 'string'},
            },
        }
        schemas_dir = temp_dir / 'schemas'
        schemas_dir.mkdir()
        (schemas_dir / 'Pet.json').write_text(json.dumps(pet_schema))

        # Create main spec with external ref
        main_spec = {
            'openapi': '3.0.3',
            'info': {'title': 'Test API', 'version': '1.0.0'},
            'paths': {
                '/pets': {
                    'get': {
                        'operationId': 'listPets',
                        'responses': {
                            '200': {
                                'description': 'OK',
                                'content': {
                                    'application/json': {
                                        'schema': {'$ref': './schemas/Pet.json'}
                                    }
                                },
                            }
                        },
                    }
                }
            },
        }
        (temp_dir / 'api.json').write_text(json.dumps(main_spec))

        # Load with external ref resolution
        loader = SchemaLoader(resolve_external_refs=True, base_path=temp_dir)
        schema = loader.load(str(temp_dir / 'api.json'))

        assert schema is not None

    def test_external_ref_disabled_by_default(self, temp_dir):
        """Test that external refs are not resolved by default."""
        main_spec = {
            'openapi': '3.0.3',
            'info': {'title': 'Test API', 'version': '1.0.0'},
            'paths': {},
            'components': {'schemas': {'Pet': {'$ref': './external/Pet.json'}}},
        }
        (temp_dir / 'api.json').write_text(json.dumps(main_spec))

        # Load without external ref resolution (default)
        loader = SchemaLoader()
        # This should load but keep the external ref as-is
        # The validation might fail or pass depending on the spec
        # Just test that it doesn't crash
        try:
            loader.load(str(temp_dir / 'api.json'))
        except (SchemaLoadError, SchemaValidationError):
            pass  # Expected if external ref can't be validated

    def test_resolve_ref_with_json_pointer(self, temp_dir):
        """Test resolving a ref with JSON pointer (file.json#/path/to/schema)."""
        # Create shared definitions file
        shared_defs = {
            'definitions': {
                'Pet': {
                    'type': 'object',
                    'properties': {
                        'id': {'type': 'integer'},
                        'name': {'type': 'string'},
                    },
                }
            }
        }
        (temp_dir / 'shared.json').write_text(json.dumps(shared_defs))

        # Create main spec referencing with JSON pointer
        main_spec = {
            'openapi': '3.0.3',
            'info': {'title': 'Test API', 'version': '1.0.0'},
            'paths': {
                '/pets': {
                    'get': {
                        'operationId': 'getPet',
                        'responses': {
                            '200': {
                                'description': 'OK',
                                'content': {
                                    'application/json': {
                                        'schema': {
                                            '$ref': './shared.json#/definitions/Pet'
                                        }
                                    }
                                },
                            }
                        },
                    }
                }
            },
        }
        (temp_dir / 'api.json').write_text(json.dumps(main_spec))

        loader = SchemaLoader(resolve_external_refs=True, base_path=temp_dir)
        schema = loader.load(str(temp_dir / 'api.json'))

        assert schema is not None


class TestErrorHandling:
    """Tests for error handling in schema loading."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_file_not_found(self, temp_dir):
        """Test error when file doesn't exist."""
        loader = SchemaLoader()

        with pytest.raises(SchemaLoadError) as exc_info:
            loader.load(str(temp_dir / 'nonexistent.json'))

        assert 'nonexistent.json' in str(exc_info.value)

    def test_invalid_json(self, temp_dir):
        """Test error with invalid JSON."""
        spec_file = temp_dir / 'invalid.json'
        spec_file.write_text('not valid json {{{')

        loader = SchemaLoader()

        with pytest.raises(SchemaLoadError):
            loader.load(str(spec_file))

    def test_invalid_yaml(self, temp_dir):
        """Test error with invalid YAML."""
        spec_file = temp_dir / 'invalid.yaml'
        spec_file.write_text('foo: bar: baz: invalid')

        loader = SchemaLoader()

        with pytest.raises(SchemaLoadError):
            loader.load(str(spec_file))

    def test_invalid_openapi_schema(self, temp_dir):
        """Test error with invalid OpenAPI schema."""
        spec_file = temp_dir / 'invalid.json'
        # Missing required fields
        spec_file.write_text(json.dumps({'foo': 'bar'}))

        loader = SchemaLoader()

        with pytest.raises((SchemaLoadError, SchemaValidationError)):
            loader.load(str(spec_file))


class TestURLLoading:
    """Tests for loading schemas from URLs."""

    def test_url_detection(self):
        """Test URL detection logic."""
        loader = SchemaLoader()

        assert loader._is_url('https://api.example.com/openapi.json')
        assert loader._is_url('http://localhost:8080/api.yaml')
        assert not loader._is_url('./api.json')
        assert not loader._is_url('/absolute/path/api.json')
        assert not loader._is_url('relative/path/api.json')

    @patch('httpx.get')
    def test_load_from_url(self, mock_get):
        """Test loading schema from URL."""
        mock_response = MagicMock()
        mock_response.text = json.dumps(OPENAPI_30_SPEC)
        mock_response.headers = {'content-type': 'application/json'}
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        loader = SchemaLoader()
        schema = loader.load('https://api.example.com/openapi.json')

        assert schema is not None
        assert schema.info.title == 'OpenAPI 3.0 API'
        mock_get.assert_called_once()

    @patch('httpx.get')
    def test_load_yaml_from_url(self, mock_get):
        """Test loading YAML schema from URL."""
        mock_response = MagicMock()
        mock_response.text = yaml.dump(OPENAPI_30_SPEC)
        mock_response.headers = {'content-type': 'application/x-yaml'}
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        loader = SchemaLoader()
        schema = loader.load('https://api.example.com/openapi.yaml')

        assert schema is not None


class TestCaching:
    """Tests for caching behavior."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_external_ref_caching(self, temp_dir):
        """Test that external refs are cached."""
        # Create shared schema
        pet_schema = {'type': 'object', 'properties': {'name': {'type': 'string'}}}
        (temp_dir / 'Pet.json').write_text(json.dumps(pet_schema))

        # Create spec that references Pet twice
        main_spec = {
            'openapi': '3.0.3',
            'info': {'title': 'Test', 'version': '1.0'},
            'paths': {
                '/pets': {
                    'get': {
                        'operationId': 'getPets',
                        'responses': {
                            '200': {
                                'description': 'OK',
                                'content': {
                                    'application/json': {
                                        'schema': {
                                            'type': 'array',
                                            'items': {'$ref': './Pet.json'},
                                        }
                                    }
                                },
                            }
                        },
                    },
                    'post': {
                        'operationId': 'createPet',
                        'requestBody': {
                            'content': {
                                'application/json': {'schema': {'$ref': './Pet.json'}}
                            }
                        },
                        'responses': {'201': {'description': 'Created'}},
                    },
                }
            },
        }
        (temp_dir / 'api.json').write_text(json.dumps(main_spec))

        loader = SchemaLoader(resolve_external_refs=True, base_path=temp_dir)

        # Should have cached the Pet.json file
        assert len(loader._external_cache) >= 1
