"""Tests for OpenAPI version upgrade functionality."""

import pytest
from otterapi.openapi.v3_1.v3_1 import OpenAPI as OpenAPI31
from otterapi.openapi.v3_2.v3_2 import OpenAPI as OpenAPI32


class TestOpenAPIUpgrade:
    """Test suite for upgrading OpenAPI 3.1 to 3.2."""

    def test_basic_upgrade(self):
        """Test basic upgrade from 3.1 to 3.2."""
        doc_31 = OpenAPI31.model_validate({
            'openapi': '3.1.0',
            'info': {'title': 'Test API', 'version': '1.0.0'}
        })

        doc_32 = doc_31.upgrade_to_v3_2()

        assert isinstance(doc_32, OpenAPI32)
        assert doc_32.openapi == '3.2.0'
        assert doc_32.info.title == 'Test API'
        assert doc_32.info.version == '1.0.0'

    def test_version_upgrade_preserves_patch(self):
        """Test that patch version is preserved during upgrade."""
        doc_31 = OpenAPI31.model_validate({
            'openapi': '3.1.5',
            'info': {'title': 'Test API', 'version': '1.0.0'}
        })

        doc_32 = doc_31.upgrade_to_v3_2()

        assert doc_32.openapi == '3.2.5'

    def test_version_upgrade_preserves_suffix(self):
        """Test that version suffix (e.g., -beta) is preserved."""
        doc_31 = OpenAPI31.model_validate({
            'openapi': '3.1.0-beta1',
            'info': {'title': 'Test API', 'version': '1.0.0'}
        })

        doc_32 = doc_31.upgrade_to_v3_2()

        assert doc_32.openapi == '3.2.0-beta1'

    def test_json_schema_dialect_upgrade(self):
        """Test that default jsonSchemaDialect is updated to 3.2."""
        doc_31 = OpenAPI31.model_validate({
            'openapi': '3.1.0',
            'info': {'title': 'Test API', 'version': '1.0.0'},
            'jsonSchemaDialect': 'https://spec.openapis.org/oas/3.1/dialect/base'
        })

        doc_32 = doc_31.upgrade_to_v3_2()

        assert doc_32.jsonSchemaDialect == 'https://spec.openapis.org/oas/3.2/dialect/base'

    def test_custom_json_schema_dialect_preserved(self):
        """Test that custom jsonSchemaDialect is preserved."""
        custom_dialect = 'https://json-schema.org/draft/2020-12/schema'
        doc_31 = OpenAPI31.model_validate({
            'openapi': '3.1.0',
            'info': {'title': 'Test API', 'version': '1.0.0'},
            'jsonSchemaDialect': custom_dialect
        })

        doc_32 = doc_31.upgrade_to_v3_2()

        assert doc_32.jsonSchemaDialect == custom_dialect

    def test_complex_document_upgrade(self):
        """Test upgrade of complex document with paths, schemas, and servers."""
        doc_31 = OpenAPI31.model_validate({
            'openapi': '3.1.0',
            'info': {
                'title': 'Pet Store API',
                'version': '1.0.0',
                'description': 'A sample API'
            },
            'servers': [
                {'url': 'https://api.example.com/v1'}
            ],
            'paths': {
                '/pets': {
                    'get': {
                        'operationId': 'listPets',
                        'summary': 'List all pets',
                        'responses': {
                            'default': {
                                'description': 'A list of pets',
                                'content': {
                                    'application/json': {
                                        'schema': {
                                            'type': 'array',
                                            'items': {'type': 'object'}
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            },
            'components': {
                'schemas': {
                    'Pet': {
                        'type': 'object',
                        'required': ['name'],
                        'properties': {
                            'name': {'type': 'string'},
                            'age': {'type': 'integer', 'minimum': 0}
                        }
                    }
                }
            }
        })

        doc_32 = doc_31.upgrade_to_v3_2()

        # Verify structure is preserved
        assert doc_32.info.title == 'Pet Store API'
        assert doc_32.info.description == 'A sample API'
        assert len(doc_32.servers) == 1
        assert doc_32.servers[0].url == 'https://api.example.com/v1'
        assert '/pets' in doc_32.paths.root
        assert 'Pet' in doc_32.components.schemas

        # Verify nested schema
        pet_schema = doc_32.components.schemas['Pet']
        assert pet_schema.type.value == 'object'
        assert pet_schema.required == ['name']
        assert 'name' in pet_schema.properties
        assert 'age' in pet_schema.properties

    def test_webhooks_upgrade(self):
        """Test that webhooks (new in 3.1) are preserved during upgrade."""
        doc_31 = OpenAPI31.model_validate({
            'openapi': '3.1.0',
            'info': {'title': 'Webhook API', 'version': '1.0.0'},
            'webhooks': {
                'newPet': {
                    'post': {
                        'summary': 'New pet notification',
                        'requestBody': {
                            'content': {
                                'application/json': {
                                    'schema': {
                                        'type': 'object',
                                        'properties': {
                                            'id': {'type': 'integer'},
                                            'name': {'type': 'string'}
                                        }
                                    }
                                }
                            }
                        },
                        'responses': {
                            'default': {
                                'description': 'Webhook received'
                            }
                        }
                    }
                }
            }
        })

        doc_32 = doc_31.upgrade_to_v3_2()

        assert doc_32.webhooks is not None
        assert 'newPet' in doc_32.webhooks
        assert doc_32.webhooks['newPet'].post.summary == 'New pet notification'

    def test_minimal_document_upgrade(self):
        """Test upgrade of minimal valid document."""
        doc_31 = OpenAPI31.model_validate({
            'openapi': '3.1.0',
            'info': {'title': 'Minimal API', 'version': '0.1.0'}
        })

        doc_32 = doc_31.upgrade_to_v3_2()

        assert isinstance(doc_32, OpenAPI32)
        assert doc_32.openapi == '3.2.0'
        assert doc_32.info.title == 'Minimal API'

    def test_security_schemes_upgrade(self):
        """Test that security schemes are preserved during upgrade."""
        doc_31 = OpenAPI31.model_validate({
            'openapi': '3.1.0',
            'info': {'title': 'Secure API', 'version': '1.0.0'},
            'components': {
                'securitySchemes': {
                    'bearerAuth': {
                        'type': 'http',
                        'scheme': 'bearer',
                        'bearerFormat': 'JWT'
                    }
                }
            },
            'security': [
                {'bearerAuth': []}
            ]
        })

        doc_32 = doc_31.upgrade_to_v3_2()

        assert doc_32.components.securitySchemes is not None
        assert 'bearerAuth' in doc_32.components.securitySchemes
        assert doc_32.security is not None
        assert len(doc_32.security) == 1

