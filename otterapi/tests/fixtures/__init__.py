"""Test fixtures for OtterAPI tests.

This module provides sample OpenAPI documents and utilities for testing
the code generation functionality.
"""

# Minimal OpenAPI 3.0 spec for basic testing
MINIMAL_OPENAPI_SPEC = {
    'openapi': '3.0.0',
    'info': {'title': 'Minimal API', 'version': '1.0.0'},
    'paths': {},
}

# Simple API with one endpoint
SIMPLE_API_SPEC = {
    'openapi': '3.0.0',
    'info': {
        'title': 'Simple API',
        'version': '1.0.0',
        'description': 'A simple API for testing',
    },
    'servers': [{'url': 'https://api.example.com/v1'}],
    'paths': {
        '/health': {
            'get': {
                'operationId': 'getHealth',
                'summary': 'Health check endpoint',
                'responses': {
                    '200': {
                        'description': 'Successful response',
                        'content': {
                            'application/json': {
                                'schema': {
                                    'type': 'object',
                                    'properties': {'status': {'type': 'string'}},
                                }
                            }
                        },
                    }
                },
            }
        }
    },
}

# Petstore-like API with models and multiple endpoints
PETSTORE_SPEC = {
    'openapi': '3.0.0',
    'info': {
        'title': 'Petstore API',
        'version': '1.0.0',
        'description': 'A sample Petstore API for testing',
    },
    'servers': [{'url': 'https://petstore.example.com/api/v1'}],
    'paths': {
        '/pets': {
            'get': {
                'operationId': 'listPets',
                'summary': 'List all pets',
                'parameters': [
                    {
                        'name': 'limit',
                        'in': 'query',
                        'description': 'Maximum number of pets to return',
                        'required': False,
                        'schema': {'type': 'integer', 'format': 'int32'},
                    },
                    {
                        'name': 'status',
                        'in': 'query',
                        'description': 'Filter by status',
                        'required': False,
                        'schema': {
                            'type': 'string',
                            'enum': ['available', 'pending', 'sold'],
                        },
                    },
                ],
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
            },
            'post': {
                'operationId': 'createPet',
                'summary': 'Create a pet',
                'requestBody': {
                    'description': 'Pet to create',
                    'required': True,
                    'content': {
                        'application/json': {
                            'schema': {'$ref': '#/components/schemas/CreatePetRequest'}
                        }
                    },
                },
                'responses': {
                    '201': {
                        'description': 'Pet created',
                        'content': {
                            'application/json': {
                                'schema': {'$ref': '#/components/schemas/Pet'}
                            }
                        },
                    },
                    '400': {
                        'description': 'Invalid input',
                        'content': {
                            'application/json': {
                                'schema': {'$ref': '#/components/schemas/Error'}
                            }
                        },
                    },
                },
            },
        },
        '/pets/{petId}': {
            'get': {
                'operationId': 'getPetById',
                'summary': 'Get a pet by ID',
                'parameters': [
                    {
                        'name': 'petId',
                        'in': 'path',
                        'description': 'ID of pet to return',
                        'required': True,
                        'schema': {'type': 'integer', 'format': 'int64'},
                    }
                ],
                'responses': {
                    '200': {
                        'description': 'Successful operation',
                        'content': {
                            'application/json': {
                                'schema': {'$ref': '#/components/schemas/Pet'}
                            }
                        },
                    },
                    '404': {
                        'description': 'Pet not found',
                        'content': {
                            'application/json': {
                                'schema': {'$ref': '#/components/schemas/Error'}
                            }
                        },
                    },
                },
            },
            'put': {
                'operationId': 'updatePet',
                'summary': 'Update a pet',
                'parameters': [
                    {
                        'name': 'petId',
                        'in': 'path',
                        'required': True,
                        'schema': {'type': 'integer', 'format': 'int64'},
                    }
                ],
                'requestBody': {
                    'required': True,
                    'content': {
                        'application/json': {
                            'schema': {'$ref': '#/components/schemas/UpdatePetRequest'}
                        }
                    },
                },
                'responses': {
                    '200': {
                        'description': 'Pet updated',
                        'content': {
                            'application/json': {
                                'schema': {'$ref': '#/components/schemas/Pet'}
                            }
                        },
                    }
                },
            },
            'delete': {
                'operationId': 'deletePet',
                'summary': 'Delete a pet',
                'parameters': [
                    {
                        'name': 'petId',
                        'in': 'path',
                        'required': True,
                        'schema': {'type': 'integer', 'format': 'int64'},
                    }
                ],
                'responses': {'204': {'description': 'Pet deleted'}},
            },
        },
    },
    'components': {
        'schemas': {
            'Pet': {
                'type': 'object',
                'required': ['id', 'name'],
                'properties': {
                    'id': {'type': 'integer', 'format': 'int64'},
                    'name': {'type': 'string'},
                    'tag': {'type': 'string'},
                    'status': {
                        'type': 'string',
                        'enum': ['available', 'pending', 'sold'],
                    },
                    'category': {'$ref': '#/components/schemas/Category'},
                },
            },
            'Category': {
                'type': 'object',
                'properties': {
                    'id': {'type': 'integer', 'format': 'int64'},
                    'name': {'type': 'string'},
                },
            },
            'CreatePetRequest': {
                'type': 'object',
                'required': ['name'],
                'properties': {
                    'name': {'type': 'string'},
                    'tag': {'type': 'string'},
                    'categoryId': {'type': 'integer', 'format': 'int64'},
                },
            },
            'UpdatePetRequest': {
                'type': 'object',
                'properties': {
                    'name': {'type': 'string'},
                    'tag': {'type': 'string'},
                    'status': {
                        'type': 'string',
                        'enum': ['available', 'pending', 'sold'],
                    },
                },
            },
            'Error': {
                'type': 'object',
                'required': ['code', 'message'],
                'properties': {
                    'code': {'type': 'integer', 'format': 'int32'},
                    'message': {'type': 'string'},
                },
            },
        }
    },
}

# API with complex types for testing type generation
COMPLEX_TYPES_SPEC = {
    'openapi': '3.0.0',
    'info': {'title': 'Complex Types API', 'version': '1.0.0'},
    'paths': {
        '/complex': {
            'post': {
                'operationId': 'createComplex',
                'requestBody': {
                    'content': {
                        'application/json': {
                            'schema': {'$ref': '#/components/schemas/ComplexObject'}
                        }
                    }
                },
                'responses': {
                    '200': {
                        'description': 'Success',
                        'content': {
                            'application/json': {
                                'schema': {'$ref': '#/components/schemas/ComplexObject'}
                            }
                        },
                    }
                },
            }
        }
    },
    'components': {
        'schemas': {
            'ComplexObject': {
                'type': 'object',
                'properties': {
                    'stringField': {'type': 'string'},
                    'intField': {'type': 'integer'},
                    'floatField': {'type': 'number', 'format': 'float'},
                    'doubleField': {'type': 'number', 'format': 'double'},
                    'boolField': {'type': 'boolean'},
                    'dateField': {'type': 'string', 'format': 'date'},
                    'dateTimeField': {'type': 'string', 'format': 'date-time'},
                    'uuidField': {'type': 'string', 'format': 'uuid'},
                    'arrayField': {'type': 'array', 'items': {'type': 'string'}},
                    'nestedObject': {'$ref': '#/components/schemas/NestedObject'},
                    'nestedArray': {
                        'type': 'array',
                        'items': {'$ref': '#/components/schemas/NestedObject'},
                    },
                    'mapField': {
                        'type': 'object',
                        'additionalProperties': {'type': 'string'},
                    },
                    'enumField': {
                        'type': 'string',
                        'enum': ['VALUE_A', 'VALUE_B', 'VALUE_C'],
                    },
                    'nullableField': {'type': 'string', 'nullable': True},
                    'unionField': {'oneOf': [{'type': 'string'}, {'type': 'integer'}]},
                },
            },
            'NestedObject': {
                'type': 'object',
                'properties': {'id': {'type': 'integer'}, 'value': {'type': 'string'}},
            },
        }
    },
}

# API with authentication definitions
AUTH_API_SPEC = {
    'openapi': '3.0.0',
    'info': {'title': 'Auth API', 'version': '1.0.0'},
    'servers': [{'url': 'https://api.auth-example.com'}],
    'paths': {
        '/protected': {
            'get': {
                'operationId': 'getProtected',
                'security': [{'bearerAuth': []}],
                'responses': {
                    '200': {
                        'description': 'Success',
                        'content': {
                            'application/json': {
                                'schema': {
                                    'type': 'object',
                                    'properties': {'data': {'type': 'string'}},
                                }
                            }
                        },
                    },
                    '401': {'description': 'Unauthorized'},
                },
            }
        },
        '/api-key-protected': {
            'get': {
                'operationId': 'getApiKeyProtected',
                'security': [{'apiKey': []}],
                'responses': {'200': {'description': 'Success'}},
            }
        },
    },
    'components': {
        'securitySchemes': {
            'bearerAuth': {'type': 'http', 'scheme': 'bearer', 'bearerFormat': 'JWT'},
            'apiKey': {'type': 'apiKey', 'in': 'header', 'name': 'X-API-Key'},
            'basicAuth': {'type': 'http', 'scheme': 'basic'},
        }
    },
}

# API with various parameter types
PARAMETERS_SPEC = {
    'openapi': '3.0.0',
    'info': {'title': 'Parameters API', 'version': '1.0.0'},
    'servers': [{'url': 'https://api.params-example.com'}],
    'paths': {
        '/items/{itemId}': {
            'get': {
                'operationId': 'getItem',
                'parameters': [
                    {
                        'name': 'itemId',
                        'in': 'path',
                        'required': True,
                        'schema': {'type': 'string'},
                    },
                    {
                        'name': 'include',
                        'in': 'query',
                        'schema': {'type': 'array', 'items': {'type': 'string'}},
                    },
                    {
                        'name': 'X-Request-ID',
                        'in': 'header',
                        'schema': {'type': 'string', 'format': 'uuid'},
                    },
                    {'name': 'session', 'in': 'cookie', 'schema': {'type': 'string'}},
                ],
                'responses': {'200': {'description': 'Success'}},
            }
        }
    },
}

# OpenAPI 3.1 spec for testing version compatibility
OPENAPI_31_SPEC = {
    'openapi': '3.1.0',
    'info': {
        'title': 'OpenAPI 3.1 API',
        'version': '1.0.0',
        'summary': 'An API using OpenAPI 3.1 features',
    },
    'paths': {
        '/items': {
            'get': {
                'operationId': 'listItems',
                'responses': {
                    '200': {
                        'description': 'Success',
                        'content': {
                            'application/json': {
                                'schema': {
                                    'type': 'array',
                                    'items': {'$ref': '#/components/schemas/Item'},
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
                    'tags': {'type': ['array', 'null'], 'items': {'type': 'string'}},
                },
            }
        }
    },
}


def get_spec_as_json(spec: dict) -> str:
    """Convert a spec dictionary to JSON string."""
    import json

    return json.dumps(spec, indent=2)


def get_spec_as_yaml(spec: dict) -> str:
    """Convert a spec dictionary to YAML string."""
    import yaml

    return yaml.dump(spec, default_flow_style=False)
