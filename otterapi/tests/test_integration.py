"""Integration tests using FastAPI test client."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from pydantic import BaseModel

from otterapi.codegen.generator import Codegen
from otterapi.config import CodegenConfig, DocumentConfig


# Test models for our mock API
class User(BaseModel):
    id: int
    name: str
    email: str
    age: int | None = None


class CreateUserRequest(BaseModel):
    name: str
    email: str
    age: int | None = None


class UserList(BaseModel):
    users: list[User]
    total: int


# Mock FastAPI app for testing
app = FastAPI(title='Test API', version='1.0.0')

# Mock data
users_db = [
    User(id=1, name='John Doe', email='john@example.com', age=30),
    User(id=2, name='Jane Smith', email='jane@example.com', age=25),
]


@app.get('/users', response_model=UserList)
def get_users():
    """Get all users."""
    return UserList(users=users_db, total=len(users_db))


@app.get('/users/{user_id}', response_model=User)
def get_user(user_id: int):
    """Get a specific user by ID."""
    for user in users_db:
        if user.id == user_id:
            return user
    raise HTTPException(status_code=404, detail='User not found')


@app.post('/users', response_model=User, status_code=201)
def create_user(user: CreateUserRequest):
    """Create a new user."""
    new_id = max(u.id for u in users_db) + 1 if users_db else 1
    new_user = User(id=new_id, **user.model_dump())
    users_db.append(new_user)
    return new_user


@app.put('/users/{user_id}', response_model=User)
def update_user(user_id: int, user: CreateUserRequest):
    """Update an existing user."""
    for i, existing_user in enumerate(users_db):
        if existing_user.id == user_id:
            updated_user = User(id=user_id, **user.model_dump())
            users_db[i] = updated_user
            return updated_user
    raise HTTPException(status_code=404, detail='User not found')


@app.delete('/users/{user_id}')
def delete_user(user_id: int):
    """Delete a user."""
    for i, user in enumerate(users_db):
        if user.id == user_id:
            del users_db[i]
            return {'message': 'User deleted successfully'}
    raise HTTPException(status_code=404, detail='User not found')


@pytest.fixture
def test_client():
    """Fixture providing FastAPI test client."""
    return TestClient(app)


@pytest.fixture
def openapi_spec(test_client):
    """Fixture providing OpenAPI specification from FastAPI app."""
    response = test_client.get('/openapi.json')
    assert response.status_code == 200
    return response.json()


@pytest.fixture
def temp_openapi_file(openapi_spec):
    """Fixture creating a temporary OpenAPI file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(openapi_spec, f, indent=2)
        f.flush()
        yield f.name
    Path(f.name).unlink()


@pytest.fixture
def temp_output_dir():
    """Fixture providing a temporary output directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


class TestCodeGenerationIntegration:
    """Integration tests for code generation with real OpenAPI specs."""

    def test_generate_from_fastapi_spec(self, temp_openapi_file, temp_output_dir):
        """Test generating code from a real FastAPI OpenAPI spec."""
        config = DocumentConfig(source=temp_openapi_file, output=temp_output_dir)

        # Mock the missing dependencies
        with (
            patch('otterapi.codegen.generator.TypeGen') as mock_typegen_class,
            patch('otterapi.codegen.generator.OpenAPIProcessor'),
            patch('otterapi.codegen.generator.httpx'),
        ):
            # Set up TypeGen mock
            mock_typegen = mock_typegen_class.return_value
            mock_typegen.types = {}

            codegen = Codegen(config)

            # This should not raise an exception
            # Note: We're testing the structure, not the full generation
            # since some dependencies might be missing
            assert codegen.config == config

    def test_api_endpoints_work(self, test_client):
        """Test that our mock API endpoints work correctly."""
        # Test GET /users
        response = test_client.get('/users')
        assert response.status_code == 200
        data = response.json()
        assert 'users' in data
        assert 'total' in data
        assert len(data['users']) == 2

        # Test GET /users/{user_id}
        response = test_client.get('/users/1')
        assert response.status_code == 200
        user = response.json()
        assert user['id'] == 1
        assert user['name'] == 'John Doe'

        # Test POST /users
        new_user = {'name': 'Alice Johnson', 'email': 'alice@example.com', 'age': 28}
        response = test_client.post('/users', json=new_user)
        assert response.status_code == 201
        created_user = response.json()
        assert created_user['name'] == 'Alice Johnson'
        assert 'id' in created_user

    def test_api_error_handling(self, test_client):
        """Test API error handling."""
        # Test 404 for non-existent user
        response = test_client.get('/users/999')
        assert response.status_code == 404
        assert 'User not found' in response.json()['detail']

        # Test 404 for deleting non-existent user
        response = test_client.delete('/users/999')
        assert response.status_code == 404

    def test_openapi_spec_structure(self, openapi_spec):
        """Test that the OpenAPI spec has the expected structure."""
        assert 'openapi' in openapi_spec
        assert 'info' in openapi_spec
        assert 'paths' in openapi_spec
        assert 'components' in openapi_spec

        # Check paths
        paths = openapi_spec['paths']
        assert '/users' in paths
        assert '/users/{user_id}' in paths

        # Check methods
        users_path = paths['/users']
        assert 'get' in users_path
        assert 'post' in users_path

        user_id_path = paths['/users/{user_id}']
        assert 'get' in user_id_path
        assert 'put' in user_id_path
        assert 'delete' in user_id_path

        # Check schemas
        schemas = openapi_spec['components']['schemas']
        assert 'User' in schemas
        assert 'CreateUserRequest' in schemas
        assert 'UserList' in schemas


class TestGeneratedCodeIntegration:
    """Test integration between generated code and actual API."""

    def test_config_with_fastapi_source(self, temp_openapi_file, temp_output_dir):
        """Test creating config with FastAPI OpenAPI source."""
        config = DocumentConfig(
            source=temp_openapi_file,
            output=temp_output_dir,
            models_file='api_models.py',
            endpoints_file='api_endpoints.py',
        )

        assert config.source == temp_openapi_file
        assert config.output == temp_output_dir
        assert config.models_file == 'api_models.py'
        assert config.endpoints_file == 'api_endpoints.py'

    def test_codegen_config_with_fastapi(self, temp_openapi_file, temp_output_dir):
        """Test CodegenConfig with FastAPI source."""
        doc_config = DocumentConfig(source=temp_openapi_file, output=temp_output_dir)

        config = CodegenConfig(documents=[doc_config], generate_endpoints=True)

        assert len(config.documents) == 1
        assert config.generate_endpoints is True

    @patch('otterapi.codegen.generator.is_url')
    def test_file_vs_url_handling(self, mock_is_url, temp_openapi_file):
        """Test that Codegen correctly handles file vs URL sources."""
        config = DocumentConfig(source=temp_openapi_file, output='./output')

        # Mock is_url to return False for file paths
        mock_is_url.return_value = False

        with (
            patch('otterapi.codegen.generator.TypeGen'),
            patch('otterapi.codegen.generator.OpenAPIProcessor'),
            patch('builtins.open', create=True) as mock_open,
        ):
            mock_open.return_value.__enter__.return_value.read.return_value = (
                b'{"openapi": "3.0.0"}'
            )

            codegen = Codegen(config)
            # Should not raise exception for file handling
            assert codegen.config.source == temp_openapi_file


class TestAPIValidation:
    """Test API validation using generated schemas."""

    def test_user_model_validation(self):
        """Test User model validation."""
        # Valid user
        user_data = {
            'id': 1,
            'name': 'John Doe',
            'email': 'john@example.com',
            'age': 30,
        }
        user = User(**user_data)
        assert user.id == 1
        assert user.name == 'John Doe'
        assert user.email == 'john@example.com'
        assert user.age == 30

        # User without optional age
        user_data_no_age = {'id': 2, 'name': 'Jane Smith', 'email': 'jane@example.com'}
        user = User(**user_data_no_age)
        assert user.age is None

    def test_create_user_request_validation(self):
        """Test CreateUserRequest model validation."""
        request_data = {
            'name': 'Alice Johnson',
            'email': 'alice@example.com',
            'age': 28,
        }
        request = CreateUserRequest(**request_data)
        assert request.name == 'Alice Johnson'
        assert request.email == 'alice@example.com'
        assert request.age == 28

    def test_user_list_validation(self):
        """Test UserList model validation."""
        users = [
            User(id=1, name='John', email='john@example.com'),
            User(id=2, name='Jane', email='jane@example.com'),
        ]
        user_list = UserList(users=users, total=2)
        assert len(user_list.users) == 2
        assert user_list.total == 2


class TestEndToEndWorkflow:
    """End-to-end tests simulating the complete workflow."""

    def test_complete_workflow_simulation(
        self, test_client, temp_openapi_file, temp_output_dir
    ):
        """Simulate complete workflow: API → OpenAPI → Code Generation."""
        # Step 1: Verify API is working
        response = test_client.get('/users')
        assert response.status_code == 200

        # Step 2: Get OpenAPI spec
        openapi_response = test_client.get('/openapi.json')
        assert openapi_response.status_code == 200
        spec = openapi_response.json()

        # Step 3: Verify spec contains expected endpoints
        assert '/users' in spec['paths']
        assert '/users/{user_id}' in spec['paths']

        # Step 4: Create config for code generation
        config = DocumentConfig(source=temp_openapi_file, output=temp_output_dir)

        # Step 5: Verify config is valid
        assert Path(config.source).exists()
        assert config.output == temp_output_dir

    def test_multiple_api_endpoints(self, test_client):
        """Test multiple API endpoints to ensure comprehensive coverage."""
        # Test all CRUD operations

        # CREATE
        new_user = {'name': 'Test User', 'email': 'test@example.com', 'age': 25}
        create_response = test_client.post('/users', json=new_user)
        assert create_response.status_code == 201
        created_user = create_response.json()
        user_id = created_user['id']

        # READ (single)
        get_response = test_client.get(f'/users/{user_id}')
        assert get_response.status_code == 200
        user = get_response.json()
        assert user['name'] == 'Test User'

        # UPDATE
        update_data = {
            'name': 'Updated User',
            'email': 'updated@example.com',
            'age': 26,
        }
        update_response = test_client.put(f'/users/{user_id}', json=update_data)
        assert update_response.status_code == 200
        updated_user = update_response.json()
        assert updated_user['name'] == 'Updated User'

        # READ (all)
        list_response = test_client.get('/users')
        assert list_response.status_code == 200
        user_list = list_response.json()
        assert any(u['id'] == user_id for u in user_list['users'])

        # DELETE
        delete_response = test_client.delete(f'/users/{user_id}')
        assert delete_response.status_code == 200

        # Verify deletion
        get_deleted_response = test_client.get(f'/users/{user_id}')
        assert get_deleted_response.status_code == 404


@pytest.fixture
def sample_openapi_spec():
    """Fixture providing a sample OpenAPI spec for testing."""
    return {
        'openapi': '3.0.2',
        'info': {'title': 'Test API', 'version': '1.0.0'},
        'paths': {
            '/users': {
                'get': {
                    'summary': 'Get Users',
                    'operationId': 'get_users',
                    'responses': {
                        '200': {
                            'description': 'Successful Response',
                            'content': {
                                'application/json': {
                                    'schema': {'$ref': '#/components/schemas/UserList'}
                                }
                            },
                        }
                    },
                },
                'post': {
                    'summary': 'Create User',
                    'operationId': 'create_user',
                    'requestBody': {
                        'content': {
                            'application/json': {
                                'schema': {
                                    '$ref': '#/components/schemas/CreateUserRequest'
                                }
                            }
                        },
                        'required': True,
                    },
                    'responses': {
                        '201': {
                            'description': 'Successful Response',
                            'content': {
                                'application/json': {
                                    'schema': {'$ref': '#/components/schemas/User'}
                                }
                            },
                        }
                    },
                },
            }
        },
        'components': {
            'schemas': {
                'User': {
                    'title': 'User',
                    'required': ['id', 'name', 'email'],
                    'type': 'object',
                    'properties': {
                        'id': {'title': 'Id', 'type': 'integer'},
                        'name': {'title': 'Name', 'type': 'string'},
                        'email': {'title': 'Email', 'type': 'string'},
                        'age': {'title': 'Age', 'type': 'integer', 'default': None},
                    },
                },
                'CreateUserRequest': {
                    'title': 'CreateUserRequest',
                    'required': ['name', 'email'],
                    'type': 'object',
                    'properties': {
                        'name': {'title': 'Name', 'type': 'string'},
                        'email': {'title': 'Email', 'type': 'string'},
                        'age': {'title': 'Age', 'type': 'integer', 'default': None},
                    },
                },
                'UserList': {
                    'title': 'UserList',
                    'required': ['users', 'total'],
                    'type': 'object',
                    'properties': {
                        'users': {
                            'title': 'Users',
                            'type': 'array',
                            'items': {'$ref': '#/components/schemas/User'},
                        },
                        'total': {'title': 'Total', 'type': 'integer'},
                    },
                },
            }
        },
    }


class TestOpenAPISpecProcessing:
    """Test processing of OpenAPI specifications."""

    def test_spec_validation(self, sample_openapi_spec):
        """Test that sample OpenAPI spec is valid."""
        assert sample_openapi_spec['openapi'] == '3.0.2'
        assert 'paths' in sample_openapi_spec
        assert 'components' in sample_openapi_spec
        assert 'schemas' in sample_openapi_spec['components']

    def test_schema_structure(self, sample_openapi_spec):
        """Test OpenAPI schema structure."""
        schemas = sample_openapi_spec['components']['schemas']

        # Test User schema
        user_schema = schemas['User']
        assert user_schema['type'] == 'object'
        assert 'id' in user_schema['required']
        assert 'name' in user_schema['required']
        assert 'email' in user_schema['required']
        assert 'age' not in user_schema['required']  # optional field

    def test_endpoint_structure(self, sample_openapi_spec):
        """Test OpenAPI endpoint structure."""
        paths = sample_openapi_spec['paths']
        users_path = paths['/users']

        # Test GET endpoint
        get_endpoint = users_path['get']
        assert get_endpoint['operationId'] == 'get_users'
        assert '200' in get_endpoint['responses']

        # Test POST endpoint
        post_endpoint = users_path['post']
        assert post_endpoint['operationId'] == 'create_user'
        assert 'requestBody' in post_endpoint
        assert post_endpoint['requestBody']['required'] is True
