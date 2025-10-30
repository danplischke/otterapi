"""Test endpoint generation functionality."""

import ast

import pytest

from otterapi.codegen.endpoints import (
    base_async_request_fn,
    base_request_fn,
    build_body_params,
    build_header_params,
    build_path_params,
    build_query_params,
    clean_docstring,
    get_base_call_keywords,
    get_parameters,
    prepare_call_from_parameters,
)


# Mock classes for testing since they might not be available
class MockType:
    """Mock Type class for testing."""

    def __init__(self, annotation_ast=None, type_name='mock_type'):
        self.annotation_ast = annotation_ast or ast.Name(id='str', ctx=ast.Load())
        self.type = type_name


class MockParameter:
    """Mock Parameter class for testing."""

    def __init__(self, name, location, required=True, type_obj=None, description=None):
        self.name = name
        self.location = location
        self.required = required
        self.type = type_obj or MockType()
        self.description = description


class TestCleanDocstring:
    """Test clean_docstring function."""

    def test_simple_docstring(self):
        """Test cleaning a simple docstring."""
        docstring = 'This is a simple docstring.'
        result = clean_docstring(docstring)
        assert result == 'This is a simple docstring.'

    def test_indented_docstring(self):
        """Test cleaning an indented docstring."""
        docstring = """
        This is an indented docstring.
        With multiple lines.
        """
        result = clean_docstring(docstring)
        assert 'This is an indented docstring.' in result
        assert 'With multiple lines.' in result
        assert not result.startswith(' ')  # Should be dedented

    def test_empty_docstring(self):
        """Test cleaning an empty docstring."""
        result = clean_docstring('')
        assert result == ''

    def test_multiline_docstring(self):
        """Test cleaning a multiline docstring."""
        docstring = """
        Get all users from the database.
        
        Returns a list of users with their details.
        """
        result = clean_docstring(docstring)
        assert 'Get all users from the database.' in result
        assert 'Returns a list of users' in result


class TestBaseRequestFunctions:
    """Test base request function generation."""

    def test_base_request_fn_structure(self):
        """Test that base_request_fn returns correct structure."""
        func_ast, imports = base_request_fn()

        assert isinstance(func_ast, ast.FunctionDef)
        assert func_ast.name == 'request_sync'
        assert isinstance(imports, dict)

        # Check required imports
        assert 'httpx' in imports
        assert 'pydantic' in imports
        assert 'typing' in imports

        # Check function arguments
        assert len(func_ast.args.args) == 2  # method, url
        assert func_ast.args.args[0].arg == 'method'
        assert func_ast.args.args[1].arg == 'url'

    def test_base_async_request_fn_structure(self):
        """Test that base_async_request_fn returns correct structure."""
        func_ast, imports = base_async_request_fn()

        assert isinstance(func_ast, ast.AsyncFunctionDef)
        assert func_ast.name == 'request_async'
        assert isinstance(imports, dict)

        # Check required imports
        assert 'httpx' in imports
        assert 'typing' in imports

    def test_base_request_fn_body(self):
        """Test that base request function has expected body structure."""
        func_ast, _ = base_request_fn()

        # Should have multiple statements in body
        assert len(func_ast.body) > 3

        # First statement should be assignment to response
        first_stmt = func_ast.body[0]
        assert isinstance(first_stmt, ast.Assign)
        assert len(first_stmt.targets) == 1
        assert isinstance(first_stmt.targets[0], ast.Name)
        assert first_stmt.targets[0].id == 'response'

    def test_base_async_request_fn_body(self):
        """Test that base async request function has expected body structure."""
        func_ast, _ = base_async_request_fn()

        # Should have async context manager
        assert len(func_ast.body) > 0
        first_stmt = func_ast.body[0]
        assert isinstance(first_stmt, ast.AsyncWith)


class TestGetParameters:
    """Test get_parameters function."""

    def test_required_parameters(self):
        """Test processing required parameters."""
        params = [
            MockParameter('user_id', 'path', required=True),
            MockParameter('name', 'query', required=True),
        ]

        args, kwonlyargs, kw_defaults = get_parameters(params)

        assert len(args) == 2
        assert len(kwonlyargs) == 0
        assert len(kw_defaults) == 0

        assert args[0].arg == 'user_id'
        assert args[1].arg == 'name'

    def test_optional_parameters(self):
        """Test processing optional parameters."""
        params = [
            MockParameter('limit', 'query', required=False),
            MockParameter('offset', 'query', required=False),
        ]

        args, kwonlyargs, kw_defaults = get_parameters(params)

        assert len(args) == 0
        assert len(kwonlyargs) == 2
        assert len(kw_defaults) == 2

        assert kwonlyargs[0].arg == 'limit'
        assert kwonlyargs[1].arg == 'offset'

        # Defaults should be None
        assert all(
            isinstance(default, ast.Constant) and default.value is None
            for default in kw_defaults
        )

    def test_mixed_parameters(self):
        """Test processing mixed required and optional parameters."""
        params = [
            MockParameter('user_id', 'path', required=True),
            MockParameter('limit', 'query', required=False),
            MockParameter('name', 'query', required=True),
        ]

        args, kwonlyargs, kw_defaults = get_parameters(params)

        assert len(args) == 2
        assert len(kwonlyargs) == 1
        assert len(kw_defaults) == 1

        # Required params in args
        required_names = [arg.arg for arg in args]
        assert 'user_id' in required_names
        assert 'name' in required_names

        # Optional params in kwonlyargs
        assert kwonlyargs[0].arg == 'limit'

    def test_empty_parameters(self):
        """Test processing empty parameter list."""
        args, kwonlyargs, kw_defaults = get_parameters([])

        assert len(args) == 0
        assert len(kwonlyargs) == 0
        assert len(kw_defaults) == 0


class TestParameterBuilders:
    """Test parameter building functions."""

    def test_build_header_params(self):
        """Test building header parameters."""
        headers = [
            MockParameter('Authorization', 'header'),
            MockParameter('Content-Type', 'header'),
        ]

        result = build_header_params(headers)

        assert isinstance(result, ast.Dict)
        assert len(result.keys) == 2
        assert len(result.values) == 2

        # Check keys are constants
        assert all(isinstance(key, ast.Constant) for key in result.keys)
        assert result.keys[0].value == 'Authorization'
        assert result.keys[1].value == 'Content-Type'

        # Check values are names
        assert all(isinstance(value, ast.Name) for value in result.values)

    def test_build_header_params_empty(self):
        """Test building header parameters with empty list."""
        result = build_header_params([])
        assert result is None

    def test_build_query_params(self):
        """Test building query parameters."""
        queries = [MockParameter('limit', 'query'), MockParameter('offset', 'query')]

        result = build_query_params(queries)

        assert isinstance(result, ast.Dict)
        assert len(result.keys) == 2
        assert len(result.values) == 2

    def test_build_query_params_empty(self):
        """Test building query parameters with empty list."""
        result = build_query_params([])
        assert result is None

    def test_build_path_params_simple(self):
        """Test building path parameters for simple path."""
        path = '/users/{user_id}'
        paths = [MockParameter('user_id', 'path')]

        result = build_path_params(paths, path)

        assert isinstance(result, ast.JoinedStr)
        assert len(result.values) >= 2  # At least literal + formatted value

    def test_build_path_params_no_params(self):
        """Test building path parameters with no parameters."""
        path = '/users'

        result = build_path_params([], path)

        assert isinstance(result, ast.Constant)
        assert result.value == '/users'

    def test_build_path_params_multiple(self):
        """Test building path parameters with multiple parameters."""
        path = '/users/{user_id}/posts/{post_id}'
        paths = [MockParameter('user_id', 'path'), MockParameter('post_id', 'path')]

        result = build_path_params(paths, path)

        assert isinstance(result, ast.JoinedStr)
        # Should have literals and formatted values
        assert len(result.values) >= 4

    def test_build_body_params_none(self):
        """Test building body parameters with None."""
        result = build_body_params(None)
        assert result is None

    def test_build_body_params_model(self):
        """Test building body parameters with model type."""
        body_param = MockParameter('user', 'body')
        body_param.type.type = 'model'

        result = build_body_params(body_param)

        assert isinstance(result, ast.Call)
        assert isinstance(result.func, ast.Attribute)
        assert result.func.attr == 'model_dump'

    def test_build_body_params_root_model(self):
        """Test building body parameters with root model type."""
        body_param = MockParameter('data', 'body')
        body_param.type.type = 'root_model'

        result = build_body_params(body_param)

        assert isinstance(result, ast.Call)
        assert isinstance(result.func, ast.Attribute)
        assert result.func.attr == 'model_dump'

    def test_build_body_params_primitive(self):
        """Test building body parameters with primitive type."""
        body_param = MockParameter('value', 'body')
        body_param.type.type = 'primitive'

        result = build_body_params(body_param)

        assert isinstance(result, ast.Name)
        assert result.id == 'value'


class TestGetBaseCallKeywords:
    """Test get_base_call_keywords function."""

    def test_basic_keywords(self):
        """Test basic keyword generation."""
        method = 'GET'
        path = ast.Constant('/users')
        response_model = MockType()

        keywords = get_base_call_keywords(method, path, response_model)

        assert len(keywords) == 4

        # Check method keyword
        method_kw = next(kw for kw in keywords if kw.arg == 'method')
        assert method_kw.value.value == 'GET'

        # Check url keyword
        url_kw = next(kw for kw in keywords if kw.arg == 'url')
        assert url_kw.value == path

    def test_keywords_with_status_codes(self):
        """Test keyword generation with status codes."""
        method = 'POST'
        path = ast.Constant('/users')
        response_model = MockType()
        status_codes = [200, 201]

        keywords = get_base_call_keywords(method, path, response_model, status_codes)

        status_kw = next(kw for kw in keywords if kw.arg == 'supported_status_codes')
        assert status_kw.value.value == [200, 201]

    def test_keywords_without_response_model(self):
        """Test keyword generation without response model."""
        method = 'DELETE'
        path = ast.Constant('/users/1')

        keywords = get_base_call_keywords(method, path, None)

        response_kw = next(kw for kw in keywords if kw.arg == 'response_model')
        assert isinstance(response_kw.value, ast.Constant)
        assert response_kw.value.value is None


class TestPrepareCallFromParameters:
    """Test prepare_call_from_parameters function."""

    def test_mixed_parameters(self):
        """Test preparing call with mixed parameter types."""
        parameters = [
            MockParameter('user_id', 'path'),
            MockParameter('limit', 'query'),
            MockParameter('Authorization', 'header'),
            MockParameter('user_data', 'body'),
        ]
        path = '/users/{user_id}'

        query_params, header_params, body_params, processed_path = (
            prepare_call_from_parameters(parameters, path)
        )

        # Query params should be dict
        assert isinstance(query_params, ast.Dict)

        # Header params should be dict
        assert isinstance(header_params, ast.Dict)

        # Body params should be name
        assert isinstance(body_params, ast.Name)
        assert body_params.id == 'user_data'

        # Path should be processed (JoinedStr for interpolation)
        assert isinstance(processed_path, ast.JoinedStr)

    def test_no_parameters(self):
        """Test preparing call with no parameters."""
        path = '/users'

        query_params, header_params, body_params, processed_path = (
            prepare_call_from_parameters(None, path)
        )

        assert query_params is None
        assert header_params is None
        assert body_params is None
        assert isinstance(processed_path, ast.Constant)
        assert processed_path.value == '/users'

    def test_only_query_parameters(self):
        """Test preparing call with only query parameters."""
        parameters = [MockParameter('limit', 'query'), MockParameter('offset', 'query')]
        path = '/users'

        query_params, header_params, body_params, processed_path = (
            prepare_call_from_parameters(parameters, path)
        )

        assert isinstance(query_params, ast.Dict)
        assert header_params is None
        assert body_params is None
        assert isinstance(processed_path, ast.Constant)

    def test_multiple_body_parameters_error(self):
        """Test that multiple body parameters raise an error."""
        parameters = [
            MockParameter('user_data', 'body'),
            MockParameter('extra_data', 'body'),
        ]
        path = '/users'

        with pytest.raises(
            ValueError, match='Multiple body parameters are not supported'
        ):
            prepare_call_from_parameters(parameters, path)


class TestEndpointGenerationIntegration:
    """Integration tests for endpoint generation."""

    def test_complex_endpoint_preparation(self):
        """Test preparing a complex endpoint with all parameter types."""
        parameters = [
            MockParameter('user_id', 'path', required=True),
            MockParameter('include_posts', 'query', required=False),
            MockParameter('limit', 'query', required=False),
            MockParameter('Authorization', 'header', required=True),
            MockParameter('X-Client-Version', 'header', required=False),
            MockParameter('update_data', 'body', required=True),
        ]
        path = '/users/{user_id}'

        # Test parameter categorization
        args, kwonlyargs, kw_defaults = get_parameters(parameters)

        # Required params: user_id, Authorization, update_data
        assert len(args) == 3
        required_names = [arg.arg for arg in args]
        assert 'user_id' in required_names
        assert 'Authorization' in required_names
        assert 'update_data' in required_names

        # Optional params: include_posts, limit, X-Client-Version
        assert len(kwonlyargs) == 3
        optional_names = [arg.arg for arg in kwonlyargs]
        assert 'include_posts' in optional_names
        assert 'limit' in optional_names
        assert 'X-Client-Version' in optional_names

    def test_realistic_get_endpoint(self):
        """Test preparing a realistic GET endpoint."""
        parameters = [
            MockParameter('user_id', 'path', required=True),
            MockParameter('include_deleted', 'query', required=False),
            MockParameter('fields', 'query', required=False),
        ]
        path = '/users/{user_id}'

        query_params, header_params, body_params, processed_path = (
            prepare_call_from_parameters(parameters, path)
        )

        # GET endpoint should have query params but no body
        assert isinstance(query_params, ast.Dict)
        assert header_params is None
        assert body_params is None
        assert isinstance(processed_path, ast.JoinedStr)

    def test_realistic_post_endpoint(self):
        """Test preparing a realistic POST endpoint."""
        parameters = [
            MockParameter('user_data', 'body', required=True),
            MockParameter('Content-Type', 'header', required=False),
        ]
        path = '/users'

        query_params, header_params, body_params, processed_path = (
            prepare_call_from_parameters(parameters, path)
        )

        # POST endpoint should have body and possibly headers
        assert query_params is None
        assert isinstance(header_params, ast.Dict)
        assert isinstance(body_params, ast.Name)
        assert isinstance(processed_path, ast.Constant)


def test_ast_generation_validity():
    """Test that generated AST nodes are valid Python."""
    # Test base request function
    func_ast, _ = base_request_fn()

    # Create a module with the function
    module = ast.Module(body=[func_ast], type_ignores=[])
    ast.fix_missing_locations(module)

    # Should compile without errors
    code = compile(module, '<test>', 'exec')
    assert code is not None

    # Test async function too
    async_func_ast, _ = base_async_request_fn()
    async_module = ast.Module(body=[async_func_ast], type_ignores=[])
    ast.fix_missing_locations(async_module)

    async_code = compile(async_module, '<test>', 'exec')
    assert async_code is not None


def test_parameter_edge_cases():
    """Test edge cases in parameter processing."""
    # Test parameter with special characters in name
    param = MockParameter('user-id', 'path')  # Hyphen in name
    args, _, _ = get_parameters([param])
    assert len(args) == 1
    assert args[0].arg == 'user-id'  # Should preserve original name

    # Test parameter with no type
    param_no_type = MockParameter('data', 'body', type_obj=None)
    args, _, _ = get_parameters([param_no_type])
    assert len(args) == 1
