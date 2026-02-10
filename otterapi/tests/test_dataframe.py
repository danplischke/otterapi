"""Tests for DataFrame configuration and code generation."""

import ast

import pytest

from otterapi.config import DataFrameConfig, EndpointDataFrameConfig


class TestEndpointDataFrameConfig:
    """Tests for EndpointDataFrameConfig model."""

    def test_default_values(self):
        """Test that default values are None (inherit from parent)."""
        config = EndpointDataFrameConfig()
        assert config.enabled is None
        assert config.path is None
        assert config.pandas is None
        assert config.polars is None

    def test_explicit_values(self):
        """Test setting explicit values."""
        config = EndpointDataFrameConfig(
            enabled=True,
            path='data.items',
            pandas=True,
            polars=False,
        )
        assert config.enabled is True
        assert config.path == 'data.items'
        assert config.pandas is True
        assert config.polars is False

    def test_rejects_extra_fields(self):
        """Test that extra fields are rejected."""
        with pytest.raises(Exception):  # Pydantic ValidationError
            EndpointDataFrameConfig(unknown_field='value')


class TestDataFrameConfig:
    """Tests for DataFrameConfig model."""

    def test_default_values(self):
        """Test default configuration values."""
        config = DataFrameConfig()
        assert config.enabled is False
        assert config.pandas is True
        assert config.polars is False
        assert config.default_path is None
        assert config.include_all is True
        assert config.endpoints == {}

    def test_enabled_with_defaults(self):
        """Test enabled config uses correct defaults."""
        config = DataFrameConfig(enabled=True)
        assert config.enabled is True
        assert config.pandas is True  # Default to pandas
        assert config.polars is False

    def test_both_libraries_enabled(self):
        """Test enabling both pandas and polars."""
        config = DataFrameConfig(
            enabled=True,
            pandas=True,
            polars=True,
        )
        assert config.pandas is True
        assert config.polars is True

    def test_polars_only(self):
        """Test enabling only polars."""
        config = DataFrameConfig(
            enabled=True,
            pandas=False,
            polars=True,
        )
        assert config.pandas is False
        assert config.polars is True

    def test_with_endpoints(self):
        """Test configuration with endpoint overrides."""
        config = DataFrameConfig(
            enabled=True,
            endpoints={
                'get_users': EndpointDataFrameConfig(
                    path='data.users',
                    pandas=True,
                    polars=True,
                ),
                'get_items': EndpointDataFrameConfig(
                    enabled=False,
                ),
            },
        )
        assert 'get_users' in config.endpoints
        assert config.endpoints['get_users'].path == 'data.users'
        assert config.endpoints['get_items'].enabled is False


class TestShouldGenerateForEndpoint:
    """Tests for DataFrameConfig.should_generate_for_endpoint method."""

    def test_disabled_returns_false(self):
        """Test that disabled config returns False for all."""
        config = DataFrameConfig(enabled=False)
        gen_pandas, gen_polars, path = config.should_generate_for_endpoint(
            'get_users', returns_list=True
        )
        assert gen_pandas is False
        assert gen_polars is False
        assert path is None

    def test_default_generates_for_list_endpoints(self):
        """Test that default config generates for list endpoints."""
        config = DataFrameConfig(enabled=True)
        gen_pandas, gen_polars, path = config.should_generate_for_endpoint(
            'get_users', returns_list=True
        )
        assert gen_pandas is True
        assert gen_polars is False  # polars is False by default
        assert path is None

    def test_does_not_generate_for_non_list(self):
        """Test that config doesn't generate for non-list endpoints."""
        config = DataFrameConfig(enabled=True)
        gen_pandas, gen_polars, path = config.should_generate_for_endpoint(
            'get_user', returns_list=False
        )
        assert gen_pandas is False
        assert gen_polars is False

    def test_both_libraries(self):
        """Test generating for both libraries."""
        config = DataFrameConfig(enabled=True, pandas=True, polars=True)
        gen_pandas, gen_polars, path = config.should_generate_for_endpoint(
            'get_users', returns_list=True
        )
        assert gen_pandas is True
        assert gen_polars is True

    def test_default_path_used(self):
        """Test that default_path is used."""
        config = DataFrameConfig(enabled=True, default_path='data.items')
        gen_pandas, gen_polars, path = config.should_generate_for_endpoint(
            'get_users', returns_list=True
        )
        assert path == 'data.items'

    def test_endpoint_override_path(self):
        """Test endpoint-specific path override."""
        config = DataFrameConfig(
            enabled=True,
            default_path='data.items',
            endpoints={
                'get_users': EndpointDataFrameConfig(path='data.users'),
            },
        )
        gen_pandas, gen_polars, path = config.should_generate_for_endpoint(
            'get_users', returns_list=True
        )
        assert path == 'data.users'

    def test_endpoint_override_disabled(self):
        """Test endpoint-specific disable override."""
        config = DataFrameConfig(
            enabled=True,
            pandas=True,
            polars=True,
            endpoints={
                'get_users': EndpointDataFrameConfig(enabled=False),
            },
        )
        gen_pandas, gen_polars, path = config.should_generate_for_endpoint(
            'get_users', returns_list=True
        )
        assert gen_pandas is False
        assert gen_polars is False

    def test_endpoint_override_libraries(self):
        """Test endpoint-specific library override."""
        config = DataFrameConfig(
            enabled=True,
            pandas=True,
            polars=True,
            endpoints={
                'get_users': EndpointDataFrameConfig(
                    pandas=False,  # Disable pandas for this endpoint
                    polars=True,
                ),
            },
        )
        gen_pandas, gen_polars, path = config.should_generate_for_endpoint(
            'get_users', returns_list=True
        )
        assert gen_pandas is False
        assert gen_polars is True

    def test_include_all_false_without_endpoint_config(self):
        """Test include_all=False skips endpoints without config."""
        config = DataFrameConfig(
            enabled=True,
            include_all=False,
        )
        gen_pandas, gen_polars, path = config.should_generate_for_endpoint(
            'get_users', returns_list=True
        )
        assert gen_pandas is False
        assert gen_polars is False

    def test_include_all_false_with_endpoint_enabled(self):
        """Test include_all=False but endpoint explicitly enabled."""
        config = DataFrameConfig(
            enabled=True,
            include_all=False,
            endpoints={
                'get_users': EndpointDataFrameConfig(enabled=True),
            },
        )
        gen_pandas, gen_polars, path = config.should_generate_for_endpoint(
            'get_users', returns_list=True
        )
        assert gen_pandas is True  # Uses default pandas=True
        assert gen_polars is False

    def test_explicitly_enabled_ignores_returns_list(self):
        """Test that explicitly enabled endpoints generate even for non-list."""
        config = DataFrameConfig(
            enabled=True,
            endpoints={
                'get_user': EndpointDataFrameConfig(enabled=True),
            },
        )
        gen_pandas, gen_polars, path = config.should_generate_for_endpoint(
            'get_user', returns_list=False
        )
        assert gen_pandas is True

    def test_inherits_parent_libraries_when_not_overridden(self):
        """Test that endpoint inherits parent library settings."""
        config = DataFrameConfig(
            enabled=True,
            pandas=True,
            polars=True,
            endpoints={
                'get_users': EndpointDataFrameConfig(
                    path='data.users',
                    # pandas and polars not set, should inherit
                ),
            },
        )
        gen_pandas, gen_polars, path = config.should_generate_for_endpoint(
            'get_users', returns_list=True
        )
        assert gen_pandas is True
        assert gen_polars is True
        assert path == 'data.users'


class TestDataFrameModuleGenerator:
    """Tests for the DataFrame module generator."""

    def test_generate_dataframe_module(self, tmp_path):
        """Test generating the _dataframe.py module."""
        from otterapi.codegen.dataframes import generate_dataframe_module

        output_path = generate_dataframe_module(tmp_path)

        assert output_path.exists()
        assert output_path.name == '_dataframe.py'

        content = output_path.read_text()
        assert 'def extract_path' in content
        assert 'def to_pandas' in content
        assert 'def to_polars' in content
        assert 'import pandas as pd' in content
        assert 'import polars as pl' in content
        assert 'def _normalize_data' in content
        assert 'def _to_dict' in content

    def test_dataframe_module_is_valid_python(self, tmp_path):
        """Test that generated module is valid Python."""
        from otterapi.codegen.dataframes import generate_dataframe_module

        output_path = generate_dataframe_module(tmp_path)
        content = output_path.read_text()

        # Should not raise SyntaxError
        ast.parse(content)


class TestDataFrameMethodConfig:
    """Tests for DataFrameMethodConfig dataclass."""

    def test_default_values(self):
        """Test default values are all False/None."""
        from otterapi.codegen.dataframes import DataFrameMethodConfig

        config = DataFrameMethodConfig()
        assert config.generate_pandas is False
        assert config.generate_polars is False
        assert config.path is None

    def test_explicit_values(self):
        """Test setting explicit values."""
        from otterapi.codegen.dataframes import DataFrameMethodConfig

        config = DataFrameMethodConfig(
            generate_pandas=True,
            generate_polars=True,
            path='data.items',
        )
        assert config.generate_pandas is True
        assert config.generate_polars is True
        assert config.path == 'data.items'


class TestDataFrameDelegatingFunction:
    """Tests for DataFrame delegating function generation."""

    def test_build_delegating_dataframe_fn_pandas(self):
        """Test building a pandas DataFrame delegating function."""
        from otterapi.codegen.endpoints import build_delegating_dataframe_fn

        fn_ast, imports = build_delegating_dataframe_fn(
            fn_name='get_users_df',
            client_method_name='get_users_df',
            parameters=None,
            request_body_info=None,
            library='pandas',
            default_path=None,
            docs='Get all users.',
            is_async=False,
        )

        assert isinstance(fn_ast, ast.FunctionDef)
        assert fn_ast.name == 'get_users_df'
        # Return type should be string annotation
        assert isinstance(fn_ast.returns, ast.Constant)
        assert fn_ast.returns.value == 'pd.DataFrame'

    def test_build_delegating_dataframe_fn_polars(self):
        """Test building a polars DataFrame delegating function."""
        from otterapi.codegen.endpoints import build_delegating_dataframe_fn

        fn_ast, imports = build_delegating_dataframe_fn(
            fn_name='get_users_pl',
            client_method_name='get_users_pl',
            parameters=None,
            request_body_info=None,
            library='polars',
            default_path=None,
            docs='Get all users.',
            is_async=False,
        )

        assert isinstance(fn_ast, ast.FunctionDef)
        assert fn_ast.name == 'get_users_pl'
        assert isinstance(fn_ast.returns, ast.Constant)
        assert fn_ast.returns.value == 'pl.DataFrame'

    def test_build_delegating_dataframe_fn_async(self):
        """Test building an async DataFrame delegating function."""
        from otterapi.codegen.endpoints import build_delegating_dataframe_fn

        fn_ast, imports = build_delegating_dataframe_fn(
            fn_name='async_get_users_df',
            client_method_name='async_get_users_df',
            parameters=None,
            request_body_info=None,
            library='pandas',
            default_path=None,
            docs='Get all users.',
            is_async=True,
        )

        assert isinstance(fn_ast, ast.AsyncFunctionDef)
        assert fn_ast.name == 'async_get_users_df'

    def test_build_delegating_dataframe_fn_with_path(self):
        """Test DataFrame function with default path."""
        from otterapi.codegen.endpoints import build_delegating_dataframe_fn

        fn_ast, imports = build_delegating_dataframe_fn(
            fn_name='get_users_df',
            client_method_name='get_users_df',
            parameters=None,
            request_body_info=None,
            library='pandas',
            default_path='data.users',
            docs='Get all users.',
            is_async=False,
        )

        # Find the path parameter default value
        path_default = None
        for i, kwarg in enumerate(fn_ast.args.kwonlyargs):
            if kwarg.arg == 'path':
                path_default = fn_ast.args.kw_defaults[i]
                break

        assert path_default is not None
        assert isinstance(path_default, ast.Constant)
        assert path_default.value == 'data.users'

    def test_build_delegating_dataframe_fn_includes_path_param(self):
        """Test that DataFrame function includes path parameter."""
        from otterapi.codegen.endpoints import build_delegating_dataframe_fn

        fn_ast, imports = build_delegating_dataframe_fn(
            fn_name='get_users_df',
            client_method_name='get_users_df',
            parameters=None,
            request_body_info=None,
            library='pandas',
            default_path=None,
            docs='Get all users.',
            is_async=False,
        )

        # Check that 'path' is in kwonlyargs
        kwarg_names = [arg.arg for arg in fn_ast.args.kwonlyargs]
        assert 'path' in kwarg_names


class TestDocumentConfigWithDataFrame:
    """Tests for DocumentConfig with DataFrame settings."""

    def test_document_config_has_dataframe(self):
        """Test that DocumentConfig has dataframe field."""
        from otterapi.config import DocumentConfig

        config = DocumentConfig(
            source='https://example.com/openapi.json',
            output='./client',
        )
        assert hasattr(config, 'dataframe')
        assert isinstance(config.dataframe, DataFrameConfig)

    def test_document_config_with_dataframe_enabled(self):
        """Test DocumentConfig with DataFrame enabled."""
        from otterapi.config import DocumentConfig

        config = DocumentConfig(
            source='https://example.com/openapi.json',
            output='./client',
            dataframe=DataFrameConfig(
                enabled=True,
                pandas=True,
                polars=True,
            ),
        )
        assert config.dataframe.enabled is True
        assert config.dataframe.pandas is True
        assert config.dataframe.polars is True

    def test_document_config_with_endpoint_overrides(self):
        """Test DocumentConfig with DataFrame endpoint overrides."""
        from otterapi.config import DocumentConfig

        config = DocumentConfig(
            source='https://example.com/openapi.json',
            output='./client',
            dataframe=DataFrameConfig(
                enabled=True,
                endpoints={
                    'get_users': EndpointDataFrameConfig(
                        path='data.users',
                    ),
                },
            ),
        )
        assert 'get_users' in config.dataframe.endpoints
        assert config.dataframe.endpoints['get_users'].path == 'data.users'


class TestDataFrameIntegration:
    """Integration tests for DataFrame code generation."""

    @pytest.fixture
    def simple_openapi_spec(self):
        """A simple OpenAPI spec with list-returning endpoints."""
        return {
            'openapi': '3.0.0',
            'info': {'title': 'Test API', 'version': '1.0.0'},
            'servers': [{'url': 'https://api.example.com'}],
            'paths': {
                '/users': {
                    'get': {
                        'operationId': 'getUsers',
                        'summary': 'Get all users',
                        'responses': {
                            '200': {
                                'description': 'Success',
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
                },
                '/users/{id}': {
                    'get': {
                        'operationId': 'getUserById',
                        'summary': 'Get user by ID',
                        'parameters': [
                            {
                                'name': 'id',
                                'in': 'path',
                                'required': True,
                                'schema': {'type': 'integer'},
                            }
                        ],
                        'responses': {
                            '200': {
                                'description': 'Success',
                                'content': {
                                    'application/json': {
                                        'schema': {'$ref': '#/components/schemas/User'}
                                    }
                                },
                            }
                        },
                    }
                },
            },
            'components': {
                'schemas': {
                    'User': {
                        'type': 'object',
                        'properties': {
                            'id': {'type': 'integer'},
                            'name': {'type': 'string'},
                            'email': {'type': 'string'},
                        },
                    }
                }
            },
        }

    def test_generate_with_dataframe_enabled(self, tmp_path, simple_openapi_spec):
        """Test generating a client with DataFrame methods enabled."""
        import json

        from otterapi.codegen.codegen import Codegen
        from otterapi.config import DataFrameConfig, DocumentConfig

        # Write the spec to a file
        spec_file = tmp_path / 'openapi.json'
        spec_file.write_text(json.dumps(simple_openapi_spec))

        output_dir = tmp_path / 'client'

        config = DocumentConfig(
            source=str(spec_file),
            output=str(output_dir),
            dataframe=DataFrameConfig(
                enabled=True,
                pandas=True,
                polars=True,
            ),
        )

        codegen = Codegen(config)
        codegen.generate()

        # Check that _dataframe.py was generated
        dataframe_file = output_dir / '_dataframe.py'
        assert dataframe_file.exists()

        # Check _client.py exists (now contains only infrastructure)
        client_file = output_dir / '_client.py'
        assert client_file.exists()

        # Check endpoints.py for DataFrame methods (new architecture)
        # DataFrame methods are now in the endpoints file with full implementations
        endpoints_file = output_dir / 'endpoints.py'
        assert endpoints_file.exists()
        endpoints_content = endpoints_file.read_text()

        # Should have DataFrame imports in endpoints file
        assert 'to_pandas' in endpoints_content
        assert 'to_polars' in endpoints_content
        assert 'TYPE_CHECKING' in endpoints_content

        # Should have _df methods for list-returning endpoint
        assert 'get_users_df' in endpoints_content
        assert 'async_get_users_df' in endpoints_content

        # Should have _pl methods for list-returning endpoint
        assert 'get_users_pl' in endpoints_content
        assert 'async_get_users_pl' in endpoints_content

        # Should NOT have DataFrame methods for non-list endpoint (getUserById)
        # because it returns a single User, not a list
        assert 'get_user_by_id_df' not in endpoints_content
        assert 'get_user_by_id_pl' not in endpoints_content

    def test_generate_with_dataframe_pandas_only(self, tmp_path, simple_openapi_spec):
        """Test generating with only pandas enabled."""
        import json

        from otterapi.codegen.codegen import Codegen
        from otterapi.config import DataFrameConfig, DocumentConfig

        spec_file = tmp_path / 'openapi.json'
        spec_file.write_text(json.dumps(simple_openapi_spec))

        output_dir = tmp_path / 'client'

        config = DocumentConfig(
            source=str(spec_file),
            output=str(output_dir),
            dataframe=DataFrameConfig(
                enabled=True,
                pandas=True,
                polars=False,
            ),
        )

        codegen = Codegen(config)
        codegen.generate()

        endpoints_file = output_dir / 'endpoints.py'
        endpoints_content = endpoints_file.read_text()

        # Should have _df methods
        assert 'get_users_df' in endpoints_content

        # Should NOT have _pl methods
        assert 'get_users_pl' not in endpoints_content

    def test_generate_with_dataframe_polars_only(self, tmp_path, simple_openapi_spec):
        """Test generating with only polars enabled."""
        import json

        from otterapi.codegen.codegen import Codegen
        from otterapi.config import DataFrameConfig, DocumentConfig

        spec_file = tmp_path / 'openapi.json'
        spec_file.write_text(json.dumps(simple_openapi_spec))

        output_dir = tmp_path / 'client'

        config = DocumentConfig(
            source=str(spec_file),
            output=str(output_dir),
            dataframe=DataFrameConfig(
                enabled=True,
                pandas=False,
                polars=True,
            ),
        )

        codegen = Codegen(config)
        codegen.generate()

        endpoints_file = output_dir / 'endpoints.py'
        endpoints_content = endpoints_file.read_text()

        # Should NOT have _df methods
        assert 'get_users_df' not in endpoints_content

        # Should have _pl methods
        assert 'get_users_pl' in endpoints_content

    def test_generate_with_endpoint_path_config(self, tmp_path, simple_openapi_spec):
        """Test generating with endpoint-specific path configuration."""
        import json

        from otterapi.codegen.codegen import Codegen
        from otterapi.config import (
            DataFrameConfig,
            DocumentConfig,
            EndpointDataFrameConfig,
        )

        spec_file = tmp_path / 'openapi.json'
        spec_file.write_text(json.dumps(simple_openapi_spec))

        output_dir = tmp_path / 'client'

        config = DocumentConfig(
            source=str(spec_file),
            output=str(output_dir),
            dataframe=DataFrameConfig(
                enabled=True,
                pandas=True,
                polars=False,
                endpoints={
                    'get_users': EndpointDataFrameConfig(
                        path='data.users',
                    ),
                },
            ),
        )

        codegen = Codegen(config)
        codegen.generate()

        endpoints_file = output_dir / 'endpoints.py'
        endpoints_content = endpoints_file.read_text()

        # Should have the configured path as default
        # The generated code uses Union[str, None] format
        assert "'data.users'" in endpoints_content
        assert 'get_users_df' in endpoints_content

    def test_generate_without_dataframe(self, tmp_path, simple_openapi_spec):
        """Test generating without DataFrame methods (default behavior)."""
        import json

        from otterapi.codegen.codegen import Codegen
        from otterapi.config import DocumentConfig

        spec_file = tmp_path / 'openapi.json'
        spec_file.write_text(json.dumps(simple_openapi_spec))

        output_dir = tmp_path / 'client'

        config = DocumentConfig(
            source=str(spec_file),
            output=str(output_dir),
            # dataframe not configured, defaults to disabled
        )

        codegen = Codegen(config)
        codegen.generate()

        # Should NOT have _dataframe.py
        dataframe_file = output_dir / '_dataframe.py'
        assert not dataframe_file.exists()

        client_file = output_dir / '_client.py'
        client_content = client_file.read_text()

        # Should NOT have DataFrame methods
        assert '_df' not in client_content
        assert '_pl' not in client_content
        assert 'to_pandas' not in client_content
        assert 'to_polars' not in client_content

    def test_generated_code_is_valid_python(self, tmp_path, simple_openapi_spec):
        """Test that all generated code is valid Python."""
        import json

        from otterapi.codegen.codegen import Codegen
        from otterapi.config import DataFrameConfig, DocumentConfig

        spec_file = tmp_path / 'openapi.json'
        spec_file.write_text(json.dumps(simple_openapi_spec))

        output_dir = tmp_path / 'client'

        config = DocumentConfig(
            source=str(spec_file),
            output=str(output_dir),
            dataframe=DataFrameConfig(
                enabled=True,
                pandas=True,
                polars=True,
            ),
        )

        codegen = Codegen(config)
        codegen.generate()

        # Check all generated Python files are valid
        for py_file in output_dir.glob('*.py'):
            content = py_file.read_text()
            try:
                ast.parse(content)
            except SyntaxError as e:
                pytest.fail(f'Invalid Python in {py_file.name}: {e}')


class TestDataFrameNormalization:
    """Tests for DataFrame normalization handling Pydantic models."""

    def test_normalize_data_with_pydantic_models(self, tmp_path):
        """Test that _normalize_data correctly converts Pydantic models to dicts."""
        from pydantic import BaseModel

        from otterapi.codegen.dataframes import generate_dataframe_module

        # Generate the module
        output_path = generate_dataframe_module(tmp_path)

        # Import the generated module
        import importlib.util

        spec = importlib.util.spec_from_file_location('_dataframe', output_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Define a test Pydantic model
        class TestModel(BaseModel):
            id: int
            name: str

        # Test with list of Pydantic models
        models = [TestModel(id=1, name='Alice'), TestModel(id=2, name='Bob')]
        normalized = module._normalize_data(models)

        assert isinstance(normalized, list)
        assert len(normalized) == 2
        assert normalized[0] == {'id': 1, 'name': 'Alice'}
        assert normalized[1] == {'id': 2, 'name': 'Bob'}

    def test_normalize_data_with_dicts(self, tmp_path):
        """Test that _normalize_data passes through dicts unchanged."""
        from otterapi.codegen.dataframes import generate_dataframe_module

        output_path = generate_dataframe_module(tmp_path)

        import importlib.util

        spec = importlib.util.spec_from_file_location('_dataframe', output_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Test with list of dicts
        dicts = [{'id': 1, 'name': 'Alice'}, {'id': 2, 'name': 'Bob'}]
        normalized = module._normalize_data(dicts)

        assert normalized == dicts

    def test_normalize_data_with_single_dict(self, tmp_path):
        """Test that _normalize_data wraps single dict in a list."""
        from otterapi.codegen.dataframes import generate_dataframe_module

        output_path = generate_dataframe_module(tmp_path)

        import importlib.util

        spec = importlib.util.spec_from_file_location('_dataframe', output_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        single_dict = {'id': 1, 'name': 'Alice'}
        normalized = module._normalize_data(single_dict)

        assert normalized == [single_dict]

    def test_normalize_data_with_empty_list(self, tmp_path):
        """Test that _normalize_data handles empty list."""
        from otterapi.codegen.dataframes import generate_dataframe_module

        output_path = generate_dataframe_module(tmp_path)

        import importlib.util

        spec = importlib.util.spec_from_file_location('_dataframe', output_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        normalized = module._normalize_data([])
        assert normalized == []

    def test_to_pandas_with_pydantic_models(self, tmp_path):
        """Test that to_pandas correctly handles Pydantic models."""
        pytest.importorskip('pandas')
        from pydantic import BaseModel

        from otterapi.codegen.dataframes import generate_dataframe_module

        output_path = generate_dataframe_module(tmp_path)

        import importlib.util

        spec = importlib.util.spec_from_file_location('_dataframe', output_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        class TestModel(BaseModel):
            id: int
            name: str

        models = [TestModel(id=1, name='Alice'), TestModel(id=2, name='Bob')]
        df = module.to_pandas(models)

        assert len(df) == 2
        assert list(df.columns) == ['id', 'name']
        assert df['id'].tolist() == [1, 2]
        assert df['name'].tolist() == ['Alice', 'Bob']

    def test_to_polars_with_pydantic_models(self, tmp_path):
        """Test that to_polars correctly handles Pydantic models."""
        pytest.importorskip('polars')
        from pydantic import BaseModel

        from otterapi.codegen.dataframes import generate_dataframe_module

        output_path = generate_dataframe_module(tmp_path)

        import importlib.util

        spec = importlib.util.spec_from_file_location('_dataframe', output_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        class TestModel(BaseModel):
            id: int
            name: str

        models = [TestModel(id=1, name='Alice'), TestModel(id=2, name='Bob')]
        df = module.to_polars(models)

        assert len(df) == 2
        assert df.columns == ['id', 'name']
        assert df['id'].to_list() == [1, 2]
        assert df['name'].to_list() == ['Alice', 'Bob']

    def test_paginated_endpoint_generates_dataframe_module(self, tmp_path):
        """Test that _dataframe.py is generated when pagination + dataframe is enabled.

        This tests the case where an endpoint doesn't return a list by itself,
        but pagination is configured which makes it return lists, and thus
        dataframe methods should be generated.
        """
        import json

        from otterapi.codegen.codegen import Codegen
        from otterapi.config import (
            DataFrameConfig,
            DocumentConfig,
            EndpointPaginationConfig,
            PaginationConfig,
        )

        # OpenAPI spec with an endpoint that returns an object (not a list)
        # but will be paginated
        spec = {
            'openapi': '3.0.0',
            'info': {'title': 'Test API', 'version': '1.0.0'},
            'servers': [{'url': 'https://api.example.com'}],
            'paths': {
                '/items': {
                    'get': {
                        'operationId': 'getItems',
                        'summary': 'Get paginated items',
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
                                'description': 'Success',
                                'content': {
                                    'application/json': {
                                        'schema': {
                                            # Returns an object, not a list
                                            '$ref': '#/components/schemas/PaginatedResponse'
                                        }
                                    }
                                },
                            }
                        },
                    }
                },
            },
            'components': {
                'schemas': {
                    'PaginatedResponse': {
                        'type': 'object',
                        'properties': {
                            'data': {
                                'type': 'array',
                                'items': {'$ref': '#/components/schemas/Item'},
                            },
                            'total': {'type': 'integer'},
                        },
                    },
                    'Item': {
                        'type': 'object',
                        'properties': {
                            'id': {'type': 'integer'},
                            'name': {'type': 'string'},
                        },
                    },
                }
            },
        }

        spec_file = tmp_path / 'openapi.json'
        spec_file.write_text(json.dumps(spec))

        output_dir = tmp_path / 'client'

        config = DocumentConfig(
            source=str(spec_file),
            output=str(output_dir),
            dataframe=DataFrameConfig(
                enabled=True,
                pandas=True,
            ),
            pagination=PaginationConfig(
                enabled=True,
                endpoints={
                    'get_items': EndpointPaginationConfig(
                        style='offset',
                        offset_param='offset',
                        limit_param='limit',
                        data_path='data',
                        total_path='total',
                    ),
                },
            ),
        )

        codegen = Codegen(config)
        codegen.generate()

        # The key assertion: _dataframe.py should be generated
        # even though the endpoint doesn't return a list by itself
        dataframe_file = output_dir / '_dataframe.py'
        assert dataframe_file.exists(), (
            '_dataframe.py should be generated when pagination + dataframe is enabled'
        )

        # Verify the content is correct
        content = dataframe_file.read_text()
        assert 'def to_pandas' in content
        assert 'def to_polars' in content
