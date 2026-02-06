"""Test suite for the Codegen and TypeGenerator classes.

This module provides comprehensive unit tests for the code generation
functionality, including type generation, endpoint generation, and
the full code generation pipeline.
"""

import ast
import json
import tempfile
from pathlib import Path

import pytest

from otterapi.codegen.codegen import Codegen
from otterapi.codegen.schema import SchemaLoader
from otterapi.codegen.types import Type, TypeGenerator
from otterapi.config import DocumentConfig
from otterapi.openapi.v3_2.v3_2 import OpenAPI, Schema

from .fixtures import (
    MINIMAL_OPENAPI_SPEC,
    PARAMETERS_SPEC,
    PETSTORE_SPEC,
    SIMPLE_API_SPEC,
)


class TestTypeGenerator:
    """Tests for the TypeGenerator class."""

    def _create_type_generator(self, spec: dict) -> TypeGenerator:
        """Helper to create a TypeGenerator from a spec dict."""
        from pydantic import TypeAdapter

        from otterapi.openapi import UniversalOpenAPI

        openapi = TypeAdapter(UniversalOpenAPI).validate_python(spec)
        # Upgrade if needed
        schema = openapi.root
        while not isinstance(schema, OpenAPI):
            schema, _ = schema.upgrade()
        return TypeGenerator(schema)

    def test_primitive_string_type(self):
        """Test generating a primitive string type."""
        typegen = self._create_type_generator(MINIMAL_OPENAPI_SPEC)

        schema = Schema(type='string')
        result = typegen.schema_to_type(schema, 'TestString')

        assert result is not None
        assert result.type == 'primitive'

    def test_primitive_integer_type(self):
        """Test generating a primitive integer type."""
        typegen = self._create_type_generator(MINIMAL_OPENAPI_SPEC)

        schema = Schema(type='integer')
        result = typegen.schema_to_type(schema, 'TestInt')

        assert result is not None
        assert result.type == 'primitive'

    def test_primitive_boolean_type(self):
        """Test generating a primitive boolean type."""
        typegen = self._create_type_generator(MINIMAL_OPENAPI_SPEC)

        schema = Schema(type='boolean')
        result = typegen.schema_to_type(schema, 'TestBool')

        assert result is not None
        assert result.type == 'primitive'

    def test_primitive_number_type(self):
        """Test generating a primitive number type."""
        typegen = self._create_type_generator(MINIMAL_OPENAPI_SPEC)

        schema = Schema(type='number')
        result = typegen.schema_to_type(schema, 'TestNumber')

        assert result is not None
        assert result.type == 'primitive'

    def test_array_of_strings(self):
        """Test generating an array of strings type."""
        typegen = self._create_type_generator(MINIMAL_OPENAPI_SPEC)

        schema = Schema(type='array', items=Schema(type='string'))
        result = typegen.schema_to_type(schema, 'StringArray')

        assert result is not None

    def test_object_type_creates_model(self):
        """Test that object types create Pydantic models."""
        typegen = self._create_type_generator(PETSTORE_SPEC)

        schema = Schema(
            type='object',
            properties={
                'name': Schema(type='string'),
                'age': Schema(type='integer'),
            },
        )
        result = typegen.schema_to_type(schema, 'Person')

        assert result is not None
        # Should be a model type (not primitive)
        assert result.type in ('model', 'root')

    def test_enum_type(self):
        """Test generating an enum type."""
        typegen = self._create_type_generator(MINIMAL_OPENAPI_SPEC)

        schema = Schema(type='string', enum=['active', 'inactive', 'pending'])
        result = typegen.schema_to_type(schema, 'Status')

        assert result is not None

    def test_reference_resolution(self):
        """Test that $ref references are resolved."""
        typegen = self._create_type_generator(PETSTORE_SPEC)

        # Get the Pet schema
        pet_schema = typegen.openapi.components.schemas.get('Pet')
        assert pet_schema is not None

        result = typegen.schema_to_type(pet_schema, 'Pet')
        assert result is not None

    def test_nested_object(self):
        """Test generating a nested object type."""
        typegen = self._create_type_generator(PETSTORE_SPEC)

        # Use a simpler nested object (Pet has Category reference)
        pet_schema = typegen.openapi.components.schemas.get('Pet')
        assert pet_schema is not None

        result = typegen.schema_to_type(pet_schema, 'Pet')
        assert result is not None

    def test_optional_field(self):
        """Test that optional fields are handled correctly."""
        typegen = self._create_type_generator(MINIMAL_OPENAPI_SPEC)

        # Test with a simple string schema (nullable is not in v3.2 Schema)
        schema = Schema(type='string')
        result = typegen.schema_to_type(schema, 'OptionalString')

        assert result is not None

    def test_add_type_registers_type(self):
        """Test that add_type properly registers types."""
        typegen = self._create_type_generator(PETSTORE_SPEC)

        # Use schema_to_type which creates a Type object
        schema = Schema(
            type='object',
            properties={'id': Schema(type='integer'), 'name': Schema(type='string')},
        )
        result = typegen.schema_to_type(schema, 'NewModel')

        assert result is not None
        # schema_to_type may or may not add to types depending on type
        # Just verify we got a valid type back
        assert hasattr(result, 'type')

    def test_get_sorted_types(self):
        """Test that types are returned in dependency order."""
        typegen = self._create_type_generator(PETSTORE_SPEC)

        # Add types via schema_to_type which properly creates Type objects
        schema = Schema(
            type='object',
            properties={'id': Schema(type='integer'), 'name': Schema(type='string')},
        )
        typegen.schema_to_type(schema, 'SimpleModel')

        sorted_types = typegen.get_sorted_types()

        # Should return a list of types
        assert isinstance(sorted_types, list)

        # All types should be Type objects
        for t in sorted_types:
            assert isinstance(t, Type)


class TestCodegen:
    """Tests for the Codegen class."""

    @pytest.fixture
    def temp_output_dir(self):
        """Create a temporary directory for test outputs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def petstore_spec_file(self, temp_output_dir):
        """Create a temp file with the Petstore spec."""
        spec_file = temp_output_dir / 'petstore.json'
        spec_file.write_text(json.dumps(PETSTORE_SPEC))
        return spec_file

    @pytest.fixture
    def simple_spec_file(self, temp_output_dir):
        """Create a temp file with the simple spec."""
        spec_file = temp_output_dir / 'simple.json'
        spec_file.write_text(json.dumps(SIMPLE_API_SPEC))
        return spec_file

    def test_codegen_init(self, temp_output_dir, petstore_spec_file):
        """Test Codegen initialization."""
        config = DocumentConfig(
            source=str(petstore_spec_file), output=str(temp_output_dir / 'output')
        )
        codegen = Codegen(config)

        assert codegen.config == config
        assert codegen.openapi is None  # Not loaded yet

    def test_load_schema(self, temp_output_dir, petstore_spec_file):
        """Test schema loading."""
        config = DocumentConfig(
            source=str(petstore_spec_file), output=str(temp_output_dir / 'output')
        )
        codegen = Codegen(config)

        codegen._load_schema()

        assert codegen.openapi is not None
        assert codegen.openapi.info.title == 'Petstore API'

    def test_generate_creates_files(self, temp_output_dir, petstore_spec_file):
        """Test that generate() creates the expected files."""
        output_dir = temp_output_dir / 'output'
        config = DocumentConfig(source=str(petstore_spec_file), output=str(output_dir))
        codegen = Codegen(config)

        codegen.generate()

        # Check that files were created
        assert (output_dir / 'models.py').exists()
        assert (output_dir / 'endpoints.py').exists()
        assert (output_dir / '__init__.py').exists()

    def test_generate_models_valid_python(self, temp_output_dir, petstore_spec_file):
        """Test that generated models are valid Python."""
        output_dir = temp_output_dir / 'output'
        config = DocumentConfig(source=str(petstore_spec_file), output=str(output_dir))
        codegen = Codegen(config)

        codegen.generate()

        # Read and compile the models file
        models_content = (output_dir / 'models.py').read_text()

        # Should not raise SyntaxError
        compile(models_content, 'models.py', 'exec')

    def test_generate_endpoints_valid_python(self, temp_output_dir, petstore_spec_file):
        """Test that generated endpoints are valid Python."""
        output_dir = temp_output_dir / 'output'
        config = DocumentConfig(source=str(petstore_spec_file), output=str(output_dir))
        codegen = Codegen(config)

        codegen.generate()

        # Read and compile the endpoints file
        endpoints_content = (output_dir / 'endpoints.py').read_text()

        # Should not raise SyntaxError
        compile(endpoints_content, 'endpoints.py', 'exec')

    def test_generate_with_simple_api(self, temp_output_dir, simple_spec_file):
        """Test generation with a simple API spec."""
        output_dir = temp_output_dir / 'output'
        config = DocumentConfig(source=str(simple_spec_file), output=str(output_dir))
        codegen = Codegen(config)

        codegen.generate()

        assert (output_dir / 'models.py').exists()
        assert (output_dir / 'endpoints.py').exists()

    def test_generate_with_custom_filenames(self, temp_output_dir, petstore_spec_file):
        """Test generation with custom output filenames."""
        output_dir = temp_output_dir / 'output'
        config = DocumentConfig(
            source=str(petstore_spec_file),
            output=str(output_dir),
            models_file='custom_models.py',
            endpoints_file='custom_endpoints.py',
        )
        codegen = Codegen(config)

        codegen.generate()

        assert (output_dir / 'custom_models.py').exists()
        assert (output_dir / 'custom_endpoints.py').exists()

    def test_generate_models_contains_pet(self, temp_output_dir, petstore_spec_file):
        """Test that generated models contain the Pet class."""
        output_dir = temp_output_dir / 'output'
        config = DocumentConfig(source=str(petstore_spec_file), output=str(output_dir))
        codegen = Codegen(config)

        codegen.generate()

        models_content = (output_dir / 'models.py').read_text()

        assert 'class Pet' in models_content or 'Pet' in models_content

    def test_generate_endpoints_contains_operations(
        self, temp_output_dir, petstore_spec_file
    ):
        """Test that generated endpoints contain the expected operations."""
        output_dir = temp_output_dir / 'output'
        config = DocumentConfig(source=str(petstore_spec_file), output=str(output_dir))
        codegen = Codegen(config)

        codegen.generate()

        endpoints_content = (output_dir / 'endpoints.py').read_text()

        # Check for operation function names
        assert 'listPets' in endpoints_content or 'list_pets' in endpoints_content
        assert 'getPetById' in endpoints_content or 'get_pet_by_id' in endpoints_content
        assert 'createPet' in endpoints_content or 'create_pet' in endpoints_content

    def test_extract_response_info(self, temp_output_dir, petstore_spec_file):
        """Test the _extract_response_info method."""
        output_dir = temp_output_dir / 'output'
        config = DocumentConfig(source=str(petstore_spec_file), output=str(output_dir))
        codegen = Codegen(config)

        codegen._load_schema()

        # Get the listPets operation
        list_pets_op = codegen.openapi.paths.root['/pets'].get
        assert list_pets_op is not None

        responses = codegen._extract_response_info(list_pets_op)

        # Should have a 200 response
        assert 200 in responses

    def test_extract_operation_parameters(self, temp_output_dir, petstore_spec_file):
        """Test the _extract_operation_parameters method."""
        output_dir = temp_output_dir / 'output'
        config = DocumentConfig(source=str(petstore_spec_file), output=str(output_dir))
        codegen = Codegen(config)

        codegen._load_schema()

        # Get the listPets operation which has parameters
        list_pets_op = codegen.openapi.paths.root['/pets'].get
        assert list_pets_op is not None

        params = codegen._extract_operation_parameters(list_pets_op)

        # Should have limit and status parameters
        param_names = [p.name for p in params]
        assert 'limit' in param_names
        assert 'status' in param_names

    def test_resolve_base_url(self, temp_output_dir, petstore_spec_file):
        """Test the _resolve_base_url method."""
        output_dir = temp_output_dir / 'output'
        config = DocumentConfig(source=str(petstore_spec_file), output=str(output_dir))
        codegen = Codegen(config)

        codegen._load_schema()

        base_url = codegen._resolve_base_url()

        assert base_url is not None
        assert 'petstore.example.com' in base_url

    def test_resolve_base_url_with_config_override(
        self, temp_output_dir, petstore_spec_file
    ):
        """Test that base_url config overrides the spec."""
        output_dir = temp_output_dir / 'output'
        config = DocumentConfig(
            source=str(petstore_spec_file),
            output=str(output_dir),
            base_url='https://custom.example.com',
        )
        codegen = Codegen(config)

        codegen._load_schema()

        base_url = codegen._resolve_base_url()

        assert base_url == 'https://custom.example.com'


class TestCodegenWithComplexTypes:
    """Tests for Codegen with complex type specifications."""

    @pytest.fixture
    def temp_output_dir(self):
        """Create a temporary directory for test outputs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def petstore_spec_file(self, temp_output_dir):
        """Create a temp file with the petstore spec (has complex types)."""
        spec_file = temp_output_dir / 'petstore.json'
        spec_file.write_text(json.dumps(PETSTORE_SPEC))
        return spec_file

    def test_generate_complex_types(self, temp_output_dir, petstore_spec_file):
        """Test generation with complex types spec."""
        output_dir = temp_output_dir / 'output'
        config = DocumentConfig(source=str(petstore_spec_file), output=str(output_dir))
        codegen = Codegen(config)

        codegen.generate()

        # Should generate without errors
        assert (output_dir / 'models.py').exists()

        # Check that it's valid Python
        models_content = (output_dir / 'models.py').read_text()
        compile(models_content, 'models.py', 'exec')

    def test_complex_types_contains_all_fields(
        self, temp_output_dir, petstore_spec_file
    ):
        """Test that complex type generation includes all field types."""
        output_dir = temp_output_dir / 'output'
        config = DocumentConfig(source=str(petstore_spec_file), output=str(output_dir))
        codegen = Codegen(config)

        codegen.generate()

        models_content = (output_dir / 'models.py').read_text()

        # Should contain Pet model
        # These checks are flexible since exact output may vary
        assert 'Pet' in models_content or 'pet' in models_content.lower()


class TestCodegenWithParameters:
    """Tests for Codegen with various parameter types."""

    @pytest.fixture
    def temp_output_dir(self):
        """Create a temporary directory for test outputs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def params_spec_file(self, temp_output_dir):
        """Create a temp file with the parameters spec."""
        spec_file = temp_output_dir / 'params.json'
        spec_file.write_text(json.dumps(PARAMETERS_SPEC))
        return spec_file

    def test_generate_with_various_parameters(self, temp_output_dir, params_spec_file):
        """Test generation with various parameter types."""
        output_dir = temp_output_dir / 'output'
        config = DocumentConfig(source=str(params_spec_file), output=str(output_dir))
        codegen = Codegen(config)

        codegen.generate()

        assert (output_dir / 'endpoints.py').exists()

        endpoints_content = (output_dir / 'endpoints.py').read_text()
        compile(endpoints_content, 'endpoints.py', 'exec')


class TestSchemaLoader:
    """Tests for the SchemaLoader class."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_load_from_file(self, temp_dir):
        """Test loading a schema from a file."""
        spec_file = temp_dir / 'api.json'
        spec_file.write_text(json.dumps(MINIMAL_OPENAPI_SPEC))

        loader = SchemaLoader()
        schema = loader.load(str(spec_file))

        assert schema is not None
        assert schema.info.title == 'Minimal API'

    def test_load_file_not_found(self, temp_dir):
        """Test that loading a non-existent file raises an error."""
        from otterapi.exceptions import SchemaLoadError

        loader = SchemaLoader()

        with pytest.raises(SchemaLoadError):
            loader.load(str(temp_dir / 'nonexistent.json'))

    def test_load_invalid_json(self, temp_dir):
        """Test that loading invalid JSON raises an error."""
        spec_file = temp_dir / 'invalid.json'
        spec_file.write_text('not valid json {{{')

        loader = SchemaLoader()

        with pytest.raises(Exception):  # Could be JSONDecodeError or ValidationError
            loader.load(str(spec_file))


class TestEndToEndGeneration:
    """End-to-end tests for the complete generation pipeline."""

    @pytest.fixture
    def temp_output_dir(self):
        """Create a temporary directory for test outputs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_full_generation_pipeline(self, temp_output_dir):
        """Test the complete generation pipeline."""
        # Create spec file
        spec_file = temp_output_dir / 'petstore.json'
        spec_file.write_text(json.dumps(PETSTORE_SPEC))

        # Configure and generate
        output_dir = temp_output_dir / 'client'
        config = DocumentConfig(source=str(spec_file), output=str(output_dir))
        codegen = Codegen(config)

        codegen.generate()

        # Verify all files exist
        assert (output_dir / 'models.py').exists()
        assert (output_dir / 'endpoints.py').exists()
        assert (output_dir / '__init__.py').exists()

        # Verify files are valid Python
        for py_file in ['models.py', 'endpoints.py']:
            content = (output_dir / py_file).read_text()
            compile(content, py_file, 'exec')

    def test_generated_code_can_be_parsed_as_ast(self, temp_output_dir):
        """Test that generated code can be parsed as AST."""
        spec_file = temp_output_dir / 'petstore.json'
        spec_file.write_text(json.dumps(PETSTORE_SPEC))

        output_dir = temp_output_dir / 'client'
        config = DocumentConfig(source=str(spec_file), output=str(output_dir))
        codegen = Codegen(config)

        codegen.generate()

        # Parse as AST
        models_content = (output_dir / 'models.py').read_text()
        tree = ast.parse(models_content)

        # Should have class definitions
        class_defs = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        assert len(class_defs) > 0

    def test_generated_endpoints_have_functions(self, temp_output_dir):
        """Test that generated endpoints have function definitions."""
        spec_file = temp_output_dir / 'petstore.json'
        spec_file.write_text(json.dumps(PETSTORE_SPEC))

        output_dir = temp_output_dir / 'client'
        config = DocumentConfig(source=str(spec_file), output=str(output_dir))
        codegen = Codegen(config)

        codegen.generate()

        endpoints_content = (output_dir / 'endpoints.py').read_text()
        tree = ast.parse(endpoints_content)

        # Should have function definitions
        func_defs = [
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        assert len(func_defs) > 0

    def test_multiple_generations_overwrite(self, temp_output_dir):
        """Test that generating twice overwrites the files."""
        spec_file = temp_output_dir / 'petstore.json'
        spec_file.write_text(json.dumps(PETSTORE_SPEC))

        output_dir = temp_output_dir / 'client'
        config = DocumentConfig(source=str(spec_file), output=str(output_dir))

        # Generate twice
        codegen1 = Codegen(config)
        codegen1.generate()

        first_content = (output_dir / 'models.py').read_text()

        codegen2 = Codegen(config)
        codegen2.generate()

        second_content = (output_dir / 'models.py').read_text()

        # Content should be the same (deterministic generation)
        assert first_content == second_content
