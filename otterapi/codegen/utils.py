import ast
import py_compile
import re
import tempfile
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from upath import UPath

from otterapi.openapi.v3_2 import OpenAPI, Reference, Schema

__all__ = ('is_url', 'sanitize_identifier', 'to_snake_case')


def capitalize(input_string):
    if not input_string:
        return ''
    return input_string[0].upper() + input_string[1:]


def is_url(text):
    try:
        result = urlparse(text)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False


def remove_accents(input_str):
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    return ''.join(c for c in nfkd_form if not unicodedata.combining(c))


def sanitize_name_python_keywords(name: str) -> str:
    import keyword

    if name in keyword.kwlist:
        return f'{name}_'
    return name


def sanitize_parameter_field_name(name: str) -> str:
    """Sanitize parameter or field names to be valid Python identifiers.

    - Replace spaces and hyphens with underscores
    - Remove other invalid characters
    - Ensure it doesn't start with a digit
    """
    if not name:
        raise ValueError('Name cannot be empty')

    sanitized = sanitize_name_python_keywords(name)
    sanitized = re.sub(r'[-\s]+', '_', remove_accents(sanitized))
    sanitized = re.sub(r'[^A-Za-z0-9_]', '', sanitized)

    if sanitized and sanitized[0].isdigit():
        sanitized = '_' + sanitized
    return sanitized


def to_snake_case(name: str) -> str:
    """Convert a camelCase or PascalCase string to snake_case.

    Args:
        name: The string to convert (e.g., 'getUserById', 'HTTPResponse').

    Returns:
        The snake_case version (e.g., 'get_user_by_id', 'http_response').
    """
    if not name:
        return name

    # Handle acronyms and consecutive uppercase letters
    # Insert underscore before uppercase letters that are followed by lowercase
    # or before uppercase letters that follow lowercase letters
    result = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', name)
    # Handle consecutive uppercase followed by lowercase (e.g., HTTPResponse -> HTTP_Response)
    result = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', result)
    # Convert to lowercase
    result = result.lower()
    # Replace any non-alphanumeric characters with underscores
    result = re.sub(r'[^a-z0-9]+', '_', result)
    # Remove leading/trailing underscores and collapse multiple underscores
    result = re.sub(r'_+', '_', result).strip('_')

    return sanitize_name_python_keywords(result)


def sanitize_identifier(name: str) -> str:
    """Convert a string into a valid Python identifier.

    - Replace spaces and hyphens with underscores
    - Remove other invalid characters
    - Ensure it doesn't start with a digit
    - Convert to PascalCase for class names
    """
    if not name:
        return 'UnnamedType'

    # Replace spaces and hyphens with underscores, then split
    parts = re.sub(r'[^A-Za-z0-9]+', '_', remove_accents(name)).split('_')

    # Capitalize each part and join (PascalCase)
    if len(parts) == 1:
        sanitized = parts[0]
    else:
        sanitized = ''.join(capitalize(part) for part in parts if part)

    # Remove any remaining invalid characters
    sanitized = re.sub(r'[^A-Za-z0-9_]', '', sanitized)

    # Ensure it doesn't start with a digit
    if sanitized and sanitized[0].isdigit():
        sanitized = '_' + sanitized

    # If empty after sanitization, return a default
    return sanitized or 'UnnamedType'


def validate_python_syntax(content: str) -> None:
    """Validate that the content is valid Python code.

    Args:
        content: Python source code as a string.

    Raises:
        SyntaxError: If the code is not valid Python.
    """
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(content)
        f.flush()
        temp_path = f.name

    try:
        # Compile to check for syntax errors
        py_compile.compile(temp_path, doraise=True)
    finally:
        # Clean up temp file
        Path(temp_path).unlink(missing_ok=True)


def format_source(source: str) -> str:
    """Format Python source code using ruff or black if available.

    Args:
        source: The source code to format.

    Returns:
        Formatted source code, or original if formatters unavailable.
    """
    # Try ruff first
    try:
        import subprocess

        result = subprocess.run(
            ['ruff', 'format', '-'],
            input=source,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return result.stdout
    except (FileNotFoundError, subprocess.SubprocessError):
        pass

    # Try black
    try:
        import black

        return black.format_str(source, mode=black.Mode())
    except ImportError:
        pass
    except Exception:
        pass

    # Return original if no formatter available
    return source


def write_mod(
    body: list[ast.stmt], path: UPath | Path | str, format_code: bool = True
) -> None:
    """Write a list of AST statements to a Python file.

    This method:
    1. Creates an AST Module from the statements
    2. Fixes missing locations in the AST
    3. Unparses the AST to Python source code
    4. Validates the code by compiling it
    5. Optionally formats the code with ruff/black
    6. Writes the code to the specified file

    Args:
        body: List of AST statement nodes to write.
        path: Path where the file should be written.
        format_code: Whether to format the code with ruff/black. Defaults to True.

    Raises:
        SyntaxError: If the generated code is not valid Python.
        OSError: If the file cannot be written.
    """
    # Convert path to string for consistency
    path = UPath(path)

    # Create and prepare the AST module
    mod = ast.Module(body=body, type_ignores=[])
    ast.fix_missing_locations(mod)

    # Convert AST to Python source code
    file_content = ast.unparse(mod)

    # Validate the generated code by compiling it
    validate_python_syntax(file_content)

    # Format code if requested
    if format_code:
        file_content = format_source(file_content)

    # Write to file
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(path), 'w', encoding='utf-8') as f:
        f.write(file_content)


def write_init_file(directory: UPath | Path | str) -> None:
    """Create an empty __init__.py file in the specified directory.

    Args:
        directory: Directory where __init__.py should be created.
    """
    directory = UPath(directory)
    init_file = directory / '__init__.py'

    if not init_file.exists():
        directory.mkdir(parents=True, exist_ok=True)
        init_file.touch()


@dataclass
class OpenAPIProcessor:
    openapi: OpenAPI | None = None

    def _resolve_reference(self, reference: Reference | Schema) -> tuple[Schema, str]:
        if isinstance(reference, Reference):
            if not reference.ref.startswith('#/components/schemas/'):
                raise ValueError(f'Unsupported reference format: {reference.ref}')

            schema_name = reference.ref.split('/')[-1]
            schemas = self.openapi.components.schemas

            if schema_name not in schemas:
                raise ValueError(
                    f"Referenced schema '{schema_name}' not found in components.schemas"
                )

            return schemas[schema_name], sanitize_identifier(schema_name)
        return reference, sanitize_identifier(reference.title) if hasattr(
            reference, 'title'
        ) and reference.title else None
