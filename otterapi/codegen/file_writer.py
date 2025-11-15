"""File writing utilities for generated Python code.

This module provides utilities for writing and validating generated Python code
to the filesystem with proper syntax checking.
"""

import ast
import py_compile
import tempfile
from pathlib import Path

from upath import UPath


class PythonFileWriter:
    """Writes Python AST modules to files with validation.
    
    This class handles the conversion of AST modules to Python source code,
    validates the syntax by compiling it, and writes it to the filesystem.
    
    Example:
        >>> writer = PythonFileWriter()
        >>> body = [ast.Import(names=[ast.alias(name='sys')])]
        >>> writer.write(body, Path('output.py'))
    """
    
    def write(self, body: list[ast.stmt], path: UPath | Path | str) -> None:
        """Write a list of AST statements to a Python file.
        
        This method:
        1. Creates an AST Module from the statements
        2. Fixes missing locations in the AST
        3. Unparses the AST to Python source code
        4. Validates the code by compiling it
        5. Writes the code to the specified file
        
        Args:
            body: List of AST statement nodes to write.
            path: Path where the file should be written.
        
        Raises:
            SyntaxError: If the generated code is not valid Python.
            OSError: If the file cannot be written.
        """
        # Convert path to string for consistency
        path = Path(path) if not isinstance(path, Path) else path
        
        # Create and prepare the AST module
        mod = ast.Module(body=body, type_ignores=[])
        ast.fix_missing_locations(mod)
        
        # Convert AST to Python source code
        file_content = ast.unparse(mod)
        
        # Validate the generated code by compiling it
        self._validate_python_syntax(file_content)
        
        # Write to file
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(str(path), 'w', encoding='utf-8') as f:
            f.write(file_content)
    
    def _validate_python_syntax(self, content: str) -> None:
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
    
    def write_init_file(self, directory: UPath | Path | str) -> None:
        """Create an empty __init__.py file in the specified directory.
        
        Args:
            directory: Directory where __init__.py should be created.
        """
        directory = Path(directory) if not isinstance(directory, Path) else directory
        init_file = directory / '__init__.py'
        
        if not init_file.exists():
            directory.mkdir(parents=True, exist_ok=True)
            init_file.touch()

