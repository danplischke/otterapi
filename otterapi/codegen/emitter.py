"""Code emitter interfaces and implementations for code generation output.

This module provides the CodeEmitter interface and concrete implementations
for emitting generated code in different formats (Python files, strings, etc.).
"""

import ast
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

from upath import UPath

if TYPE_CHECKING:
    from otterapi.codegen.types import Type


class CodeEmitter(ABC):
    """Abstract base class for code emitters.

    A CodeEmitter is responsible for taking generated AST nodes and
    outputting them in a specific format (files, strings, etc.).

    Implementations can handle formatting, validation, and writing
    of generated code.
    """

    @abstractmethod
    def emit_module(
        self,
        body: list[ast.stmt],
        name: str,
        docstring: str | None = None,
    ) -> str | None:
        """Emit a complete Python module.

        Args:
            body: List of AST statements forming the module body.
            name: The module name (used for file naming or identification).
            docstring: Optional module-level docstring.

        Returns:
            The path to the emitted file, or the code string, depending
            on the implementation. May return None if emission fails.
        """
        pass

    @abstractmethod
    def emit_models(
        self,
        types: list['Type'],
        module_name: str = 'models',
    ) -> str | None:
        """Emit a module containing model definitions.

        Args:
            types: List of Type objects to emit.
            module_name: Name for the models module.

        Returns:
            The path or code string for the emitted models.
        """
        pass

    @abstractmethod
    def emit_endpoints(
        self,
        body: list[ast.stmt],
        module_name: str = 'endpoints',
    ) -> str | None:
        """Emit a module containing endpoint functions.

        Args:
            body: List of AST statements for endpoint functions.
            module_name: Name for the endpoints module.

        Returns:
            The path or code string for the emitted endpoints.
        """
        pass


class FileEmitter(CodeEmitter):
    """Emits generated code to Python files on disk.

    This emitter writes generated code to files in a specified output
    directory, handling directory creation and file formatting.
    """

    def __init__(
        self,
        output_dir: str | Path | UPath,
        format_code: bool = True,
        validate_syntax: bool = True,
        create_init: bool = True,
    ):
        """Initialize the file emitter.

        Args:
            output_dir: Directory where files will be written.
            format_code: Whether to format code with black/ruff (if available).
            validate_syntax: Whether to validate Python syntax before writing.
            create_init: Whether to create an __init__.py file.
        """
        self.output_dir = UPath(output_dir)
        self.format_code = format_code
        self.validate_syntax = validate_syntax
        self.create_init = create_init
        self._written_files: list[str] = []

    def emit_module(
        self,
        body: list[ast.stmt],
        name: str,
        docstring: str | None = None,
    ) -> str | None:
        """Emit a complete Python module to a file.

        Args:
            body: List of AST statements forming the module body.
            name: The module name (used for file naming).
            docstring: Optional module-level docstring.

        Returns:
            The path to the written file, or None if emission fails.
        """
        # Add docstring if provided
        if docstring:
            doc_node = ast.Expr(value=ast.Constant(value=docstring))
            body = [doc_node] + body

        # Create module AST
        module = ast.Module(body=body, type_ignores=[])
        ast.fix_missing_locations(module)

        # Convert to source code
        try:
            source = ast.unparse(module)
        except Exception as e:
            raise RuntimeError(f'Failed to unparse AST for module {name}: {e}')

        # Validate syntax if requested
        if self.validate_syntax:
            self._validate_syntax(source, name)

        # Format code if requested
        if self.format_code:
            source = self._format_source(source)

        # Write to file
        return self._write_file(f'{name}.py', source)

    def emit_models(
        self,
        types: list['Type'],
        module_name: str = 'models',
    ) -> str | None:
        """Emit a module containing model definitions.

        Args:
            types: List of Type objects to emit.
            module_name: Name for the models module.

        Returns:
            The path to the written models file.
        """
        # Collect all AST nodes and imports
        body: list[ast.stmt] = []
        imports: dict[str, set[str]] = {}

        for type_obj in types:
            if type_obj.implementation_ast:
                body.append(type_obj.implementation_ast)

            # Collect imports from type
            if hasattr(type_obj, 'implementation_imports'):
                for module, names in type_obj.implementation_imports.items():
                    if module not in imports:
                        imports[module] = set()
                    imports[module].update(names)

        # Build import statements
        import_stmts = self._build_imports(imports)

        # Combine imports and body
        full_body = import_stmts + body

        return self.emit_module(
            full_body, module_name, docstring='Generated models from OpenAPI schema.'
        )

    def emit_endpoints(
        self,
        body: list[ast.stmt],
        module_name: str = 'endpoints',
    ) -> str | None:
        """Emit a module containing endpoint functions.

        Args:
            body: List of AST statements for endpoint functions.
            module_name: Name for the endpoints module.

        Returns:
            The path to the written endpoints file.
        """
        return self.emit_module(
            body, module_name, docstring='Generated API endpoints from OpenAPI schema.'
        )

    def emit_init(self, exports: list[str] | None = None) -> str | None:
        """Emit an __init__.py file.

        Args:
            exports: Optional list of names to include in __all__.

        Returns:
            The path to the written __init__.py file.
        """
        body: list[ast.stmt] = []

        if exports:
            # Create __all__ = [...]
            all_assign = ast.Assign(
                targets=[ast.Name(id='__all__', ctx=ast.Store())],
                value=ast.List(
                    elts=[ast.Constant(value=name) for name in exports],
                    ctx=ast.Load(),
                ),
            )
            body.append(all_assign)

        if not body:
            # Empty __init__.py
            return self._write_file('__init__.py', '')

        module = ast.Module(body=body, type_ignores=[])
        ast.fix_missing_locations(module)
        source = ast.unparse(module)

        return self._write_file('__init__.py', source)

    def emit_py_typed(self) -> str | None:
        """Emit a py.typed marker file for PEP 561.

        Returns:
            The path to the written py.typed file.
        """
        return self._write_file('py.typed', '')

    def _write_file(self, filename: str, content: str) -> str:
        """Write content to a file in the output directory.

        Args:
            filename: Name of the file to write.
            content: Content to write to the file.

        Returns:
            The path to the written file.
        """
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        file_path = self.output_dir / filename
        file_path.write_text(content, encoding='utf-8')
        self._written_files.append(str(file_path))

        return str(file_path)

    def _validate_syntax(self, source: str, name: str) -> None:
        """Validate that source code has valid Python syntax.

        Args:
            source: The source code to validate.
            name: Name of the module (for error messages).

        Raises:
            SyntaxError: If the source code is not valid Python.
        """
        try:
            compile(source, f'{name}.py', 'exec')
        except SyntaxError as e:
            raise SyntaxError(f'Generated code for {name} has invalid syntax: {e}')

    def _format_source(self, source: str) -> str:
        """Format source code using black or ruff if available.

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

    def _build_imports(self, imports: dict[str, set[str]]) -> list[ast.ImportFrom]:
        """Build import statements from an imports dictionary.

        Args:
            imports: Dictionary mapping module names to sets of names to import.

        Returns:
            List of ImportFrom AST nodes.
        """
        import_stmts = []
        for module, names in sorted(imports.items()):
            import_stmt = ast.ImportFrom(
                module=module,
                names=[ast.alias(name=name, asname=None) for name in sorted(names)],
                level=0,
            )
            import_stmts.append(import_stmt)
        return import_stmts

    def get_written_files(self) -> list[str]:
        """Get list of all files written by this emitter.

        Returns:
            List of file paths that have been written.
        """
        return self._written_files.copy()


class StringEmitter(CodeEmitter):
    """Emits generated code as strings.

    This emitter is useful for testing or when you need to manipulate
    the generated code before writing it.
    """

    def __init__(self, format_code: bool = False):
        """Initialize the string emitter.

        Args:
            format_code: Whether to format code with black (if available).
        """
        self.format_code = format_code
        self._modules: dict[str, str] = {}

    def emit_module(
        self,
        body: list[ast.stmt],
        name: str,
        docstring: str | None = None,
    ) -> str:
        """Emit a complete Python module as a string.

        Args:
            body: List of AST statements forming the module body.
            name: The module name (used for identification).
            docstring: Optional module-level docstring.

        Returns:
            The generated source code as a string.
        """
        # Add docstring if provided
        if docstring:
            doc_node = ast.Expr(value=ast.Constant(value=docstring))
            body = [doc_node] + body

        # Create module AST
        module = ast.Module(body=body, type_ignores=[])
        ast.fix_missing_locations(module)

        # Convert to source code
        source = ast.unparse(module)

        # Format if requested
        if self.format_code:
            source = self._format_source(source)

        self._modules[name] = source
        return source

    def emit_models(
        self,
        types: list['Type'],
        module_name: str = 'models',
    ) -> str:
        """Emit a module containing model definitions as a string.

        Args:
            types: List of Type objects to emit.
            module_name: Name for the models module.

        Returns:
            The generated source code as a string.
        """
        body: list[ast.stmt] = []

        for type_obj in types:
            if type_obj.implementation_ast:
                body.append(type_obj.implementation_ast)

        return self.emit_module(body, module_name)

    def emit_endpoints(
        self,
        body: list[ast.stmt],
        module_name: str = 'endpoints',
    ) -> str:
        """Emit a module containing endpoint functions as a string.

        Args:
            body: List of AST statements for endpoint functions.
            module_name: Name for the endpoints module.

        Returns:
            The generated source code as a string.
        """
        return self.emit_module(body, module_name)

    def _format_source(self, source: str) -> str:
        """Format source code using black if available.

        Args:
            source: The source code to format.

        Returns:
            Formatted source code, or original if black unavailable.
        """
        try:
            import black

            return black.format_str(source, mode=black.Mode())
        except ImportError:
            return source
        except Exception:
            return source

    def get_module(self, name: str) -> str | None:
        """Get a previously emitted module by name.

        Args:
            name: The module name.

        Returns:
            The source code string, or None if not found.
        """
        return self._modules.get(name)

    def get_all_modules(self) -> dict[str, str]:
        """Get all emitted modules.

        Returns:
            Dictionary mapping module names to source code strings.
        """
        return self._modules.copy()
