"""Split module emitter for generating organized endpoint modules.

This module provides the SplitModuleEmitter class that takes a ModuleTree
and emits the organized endpoint modules to the filesystem, handling
imports, exports, and directory structure.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from upath import UPath

from otterapi.codegen.ast_utils import _all
from otterapi.codegen.import_collector import ImportCollector
from otterapi.codegen.utils import write_mod

if TYPE_CHECKING:
    from otterapi.codegen.splitting.tree import ModuleTree
    from otterapi.codegen.types import Endpoint, Type
    from otterapi.config import ModuleSplitConfig


@dataclass
class EmittedModule:
    """Information about an emitted module.

    Attributes:
        path: The file path where the module was written.
        module_path: The module path components, e.g., ["api", "v1", "users"].
        endpoint_names: Names of endpoints (functions) in this module.
    """

    path: Path
    module_path: list[str]
    endpoint_names: list[str] = field(default_factory=list)


class SplitModuleEmitter:
    """Emits split modules based on a ModuleTree structure.

    This emitter generates:
    - Endpoint files organized by module hierarchy
    - __init__.py files with proper exports
    - Handles both flat and nested directory structures

    Example:
        >>> from otterapi.config import ModuleSplitConfig
        >>> config = ModuleSplitConfig(enabled=True)
        >>> emitter = SplitModuleEmitter(config, output_dir, models_file)
        >>> emitter.emit(tree, base_url)
    """

    def __init__(
        self,
        config: ModuleSplitConfig,
        output_dir: Path | UPath,
        models_file: Path | UPath,
        models_import_path: str | None = None,
        client_class_name: str = 'APIClient',
    ):
        """Initialize the split module emitter.

        Args:
            config: The module split configuration.
            output_dir: The root output directory for generated files.
            models_file: Path to the models file for import generation.
            models_import_path: Optional custom import path for models.
            client_class_name: Name of the client class (e.g., 'SwaggerPetstoreOpenAPI30Client').
        """
        self.config = config
        self.output_dir = UPath(output_dir)
        self.models_file = UPath(models_file)
        self.models_import_path = models_import_path
        self.client_class_name = client_class_name
        self._emitted_modules: list[EmittedModule] = []

    def emit(
        self,
        tree: ModuleTree,
        base_url: str,
        typegen_types: dict[str, Type] | None = None,
    ) -> list[EmittedModule]:
        """Emit all modules from the tree.

        Args:
            tree: The ModuleTree containing organized endpoints.
            base_url: The base URL for API requests.
            typegen_types: Optional dict of types for collecting model imports.

        Returns:
            List of EmittedModule objects describing what was written.
        """
        self._emitted_modules = []
        self._typegen_types = typegen_types or {}

        if self.config.flat_structure:
            self._emit_flat(tree, base_url)
        else:
            self._emit_nested(tree, base_url)

        return self._emitted_modules

    def _emit_flat(self, tree: ModuleTree, base_url: str) -> None:
        """Emit modules as flat files (no subdirectories).

        Args:
            tree: The ModuleTree to emit.
            base_url: The base URL for API requests.
        """
        # Collect all endpoint exports for the main __init__.py
        all_exports: dict[str, list[str]] = {}  # module_name -> [endpoint_names]

        for module_path, node in tree.walk_leaves():
            if not node.endpoints:
                continue

            # Generate flat filename: api_v1_users.py
            flat_name = '_'.join(module_path) if module_path else 'endpoints'
            file_path = self.output_dir / f'{flat_name}.py'

            # Emit the module
            endpoint_names = self._emit_module_file(
                file_path=file_path,
                endpoints=node.endpoints,
                base_url=base_url,
                description=node.description,
                module_path=module_path,
            )

            all_exports[flat_name] = endpoint_names

            self._emitted_modules.append(
                EmittedModule(
                    path=file_path,
                    module_path=module_path,
                    endpoint_names=endpoint_names,
                )
            )

        # Generate main __init__.py that imports from all flat modules
        self._emit_flat_init(all_exports)

    def _emit_nested(self, tree: ModuleTree, base_url: str) -> None:
        """Emit modules as nested directories.

        Args:
            tree: The ModuleTree to emit.
            base_url: The base URL for API requests.
        """
        # Track which directories need __init__.py files
        directories: set[tuple[str, ...]] = set()

        for module_path, node in tree.walk_leaves():
            if not node.endpoints:
                continue

            # Create directory structure if needed
            if len(module_path) > 1:
                # Module is in a subdirectory
                dir_path = self.output_dir / '/'.join(module_path[:-1])
                dir_path.mkdir(parents=True, exist_ok=True)
                directories.add(tuple(module_path[:-1]))

                # Add all parent directories
                for i in range(1, len(module_path) - 1):
                    directories.add(tuple(module_path[:i]))

                file_path = dir_path / f'{module_path[-1]}.py'
            else:
                # Module is a top-level file
                module_name = module_path[0] if module_path else 'endpoints'
                file_path = self.output_dir / f'{module_name}.py'

            # Emit the module
            endpoint_names = self._emit_module_file(
                file_path=file_path,
                endpoints=node.endpoints,
                base_url=base_url,
                description=node.description,
                module_path=module_path,
            )

            self._emitted_modules.append(
                EmittedModule(
                    path=file_path,
                    module_path=module_path,
                    endpoint_names=endpoint_names,
                )
            )

        # Generate __init__.py files for all directories
        self._emit_nested_inits(tree, directories)

    def _emit_module_file(
        self,
        file_path: Path | UPath,
        endpoints: list[Endpoint],
        base_url: str,
        description: str | None = None,
        module_path: list[str] | None = None,
    ) -> list[str]:
        """Emit a single endpoint module file.

        Generates thin wrapper functions that delegate to the Client class methods.

        Args:
            file_path: Where to write the file.
            endpoints: List of endpoints to include.
            base_url: The base URL for API requests (unused, kept for API compat).
            description: Optional module docstring.
            module_path: The module path (for calculating relative imports).

        Returns:
            List of endpoint function names in this module.
        """
        from otterapi.codegen.endpoints import (
            build_default_client_code,
            build_delegating_endpoint_fn,
        )

        body: list[ast.stmt] = []

        # Add docstring if provided
        if description:
            body.append(ast.Expr(value=ast.Constant(value=description)))

        import_collector = ImportCollector()

        # Add default client variable and _get_client() function
        client_stmts, client_imports = build_default_client_code()
        body.extend(client_stmts)
        import_collector.add_imports(client_imports)

        # Add delegating endpoint functions
        endpoint_names: list[str] = []
        for endpoint in endpoints:
            # Build sync delegating function
            sync_fn, sync_imports = build_delegating_endpoint_fn(
                fn_name=endpoint.sync_fn_name,
                client_method_name=endpoint.sync_fn_name,
                parameters=endpoint.parameters,
                request_body_info=endpoint.request_body,
                response_type=endpoint.response_type,
                docs=endpoint.description,
                is_async=False,
            )
            endpoint_names.append(endpoint.sync_fn_name)
            body.append(sync_fn)
            import_collector.add_imports(sync_imports)

            # Build async delegating function
            async_fn, async_imports = build_delegating_endpoint_fn(
                fn_name=endpoint.async_fn_name,
                client_method_name=endpoint.async_fn_name,
                parameters=endpoint.parameters,
                request_body_info=endpoint.request_body,
                response_type=endpoint.response_type,
                docs=endpoint.description,
                is_async=True,
            )
            endpoint_names.append(endpoint.async_fn_name)
            body.append(async_fn)
            import_collector.add_imports(async_imports)

        # Add __all__ export
        body.insert(0, _all(sorted(endpoint_names)))

        # Add model imports
        model_names = self._collect_used_model_names(endpoints)
        if model_names:
            model_import = self._create_model_import(model_names, module_path)
            body.insert(0, model_import)

        # Add Client import (relative import based on module depth)
        client_import = self._create_client_import(module_path)
        body.insert(0, client_import)

        # Add all other imports at the beginning
        for import_stmt in import_collector.to_ast():
            body.insert(0, import_stmt)

        # Write the file
        file_path = UPath(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        write_mod(body, file_path)

        return endpoint_names

    def _create_client_import(
        self,
        module_path: list[str] | None = None,
    ) -> ast.ImportFrom:
        """Create an import statement for the Client class.

        Args:
            module_path: The current module path (for relative import calculation).

        Returns:
            AST ImportFrom statement.
        """
        # Calculate relative import level
        if self.config.flat_structure or not module_path:
            # Flat structure or top-level: same directory as client.py
            level = 1
        else:
            # Nested: go up by the depth of the module path
            level = len(module_path)

        return ast.ImportFrom(
            module='client',
            names=[ast.alias(name='Client', asname=None)],
            level=level,
        )

    def _collect_used_model_names(self, endpoints: list[Endpoint]) -> set[str]:
        """Collect model names used in endpoint signatures.

        Args:
            endpoints: List of endpoints to check.

        Returns:
            Set of model names used.
        """
        # Get all model names that have implementations
        available_models = {
            type_.name
            for type_ in self._typegen_types.values()
            if type_.name and type_.implementation_ast
        }

        used_models: set[str] = set()

        class NameCollector(ast.NodeVisitor):
            def visit_Name(self, node: ast.Name) -> None:
                if node.id in available_models:
                    used_models.add(node.id)
                self.generic_visit(node)

        collector = NameCollector()
        for endpoint in endpoints:
            collector.visit(endpoint.sync_ast)
            collector.visit(endpoint.async_ast)

        return used_models

    def _create_model_import(
        self,
        model_names: set[str],
        module_path: list[str] | None = None,
    ) -> ast.ImportFrom:
        """Create an import statement for models.

        Calculates the correct relative import level based on module depth.

        Args:
            model_names: Set of model names to import.
            module_path: The current module path (for relative import calculation).

        Returns:
            AST ImportFrom statement.
        """
        if self.models_import_path:
            # Use absolute import path
            return ast.ImportFrom(
                module=self.models_import_path,
                names=[
                    ast.alias(name=name, asname=None) for name in sorted(model_names)
                ],
                level=0,
            )

        # Calculate relative import level
        # If we're in api/v1/users.py, we need to go up 2 levels to reach models.py
        if self.config.flat_structure or not module_path:
            # Flat structure or top-level: same directory as models
            level = 1
        else:
            # Nested: go up by the depth of the module path minus 1
            # e.g., api/v1/users.py -> level 2 (up to api/, then to root)
            level = len(module_path)

        return ast.ImportFrom(
            module=self.models_file.stem,
            names=[ast.alias(name=name, asname=None) for name in sorted(model_names)],
            level=level,
        )

    def _emit_flat_init(self, module_exports: dict[str, list[str]]) -> None:
        """Emit main __init__.py for flat structure.

        Imports and re-exports all endpoints from all modules.

        Args:
            module_exports: Dict mapping module names to their exported function names.
        """
        body: list[ast.stmt] = []
        all_names: list[str] = []

        for module_name, endpoint_names in sorted(module_exports.items()):
            # Import from each module
            body.append(
                ast.ImportFrom(
                    module=module_name,
                    names=[
                        ast.alias(name=name, asname=None)
                        for name in sorted(endpoint_names)
                    ],
                    level=1,  # relative import
                )
            )
            all_names.extend(endpoint_names)

        # Add __all__
        if all_names:
            body.insert(0, _all(sorted(all_names)))

        # Write __init__.py
        init_path = self.output_dir / '__init__.py'
        if body:
            write_mod(body, init_path)
        else:
            init_path.touch()

    def _emit_nested_inits(
        self,
        tree: ModuleTree,
        directories: set[tuple[str, ...]],
    ) -> None:
        """Emit __init__.py files for nested directory structure.

        Args:
            tree: The ModuleTree for gathering export information.
            directories: Set of directory paths that need __init__.py files.
        """
        # First, emit __init__.py for each subdirectory
        for dir_parts in sorted(directories):
            dir_path = self.output_dir / '/'.join(dir_parts)
            self._emit_directory_init(dir_path, dir_parts, tree)

        # Emit main __init__.py that exports from all top-level modules
        self._emit_root_init(tree)

    def _emit_directory_init(
        self,
        dir_path: Path | UPath,
        dir_parts: tuple[str, ...],
        tree: ModuleTree,
    ) -> None:
        """Emit __init__.py for a subdirectory.

        Re-exports all endpoints from child modules.

        Args:
            dir_path: Path to the directory.
            dir_parts: The directory path components.
            tree: The ModuleTree for gathering information.
        """
        body: list[ast.stmt] = []
        all_names: list[str] = []

        # Find all child modules
        for emitted in self._emitted_modules:
            # Check if this module is a direct child of this directory
            if (
                len(emitted.module_path) == len(dir_parts) + 1
                and tuple(emitted.module_path[:-1]) == dir_parts
            ):
                module_name = emitted.module_path[-1]
                # Import from child module
                body.append(
                    ast.ImportFrom(
                        module=module_name,
                        names=[
                            ast.alias(name=name, asname=None)
                            for name in sorted(emitted.endpoint_names)
                        ],
                        level=1,
                    )
                )
                all_names.extend(emitted.endpoint_names)

        # Add __all__
        if all_names:
            body.insert(0, _all(sorted(all_names)))

        # Write __init__.py
        init_path = UPath(dir_path) / '__init__.py'
        if body:
            write_mod(body, init_path)
        else:
            init_path.touch()

    def _emit_root_init(self, tree: ModuleTree) -> None:
        """Emit the root __init__.py file.

        Re-exports from all top-level modules, subdirectories, client, and models.

        Args:
            tree: The ModuleTree for gathering information.
        """
        body: list[ast.stmt] = []
        all_names: list[str] = []
        imported_modules: set[str] = set()

        for emitted in self._emitted_modules:
            if len(emitted.module_path) == 1:
                # Top-level module file
                module_name = emitted.module_path[0]
                if module_name not in imported_modules:
                    body.append(
                        ast.ImportFrom(
                            module=module_name,
                            names=[
                                ast.alias(name=name, asname=None)
                                for name in sorted(emitted.endpoint_names)
                            ],
                            level=1,
                        )
                    )
                    all_names.extend(emitted.endpoint_names)
                    imported_modules.add(module_name)
            elif len(emitted.module_path) > 1:
                # Nested module - import from the top-level package
                top_module = emitted.module_path[0]
                if top_module not in imported_modules:
                    # Import the package as a module
                    body.append(
                        ast.ImportFrom(
                            module=f'.{top_module}',
                            names=[ast.alias(name='*', asname=None)],
                            level=0,
                        )
                    )
                    imported_modules.add(top_module)

        # Also collect names from nested modules for __all__
        for emitted in self._emitted_modules:
            if len(emitted.module_path) > 1:
                all_names.extend(emitted.endpoint_names)

        # Import and export Client from client.py
        body.append(
            ast.ImportFrom(
                module='client',
                names=[ast.alias(name='Client', asname=None)],
                level=1,
            )
        )
        all_names.append('Client')

        # Import and export BaseClient from _client.py
        base_client_name = f'Base{self.client_class_name}'
        body.append(
            ast.ImportFrom(
                module='_client',
                names=[ast.alias(name=base_client_name, asname=None)],
                level=1,
            )
        )
        all_names.append(base_client_name)

        # Import and export all models from models.py
        model_names = self._get_model_names()
        if model_names:
            body.append(
                ast.ImportFrom(
                    module='models',
                    names=[
                        ast.alias(name=name, asname=None)
                        for name in sorted(model_names)
                    ],
                    level=1,
                )
            )
            all_names.extend(model_names)

        # Add __all__ with unique names
        unique_names = sorted(set(all_names))
        if unique_names:
            body.insert(0, _all(unique_names))

        # Write __init__.py
        init_path = self.output_dir / '__init__.py'
        if body:
            write_mod(body, init_path)

    def _get_model_names(self) -> list[str]:
        """Get model names from the typegen_types.

        Returns:
            List of model names that have implementations.
        """
        if not self._typegen_types:
            return []

        return [
            type_.name
            for type_ in self._typegen_types.values()
            if type_.name and type_.implementation_ast
        ]

    def get_emitted_modules(self) -> list[EmittedModule]:
        """Get list of all emitted modules.

        Returns:
            List of EmittedModule objects.
        """
        return self._emitted_modules.copy()
