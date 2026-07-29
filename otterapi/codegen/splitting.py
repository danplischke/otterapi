"""Module splitting utilities for organizing endpoints into a hierarchy.

This module provides utilities for:
- Building a hierarchical structure of modules from endpoints (ModuleTree)
- Resolving endpoint paths to target modules (ModuleMapResolver)
- Emitting organized endpoint modules to the filesystem (SplitModuleEmitter)
"""

from __future__ import annotations

import ast
import fnmatch
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, cast

from upath import UPath

from otterapi.codegen.ast_utils import _all
from otterapi.codegen.emit import (
    EmitConfig,
    TypeResolver,
    build_endpoints_module_body,
)
from otterapi.codegen.utils import write_mod

if TYPE_CHECKING:
    from otterapi.codegen.types import Endpoint, Type
    from otterapi.config import (
        DataFrameConfig,
        ExportConfig,
        ModuleDefinition,
        ModuleSplitConfig,
        PaginationConfig,
        ResponseUnwrapConfig,
    )

__all__ = [
    # Tree classes
    'ModuleTree',
    'ModuleTreeBuilder',
    'build_module_tree',
    # Resolver classes
    'ResolvedModule',
    'ModuleMapResolver',
    # Emitter classes
    'EmittedModule',
    'SplitModuleEmitter',
]

# =============================================================================
# Module Tree
# =============================================================================


@dataclass
class ModuleTree:
    """Tree structure representing the module hierarchy.

    Each node in the tree can contain endpoints and/or child modules.
    The root node typically has an empty name and contains top-level modules.

    Attributes:
        name: The name of this module node.
        endpoints: List of endpoints that belong directly to this module.
        children: Child modules keyed by their name.
        definition: The ModuleDefinition associated with this node (if any).
        description: Optional description for this module.
    """

    name: str
    endpoints: list[Endpoint] = field(default_factory=list)
    children: dict[str, ModuleTree] = field(default_factory=dict)
    definition: ModuleDefinition | None = None
    description: str | None = None

    def add_endpoint(self, module_path: list[str], endpoint: Endpoint) -> None:
        """Add an endpoint to the tree at the specified module path.

        Creates intermediate nodes as needed.

        Args:
            module_path: List of module path components, e.g., ["api", "v1", "users"].
            endpoint: The endpoint to add.
        """
        if not module_path:
            self.endpoints.append(endpoint)
            return

        current = self
        for part in module_path:
            if part not in current.children:
                current.children[part] = ModuleTree(name=part)
            current = current.children[part]

        current.endpoints.append(endpoint)

    def get_node(self, module_path: list[str]) -> ModuleTree | None:
        """Get a node at the specified path.

        Args:
            module_path: List of module path components.

        Returns:
            The ModuleTree node at the path, or None if not found.
        """
        if not module_path:
            return self

        current = self
        for part in module_path:
            if part not in current.children:
                return None
            current = current.children[part]

        return current

    def walk(self) -> Iterator[tuple[list[str], ModuleTree]]:
        """Iterate over all nodes in the tree depth-first.

        Yields:
            Tuples of (module_path, node) for each node in the tree.
        """
        yield from self._walk_recursive([])

    def _walk_recursive(
        self, current_path: list[str]
    ) -> Iterator[tuple[list[str], ModuleTree]]:
        """Recursively walk the tree."""
        yield current_path, self

        for child_name, child_node in sorted(self.children.items()):
            child_path = current_path + [child_name]
            yield from child_node._walk_recursive(child_path)

    def walk_leaves(self) -> Iterator[tuple[list[str], ModuleTree]]:
        """Iterate over leaf nodes (nodes with endpoints).

        Yields:
            Tuples of (module_path, node) for leaf nodes with endpoints.
        """
        for path, node in self.walk():
            if node.endpoints:
                yield path, node

    def count_endpoints(self) -> int:
        """Count the total number of endpoints in this subtree."""
        total = len(self.endpoints)
        for child in self.children.values():
            total += child.count_endpoints()
        return total

    def is_empty(self) -> bool:
        """Check if this subtree has no endpoints."""
        return self.count_endpoints() == 0

    def flatten(self) -> dict[str, list[Endpoint]]:
        """Flatten the tree into a dictionary mapping module paths to endpoints."""
        result: dict[str, list[Endpoint]] = {}

        for path, node in self.walk():
            if node.endpoints:
                module_name = '.'.join(path) if path else '__root__'
                result[module_name] = node.endpoints

        return result


class ModuleTreeBuilder:
    """Builds a ModuleTree from a list of endpoints and configuration.

    Example:
        >>> from otterapi.config import ModuleSplitConfig
        >>> config = ModuleSplitConfig(enabled=True, strategy="tag")
        >>> builder = ModuleTreeBuilder(config)
        >>> tree = builder.build(endpoints)
    """

    def __init__(self, config: ModuleSplitConfig):
        """Initialize the tree builder.

        Args:
            config: The module split configuration.
        """
        self.config = config
        self.resolver = ModuleMapResolver(config)

    def build(self, endpoints: list[Endpoint]) -> ModuleTree:
        """Build a module tree from a list of endpoints.

        Args:
            endpoints: List of Endpoint objects to organize.

        Returns:
            A ModuleTree with endpoints organized according to the configuration.
        """
        root = ModuleTree(name='__root__')

        for endpoint in endpoints:
            tags = getattr(endpoint, 'tags', None)

            resolved = self.resolver.resolve(
                path=endpoint.path,
                method=endpoint.method,
                tags=tags,
            )

            root.add_endpoint(resolved.module_path, endpoint)

            if resolved.definition:
                node = root.get_node(resolved.module_path)
                if node and not node.definition:
                    node.definition = resolved.definition
                    if resolved.definition.description:
                        node.description = resolved.definition.description

        if self.config.min_endpoints > 1:
            self._consolidate_small_modules(root)

        return root

    def _consolidate_small_modules(self, root: ModuleTree) -> None:
        """Consolidate modules with fewer than min_endpoints into fallback."""
        to_consolidate: list[tuple[list[str], ModuleTree]] = []

        for path, node in root.walk():
            if not path or path == [self.config.fallback_module]:
                continue

            if node.endpoints and not node.children:
                if len(node.endpoints) < self.config.min_endpoints:
                    to_consolidate.append((path, node))

        for path, node in to_consolidate:
            for endpoint in node.endpoints:
                root.add_endpoint([self.config.fallback_module], endpoint)
            node.endpoints = []

        self._remove_empty_nodes(root)

    def _remove_empty_nodes(self, node: ModuleTree) -> bool:
        """Remove empty nodes from the tree."""
        empty_children = []
        for child_name, child_node in node.children.items():
            if self._remove_empty_nodes(child_node):
                empty_children.append(child_name)

        for child_name in empty_children:
            del node.children[child_name]

        return not node.endpoints and not node.children


def build_module_tree(
    endpoints: list[Endpoint],
    config: ModuleSplitConfig,
) -> ModuleTree:
    """Convenience function to build a module tree.

    Args:
        endpoints: List of Endpoint objects to organize.
        config: The module split configuration.

    Returns:
        A ModuleTree with endpoints organized according to the configuration.
    """
    builder = ModuleTreeBuilder(config)
    return builder.build(endpoints)


# =============================================================================
# Module Map Resolver
# =============================================================================


@dataclass
class ResolvedModule:
    """Result of resolving an endpoint to a module.

    Attributes:
        module_path: List of module path components, e.g., ["api", "v1", "users"].
        definition: The ModuleDefinition that matched (if any).
        resolution: How the module was resolved: "custom", "tag", "path", "fallback".
        stripped_path: The endpoint path after applying strip_prefix transformations.
    """

    module_path: list[str]
    definition: ModuleDefinition | None = None
    resolution: str = 'fallback'
    stripped_path: str = ''

    @property
    def module_name(self) -> str:
        """Get the dotted module name, e.g., 'api.v1.users'."""
        return '.'.join(self.module_path)

    @property
    def file_path(self) -> str:
        """Get the relative file path for this module."""
        if len(self.module_path) == 1:
            return f'{self.module_path[0]}.py'
        return '/'.join(self.module_path[:-1]) + f'/{self.module_path[-1]}.py'

    @property
    def flat_file_path(self) -> str:
        """Get the flat file path for this module, e.g., 'api_v1_users.py'."""
        return '_'.join(self.module_path) + '.py'


class ModuleMapResolver:
    """Resolves endpoint paths to target modules based on configuration.

    The resolver uses the following priority:
    1. Custom module_map patterns (if strategy is 'custom' or 'hybrid')
    2. OpenAPI tags (if strategy is 'tag' or 'hybrid')
    3. Path segments (if strategy is 'path' or 'hybrid')
    4. Fallback module

    Example:
        >>> from otterapi.config import ModuleSplitConfig
        >>> config = ModuleSplitConfig(
        ...     enabled=True,
        ...     strategy="custom",
        ...     module_map={"users": ["/users/*", "/user/*"]}
        ... )
        >>> resolver = ModuleMapResolver(config)
        >>> result = resolver.resolve("/users/123", "GET", tags=None)
        >>> result.module_name
        'users'
    """

    def __init__(self, config: ModuleSplitConfig):
        """Initialize the resolver with configuration."""
        self.config = config

    def resolve(
        self,
        path: str,
        method: str,
        tags: list[str] | None = None,
    ) -> ResolvedModule:
        """Resolve an endpoint path to a target module.

        Args:
            path: The API endpoint path, e.g., "/users/{id}".
            method: The HTTP method (GET, POST, etc.).
            tags: Optional list of OpenAPI tags for this operation.

        Returns:
            A ResolvedModule indicating where the endpoint should be placed.
        """
        stripped_path = self._strip_global_prefixes(path)

        if self.config.module_map and self._should_use_module_map():
            result = self._match_module_map(
                stripped_path,
                cast('dict[str, ModuleDefinition]', self.config.module_map),
                [],
            )
            if result:
                result.stripped_path = stripped_path
                return result

        if self._should_use_tags() and tags:
            module_name = self._sanitize(tags[0])
            return ResolvedModule(
                module_path=[module_name],
                resolution='tag',
                stripped_path=stripped_path,
            )

        if self._should_use_path():
            path_module = self._extract_from_path(stripped_path)
            if path_module:
                return ResolvedModule(
                    module_path=[path_module],
                    resolution='path',
                    stripped_path=stripped_path,
                )

        return ResolvedModule(
            module_path=[self.config.fallback_module],
            resolution='fallback',
            stripped_path=stripped_path,
        )

    def _should_use_module_map(self) -> bool:
        """Check if module_map should be used based on strategy."""
        from otterapi.config import SplitStrategy

        strategy = self.config.strategy
        if isinstance(strategy, str):
            strategy = SplitStrategy(strategy)
        return strategy in (SplitStrategy.CUSTOM, SplitStrategy.HYBRID)

    def _should_use_tags(self) -> bool:
        """Check if tags should be used based on strategy."""
        from otterapi.config import SplitStrategy

        strategy = self.config.strategy
        if isinstance(strategy, str):
            strategy = SplitStrategy(strategy)
        return strategy in (SplitStrategy.TAG, SplitStrategy.HYBRID)

    def _should_use_path(self) -> bool:
        """Check if path extraction should be used based on strategy."""
        from otterapi.config import SplitStrategy

        strategy = self.config.strategy
        if isinstance(strategy, str):
            strategy = SplitStrategy(strategy)
        return strategy in (SplitStrategy.PATH, SplitStrategy.HYBRID)

    def _strip_global_prefixes(self, path: str) -> str:
        """Strip configured global prefixes from the path."""
        for prefix in self.config.global_strip_prefixes:
            if path.startswith(prefix):
                stripped = path[len(prefix) :]
                if not stripped.startswith('/'):
                    stripped = '/' + stripped
                return stripped
        return path

    @staticmethod
    def _strip_prefix(path: str, prefix: str) -> str:
        """Strip ``prefix`` from ``path`` if present, preserving a leading slash."""
        if not path.startswith(prefix):
            return path
        stripped = path[len(prefix) :]
        if not stripped.startswith('/'):
            stripped = '/' + stripped
        return stripped

    def _match_module_definition(
        self,
        working_path: str,
        match_path: str,
        current_path: list[str],
        definition: ModuleDefinition,
    ) -> ResolvedModule | None:
        """Try to resolve ``working_path`` against a single module definition."""
        if definition.paths:
            for pattern in definition.paths:
                if not self._path_matches(working_path, pattern):
                    continue
                if definition.modules:
                    nested_result = self._match_module_map(
                        match_path, definition.modules, current_path, definition
                    )
                    if nested_result:
                        return nested_result
                return ResolvedModule(
                    module_path=current_path,
                    definition=definition,
                    resolution='custom',
                )
            return None

        if definition.modules:
            return self._match_module_map(
                match_path, definition.modules, current_path, definition
            )

        return None

    def _match_module_map(
        self,
        path: str,
        module_map: dict[str, ModuleDefinition],
        parent_path: list[str],
        parent_definition: ModuleDefinition | None = None,
    ) -> ResolvedModule | None:
        """Recursively match a path against the module_map."""
        working_path = path
        if parent_definition and parent_definition.strip_prefix:
            working_path = self._strip_prefix(
                working_path, parent_definition.strip_prefix
            )

        for module_name, definition in module_map.items():
            current_path = parent_path + [module_name]

            match_path = working_path
            if definition.strip_prefix:
                match_path = self._strip_prefix(match_path, definition.strip_prefix)

            result = self._match_module_definition(
                working_path, match_path, current_path, definition
            )
            if result:
                return result

        return None

    def _path_matches(self, path: str, pattern: str) -> bool:
        """Check if a path matches a glob pattern."""
        path = path.rstrip('/')
        pattern = pattern.rstrip('/')

        if '**' in pattern:
            regex_pattern = self._glob_to_regex(pattern)
            return bool(re.match(regex_pattern, path))

        return fnmatch.fnmatch(path, pattern)

    def _glob_to_regex(self, pattern: str) -> str:
        """Convert a glob pattern to a regex pattern."""
        result = ''
        i = 0
        while i < len(pattern):
            if pattern[i : i + 2] == '**':
                result += '.*'
                i += 2
            elif pattern[i] == '*':
                result += '[^/]*'
                i += 1
            elif pattern[i] == '?':
                result += '[^/]'
                i += 1
            else:
                if pattern[i] in r'\.^$+{}[]|()':
                    result += '\\' + pattern[i]
                else:
                    result += pattern[i]
                i += 1

        return f'^{result}$'

    def _extract_from_path(self, path: str) -> str | None:
        """Extract a module name from the path based on path_depth."""
        segments = [s for s in path.split('/') if s and not s.startswith('{')]

        if not segments:
            return None

        depth = min(self.config.path_depth, len(segments))
        if depth == 1:
            return self._sanitize(segments[0])
        else:
            return '_'.join(self._sanitize(s) for s in segments[:depth])

    def _sanitize(self, name: str) -> str:
        """Sanitize a name to be a valid Python identifier."""
        import keyword

        sanitized = re.sub(r'[-\s]+', '_', name.lower())
        sanitized = re.sub(r'[^a-z0-9_]', '', sanitized)

        if sanitized and sanitized[0].isdigit():
            sanitized = '_' + sanitized

        if keyword.iskeyword(sanitized):
            sanitized = sanitized + '_'

        return sanitized or 'module'


# =============================================================================
# Split Module Emitter
# =============================================================================


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
        >>> emitter.emit(tree)
    """

    def __init__(
        self,
        config: ModuleSplitConfig,
        output_dir: Path | UPath,
        models_file: Path | UPath,
        models_import_path: str | None = None,
        client_class_name: str = 'APIClient',
        dataframe_config: DataFrameConfig | None = None,
        pagination_config: PaginationConfig | None = None,
        response_unwrap_config: ResponseUnwrapConfig | None = None,
        export_config: ExportConfig | None = None,
        generate_sync: bool = True,
        generate_async: bool = True,
        client_style: str = 'functions',
        result_objects: bool = False,
        reexport_models: bool = False,
        reexport_model_exclude_patterns: list[str] | None = None,
        format_output: bool = True,
        validate_output: bool = True,
    ):
        """Initialize the split module emitter.

        Args:
            config: The module split configuration.
            output_dir: The root output directory for generated files.
            models_file: Path to the models file for import generation.
            models_import_path: Optional custom import path for models.
            client_class_name: Name of the client class.
            dataframe_config: Optional DataFrame configuration.
            pagination_config: Optional pagination configuration.
            response_unwrap_config: Optional response unwrap configuration.
            export_config: Optional export configuration.
            generate_sync: Whether to emit synchronous endpoint functions.
            generate_async: Whether to emit asynchronous endpoint functions.
            client_style: Public-API shape. With ``client``/``resource`` the root
                ``__init__`` exports ``Client`` / ``AsyncClient`` (from
                ``_clients.py``, written by the caller) instead of re-exporting the
                per-module free functions.
            reexport_models: Whether to include model names in __all__.
            reexport_model_exclude_patterns: Glob patterns of model names to exclude.
            format_output: Whether to format generated code with ruff/black.
            validate_output: Whether to validate generated code syntax.
        """
        self.config = config
        self.output_dir = UPath(output_dir)
        self.models_file = UPath(models_file)
        self.models_import_path = models_import_path
        self.client_class_name = client_class_name
        self.dataframe_config = dataframe_config
        self.pagination_config = pagination_config
        self.response_unwrap_config = response_unwrap_config
        self.export_config = export_config
        self.generate_sync = generate_sync
        self.generate_async = generate_async
        self.client_style = client_style
        self.result_objects = result_objects
        self.reexport_models = reexport_models
        self.reexport_model_exclude_patterns: list[str] = (
            reexport_model_exclude_patterns or []
        )
        self.format_output = format_output
        self.validate_output = validate_output
        self._emitted_modules: list[EmittedModule] = []
        self._typegen_types: dict[str, Type] = {}

    def emit(
        self,
        tree: ModuleTree,
        typegen_types: dict[str, Type] | None = None,
    ) -> list[EmittedModule]:
        """Emit all modules from the tree.

        Args:
            tree: The ModuleTree containing organized endpoints.
            typegen_types: Optional dict of types for collecting model imports.

        Returns:
            List of EmittedModule objects describing what was written.
        """
        self._emitted_modules = []
        self._typegen_types = typegen_types or {}

        if self.config.flat_structure:
            self._emit_flat(tree)
        else:
            self._emit_nested(tree)

        return self._emitted_modules

    def _emit_flat(self, tree: ModuleTree) -> None:
        """Emit modules as flat files (no subdirectories)."""
        all_exports: dict[str, list[str]] = {}

        for module_path, node in tree.walk_leaves():
            if not node.endpoints:
                continue

            flat_name = '_'.join(module_path) if module_path else 'endpoints'
            file_path = self.output_dir / f'{flat_name}.py'

            endpoint_names = self._emit_module_file(
                file_path=file_path,
                endpoints=node.endpoints,
                description=node.description,
            )

            all_exports[flat_name] = endpoint_names

            self._emitted_modules.append(
                EmittedModule(
                    path=file_path,
                    module_path=module_path,
                    endpoint_names=endpoint_names,
                )
            )

        self._emit_flat_init(all_exports)

    def _emit_nested(self, tree: ModuleTree) -> None:
        """Emit modules as nested directories."""
        directories: set[tuple[str, ...]] = set()

        for module_path, node in tree.walk_leaves():
            if not node.endpoints:
                continue

            if len(module_path) > 1:
                dir_path = self.output_dir / '/'.join(module_path[:-1])
                dir_path.mkdir(parents=True, exist_ok=True)
                directories.add(tuple(module_path[:-1]))

                for i in range(1, len(module_path) - 1):
                    directories.add(tuple(module_path[:i]))

                file_path = dir_path / f'{module_path[-1]}.py'
            else:
                module_name = module_path[0] if module_path else 'endpoints'
                file_path = self.output_dir / f'{module_name}.py'

            endpoint_names = self._emit_module_file(
                file_path=file_path,
                endpoints=node.endpoints,
                description=node.description,
            )

            self._emitted_modules.append(
                EmittedModule(
                    path=file_path,
                    module_path=module_path,
                    endpoint_names=endpoint_names,
                )
            )

        self._emit_nested_inits(tree, directories)

    def _emit_module_file(
        self,
        file_path: Path | UPath,
        endpoints: list[Endpoint],
        description: str | None = None,
    ) -> list[str]:
        """Emit a single endpoint module file.

        Per-endpoint emission is delegated to the shared, layout-agnostic core in
        :mod:`otterapi.codegen.emit`, which the non-split writer uses too.
        """
        emit_config = EmitConfig(
            pagination=self.pagination_config,
            dataframe=self.dataframe_config,
            response_unwrap=self.response_unwrap_config,
            export=self.export_config,
            generate_sync=self.generate_sync,
            generate_async=self.generate_async,
        )
        body, endpoint_names = build_endpoints_module_body(
            endpoints,
            emit_config,
            TypeResolver(self._typegen_types),
            reexport_models=self.reexport_models,
            reexport_model_exclude_patterns=self.reexport_model_exclude_patterns,
            description=description,
        )

        file_path = UPath(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        write_mod(
            body,
            file_path,
            format_code=self.format_output,
            validate_code=self.validate_output,
        )

        return endpoint_names

    def _emit_flat_init(self, all_exports: dict[str, list[str]]) -> None:
        """Emit __init__.py for flat structure."""
        body: list[ast.stmt] = []
        all_names: list[str] = []

        for module_name, endpoint_names in sorted(all_exports.items()):
            body.append(
                ast.ImportFrom(
                    module=module_name,
                    names=[
                        ast.alias(name=name, asname=None)
                        for name in sorted(endpoint_names)
                    ],
                    level=1,
                )
            )
            all_names.extend(endpoint_names)

        if all_names:
            body.insert(0, _all(sorted(all_names)))

        self._emit_root_init_base(body)

    def _emit_nested_inits(
        self, tree: ModuleTree, directories: set[tuple[str, ...]]
    ) -> None:
        """Emit __init__.py files for nested structure."""
        for dir_tuple in sorted(directories, key=len, reverse=True):
            dir_path = self.output_dir / '/'.join(dir_tuple)
            self._emit_directory_init(dir_path, list(dir_tuple))

        self._emit_root_init()

    def _emit_directory_init(
        self, dir_path: Path | UPath, module_path: list[str]
    ) -> None:
        """Emit __init__.py for a directory."""
        body: list[ast.stmt] = []
        all_names: list[str] = []

        for emitted in self._emitted_modules:
            if (
                len(emitted.module_path) > len(module_path)
                and emitted.module_path[: len(module_path)] == module_path
            ):
                remaining = emitted.module_path[len(module_path) :]
                if len(remaining) == 1:
                    body.append(
                        ast.ImportFrom(
                            module=remaining[0],
                            names=[
                                ast.alias(name=name, asname=None)
                                for name in sorted(emitted.endpoint_names)
                            ],
                            level=1,
                        )
                    )
                    all_names.extend(emitted.endpoint_names)

        if all_names:
            body.insert(0, _all(sorted(all_names)))

        init_path = UPath(dir_path) / '__init__.py'
        if body:
            write_mod(
                body,
                init_path,
                format_code=self.format_output,
                validate_code=self.validate_output,
            )
        else:
            init_path.touch()

    def _emit_root_init(self) -> None:
        """Emit the root __init__.py file."""
        body: list[ast.stmt] = []
        all_names: list[str] = []
        imported_modules: set[str] = set()

        # In client/resource style the public surface is Client / AsyncClient
        # (emitted into _clients.py by the caller); the per-module free functions
        # stay importable from their modules but are not re-exported here.
        if self.client_style in ('client', 'resource'):
            self._emit_root_init_base(body, all_names)
            return

        for emitted in self._emitted_modules:
            if len(emitted.module_path) == 1:
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
                top_module = emitted.module_path[0]
                if top_module not in imported_modules:
                    body.append(
                        ast.ImportFrom(
                            module=f'.{top_module}',
                            names=[ast.alias(name='*', asname=None)],
                            level=0,
                        )
                    )
                    imported_modules.add(top_module)

        for emitted in self._emitted_modules:
            if len(emitted.module_path) > 1:
                all_names.extend(emitted.endpoint_names)

        self._emit_root_init_base(body, all_names)

    def _emit_root_init_base(
        self, body: list[ast.stmt], all_names: list[str] | None = None
    ) -> None:
        """Emit the base root __init__.py content."""
        if all_names is None:
            all_names = []

        # Import and export the public client(s). In client/resource style the
        # method-carrying Client / AsyncClient live in _clients.py; otherwise the
        # infrastructure Client from client.py is the export.
        if self.client_style in ('client', 'resource'):
            body.append(
                ast.ImportFrom(
                    module='_clients',
                    names=[
                        ast.alias(name='AsyncClient', asname=None),
                        ast.alias(name='Client', asname=None),
                    ],
                    level=1,
                )
            )
            all_names.extend(['AsyncClient', 'Client'])
            if self.result_objects:
                body.append(
                    ast.ImportFrom(
                        module='_query',
                        names=[
                            ast.alias(name='AsyncQuery', asname=None),
                            ast.alias(name='Query', asname=None),
                        ],
                        level=1,
                    )
                )
                all_names.extend(['AsyncQuery', 'Query'])
        else:
            body.append(
                ast.ImportFrom(
                    module='client',
                    names=[ast.alias(name='Client', asname=None)],
                    level=1,
                )
            )
            all_names.append('Client')

        # Import and export BaseClient
        base_client_name = f'Base{self.client_class_name}'
        body.append(
            ast.ImportFrom(
                module='_client',
                names=[ast.alias(name=base_client_name, asname=None)],
                level=1,
            )
        )
        all_names.append(base_client_name)

        # Import and export all models
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

        # Add __all__
        unique_names = sorted(set(all_names))
        if unique_names:
            body.insert(0, _all(unique_names))

        # Write __init__.py
        init_path = self.output_dir / '__init__.py'
        if body:
            write_mod(
                body,
                init_path,
                format_code=self.format_output,
                validate_code=self.validate_output,
            )

    def _get_model_names(self) -> list[str]:
        """Get model names from the typegen_types."""
        if not self._typegen_types:
            return []

        return [
            type_.name
            for type_ in self._typegen_types.values()
            if type_.name and type_.implementation_ast
        ]
