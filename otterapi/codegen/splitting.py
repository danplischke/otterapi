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
from typing import TYPE_CHECKING

from upath import UPath

from otterapi.codegen.ast_utils import MODELS_MODULE, ImportCollector, _all, _name
from otterapi.codegen.dataframes import get_dataframe_config_for_endpoint
from otterapi.codegen.endpoints import (
    build_default_client_code,
    build_standalone_dataframe_fn,
    build_standalone_endpoint_fn,
    build_standalone_paginated_dataframe_fn,
    build_standalone_paginated_fn,
    build_standalone_paginated_iter_fn,
)
from otterapi.codegen.pagination import (
    PaginationMethodConfig,
    get_pagination_config_for_endpoint,
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
            result = self._match_module_map(stripped_path, self.config.module_map, [])
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
            module_name = self._extract_from_path(stripped_path)
            if module_name:
                return ResolvedModule(
                    module_path=[module_name],
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
        reexport_models: bool = False,
        reexport_model_exclude_patterns: list[str] | None = None,
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
            reexport_models: Whether to include model names in __all__.
            reexport_model_exclude_patterns: Glob patterns of model names to exclude.
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
        self.reexport_models = reexport_models
        self.reexport_model_exclude_patterns: list[str] = (
            reexport_model_exclude_patterns or []
        )
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

    @staticmethod
    def _build_pagination_config_dict(pag_config: PaginationMethodConfig) -> dict:
        """Build the plain-dict pagination config passed to the AST builders."""
        return {
            'offset_param': pag_config.offset_param,
            'limit_param': pag_config.limit_param,
            'cursor_param': pag_config.cursor_param,
            'page_param': pag_config.page_param,
            'per_page_param': pag_config.per_page_param,
            'data_path': pag_config.data_path,
            'total_path': pag_config.total_path,
            'next_cursor_path': pag_config.next_cursor_path,
            'total_pages_path': pag_config.total_pages_path,
            'default_page_size': pag_config.default_page_size,
        }

    @staticmethod
    def _emit_paginated_fn_pair(
        builder_fn,
        sync_name: str,
        async_name: str,
        endpoint: Endpoint,
        pag_config: PaginationMethodConfig,
        pag_dict: dict,
        item_type_ast: ast.expr | None,
        item_type_imports: dict[str, set[str]],
        body: list[ast.stmt],
        import_collector: ImportCollector,
        endpoint_names: list[str],
    ) -> None:
        """Emit sync+async pairs from a ``build_standalone_paginated{,_iter}_fn`` builder."""
        for fn_name, is_async in ((sync_name, False), (async_name, True)):
            fn, fn_imports = builder_fn(
                fn_name=fn_name,
                method=endpoint.method,
                path=endpoint.path,
                parameters=endpoint.parameters,
                request_body_info=endpoint.request_body,
                response_type=endpoint.response_type,
                pagination_style=pag_config.style,
                pagination_config=pag_dict,
                item_type_ast=item_type_ast,
                item_type_imports=item_type_imports,
                docs=endpoint.description,
                is_async=is_async,
            )
            endpoint_names.append(fn_name)
            body.append(fn)
            import_collector.add_imports(fn_imports)

    @staticmethod
    def _emit_paginated_dataframe_fn_pair(
        library: str,
        suffix: str,
        endpoint: Endpoint,
        pag_config: PaginationMethodConfig,
        pag_dict: dict,
        item_type_ast: ast.expr | None,
        item_type_imports: dict[str, set[str]],
        body: list[ast.stmt],
        import_collector: ImportCollector,
        endpoint_names: list[str],
    ) -> None:
        """Emit sync+async ``build_standalone_paginated_dataframe_fn`` for one library."""
        for fn_name, is_async in (
            (f'{endpoint.sync_fn_name}{suffix}', False),
            (f'{endpoint.async_fn_name}{suffix}', True),
        ):
            fn, fn_imports = build_standalone_paginated_dataframe_fn(
                fn_name=fn_name,
                method=endpoint.method,
                path=endpoint.path,
                parameters=endpoint.parameters,
                request_body_info=endpoint.request_body,
                response_type=endpoint.response_type,
                pagination_style=pag_config.style,
                pagination_config=pag_dict,
                library=library,
                item_type_ast=item_type_ast,
                item_type_imports=item_type_imports,
                docs=endpoint.description,
                is_async=is_async,
            )
            endpoint_names.append(fn_name)
            body.append(fn)
            import_collector.add_imports(fn_imports)

    def _emit_paginated_dataframe_methods(
        self,
        endpoint: Endpoint,
        pag_config: PaginationMethodConfig,
        pag_dict: dict,
        item_type_ast: ast.expr | None,
        item_type_imports: dict[str, set[str]],
        body: list[ast.stmt],
        import_collector: ImportCollector,
        endpoint_names: list[str],
    ) -> bool:
        """Emit paginated DataFrame methods for a paginated endpoint, if configured.

        Returns whether any were generated (callers use this to skip the
        regular, non-paginated DataFrame methods for the same endpoint).
        """
        if not (self.dataframe_config and self.dataframe_config.enabled):
            return False

        endpoint_df_config = self.dataframe_config.endpoints.get(endpoint.sync_fn_name)
        endpoint_disabled = bool(
            endpoint_df_config and endpoint_df_config.enabled is False
        )

        generated = False
        if not endpoint_disabled and self.dataframe_config.pandas:
            generated = True
            self._emit_paginated_dataframe_fn_pair(
                'pandas',
                '_df',
                endpoint,
                pag_config,
                pag_dict,
                item_type_ast,
                item_type_imports,
                body,
                import_collector,
                endpoint_names,
            )

        if self.dataframe_config.polars:
            generated = True
            self._emit_paginated_dataframe_fn_pair(
                'polars',
                '_pl',
                endpoint,
                pag_config,
                pag_dict,
                item_type_ast,
                item_type_imports,
                body,
                import_collector,
                endpoint_names,
            )

        return generated

    def _emit_paginated_endpoint_methods(
        self,
        endpoint: Endpoint,
        pag_config: PaginationMethodConfig,
        body: list[ast.stmt],
        import_collector: ImportCollector,
        endpoint_names: list[str],
    ) -> bool:
        """Emit paginate/iterate functions (and DataFrame variants) for a paginated endpoint.

        Returns whether paginated DataFrame methods were generated.
        """
        item_type_ast, item_type_imports = self._get_item_type_ast(
            endpoint, pag_config.data_path
        )
        pag_dict = self._build_pagination_config_dict(pag_config)

        self._emit_paginated_fn_pair(
            build_standalone_paginated_fn,
            endpoint.sync_fn_name,
            endpoint.async_fn_name,
            endpoint,
            pag_config,
            pag_dict,
            item_type_ast,
            item_type_imports,
            body,
            import_collector,
            endpoint_names,
        )
        self._emit_paginated_fn_pair(
            build_standalone_paginated_iter_fn,
            f'{endpoint.sync_fn_name}_iter',
            f'{endpoint.async_fn_name}_iter',
            endpoint,
            pag_config,
            pag_dict,
            item_type_ast,
            item_type_imports,
            body,
            import_collector,
            endpoint_names,
        )

        return self._emit_paginated_dataframe_methods(
            endpoint,
            pag_config,
            pag_dict,
            item_type_ast,
            item_type_imports,
            body,
            import_collector,
            endpoint_names,
        )

    def _emit_standalone_endpoint_methods(
        self,
        endpoint: Endpoint,
        body: list[ast.stmt],
        import_collector: ImportCollector,
        endpoint_names: list[str],
    ) -> None:
        """Emit sync+async standalone endpoint functions for a non-paginated endpoint."""
        should_unwrap, unwrap_path = self._get_unwrap_config(endpoint)
        unwrap_type_ast = None
        unwrap_type_imports = None
        if should_unwrap and unwrap_path:
            unwrap_type_ast, unwrap_type_imports = self._get_unwrapped_type_ast(
                endpoint, unwrap_path
            )

        response_type_imports = None
        if endpoint.response_type and endpoint.response_type.annotation_ast:
            response_type_imports = self._collect_model_imports_from_ast(
                endpoint.response_type.annotation_ast
            )

        for fn_name, is_async in (
            (endpoint.sync_fn_name, False),
            (endpoint.async_fn_name, True),
        ):
            fn, fn_imports = build_standalone_endpoint_fn(
                fn_name=fn_name,
                method=endpoint.method,
                path=endpoint.path,
                parameters=endpoint.parameters,
                request_body_info=endpoint.request_body,
                response_type=endpoint.response_type,
                response_infos=endpoint.response_infos,
                docs=endpoint.description,
                is_async=is_async,
                unwrap_data_path=unwrap_path if should_unwrap else None,
                unwrap_type_ast=unwrap_type_ast,
                unwrap_type_imports=unwrap_type_imports,
                response_type_imports=response_type_imports,
            )
            endpoint_names.append(fn_name)
            body.append(fn)
            import_collector.add_imports(fn_imports)

    @staticmethod
    def _emit_dataframe_fn_pair(
        library: str,
        suffix: str,
        endpoint: Endpoint,
        default_path: str | None,
        body: list[ast.stmt],
        import_collector: ImportCollector,
        endpoint_names: list[str],
    ) -> None:
        """Emit sync+async ``build_standalone_dataframe_fn`` for one library."""
        for fn_name, is_async in (
            (f'{endpoint.sync_fn_name}{suffix}', False),
            (f'{endpoint.async_fn_name}{suffix}', True),
        ):
            fn, fn_imports = build_standalone_dataframe_fn(
                fn_name=fn_name,
                method=endpoint.method,
                path=endpoint.path,
                parameters=endpoint.parameters,
                request_body_info=endpoint.request_body,
                library=library,
                default_path=default_path,
                docs=endpoint.description,
                is_async=is_async,
            )
            endpoint_names.append(fn_name)
            body.append(fn)
            import_collector.add_imports(fn_imports)

    def _emit_dataframe_methods(
        self,
        endpoint: Endpoint,
        body: list[ast.stmt],
        import_collector: ImportCollector,
        endpoint_names: list[str],
    ) -> bool:
        """Emit regular (non-paginated) DataFrame methods for an endpoint, if configured.

        Returns whether any were generated.
        """
        if not (self.dataframe_config and self.dataframe_config.enabled):
            return False

        # When response unwrapping is active the unwrapped data type is the
        # endpoint's real return type, so feed it to list detection -- without
        # this, non-paginated envelope list endpoints (e.g.
        # ``ResponseWithStatusEnvelope*``) are misclassified as non-list and
        # silently lose their DataFrame variants.
        unwrap_type_ast = None
        should_unwrap, unwrap_path = self._get_unwrap_config(endpoint)
        if should_unwrap and unwrap_path:
            unwrap_type_ast, _ = self._get_unwrapped_type_ast(endpoint, unwrap_path)

        df_config = get_dataframe_config_for_endpoint(
            endpoint, self.dataframe_config, unwrap_type_ast=unwrap_type_ast
        )

        generated = False
        if df_config.generate_pandas:
            generated = True
            self._emit_dataframe_fn_pair(
                'pandas',
                '_df',
                endpoint,
                df_config.path,
                body,
                import_collector,
                endpoint_names,
            )

        if df_config.generate_polars:
            generated = True
            self._emit_dataframe_fn_pair(
                'polars',
                '_pl',
                endpoint,
                df_config.path,
                body,
                import_collector,
                endpoint_names,
            )

        return generated

    def _emit_paginated_export_pair(
        self,
        endpoint: Endpoint,
        item_type_ast: ast.expr | None,
        item_type_imports: dict[str, set[str]],
        body: list[ast.stmt],
        import_collector: ImportCollector,
        endpoint_names: list[str],
    ) -> bool:
        """Emit sync+async export wrappers around the paginated ``_iter`` fns."""
        if (
            not self.export_config
            or not self.export_config.enabled
            or item_type_ast is None
        ):
            return False
        should_generate, formats, _path = (
            self.export_config.should_generate_for_endpoint(
                endpoint_name=endpoint.sync_fn_name,
                returns_list=True,
            )
        )
        if not should_generate:
            return False

        from otterapi.codegen.export import build_standalone_paginated_export_fn

        default_format = formats[0] if formats else 'csv'
        for fn_name, is_async in (
            (f'{endpoint.sync_fn_name}_export', False),
            (f'{endpoint.async_fn_name}_export', True),
        ):
            fn, imports = build_standalone_paginated_export_fn(
                fn_name=fn_name,
                target_iter_fn_name=f'{endpoint.sync_fn_name}_iter'
                if not is_async
                else f'{endpoint.async_fn_name}_iter',
                parameters=endpoint.parameters,
                request_body_info=endpoint.request_body,
                item_type_ast=item_type_ast,
                item_type_imports=item_type_imports,
                docs=endpoint.description,
                is_async=is_async,
                default_format=default_format,
                default_batch_size=self.export_config.batch_size,
            )
            endpoint_names.append(fn_name)
            body.append(fn)
            import_collector.add_imports(imports)
        return True

    def _emit_standalone_export_pair(
        self,
        endpoint: Endpoint,
        body: list[ast.stmt],
        import_collector: ImportCollector,
        endpoint_names: list[str],
    ) -> bool:
        """Emit sync+async export wrappers for a non-paginated list endpoint."""
        if not self.export_config or not self.export_config.enabled:
            return False
        # When response unwrapping is active, the list lives behind the data
        # path (e.g. envelope.data) rather than directly on the response type,
        # so resolve the item type through that path.
        should_unwrap, unwrap_path = self._get_unwrap_config(endpoint)
        data_path = unwrap_path if should_unwrap else None
        item_type_ast, item_type_imports = self._get_item_type_ast(endpoint, data_path)
        if item_type_ast is None:
            return False
        should_generate, formats, _path = (
            self.export_config.should_generate_for_endpoint(
                endpoint_name=endpoint.sync_fn_name,
                returns_list=True,
            )
        )
        if not should_generate:
            return False

        from otterapi.codegen.export import build_standalone_export_fn

        default_format = formats[0] if formats else 'csv'
        for fn_name, is_async in (
            (f'{endpoint.sync_fn_name}_export', False),
            (f'{endpoint.async_fn_name}_export', True),
        ):
            fn, imports = build_standalone_export_fn(
                fn_name=fn_name,
                target_fn_name=endpoint.sync_fn_name
                if not is_async
                else endpoint.async_fn_name,
                parameters=endpoint.parameters,
                request_body_info=endpoint.request_body,
                item_type_ast=item_type_ast,
                item_type_imports=item_type_imports,
                docs=endpoint.description,
                is_async=is_async,
                default_format=default_format,
                default_batch_size=self.export_config.batch_size,
            )
            endpoint_names.append(fn_name)
            body.append(fn)
            import_collector.add_imports(imports)
        return True

    def _emit_endpoint_functions(
        self,
        endpoint: Endpoint,
        body: list[ast.stmt],
        import_collector: ImportCollector,
        endpoint_names: list[str],
    ) -> tuple[bool, bool, bool]:
        """Emit every function for a single endpoint.

        Returns (has_dataframe_methods, has_pagination_methods, has_export_methods) for this endpoint.
        """
        pag_config = None
        if self.pagination_config and self.pagination_config.enabled:
            pag_config = get_pagination_config_for_endpoint(
                endpoint.sync_fn_name, self.pagination_config, endpoint.parameters
            )

        has_export_methods = False
        if pag_config:
            generated_paginated_df = self._emit_paginated_endpoint_methods(
                endpoint, pag_config, body, import_collector, endpoint_names
            )
            # Export for paginated endpoints (wraps _iter functions)
            item_type_ast, item_type_imports = self._get_item_type_ast(
                endpoint, pag_config.data_path
            )
            has_export_methods = self._emit_paginated_export_pair(
                endpoint,
                item_type_ast,
                item_type_imports,
                body,
                import_collector,
                endpoint_names,
            )
        else:
            generated_paginated_df = False
            self._emit_standalone_endpoint_methods(
                endpoint, body, import_collector, endpoint_names
            )
            # Export for non-paginated endpoints
            has_export_methods = self._emit_standalone_export_pair(
                endpoint,
                body,
                import_collector,
                endpoint_names,
            )

        has_dataframe_methods = generated_paginated_df
        if not generated_paginated_df:
            if self._emit_dataframe_methods(
                endpoint, body, import_collector, endpoint_names
            ):
                has_dataframe_methods = True

        return has_dataframe_methods, pag_config is not None, has_export_methods

    def _collect_module_file_imports(
        self,
        import_collector: ImportCollector,
        endpoints: list[Endpoint],
        has_dataframe_methods: bool,
        has_pagination_methods: bool,
        has_export_methods: bool = False,
    ) -> ast.If | None:
        """Register Client/model/DataFrame/pagination imports; return the optional TYPE_CHECKING block."""
        import_collector.add_imports({'.client': {'Client'}})

        model_names = self._collect_used_model_names(endpoints)
        if model_names:
            for name in model_names:
                import_collector.add_imports({MODELS_MODULE: {name}})

        type_checking_block = None
        if has_dataframe_methods:
            import_collector.add_imports({'typing': {'TYPE_CHECKING'}})
            type_checking_block = ast.If(
                test=_name('TYPE_CHECKING'),
                body=[
                    ast.Import(names=[ast.alias(name='pandas', asname='pd')]),
                    ast.Import(names=[ast.alias(name='polars', asname='pl')]),
                ],
                orelse=[],
            )
            import_collector.add_imports({'._dataframe': {'to_pandas', 'to_polars'}})

        if has_pagination_methods:
            import_collector.add_imports(
                {'collections.abc': {'Iterator', 'AsyncIterator'}}
            )
            import_collector.add_imports(
                {
                    '._pagination': {
                        'paginate_offset',
                        'paginate_offset_async',
                        'paginate_cursor',
                        'paginate_cursor_async',
                        'paginate_page',
                        'paginate_page_async',
                        'iterate_offset',
                        'iterate_offset_async',
                        'iterate_cursor',
                        'iterate_cursor_async',
                        'iterate_page',
                        'iterate_page_async',
                        'extract_path',
                    }
                }
            )

        return type_checking_block

    def _emit_module_file(
        self,
        file_path: Path | UPath,
        endpoints: list[Endpoint],
        description: str | None = None,
    ) -> list[str]:
        """Emit a single endpoint module file."""
        body: list[ast.stmt] = []

        if description:
            body.append(ast.Expr(value=ast.Constant(value=description)))

        import_collector = ImportCollector()

        client_stmts, client_imports = build_default_client_code()
        body.extend(client_stmts)
        import_collector.add_imports(client_imports)

        has_dataframe_methods = False
        has_pagination_methods = False
        has_export_methods = False
        endpoint_names: list[str] = []

        for endpoint in endpoints:
            endpoint_has_df, endpoint_has_pagination, endpoint_has_export = (
                self._emit_endpoint_functions(
                    endpoint, body, import_collector, endpoint_names
                )
            )
            has_dataframe_methods = has_dataframe_methods or endpoint_has_df
            has_pagination_methods = has_pagination_methods or endpoint_has_pagination
            has_export_methods = has_export_methods or endpoint_has_export

        type_checking_block = self._collect_module_file_imports(
            import_collector,
            endpoints,
            has_dataframe_methods,
            has_pagination_methods,
            has_export_methods,
        )

        final_body: list[ast.stmt] = []
        final_body.extend(import_collector.to_ast())

        if type_checking_block:
            final_body.append(type_checking_block)

        all_names = list(endpoint_names)
        if self.reexport_models:
            model_names = import_collector._imports.get(MODELS_MODULE, set())
            if self.reexport_model_exclude_patterns:
                model_names = {
                    n
                    for n in model_names
                    if not any(
                        fnmatch.fnmatch(n, pat)
                        for pat in self.reexport_model_exclude_patterns
                    )
                }
            all_names = sorted(set(all_names) | model_names)

        final_body.append(_all(sorted(all_names)))
        final_body.extend(body)

        file_path = UPath(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        write_mod(final_body, file_path)

        return endpoint_names

    @staticmethod
    def _list_item_type(type_ast: ast.expr | None) -> ast.expr | None:
        """Return the item-type AST if ``type_ast`` is ``list[ItemType]``, else None."""
        if (
            isinstance(type_ast, ast.Subscript)
            and isinstance(type_ast.value, ast.Name)
            and type_ast.value.id == 'list'
        ):
            return type_ast.slice
        return None

    def _get_item_type_ast(
        self, endpoint: Endpoint, data_path: str | None = None
    ) -> tuple[ast.expr | None, dict[str, set[str]]]:
        """Extract the item type AST from a list response type.

        For example, if response_type is list[User], returns the AST for User.
        For paginated endpoints with envelope response types (e.g.,
        PaginatedResponse with a 'data' field), this method uses data_path
        to look up the field type from the response model.

        Args:
            endpoint: The endpoint to check.
            data_path: Optional path to the data field in envelope response types
                (e.g., "data"). If provided and the response type is not directly
                a list, this will look up the field type from the response model.

        Returns:
            A tuple of (ast_expression, imports) where:
            - ast_expression: The AST for the item type, or None if not determinable.
            - imports: Dictionary of imports needed for the item type.
        """
        if not endpoint.response_type or not endpoint.response_type.annotation_ast:
            return None, {}

        item_type = self._list_item_type(endpoint.response_type.annotation_ast)

        if item_type is None and data_path:
            field_type_ast = self._get_field_type_from_response(endpoint, data_path)
            item_type = self._list_item_type(field_type_ast)

        if item_type is None:
            return None, {}

        return item_type, self._collect_model_imports_from_ast(item_type)

    @staticmethod
    def _resolve_response_type_name(endpoint: Endpoint) -> str | None:
        """Resolve the type name to look up in ``_typegen_types`` for a response.

        Falls back to the 2xx success response's type when ``response_type``
        itself has no name (i.e. it's a union of success/error types).
        """
        type_name = endpoint.response_type.name if endpoint.response_type else None
        if not type_name and endpoint.response_infos:
            for response_info in endpoint.response_infos:
                if 200 <= response_info.status_code < 300 and response_info.type:
                    type_name = response_info.type.name
                    break
        return type_name

    @staticmethod
    def _find_field_annotation(
        class_def: ast.ClassDef, field_name: str
    ) -> ast.expr | None:
        """Find the annotation for a field assigned directly in a class body."""
        for stmt in class_def.body:
            if (
                isinstance(stmt, ast.AnnAssign)
                and isinstance(stmt.target, ast.Name)
                and stmt.target.id == field_name
            ):
                return stmt.annotation
        return None

    def _lookup_response_field_annotation(
        self, endpoint: Endpoint, field_path: str
    ) -> ast.expr | None:
        """Resolve the annotation for the first segment of ``field_path`` on the response model.

        Shared by :meth:`_get_field_type_from_response` and
        :meth:`_get_unwrapped_type_ast`, which differ only in whether they
        also collect model imports for the resolved annotation.
        """
        if not endpoint.response_type:
            return None

        field_name = field_path.split('.')[0]
        type_name = self._resolve_response_type_name(endpoint)
        if not type_name:
            return None

        type_def = self._typegen_types.get(type_name)
        if not type_def or not type_def.implementation_ast:
            return None

        impl = type_def.implementation_ast
        if not isinstance(impl, ast.ClassDef):
            return None

        return self._find_field_annotation(impl, field_name)

    def _get_field_type_from_response(
        self, endpoint: Endpoint, field_path: str
    ) -> ast.expr | None:
        """Extract the type AST for a field from the response model.

        For union response types (e.g., SuccessResponse | ErrorResponse),
        this method looks for the 2xx success response in response_infos
        to find the correct type.

        Args:
            endpoint: The endpoint to check.
            field_path: The dotted path to the field (e.g., "data").

        Returns:
            The AST expression for the field type, or None if not found.
        """
        return self._lookup_response_field_annotation(endpoint, field_path)

    def _get_unwrap_config(self, endpoint: Endpoint) -> tuple[bool, str | None]:
        """Get the response unwrap configuration for an endpoint.

        Args:
            endpoint: The endpoint to check.

        Returns:
            A tuple of (should_unwrap, data_path).
        """
        if not self.response_unwrap_config:
            return False, None
        return self.response_unwrap_config.get_unwrap_config_for_endpoint(
            endpoint.sync_fn_name
        )

    def _collect_model_imports_from_ast(
        self, annotation_ast: ast.expr
    ) -> dict[str, set[str]]:
        """Collect model imports needed for an AST annotation.

        Walks the AST and finds all Name nodes that correspond to
        model types in _typegen_types, then collects their annotation imports.

        Args:
            annotation_ast: The annotation AST to scan for model references.

        Returns:
            Dictionary mapping module names to sets of import names.
        """
        imports: dict[str, set[str]] = {}

        # Get all available model names
        available_models = {
            name
            for name, type_ in self._typegen_types.items()
            if type_.implementation_ast is not None
        }

        # Walk the AST to find Name nodes
        for node in ast.walk(annotation_ast):
            if isinstance(node, ast.Name) and node.id in available_models:
                # This is a model reference - add it to imports
                # Models are imported from the models module
                if MODELS_MODULE not in imports:
                    imports[MODELS_MODULE] = set()
                imports[MODELS_MODULE].add(node.id)

        return imports

    def _get_unwrapped_type_ast(
        self,
        endpoint: Endpoint,
        data_path: str,
    ) -> tuple[ast.expr | None, dict[str, set[str]]]:
        """Extract the type AST for the unwrapped data field.

        Looks up the response type model in typegen and finds the field
        matching the data_path to determine its type.

        For union response types (e.g., SuccessResponse | ErrorResponse),
        this method looks for the 2xx success response in response_infos
        to find the correct type to unwrap.

        Args:
            endpoint: The endpoint to check.
            data_path: The dotted path to the data field (e.g., "data").

        Returns:
            A tuple of (ast_expression, imports) where:
            - ast_expression: The AST for the unwrapped type, or None if not found.
            - imports: Dictionary of imports needed for the unwrapped type.
        """
        annotation = self._lookup_response_field_annotation(endpoint, data_path)
        if annotation is None:
            return None, {}
        return annotation, self._collect_model_imports_from_ast(annotation)

    def _collect_used_model_names(self, endpoints: list[Endpoint]) -> set[str]:
        """Collect model names used in the given endpoints."""
        from otterapi.codegen.types import collect_used_model_names

        return collect_used_model_names(endpoints, self._typegen_types)

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
            write_mod(body, init_path)
        else:
            init_path.touch()

    def _emit_root_init(self) -> None:
        """Emit the root __init__.py file."""
        body: list[ast.stmt] = []
        all_names: list[str] = []
        imported_modules: set[str] = set()

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

        # Import and export Client
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
            write_mod(body, init_path)

    def _get_model_names(self) -> list[str]:
        """Get model names from the typegen_types."""
        if not self._typegen_types:
            return []

        return [
            type_.name
            for type_ in self._typegen_types.values()
            if type_.name and type_.implementation_ast
        ]
