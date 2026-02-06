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

from otterapi.codegen.ast_utils import ImportCollector, _all, _name
from otterapi.codegen.utils import write_mod

if TYPE_CHECKING:
    from otterapi.codegen.types import Endpoint, Type
    from otterapi.config import (
        DataFrameConfig,
        ModuleDefinition,
        ModuleSplitConfig,
        PaginationConfig,
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
        self._compiled_patterns: dict[str, list[re.Pattern]] = {}

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
            if working_path.startswith(parent_definition.strip_prefix):
                working_path = working_path[len(parent_definition.strip_prefix) :]
                if not working_path.startswith('/'):
                    working_path = '/' + working_path

        for module_name, definition in module_map.items():
            current_path = parent_path + [module_name]

            match_path = working_path
            if definition.strip_prefix:
                if match_path.startswith(definition.strip_prefix):
                    match_path = match_path[len(definition.strip_prefix) :]
                    if not match_path.startswith('/'):
                        match_path = '/' + match_path

            if definition.paths:
                for pattern in definition.paths:
                    if self._path_matches(working_path, pattern):
                        if definition.modules:
                            nested_result = self._match_module_map(
                                match_path,
                                definition.modules,
                                current_path,
                                definition,
                            )
                            if nested_result:
                                return nested_result

                        return ResolvedModule(
                            module_path=current_path,
                            definition=definition,
                            resolution='custom',
                        )

            elif definition.modules:
                nested_result = self._match_module_map(
                    match_path,
                    definition.modules,
                    current_path,
                    definition,
                )
                if nested_result:
                    return nested_result

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
        >>> emitter.emit(tree, base_url)
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
        """
        self.config = config
        self.output_dir = UPath(output_dir)
        self.models_file = UPath(models_file)
        self.models_import_path = models_import_path
        self.client_class_name = client_class_name
        self.dataframe_config = dataframe_config
        self.pagination_config = pagination_config
        self._emitted_modules: list[EmittedModule] = []
        self._typegen_types: dict[str, Type] = {}
        self._is_flat: bool = False  # Track if we're emitting flat structure

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
            self._is_flat = True
            self._emit_flat(tree, base_url)
        else:
            self._is_flat = False
            self._emit_nested(tree, base_url)

        return self._emitted_modules

    def _emit_flat(self, tree: ModuleTree, base_url: str) -> None:
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

        self._emit_flat_init(all_exports)

    def _emit_nested(self, tree: ModuleTree, base_url: str) -> None:
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

        self._emit_nested_inits(tree, directories)

    def _emit_module_file(
        self,
        file_path: Path | UPath,
        endpoints: list[Endpoint],
        base_url: str,
        description: str | None = None,
        module_path: list[str] | None = None,
    ) -> list[str]:
        """Emit a single endpoint module file."""
        from otterapi.codegen.dataframes import get_dataframe_config_for_endpoint
        from otterapi.codegen.endpoints import (
            build_default_client_code,
            build_standalone_dataframe_fn,
            build_standalone_endpoint_fn,
            build_standalone_paginated_dataframe_fn,
            build_standalone_paginated_fn,
            build_standalone_paginated_iter_fn,
        )
        from otterapi.codegen.pagination import get_pagination_config_for_endpoint

        body: list[ast.stmt] = []

        if description:
            body.append(ast.Expr(value=ast.Constant(value=description)))

        import_collector = ImportCollector()

        client_stmts, client_imports = build_default_client_code()
        body.extend(client_stmts)
        import_collector.add_imports(client_imports)

        has_dataframe_methods = False
        has_pagination_methods = False
        endpoint_names: list[str] = []

        for endpoint in endpoints:
            # Track whether we generated paginated DataFrame methods for this endpoint
            generated_paginated_df = False

            # Check if this endpoint has pagination configured
            pag_config = None
            if self.pagination_config and self.pagination_config.enabled:
                pag_config = get_pagination_config_for_endpoint(
                    endpoint.sync_fn_name,
                    self.pagination_config,
                    endpoint.parameters,
                )

            # Generate pagination methods if configured, otherwise regular functions
            if pag_config:
                has_pagination_methods = True

                # Get item type from response type if it's a list
                item_type_ast = self._get_item_type_ast(endpoint)

                # Build pagination config dict
                pag_dict = {
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

                # Sync paginated function
                pag_fn, pag_imports = build_standalone_paginated_fn(
                    fn_name=endpoint.sync_fn_name,
                    method=endpoint.method,
                    path=endpoint.path,
                    parameters=endpoint.parameters,
                    request_body_info=endpoint.request_body,
                    response_type=endpoint.response_type,
                    pagination_style=pag_config.style,
                    pagination_config=pag_dict,
                    item_type_ast=item_type_ast,
                    docs=endpoint.description,
                    is_async=False,
                )
                endpoint_names.append(endpoint.sync_fn_name)
                body.append(pag_fn)
                import_collector.add_imports(pag_imports)

                # Async paginated function
                async_pag_fn, async_pag_imports = build_standalone_paginated_fn(
                    fn_name=endpoint.async_fn_name,
                    method=endpoint.method,
                    path=endpoint.path,
                    parameters=endpoint.parameters,
                    request_body_info=endpoint.request_body,
                    response_type=endpoint.response_type,
                    pagination_style=pag_config.style,
                    pagination_config=pag_dict,
                    item_type_ast=item_type_ast,
                    docs=endpoint.description,
                    is_async=True,
                )
                endpoint_names.append(endpoint.async_fn_name)
                body.append(async_pag_fn)
                import_collector.add_imports(async_pag_imports)

                # Sync iterator function
                iter_fn_name = f'{endpoint.sync_fn_name}_iter'
                iter_fn, iter_imports = build_standalone_paginated_iter_fn(
                    fn_name=iter_fn_name,
                    method=endpoint.method,
                    path=endpoint.path,
                    parameters=endpoint.parameters,
                    request_body_info=endpoint.request_body,
                    response_type=endpoint.response_type,
                    pagination_style=pag_config.style,
                    pagination_config=pag_dict,
                    item_type_ast=item_type_ast,
                    docs=endpoint.description,
                    is_async=False,
                )
                endpoint_names.append(iter_fn_name)
                body.append(iter_fn)
                import_collector.add_imports(iter_imports)

                # Async iterator function
                async_iter_fn_name = f'{endpoint.async_fn_name}_iter'
                async_iter_fn, async_iter_imports = build_standalone_paginated_iter_fn(
                    fn_name=async_iter_fn_name,
                    method=endpoint.method,
                    path=endpoint.path,
                    parameters=endpoint.parameters,
                    request_body_info=endpoint.request_body,
                    response_type=endpoint.response_type,
                    pagination_style=pag_config.style,
                    pagination_config=pag_dict,
                    item_type_ast=item_type_ast,
                    docs=endpoint.description,
                    is_async=True,
                )
                endpoint_names.append(async_iter_fn_name)
                body.append(async_iter_fn)
                import_collector.add_imports(async_iter_imports)

                # Generate paginated DataFrame methods if dataframe is enabled
                # For paginated endpoints, we know they return lists, so check config directly

                if self.dataframe_config and self.dataframe_config.enabled:
                    # Check if endpoint is explicitly disabled
                    endpoint_df_config = self.dataframe_config.endpoints.get(
                        endpoint.sync_fn_name
                    )
                    if endpoint_df_config and endpoint_df_config.enabled is False:
                        pass  # Skip DataFrame generation for this endpoint
                    elif self.dataframe_config.pandas:
                        generated_paginated_df = True
                        has_dataframe_methods = True
                        # Sync pandas paginated method
                        pandas_fn_name = f'{endpoint.sync_fn_name}_df'
                        pandas_fn, pandas_imports = (
                            build_standalone_paginated_dataframe_fn(
                                fn_name=pandas_fn_name,
                                method=endpoint.method,
                                path=endpoint.path,
                                parameters=endpoint.parameters,
                                request_body_info=endpoint.request_body,
                                response_type=endpoint.response_type,
                                pagination_style=pag_config.style,
                                pagination_config=pag_dict,
                                library='pandas',
                                item_type_ast=item_type_ast,
                                docs=endpoint.description,
                                is_async=False,
                            )
                        )
                        endpoint_names.append(pandas_fn_name)
                        body.append(pandas_fn)
                        import_collector.add_imports(pandas_imports)

                        # Async pandas paginated method
                        async_pandas_fn_name = f'{endpoint.async_fn_name}_df'
                        async_pandas_fn, async_pandas_imports = (
                            build_standalone_paginated_dataframe_fn(
                                fn_name=async_pandas_fn_name,
                                method=endpoint.method,
                                path=endpoint.path,
                                parameters=endpoint.parameters,
                                request_body_info=endpoint.request_body,
                                response_type=endpoint.response_type,
                                pagination_style=pag_config.style,
                                pagination_config=pag_dict,
                                library='pandas',
                                item_type_ast=item_type_ast,
                                docs=endpoint.description,
                                is_async=True,
                            )
                        )
                        endpoint_names.append(async_pandas_fn_name)
                        body.append(async_pandas_fn)
                        import_collector.add_imports(async_pandas_imports)

                    if self.dataframe_config.polars:
                        generated_paginated_df = True
                        has_dataframe_methods = True
                        # Sync polars paginated method
                        polars_fn_name = f'{endpoint.sync_fn_name}_pl'
                        polars_fn, polars_imports = (
                            build_standalone_paginated_dataframe_fn(
                                fn_name=polars_fn_name,
                                method=endpoint.method,
                                path=endpoint.path,
                                parameters=endpoint.parameters,
                                request_body_info=endpoint.request_body,
                                response_type=endpoint.response_type,
                                pagination_style=pag_config.style,
                                pagination_config=pag_dict,
                                library='polars',
                                item_type_ast=item_type_ast,
                                docs=endpoint.description,
                                is_async=False,
                            )
                        )
                        endpoint_names.append(polars_fn_name)
                        body.append(polars_fn)
                        import_collector.add_imports(polars_imports)

                        # Async polars paginated method
                        async_polars_fn_name = f'{endpoint.async_fn_name}_pl'
                        async_polars_fn, async_polars_imports = (
                            build_standalone_paginated_dataframe_fn(
                                fn_name=async_polars_fn_name,
                                method=endpoint.method,
                                path=endpoint.path,
                                parameters=endpoint.parameters,
                                request_body_info=endpoint.request_body,
                                response_type=endpoint.response_type,
                                pagination_style=pag_config.style,
                                pagination_config=pag_dict,
                                library='polars',
                                item_type_ast=item_type_ast,
                                docs=endpoint.description,
                                is_async=True,
                            )
                        )
                        endpoint_names.append(async_polars_fn_name)
                        body.append(async_polars_fn)
                        import_collector.add_imports(async_polars_imports)
            else:
                # Build sync standalone function
                sync_fn, sync_imports = build_standalone_endpoint_fn(
                    fn_name=endpoint.sync_fn_name,
                    method=endpoint.method,
                    path=endpoint.path,
                    parameters=endpoint.parameters,
                    request_body_info=endpoint.request_body,
                    response_type=endpoint.response_type,
                    response_infos=endpoint.response_infos,
                    docs=endpoint.description,
                    is_async=False,
                )
                endpoint_names.append(endpoint.sync_fn_name)
                body.append(sync_fn)
                import_collector.add_imports(sync_imports)

                # Build async standalone function
                async_fn, async_imports = build_standalone_endpoint_fn(
                    fn_name=endpoint.async_fn_name,
                    method=endpoint.method,
                    path=endpoint.path,
                    parameters=endpoint.parameters,
                    request_body_info=endpoint.request_body,
                    response_type=endpoint.response_type,
                    response_infos=endpoint.response_infos,
                    docs=endpoint.description,
                    is_async=True,
                )
                endpoint_names.append(endpoint.async_fn_name)
                body.append(async_fn)
                import_collector.add_imports(async_imports)

            # Generate DataFrame methods if configured
            # Skip if paginated DataFrame methods were already generated for this endpoint
            if (
                self.dataframe_config
                and self.dataframe_config.enabled
                and not generated_paginated_df
            ):
                df_config = get_dataframe_config_for_endpoint(
                    endpoint, self.dataframe_config
                )

                if df_config.generate_pandas:
                    has_dataframe_methods = True
                    pandas_fn_name = f'{endpoint.sync_fn_name}_df'
                    pandas_fn, pandas_imports = build_standalone_dataframe_fn(
                        fn_name=pandas_fn_name,
                        method=endpoint.method,
                        path=endpoint.path,
                        parameters=endpoint.parameters,
                        request_body_info=endpoint.request_body,
                        library='pandas',
                        default_path=df_config.path,
                        docs=endpoint.description,
                        is_async=False,
                    )
                    endpoint_names.append(pandas_fn_name)
                    body.append(pandas_fn)
                    import_collector.add_imports(pandas_imports)

                    async_pandas_fn_name = f'{endpoint.async_fn_name}_df'
                    async_pandas_fn, async_pandas_imports = (
                        build_standalone_dataframe_fn(
                            fn_name=async_pandas_fn_name,
                            method=endpoint.method,
                            path=endpoint.path,
                            parameters=endpoint.parameters,
                            request_body_info=endpoint.request_body,
                            library='pandas',
                            default_path=df_config.path,
                            docs=endpoint.description,
                            is_async=True,
                        )
                    )
                    endpoint_names.append(async_pandas_fn_name)
                    body.append(async_pandas_fn)
                    import_collector.add_imports(async_pandas_imports)

                if df_config.generate_polars:
                    has_dataframe_methods = True
                    polars_fn_name = f'{endpoint.sync_fn_name}_pl'
                    polars_fn, polars_imports = build_standalone_dataframe_fn(
                        fn_name=polars_fn_name,
                        method=endpoint.method,
                        path=endpoint.path,
                        parameters=endpoint.parameters,
                        request_body_info=endpoint.request_body,
                        library='polars',
                        default_path=df_config.path,
                        docs=endpoint.description,
                        is_async=False,
                    )
                    endpoint_names.append(polars_fn_name)
                    body.append(polars_fn)
                    import_collector.add_imports(polars_imports)

                    async_polars_fn_name = f'{endpoint.async_fn_name}_pl'
                    async_polars_fn, async_polars_imports = (
                        build_standalone_dataframe_fn(
                            fn_name=async_polars_fn_name,
                            method=endpoint.method,
                            path=endpoint.path,
                            parameters=endpoint.parameters,
                            request_body_info=endpoint.request_body,
                            library='polars',
                            default_path=df_config.path,
                            docs=endpoint.description,
                            is_async=True,
                        )
                    )
                    endpoint_names.append(async_polars_fn_name)
                    body.append(async_polars_fn)
                    import_collector.add_imports(async_polars_imports)

        # Add __all__ export
        body.insert(0, _all(sorted(endpoint_names)))

        # Add model imports
        model_names = self._collect_used_model_names(endpoints)
        if model_names:
            model_import = self._create_model_import(model_names, module_path)
            body.insert(0, model_import)

        # Add Client import
        client_import = self._create_client_import(module_path)
        body.insert(0, client_import)

        # Add TYPE_CHECKING block for DataFrame type hints if needed
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
            body.insert(0, type_checking_block)

            dataframe_import = ast.ImportFrom(
                module='_dataframe',
                names=[
                    ast.alias(name='to_pandas', asname=None),
                    ast.alias(name='to_polars', asname=None),
                ],
                level=1,
            )
            body.insert(0, dataframe_import)

        # Add pagination imports if needed
        if has_pagination_methods:
            import_collector.add_imports(
                {'collections.abc': {'Iterator', 'AsyncIterator'}}
            )
            pagination_import = ast.ImportFrom(
                module='_pagination',
                names=[
                    ast.alias(name='paginate_offset', asname=None),
                    ast.alias(name='paginate_offset_async', asname=None),
                    ast.alias(name='paginate_cursor', asname=None),
                    ast.alias(name='paginate_cursor_async', asname=None),
                    ast.alias(name='paginate_page', asname=None),
                    ast.alias(name='paginate_page_async', asname=None),
                    ast.alias(name='iterate_offset', asname=None),
                    ast.alias(name='iterate_offset_async', asname=None),
                    ast.alias(name='iterate_cursor', asname=None),
                    ast.alias(name='iterate_cursor_async', asname=None),
                    ast.alias(name='extract_path', asname=None),
                ],
                level=1,
            )
            body.insert(0, pagination_import)

        # Add all other imports at the beginning
        for import_stmt in import_collector.to_ast():
            body.insert(0, import_stmt)

        # Write the file
        file_path = UPath(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        write_mod(body, file_path)

        return endpoint_names

    def _get_item_type_ast(self, endpoint: Endpoint) -> ast.expr | None:
        """Extract the item type AST from a list response type.

        For example, if response_type is list[User], returns the AST for User.

        Args:
            endpoint: The endpoint to check.

        Returns:
            The AST expression for the item type, or None if not a list type.
        """
        if not endpoint.response_type or not endpoint.response_type.annotation_ast:
            return None

        ann = endpoint.response_type.annotation_ast
        if isinstance(ann, ast.Subscript):
            if isinstance(ann.value, ast.Name) and ann.value.id == 'list':
                return ann.slice

        return None

    def _create_client_import(
        self, module_path: list[str] | None = None
    ) -> ast.ImportFrom:
        """Create import statement for the Client class.

        For flat structure, all files are in the same directory as client.py,
        so we always use level=1 (single dot relative import).

        For nested structure, we need to go up the directory tree based on
        the actual file location:
        - module_path=['loot'] -> file at output_dir/loot.py -> level=1
        - module_path=['api', 'users'] -> file at output_dir/api/users.py -> level=2
        """
        if self._is_flat:
            # Flat structure: all files in same directory as client.py
            level = 1
        else:
            # Nested structure: actual directory depth is len(module_path) - 1
            # because the last element is the filename, not a directory
            # e.g., ['loot'] -> depth 0 -> level 1
            # e.g., ['api', 'users'] -> depth 1 -> level 2
            depth = len(module_path) - 1 if module_path and len(module_path) > 1 else 0
            level = depth + 1

        return ast.ImportFrom(
            module='client',
            names=[ast.alias(name='Client', asname=None)],
            level=level,
        )

    def _collect_used_model_names(self, endpoints: list[Endpoint]) -> set[str]:
        """Collect model names used in the given endpoints."""
        from otterapi.codegen.types import collect_used_model_names

        return collect_used_model_names(endpoints, self._typegen_types)

    def _create_model_import(
        self, model_names: set[str], module_path: list[str] | None = None
    ) -> ast.ImportFrom:
        """Create import statement for models.

        For flat structure, all files are in the same directory as models.py,
        so we always use level=1 (single dot relative import).

        For nested structure, we need to go up the directory tree based on
        the actual file location:
        - module_path=['loot'] -> file at output_dir/loot.py -> level=1
        - module_path=['api', 'users'] -> file at output_dir/api/users.py -> level=2
        """
        if self._is_flat:
            # Flat structure: all files in same directory as models.py
            level = 1
        else:
            # Nested structure: actual directory depth is len(module_path) - 1
            # because the last element is the filename, not a directory
            # e.g., ['loot'] -> depth 0 -> level 1
            # e.g., ['api', 'users'] -> depth 1 -> level 2
            depth = len(module_path) - 1 if module_path and len(module_path) > 1 else 0
            level = depth + 1

        return ast.ImportFrom(
            module='models',
            names=[ast.alias(name=name, asname=None) for name in sorted(model_names)],
            level=level,
        )

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

        self._emit_root_init(tree)

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

    def _emit_root_init(self, tree: ModuleTree) -> None:
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

    def get_emitted_modules(self) -> list[EmittedModule]:
        """Get list of all emitted modules."""
        return self._emitted_modules.copy()
