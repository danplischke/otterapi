"""Module tree structure for organizing endpoints into a hierarchy.

This module provides the ModuleTree dataclass and ModuleTreeBuilder class
for building a hierarchical structure of modules from a list of endpoints
based on the configured module splitting strategy.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from otterapi.codegen.types import Endpoint
    from otterapi.config import ModuleDefinition, ModuleSplitConfig


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
            # Add to this node
            self.endpoints.append(endpoint)
            return

        # Navigate/create the path
        current = self
        for part in module_path:
            if part not in current.children:
                current.children[part] = ModuleTree(name=part)
            current = current.children[part]

        # Add endpoint to the leaf node
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
        """Recursively walk the tree.

        Args:
            current_path: The path to the current node.

        Yields:
            Tuples of (module_path, node).
        """
        yield current_path, self

        for child_name, child_node in sorted(self.children.items()):
            child_path = current_path + [child_name]
            yield from child_node._walk_recursive(child_path)

    def walk_leaves(self) -> Iterator[tuple[list[str], ModuleTree]]:
        """Iterate over leaf nodes (nodes with endpoints but no children, or just endpoints).

        Yields:
            Tuples of (module_path, node) for leaf nodes with endpoints.
        """
        for path, node in self.walk():
            # A leaf is a node that has endpoints
            if node.endpoints:
                yield path, node

    def count_endpoints(self) -> int:
        """Count the total number of endpoints in this subtree.

        Returns:
            Total endpoint count including all descendants.
        """
        total = len(self.endpoints)
        for child in self.children.values():
            total += child.count_endpoints()
        return total

    def is_empty(self) -> bool:
        """Check if this subtree has no endpoints.

        Returns:
            True if there are no endpoints in this node or any descendants.
        """
        return self.count_endpoints() == 0

    def flatten(self) -> dict[str, list[Endpoint]]:
        """Flatten the tree into a dictionary mapping module paths to endpoints.

        Returns:
            Dictionary with dotted module paths as keys and endpoint lists as values.
        """
        result: dict[str, list[Endpoint]] = {}

        for path, node in self.walk():
            if node.endpoints:
                module_name = '.'.join(path) if path else '__root__'
                result[module_name] = node.endpoints

        return result


class ModuleTreeBuilder:
    """Builds a ModuleTree from a list of endpoints and configuration.

    The builder uses a ModuleMapResolver to determine which module each
    endpoint belongs to, then constructs the tree structure.

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
        from otterapi.codegen.splitting.resolver import ModuleMapResolver

        self.config = config
        self.resolver = ModuleMapResolver(config)

    def build(self, endpoints: list[Endpoint]) -> ModuleTree:
        """Build a module tree from a list of endpoints.

        Args:
            endpoints: List of Endpoint objects to organize.

        Returns:
            A ModuleTree with endpoints organized according to the configuration.
        """
        # Create root node
        root = ModuleTree(name='__root__')

        # Resolve and add each endpoint
        for endpoint in endpoints:
            # Get tags from endpoint if available
            tags = getattr(endpoint, 'tags', None)

            # Resolve the endpoint to a module
            resolved = self.resolver.resolve(
                path=endpoint.path,
                method=endpoint.method,
                tags=tags,
            )

            # Add to tree
            root.add_endpoint(resolved.module_path, endpoint)

            # Store definition on the node if we have one
            if resolved.definition:
                node = root.get_node(resolved.module_path)
                if node and not node.definition:
                    node.definition = resolved.definition
                    if resolved.definition.description:
                        node.description = resolved.definition.description

        # Apply consolidation for small modules
        if self.config.min_endpoints > 1:
            self._consolidate_small_modules(root)

        return root

    def _consolidate_small_modules(self, root: ModuleTree) -> None:
        """Consolidate modules with fewer than min_endpoints into fallback.

        Args:
            root: The root ModuleTree to consolidate.
        """
        # Collect modules that are too small
        to_consolidate: list[tuple[list[str], ModuleTree]] = []

        for path, node in root.walk():
            # Skip root and fallback module
            if not path or path == [self.config.fallback_module]:
                continue

            # Check if this is a leaf module with too few endpoints
            if node.endpoints and not node.children:
                if len(node.endpoints) < self.config.min_endpoints:
                    to_consolidate.append((path, node))

        # Move endpoints from small modules to fallback
        for path, node in to_consolidate:
            for endpoint in node.endpoints:
                root.add_endpoint([self.config.fallback_module], endpoint)
            node.endpoints = []

        # Clean up empty nodes (from bottom up)
        self._remove_empty_nodes(root)

    def _remove_empty_nodes(self, node: ModuleTree) -> bool:
        """Remove empty nodes from the tree.

        Args:
            node: The node to clean up.

        Returns:
            True if this node is empty and should be removed.
        """
        # Recursively clean children first
        empty_children = []
        for child_name, child_node in node.children.items():
            if self._remove_empty_nodes(child_node):
                empty_children.append(child_name)

        # Remove empty children
        for child_name in empty_children:
            del node.children[child_name]

        # This node is empty if it has no endpoints and no children
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
