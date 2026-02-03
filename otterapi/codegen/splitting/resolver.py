"""Module map resolver for matching endpoint paths to target modules.

This module provides the ModuleMapResolver class that determines which
module an endpoint should be placed in based on the configured module
splitting strategy and module map.
"""

from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from otterapi.config import ModuleDefinition, ModuleSplitConfig


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
        """Get the relative file path for this module.

        For nested modules, returns a path like 'api/v1/users.py'.
        For flat structure, returns 'api_v1_users.py'.
        """
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
        """Initialize the resolver with configuration.

        Args:
            config: The module split configuration.
        """
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
        # Step 1: Strip global prefixes
        stripped_path = self._strip_global_prefixes(path)

        # Step 2: Try custom module_map (for custom or hybrid strategies)
        if self.config.module_map and self._should_use_module_map():
            result = self._match_module_map(stripped_path, self.config.module_map, [])
            if result:
                result.stripped_path = stripped_path
                return result

        # Step 3: Try tag-based resolution (for tag or hybrid strategies)
        if self._should_use_tags() and tags:
            module_name = self._sanitize(tags[0])
            return ResolvedModule(
                module_path=[module_name],
                resolution='tag',
                stripped_path=stripped_path,
            )

        # Step 4: Try path-based resolution (for path or hybrid strategies)
        if self._should_use_path():
            module_name = self._extract_from_path(stripped_path)
            if module_name:
                return ResolvedModule(
                    module_path=[module_name],
                    resolution='path',
                    stripped_path=stripped_path,
                )

        # Step 5: Fallback module
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
        """Strip configured global prefixes from the path.

        Args:
            path: The original endpoint path.

        Returns:
            The path with global prefixes removed.
        """
        for prefix in self.config.global_strip_prefixes:
            if path.startswith(prefix):
                stripped = path[len(prefix) :]
                # Ensure path still starts with /
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
        """Recursively match a path against the module_map.

        Args:
            path: The (possibly stripped) endpoint path.
            module_map: The module map to match against.
            parent_path: The module path accumulated from parent levels.
            parent_definition: The parent ModuleDefinition (for strip_prefix).

        Returns:
            A ResolvedModule if a match is found, None otherwise.
        """
        # Apply parent's strip_prefix if present
        working_path = path
        if parent_definition and parent_definition.strip_prefix:
            if working_path.startswith(parent_definition.strip_prefix):
                working_path = working_path[len(parent_definition.strip_prefix) :]
                if not working_path.startswith('/'):
                    working_path = '/' + working_path

        for module_name, definition in module_map.items():
            current_path = parent_path + [module_name]

            # Apply this definition's strip_prefix for matching
            match_path = working_path
            if definition.strip_prefix:
                if match_path.startswith(definition.strip_prefix):
                    match_path = match_path[len(definition.strip_prefix) :]
                    if not match_path.startswith('/'):
                        match_path = '/' + match_path

            # Check if any of the patterns match
            if definition.paths:
                for pattern in definition.paths:
                    if self._path_matches(working_path, pattern):
                        # If there are nested modules, try to match deeper
                        if definition.modules:
                            nested_result = self._match_module_map(
                                match_path,
                                definition.modules,
                                current_path,
                                definition,
                            )
                            if nested_result:
                                return nested_result

                        # Return match at this level
                        return ResolvedModule(
                            module_path=current_path,
                            definition=definition,
                            resolution='custom',
                        )

            # If no paths defined but has nested modules, check them
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
        """Check if a path matches a glob pattern.

        Supports:
        - `*` to match any single path segment
        - `**` to match any number of path segments
        - Exact matches

        Args:
            path: The endpoint path to match.
            pattern: The glob pattern.

        Returns:
            True if the path matches the pattern.
        """
        # Normalize both path and pattern
        path = path.rstrip('/')
        pattern = pattern.rstrip('/')

        # Handle ** (recursive match)
        if '**' in pattern:
            # Convert ** to regex pattern
            regex_pattern = self._glob_to_regex(pattern)
            return bool(re.match(regex_pattern, path))

        # Use fnmatch for simpler patterns
        return fnmatch.fnmatch(path, pattern)

    def _glob_to_regex(self, pattern: str) -> str:
        """Convert a glob pattern to a regex pattern.

        Args:
            pattern: The glob pattern.

        Returns:
            A regex pattern string.
        """
        # Escape special regex characters except * and **
        result = ''
        i = 0
        while i < len(pattern):
            if pattern[i : i + 2] == '**':
                # ** matches any number of path segments
                result += '.*'
                i += 2
            elif pattern[i] == '*':
                # * matches a single path segment (no slashes)
                result += '[^/]*'
                i += 1
            elif pattern[i] == '?':
                # ? matches a single character
                result += '[^/]'
                i += 1
            else:
                # Escape other special regex characters
                if pattern[i] in r'\.^$+{}[]|()':
                    result += '\\' + pattern[i]
                else:
                    result += pattern[i]
                i += 1

        return f'^{result}$'

    def _extract_from_path(self, path: str) -> str | None:
        """Extract a module name from the path based on path_depth.

        Args:
            path: The endpoint path.

        Returns:
            The extracted module name, or None if extraction fails.
        """
        # Split path into segments, filtering empty strings
        segments = [s for s in path.split('/') if s and not s.startswith('{')]

        if not segments:
            return None

        # Use path_depth to determine how many segments to use
        depth = min(self.config.path_depth, len(segments))
        if depth == 1:
            return self._sanitize(segments[0])
        else:
            # Join multiple segments with underscore
            return '_'.join(self._sanitize(s) for s in segments[:depth])

    def _sanitize(self, name: str) -> str:
        """Sanitize a name to be a valid Python identifier.

        Args:
            name: The name to sanitize.

        Returns:
            A valid Python identifier.
        """
        import keyword
        import re

        # Replace hyphens and spaces with underscores
        sanitized = re.sub(r'[-\s]+', '_', name.lower())

        # Remove invalid characters
        sanitized = re.sub(r'[^a-z0-9_]', '', sanitized)

        # Ensure it doesn't start with a digit
        if sanitized and sanitized[0].isdigit():
            sanitized = '_' + sanitized

        # Handle Python keywords
        if keyword.iskeyword(sanitized):
            sanitized = sanitized + '_'

        return sanitized or 'module'
