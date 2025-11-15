"""Import collection and management for code generation.

This module provides utilities for collecting and organizing imports
during code generation, ensuring proper deduplication and formatting.
"""

import ast


class ImportCollector:
    """Collects and manages imports for generated Python code.
    
    This class provides a centralized way to collect imports from various
    sources during code generation and convert them to AST import statements.
    It automatically deduplicates imports and sorts them for consistent output.
    
    Example:
        >>> collector = ImportCollector()
        >>> collector.add_imports({'typing': {'List', 'Dict'}})
        >>> collector.add_imports({'typing': {'Optional'}})
        >>> imports = collector.to_ast()
        >>> # Returns [ImportFrom(module='typing', names=['Dict', 'List', 'Optional'])]
    """
    
    def __init__(self):
        """Initialize an empty import collector."""
        self._imports: dict[str, set[str]] = {}
    
    def add_imports(self, imports: dict[str, set[str]]) -> None:
        """Add imports from a dictionary mapping modules to sets of names.
        
        Args:
            imports: Dictionary mapping module names to sets of imported names.
                    Example: {'typing': {'List', 'Dict'}, 'pydantic': {'BaseModel'}}
        """
        for module, names in imports.items():
            if module not in self._imports:
                self._imports[module] = set()
            self._imports[module].update(names)
    
    def add_import(self, module: str, name: str) -> None:
        """Add a single import.
        
        Args:
            module: The module to import from (e.g., 'typing', 'pydantic').
            name: The name to import (e.g., 'List', 'BaseModel').
        """
        if module not in self._imports:
            self._imports[module] = set()
        self._imports[module].add(name)
    
    def to_ast(self, reverse_sort: bool = True) -> list[ast.ImportFrom]:
        """Convert collected imports to AST ImportFrom statements.
        
        Args:
            reverse_sort: If True, sort modules in reverse order (default).
                         This is useful for placing standard library imports last.
        
        Returns:
            List of ast.ImportFrom statements, sorted by module name.
            Names within each import are also sorted alphabetically.
        """
        import_stmts = []
        for module, names in sorted(self._imports.items(), reverse=reverse_sort):
            import_stmt = ast.ImportFrom(
                module=module,
                names=[ast.alias(name=name, asname=None) for name in sorted(names)],
                level=0,
            )
            import_stmts.append(import_stmt)
        return import_stmts
    
    def has_imports(self) -> bool:
        """Check if any imports have been collected.
        
        Returns:
            True if imports exist, False otherwise.
        """
        return bool(self._imports)
    
    def clear(self) -> None:
        """Clear all collected imports."""
        self._imports.clear()
    
    def get_modules(self) -> set[str]:
        """Get the set of all modules that have been imported.
        
        Returns:
            Set of module names.
        """
        return set(self._imports.keys())

