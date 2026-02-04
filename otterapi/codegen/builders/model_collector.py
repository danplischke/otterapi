"""Model name collection utilities.

This module provides utilities for collecting model names that are
actually used in endpoint function signatures by walking the AST.
"""

import ast
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from otterapi.codegen.types import Endpoint, Type

__all__ = ['ModelNameCollector', 'collect_used_model_names']


class ModelNameCollector(ast.NodeVisitor):
    """AST visitor that collects model names from function definitions.

    This visitor walks AST nodes and identifies Name nodes that match
    a set of available model names, allowing us to determine which
    models are actually referenced in generated code.

    Example:
        >>> available = {'Pet', 'User', 'Order'}
        >>> collector = ModelNameCollector(available)
        >>> collector.visit(some_function_ast)
        >>> print(collector.used_models)
        {'Pet', 'User'}
    """

    def __init__(self, available_models: set[str]):
        """Initialize the collector.

        Args:
            available_models: Set of model names that are available for import.
        """
        self.available_models = available_models
        self.used_models: set[str] = set()

    def visit_Name(self, node: ast.Name) -> None:
        """Visit a Name node and check if it's an available model.

        Args:
            node: The AST Name node to check.
        """
        if node.id in self.available_models:
            self.used_models.add(node.id)
        self.generic_visit(node)

    @classmethod
    def collect_from_endpoints(
        cls,
        endpoints: list['Endpoint'],
        available_models: set[str],
    ) -> set[str]:
        """Collect model names used across multiple endpoints.

        Args:
            endpoints: List of Endpoint objects to scan.
            available_models: Set of model names that are available.

        Returns:
            Set of model names that are actually used in the endpoints.
        """
        collector = cls(available_models)
        for endpoint in endpoints:
            collector.visit(endpoint.sync_ast)
            collector.visit(endpoint.async_ast)
        return collector.used_models


def collect_used_model_names(
    endpoints: list['Endpoint'],
    typegen_types: dict[str, 'Type'],
) -> set[str]:
    """Collect model names that are actually used in endpoint signatures.

    Only collects models that have implementations (defined in models.py)
    and are referenced in endpoint parameters, request bodies, or responses.

    Args:
        endpoints: List of Endpoint objects to check for model usage.
        typegen_types: Dictionary mapping type names to Type objects.

    Returns:
        Set of model names actually used in endpoints.
    """
    # Get all model names that have implementations
    available_models = {
        type_.name
        for type_ in typegen_types.values()
        if type_.name and type_.implementation_ast
    }

    return ModelNameCollector.collect_from_endpoints(endpoints, available_models)
