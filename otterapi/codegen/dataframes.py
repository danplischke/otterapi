"""DataFrame utilities for OtterAPI code generation.

This module provides utilities for:
- Generating the _dataframe.py utility file for runtime DataFrame conversion
- Determining DataFrame generation configuration for endpoints
- Checking if endpoints return list types suitable for DataFrame conversion
"""

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from upath import UPath

if TYPE_CHECKING:
    from otterapi.codegen.types import Endpoint, Type
    from otterapi.config import DataFrameConfig

__all__ = [
    'generate_dataframe_module',
    'DataFrameMethodConfig',
    'get_dataframe_imports',
    'get_dataframe_type_checking_imports',
    'get_dataframe_config_for_endpoint',
    'get_dataframe_config_from_parts',
    'endpoint_returns_list',
    'response_type_returns_list',
    'annotation_ast_returns_list',
]


def generate_dataframe_module(output_dir: Path | UPath) -> Path | UPath:
    """Write ``_dataframe.py`` into *output_dir* and return the written path."""
    from otterapi.codegen._features import DataFrameFeature

    return DataFrameFeature().write(output_dir)


def get_dataframe_imports() -> dict[str, set[str]]:
    """Get the imports needed for DataFrame method generation.

    Returns:
        A dictionary mapping module names to sets of imported names.
    """
    return {
        'typing': {'TYPE_CHECKING'},
    }


def get_dataframe_type_checking_imports() -> list[str]:
    """Get the TYPE_CHECKING imports for pandas and polars.

    Returns:
        List of import statements to include inside TYPE_CHECKING block.
    """
    return [
        'import pandas as pd',
        'import polars as pl',
    ]


# =============================================================================
# DataFrame Configuration
# =============================================================================


@dataclass
class DataFrameMethodConfig:
    """Configuration for DataFrame method generation.

    Attributes:
        generate_pandas: Whether to generate pandas DataFrame methods.
        generate_polars: Whether to generate polars DataFrame methods.
        path: Default JSONPath for extracting data from responses.
    """

    generate_pandas: bool = False
    generate_polars: bool = False
    path: str | None = None


def annotation_ast_returns_list(annotation_ast: 'ast.expr | None') -> bool:
    """Check if a raw annotation AST represents a ``list[...]`` type.

    Args:
        annotation_ast: The annotation AST node to check, or None.

    Returns:
        True if the annotation is a ``list[...]`` subscript, False otherwise.
    """
    if isinstance(annotation_ast, ast.Subscript):
        if isinstance(annotation_ast.value, ast.Name) and (
            annotation_ast.value.id == 'list'
        ):
            return True

    return False


def endpoint_returns_list(
    endpoint: 'Endpoint',
    unwrap_type_ast: 'ast.expr | None' = None,
) -> bool:
    """Check if an endpoint returns a list type.

    Examines the endpoint's response type annotation AST to determine
    if it represents a list type.

    When response unwrapping is active for the endpoint, the function's actual
    return type is the unwrapped data field rather than the envelope model on
    ``endpoint.response_type``. In that case the caller passes the unwrapped
    type AST (e.g. ``list[Pet]`` extracted from ``data``) via
    *unwrap_type_ast*, and it is consulted instead of the envelope type.
    Without this, every non-paginated ``ResponseWithStatusEnvelope*`` list
    endpoint would be misclassified as non-list and silently lose its
    DataFrame variants.

    Args:
        endpoint: The endpoint to check.
        unwrap_type_ast: The AST of the unwrapped return type when response
            unwrapping is active, or None when it is not.

    Returns:
        True if the endpoint returns a list, False otherwise.

    Example:
        >>> # Endpoint with response_type annotation of list[Pet]
        >>> endpoint_returns_list(endpoint)
        True
    """
    if unwrap_type_ast is not None:
        return annotation_ast_returns_list(unwrap_type_ast)

    if not endpoint.response_type:
        return False

    return response_type_returns_list(endpoint.response_type)


def response_type_returns_list(response_type: 'Type | None') -> bool:
    """Check if a response type represents a list.

    This is a lower-level version of endpoint_returns_list that works
    directly with Type objects.

    Args:
        response_type: The Type object to check, or None.

    Returns:
        True if the type represents a list, False otherwise.
    """
    if not response_type:
        return False

    return annotation_ast_returns_list(response_type.annotation_ast)


def get_dataframe_config_for_endpoint(
    endpoint: 'Endpoint',
    dataframe_config: 'DataFrameConfig',
    unwrap_type_ast: 'ast.expr | None' = None,
) -> DataFrameMethodConfig:
    """Get the DataFrame method configuration for an endpoint.

    Determines whether to generate pandas and/or polars DataFrame methods
    for a given endpoint based on the configuration and whether the
    endpoint returns a list type.

    Args:
        endpoint: The endpoint to get configuration for.
        dataframe_config: The global DataFrame configuration.
        unwrap_type_ast: The AST of the unwrapped return type when response
            unwrapping is active for the endpoint, or None. When provided it
            is used for list detection instead of the envelope response type.

    Returns:
        DataFrameMethodConfig with generation flags and default path.

    Example:
        >>> config = get_dataframe_config_for_endpoint(endpoint, df_config)
        >>> if config.generate_pandas:
        ...     # Generate pandas method
    """
    if not dataframe_config.enabled:
        return DataFrameMethodConfig()

    # Check if this endpoint returns a list type. When response unwrapping is
    # active the unwrapped data type is the real return type, so prefer it.
    returns_list = endpoint_returns_list(endpoint, unwrap_type_ast=unwrap_type_ast)

    # Get the sync function name for config lookup
    endpoint_name = endpoint.sync_fn_name

    # Use the config method to determine what to generate
    gen_pandas, gen_polars, path = dataframe_config.should_generate_for_endpoint(
        endpoint_name=endpoint_name,
        returns_list=returns_list,
    )

    return DataFrameMethodConfig(
        generate_pandas=gen_pandas,
        generate_polars=gen_polars,
        path=path,
    )


def get_dataframe_config_from_parts(
    endpoint_name: str,
    response_type: 'Type | None',
    dataframe_config: 'DataFrameConfig',
) -> DataFrameMethodConfig:
    """Get DataFrame configuration using individual endpoint parts.

    This variant is useful when you have the endpoint name and response
    type separately rather than a full Endpoint object.

    Args:
        endpoint_name: The name of the endpoint function.
        response_type: The response Type object, or None.
        dataframe_config: The global DataFrame configuration.

    Returns:
        DataFrameMethodConfig with generation flags and default path.
    """
    if not dataframe_config.enabled:
        return DataFrameMethodConfig()

    returns_list = response_type_returns_list(response_type)

    gen_pandas, gen_polars, path = dataframe_config.should_generate_for_endpoint(
        endpoint_name=endpoint_name,
        returns_list=returns_list,
    )

    return DataFrameMethodConfig(
        generate_pandas=gen_pandas,
        generate_polars=gen_polars,
        path=path,
    )
