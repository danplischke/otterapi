"""Shared DataFrame configuration utilities.

This module provides utilities for determining DataFrame generation
configuration for endpoints. It extracts logic that was previously
duplicated in codegen.py and splitting/emitter.py.
"""

import ast
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from otterapi.codegen.types import Endpoint, Type
    from otterapi.config import DataFrameConfig

__all__ = [
    'DataFrameMethodConfig',
    'get_dataframe_config_for_endpoint',
    'endpoint_returns_list',
]


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


def endpoint_returns_list(endpoint: 'Endpoint') -> bool:
    """Check if an endpoint returns a list type.

    Examines the endpoint's response type annotation AST to determine
    if it represents a list type.

    Args:
        endpoint: The endpoint to check.

    Returns:
        True if the endpoint returns a list, False otherwise.

    Example:
        >>> # Endpoint with response_type annotation of list[Pet]
        >>> endpoint_returns_list(endpoint)
        True
    """
    if not endpoint.response_type:
        return False

    response_type = endpoint.response_type

    # Check the annotation AST for list type
    # Array types have annotation_ast as a Subscript with value.id='list'
    if response_type.annotation_ast:
        ann = response_type.annotation_ast
        if isinstance(ann, ast.Subscript):
            if isinstance(ann.value, ast.Name) and ann.value.id == 'list':
                return True

    return False


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

    if response_type.annotation_ast:
        ann = response_type.annotation_ast
        if isinstance(ann, ast.Subscript):
            if isinstance(ann.value, ast.Name) and ann.value.id == 'list':
                return True

    return False


def get_dataframe_config_for_endpoint(
    endpoint: 'Endpoint',
    dataframe_config: 'DataFrameConfig',
) -> DataFrameMethodConfig:
    """Get the DataFrame method configuration for an endpoint.

    Determines whether to generate pandas and/or polars DataFrame methods
    for a given endpoint based on the configuration and whether the
    endpoint returns a list type.

    Args:
        endpoint: The endpoint to get configuration for.
        dataframe_config: The global DataFrame configuration.

    Returns:
        DataFrameMethodConfig with generation flags and default path.

    Example:
        >>> config = get_dataframe_config_for_endpoint(endpoint, df_config)
        >>> if config.generate_pandas:
        ...     # Generate pandas method
    """
    if not dataframe_config.enabled:
        return DataFrameMethodConfig()

    # Check if this endpoint returns a list type
    returns_list = endpoint_returns_list(endpoint)

    # Get the sync function name for config lookup
    endpoint_name = endpoint.fn.name

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
