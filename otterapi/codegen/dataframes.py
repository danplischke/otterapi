"""DataFrame utilities for OtterAPI code generation.

This module provides utilities for:
- Generating the _dataframe.py utility file for runtime DataFrame conversion
- Determining DataFrame generation configuration for endpoints
- Checking if endpoints return list types suitable for DataFrame conversion
"""

import ast
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING

from upath import UPath

if TYPE_CHECKING:
    from otterapi.codegen.types import Endpoint, Type
    from otterapi.config import DataFrameConfig

__all__ = [
    'DataFrameMethodConfig',
    'generate_dataframe_module',
    'get_dataframe_imports',
    'get_dataframe_type_checking_imports',
    'get_dataframe_config_for_endpoint',
    'get_dataframe_config_from_parts',
    'endpoint_returns_list',
    'response_type_returns_list',
]


# =============================================================================
# DataFrame Module Generation
# =============================================================================


def generate_dataframe_module(output_dir: Path | UPath) -> Path | UPath:
    """Generate the _dataframe.py utility module.

    Args:
        output_dir: The output directory where the module should be written.

    Returns:
        The path to the generated file.
    """
    output_dir = UPath(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    file_path = output_dir / '_dataframe.py'
    file_path.write_text(
        files('otterapi.codegen.runtime').joinpath('_dataframe.py').read_text('utf-8')
    )

    return file_path


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
