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
    'DATAFRAME_MODULE_CONTENT',
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
# DataFrame Module Content
# =============================================================================

DATAFRAME_MODULE_CONTENT = '''\
"""DataFrame conversion utilities for OtterAPI generated clients."""

from typing import Any


def extract_path(data: dict | list, path: str | None) -> list | dict:
    """Extract nested data using dot notation path.

    Args:
        data: The JSON response data
        path: Dot notation path (e.g., "data.users", "response.items")

    Returns:
        The extracted data at the specified path

    Raises:
        KeyError: If the path does not exist in the data

    Examples:
        >>> extract_path({"data": {"users": [1, 2, 3]}}, "data.users")
        [1, 2, 3]
        >>> extract_path([1, 2, 3], None)
        [1, 2, 3]
    """
    if path is None:
        return data

    current = data
    for key in path.split("."):
        if isinstance(current, dict):
            if key not in current:
                raise KeyError(
                    f"Key \\'{key}\\' not found in response. "
                    f"Available keys: {list(current.keys())}. Full path: {path}"
                )
            current = current[key]
        elif isinstance(current, list) and key.isdigit():
            current = current[int(key)]
        else:
            raise KeyError(
                f"Cannot access \\'{key}\\' on {type(current).__name__}. "
                f"Full path: {path}"
            )

    return current


def to_pandas(data: list | dict, path: str | None = None):
    """Convert JSON data to a pandas DataFrame.

    Args:
        data: The JSON data to convert (dict or list)
        path: Optional dot notation path to extract data first

    Returns:
        pandas.DataFrame

    Raises:
        ImportError: If pandas is not installed
        TypeError: If data cannot be converted to DataFrame
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError(
            "pandas is required for DataFrame conversion. "
            "Install with: pip install pandas"
        )

    # Extract nested data if path specified
    target_data = extract_path(data, path)

    # Ensure we have a list for DataFrame conversion
    if isinstance(target_data, dict):
        target_data = [target_data]
    elif not isinstance(target_data, list):
        raise TypeError(
            f"Cannot convert {type(target_data).__name__} to DataFrame. "
            f"Expected list or dict."
        )

    # Use json_normalize for nested structure support
    return pd.json_normalize(target_data)


def to_polars(data: list | dict, path: str | None = None):
    """Convert JSON data to a polars DataFrame.

    Args:
        data: The JSON data to convert (dict or list)
        path: Optional dot notation path to extract data first

    Returns:
        polars.DataFrame

    Raises:
        ImportError: If polars is not installed
        TypeError: If data cannot be converted to DataFrame
    """
    try:
        import polars as pl
    except ImportError:
        raise ImportError(
            "polars is required for DataFrame conversion. "
            "Install with: pip install polars"
        )

    # Extract nested data if path specified
    target_data = extract_path(data, path)

    # Ensure we have a list for DataFrame conversion
    if isinstance(target_data, dict):
        target_data = [target_data]
    elif not isinstance(target_data, list):
        raise TypeError(
            f"Cannot convert {type(target_data).__name__} to DataFrame. "
            f"Expected list or dict."
        )

    return pl.DataFrame(target_data)
'''


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
    file_path.write_text(DATAFRAME_MODULE_CONTENT)

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
