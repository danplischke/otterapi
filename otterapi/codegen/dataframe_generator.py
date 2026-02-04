"""DataFrame utility module generation for OtterAPI.

This module generates the _dataframe.py utility file that provides
functions for converting JSON responses to pandas and polars DataFrames.
"""

from pathlib import Path

from upath import UPath

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


def generate_dataframe_module(output_dir: Path | UPath) -> Path:
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
