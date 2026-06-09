"""DataFrame conversion utilities for OtterAPI generated clients."""

from typing import Any


def _to_dict(obj: Any) -> Any:
    """Convert an object to a dictionary if it has a model_dump method (Pydantic).

    Args:
        obj: The object to convert.

    Returns:
        A dictionary if the object is a Pydantic model, otherwise the original object.
    """
    if hasattr(obj, 'model_dump'):
        return obj.model_dump()
    elif hasattr(obj, 'dict'):
        # Pydantic v1 compatibility
        return obj.dict()
    return obj


def _normalize_data(data: list | dict) -> list[dict]:
    """Normalize data to a list of dictionaries for DataFrame conversion.

    Handles:
    - Lists of Pydantic models
    - Lists of dictionaries
    - Single dictionaries
    - Single Pydantic models

    Args:
        data: The data to normalize.

    Returns:
        A list of dictionaries.
    """
    if isinstance(data, dict):
        return [data]

    if isinstance(data, list):
        if not data:
            return []
        # Check if items need conversion (Pydantic models)
        first = data[0]
        if hasattr(first, 'model_dump') or hasattr(first, 'dict'):
            return [_to_dict(item) for item in data]
        return data

    # Single Pydantic model
    if hasattr(data, 'model_dump') or hasattr(data, 'dict'):
        return [_to_dict(data)]

    raise TypeError(
        f'Cannot convert {type(data).__name__} to DataFrame. '
        f'Expected list, dict, or Pydantic model.'
    )


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
    for key in path.split('.'):
        if isinstance(current, dict):
            if key not in current:
                raise KeyError(
                    f"Key '{key}' not found in response. "
                    f"Available keys: {list(current.keys())}. Full path: {path}"
                )
            current = current[key]
        elif isinstance(current, list) and key.isdigit():
            current = current[int(key)]
        else:
            raise KeyError(
                f"Cannot access '{key}' on {type(current).__name__}. "
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
            'pandas is required for DataFrame conversion. '
            'Install with: pip install pandas'
        )

    # Extract nested data if path specified
    target_data = extract_path(data, path)

    # Normalize data to list of dicts (handles Pydantic models)
    normalized = _normalize_data(target_data)

    # Use json_normalize for nested structure support
    return pd.json_normalize(normalized)


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
            'polars is required for DataFrame conversion. '
            'Install with: pip install polars'
        )

    # Extract nested data if path specified
    target_data = extract_path(data, path)

    # Normalize data to list of dicts (handles Pydantic models)
    normalized = _normalize_data(target_data)

    return pl.DataFrame(normalized)
