"""Pagination utilities for OtterAPI code generation.

This module provides utilities for:
- Generating the _pagination.py utility file for runtime pagination
- Determining pagination configuration for endpoints
- Building paginated endpoint functions
"""

from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING

from upath import UPath

if TYPE_CHECKING:
    from otterapi.config import PaginationConfig

__all__ = [
    'PaginationMethodConfig',
    'generate_pagination_module',
    'get_pagination_imports',
    'get_pagination_type_checking_imports',
    'get_pagination_config_for_endpoint',
    'endpoint_is_paginated',
]


# =============================================================================
# Pagination Module Generation
# =============================================================================


def generate_pagination_module(output_dir: Path | UPath) -> Path | UPath:
    """Generate the _pagination.py utility module.

    Args:
        output_dir: The output directory where the module should be written.

    Returns:
        The path to the generated file.
    """
    output_dir = UPath(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    file_path = output_dir / '_pagination.py'
    file_path.write_text(
        files('otterapi.codegen.runtime').joinpath('_pagination.py').read_text('utf-8')
    )

    return file_path


def get_pagination_imports() -> dict[str, set[str]]:
    """Get the imports needed for pagination method generation.

    Returns:
        A dictionary mapping module names to sets of imported names.
    """
    return {
        'typing': {'TYPE_CHECKING'},
        'collections.abc': {'Iterator', 'AsyncIterator'},
    }


def get_pagination_type_checking_imports() -> list[str]:
    """Get the TYPE_CHECKING imports for pagination.

    Returns:
        List of import statements to include inside TYPE_CHECKING block.
    """
    return []


# =============================================================================
# Pagination Configuration
# =============================================================================


@dataclass
class PaginationMethodConfig:
    """Configuration for pagination method generation.

    Attributes:
        style: The pagination style to use.
        offset_param: Name of offset parameter (for offset style).
        limit_param: Name of limit parameter.
        cursor_param: Name of cursor parameter (for cursor style).
        page_param: Name of page parameter (for page style).
        per_page_param: Name of per_page parameter (for page style).
        data_path: JSON path to items array in response.
        total_path: JSON path to total count in response.
        next_cursor_path: JSON path to next cursor in response.
        total_pages_path: JSON path to total pages in response.
        default_page_size: Default page size.
        max_page_size: Maximum page size.
    """

    style: str = 'offset'
    offset_param: str = 'offset'
    limit_param: str = 'limit'
    cursor_param: str = 'cursor'
    page_param: str = 'page'
    per_page_param: str = 'per_page'
    data_path: str | None = None
    total_path: str | None = None
    next_cursor_path: str | None = None
    total_pages_path: str | None = None
    default_page_size: int = 100
    max_page_size: int | None = None


def endpoint_is_paginated(
    endpoint_name: str,
    pagination_config: 'PaginationConfig',
    endpoint_parameters: list | None = None,
) -> bool:
    """Check if an endpoint is configured for pagination.

    Args:
        endpoint_name: The name of the endpoint function.
        pagination_config: The global pagination configuration.
        endpoint_parameters: Optional list of endpoint parameters for auto-detection.

    Returns:
        True if the endpoint should have pagination methods generated.
    """
    if not pagination_config.enabled:
        return False

    should_generate, _ = pagination_config.should_generate_for_endpoint(
        endpoint_name, endpoint_parameters
    )
    return should_generate


def get_pagination_config_for_endpoint(
    endpoint_name: str,
    pagination_config: 'PaginationConfig',
    endpoint_parameters: list | None = None,
) -> PaginationMethodConfig | None:
    """Get the pagination method configuration for an endpoint.

    Args:
        endpoint_name: The name of the endpoint function.
        pagination_config: The global pagination configuration.
        endpoint_parameters: Optional list of endpoint parameters for auto-detection.

    Returns:
        PaginationMethodConfig if pagination is configured, None otherwise.
    """
    if not pagination_config.enabled:
        return None

    should_generate, resolved = pagination_config.should_generate_for_endpoint(
        endpoint_name, endpoint_parameters
    )

    if not should_generate or resolved is None:
        return None

    return PaginationMethodConfig(
        style=resolved.style.value,
        offset_param=resolved.offset_param,
        limit_param=resolved.limit_param,
        cursor_param=resolved.cursor_param,
        page_param=resolved.page_param,
        per_page_param=resolved.per_page_param,
        data_path=resolved.data_path,
        total_path=resolved.total_path,
        next_cursor_path=resolved.next_cursor_path,
        total_pages_path=resolved.total_pages_path,
        default_page_size=resolved.default_page_size,
        max_page_size=resolved.max_page_size,
    )
