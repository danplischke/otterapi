"""Configuration management for OtterAPI.

This module provides configuration loading and validation for OtterAPI,
supporting multiple configuration formats (YAML, JSON, TOML) and
environment variable overrides.
"""

from __future__ import annotations

import json
import os
import re
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings

DEFAULT_FILENAMES = ['otter.yaml', 'otter.yml', 'otter.json']


def _expand_env_vars(value: str) -> str:
    """Expand environment variables in a string.

    Supports both ${VAR} and ${VAR:-default} syntax.

    Args:
        value: String potentially containing environment variables.

    Returns:
        String with environment variables expanded.
    """
    # Pattern matches ${VAR} or ${VAR:-default}
    pattern = r'\$\{([^}:]+)(?::-([^}]*))?\}'

    def replacer(match: re.Match) -> str:
        var_name = match.group(1)
        default = match.group(2)
        return os.environ.get(var_name, default if default is not None else '')

    return re.sub(pattern, replacer, value)


def _expand_env_vars_recursive(obj: Any) -> Any:
    """Recursively expand environment variables in a data structure.

    Args:
        obj: Data structure (dict, list, or scalar).

    Returns:
        Data structure with all string values having env vars expanded.
    """
    if isinstance(obj, dict):
        return {k: _expand_env_vars_recursive(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_expand_env_vars_recursive(item) for item in obj]
    elif isinstance(obj, str):
        return _expand_env_vars(obj)
    else:
        return obj


class SplitStrategy(str, Enum):
    """Strategy for splitting endpoints into modules."""

    NONE = 'none'  # No splitting, all endpoints in one file
    PATH = 'path'  # Split based on URL path segments
    TAG = 'tag'  # Split based on OpenAPI tags
    HYBRID = 'hybrid'  # Combine tag and path strategies
    CUSTOM = 'custom'  # Use custom module_map only


class EndpointDataFrameConfig(BaseModel):
    """Per-endpoint DataFrame configuration.

    Allows overriding the default DataFrame settings for specific endpoints.

    Attributes:
        enabled: Override whether to generate DataFrame methods for this endpoint.
        path: JSON path to extract data from response (e.g., "data.users").
        pandas: Override whether to generate _df method (pandas).
        polars: Override whether to generate _pl method (polars).
    """

    enabled: bool | None = Field(
        default=None,
        description='Override whether to generate DataFrame methods.',
    )

    path: str | None = Field(
        default=None,
        description='JSON path to extract data from response.',
    )

    pandas: bool | None = Field(
        default=None,
        description='Override whether to generate _df method (pandas).',
    )

    polars: bool | None = Field(
        default=None,
        description='Override whether to generate _pl method (polars).',
    )

    model_config = {'extra': 'forbid'}


class PaginationStyle(str, Enum):
    """Style of pagination used by an API endpoint."""

    OFFSET = 'offset'  # Offset/limit based pagination
    CURSOR = 'cursor'  # Cursor-based pagination
    PAGE = 'page'  # Page number based pagination
    LINK = 'link'  # Link header pagination (RFC 5988)


class EndpointPaginationConfig(BaseModel):
    """Per-endpoint pagination configuration.

    Allows configuring pagination behavior for specific endpoints.

    Attributes:
        enabled: Override whether to generate pagination methods for this endpoint.
        style: Pagination style for this endpoint.
        offset_param: Name of offset parameter (for offset style).
        limit_param: Name of limit parameter.
        cursor_param: Name of cursor parameter (for cursor style).
        page_param: Name of page parameter (for page style).
        per_page_param: Name of per_page parameter (for page style).
        data_path: JSON path to items array in response.
        total_path: JSON path to total count in response.
        next_cursor_path: JSON path to next cursor in response (for cursor style).
        total_pages_path: JSON path to total pages in response (for page style).
        default_page_size: Default page size for this endpoint.
        max_page_size: Maximum page size for this endpoint.
    """

    enabled: bool | None = Field(
        default=None,
        description='Override whether to generate pagination methods.',
    )

    style: PaginationStyle | Literal['offset', 'cursor', 'page', 'link'] | None = Field(
        default=None,
        description='Pagination style for this endpoint.',
    )

    # Parameter mappings
    offset_param: str | None = Field(
        default=None,
        description='Name of offset parameter.',
    )

    limit_param: str | None = Field(
        default=None,
        description='Name of limit parameter.',
    )

    cursor_param: str | None = Field(
        default=None,
        description='Name of cursor parameter.',
    )

    page_param: str | None = Field(
        default=None,
        description='Name of page parameter.',
    )

    per_page_param: str | None = Field(
        default=None,
        description='Name of per_page parameter.',
    )

    # Response mappings
    data_path: str | None = Field(
        default=None,
        description='JSON path to items array in response.',
    )

    total_path: str | None = Field(
        default=None,
        description='JSON path to total count in response.',
    )

    next_cursor_path: str | None = Field(
        default=None,
        description='JSON path to next cursor in response.',
    )

    total_pages_path: str | None = Field(
        default=None,
        description='JSON path to total pages in response.',
    )

    # Limits
    default_page_size: int | None = Field(
        default=None,
        description='Default page size for this endpoint.',
    )

    max_page_size: int | None = Field(
        default=None,
        description='Maximum page size for this endpoint.',
    )

    model_config = {'extra': 'forbid'}

    @field_validator('style', mode='before')
    @classmethod
    def normalize_style(cls, v: Any) -> PaginationStyle | None:
        """Convert string style to enum."""
        if v is None:
            return None
        if isinstance(v, str):
            return PaginationStyle(v.lower())
        return v


class PaginationConfig(BaseModel):
    """Global pagination configuration.

    When enabled, generates pagination methods for configured endpoints.

    Attributes:
        enabled: Enable pagination method generation.
        auto_detect: Automatically detect and enable pagination for endpoints
            that have pagination parameters (offset/limit, cursor, page/per_page).
        default_style: Default pagination style when not explicitly configured.
        default_page_size: Default page size for iteration.
        default_data_path: Default JSON path to items array.
        endpoints: Per-endpoint pagination configuration.
    """

    enabled: bool = Field(
        default=False,
        description='Enable pagination method generation.',
    )

    auto_detect: bool = Field(
        default=True,
        description=(
            'Automatically detect and enable pagination for endpoints '
            'that have pagination parameters (offset/limit, cursor, page/per_page). '
            'When enabled, endpoints with matching parameters will automatically '
            'get pagination methods generated without explicit configuration.'
        ),
    )

    default_style: PaginationStyle | Literal['offset', 'cursor', 'page', 'link'] = (
        Field(
            default=PaginationStyle.OFFSET,
            description='Default pagination style.',
        )
    )

    default_page_size: int = Field(
        default=100,
        description='Default page size for iteration.',
    )

    default_data_path: str | None = Field(
        default=None,
        description='Default JSON path to items array.',
    )

    default_total_path: str | None = Field(
        default=None,
        description='Default JSON path to total count in response.',
    )

    # Default parameter names
    default_offset_param: str = Field(
        default='offset',
        description='Default name of offset parameter.',
    )

    default_limit_param: str = Field(
        default='limit',
        description='Default name of limit parameter.',
    )

    default_cursor_param: str = Field(
        default='cursor',
        description='Default name of cursor parameter.',
    )

    default_page_param: str = Field(
        default='page',
        description='Default name of page parameter.',
    )

    default_per_page_param: str = Field(
        default='per_page',
        description='Default name of per_page parameter.',
    )

    # Per-endpoint configuration
    endpoints: dict[str, EndpointPaginationConfig] = Field(
        default_factory=dict,
        description='Per-endpoint pagination configuration.',
    )

    model_config = {'extra': 'forbid'}

    @field_validator('default_style', mode='before')
    @classmethod
    def normalize_default_style(cls, v: Any) -> PaginationStyle:
        """Convert string style to enum."""
        if isinstance(v, str):
            return PaginationStyle(v.lower())
        return v

    def should_generate_for_endpoint(
        self,
        endpoint_name: str,
        endpoint_parameters: list | None = None,
    ) -> tuple[bool, ResolvedPaginationConfig | None]:
        """Determine if pagination methods should be generated for an endpoint.

        Args:
            endpoint_name: The name of the endpoint function.
            endpoint_parameters: Optional list of endpoint parameters for auto-detection.

        Returns:
            A tuple of (should_generate, resolved_config) indicating
            whether to generate and the resolved configuration.
        """
        if not self.enabled:
            return False, None

        # Check for endpoint-specific configuration
        endpoint_config = self.endpoints.get(endpoint_name)

        if endpoint_config is None:
            # No explicit config - check if auto_detect is enabled
            if not self.auto_detect or endpoint_parameters is None:
                return False, None

            # Auto-detect pagination based on parameters
            detected_style = self._detect_pagination_style(endpoint_parameters)
            if detected_style is None:
                return False, None

            # Use defaults for auto-detected endpoints
            resolved = ResolvedPaginationConfig(
                style=detected_style,
                offset_param=self.default_offset_param,
                limit_param=self.default_limit_param,
                cursor_param=self.default_cursor_param,
                page_param=self.default_page_param,
                per_page_param=self.default_per_page_param,
                data_path=self.default_data_path,
                total_path=self.default_total_path,
                next_cursor_path=None,
                total_pages_path=None,
                default_page_size=self.default_page_size,
                max_page_size=None,
            )
            return True, resolved

        # Check if explicitly disabled
        if endpoint_config.enabled is False:
            return False, None

        # Resolve the configuration with defaults
        style = endpoint_config.style or self.default_style
        if isinstance(style, str):
            style = PaginationStyle(style.lower())

        resolved = ResolvedPaginationConfig(
            style=style,
            offset_param=endpoint_config.offset_param or self.default_offset_param,
            limit_param=endpoint_config.limit_param or self.default_limit_param,
            cursor_param=endpoint_config.cursor_param or self.default_cursor_param,
            page_param=endpoint_config.page_param or self.default_page_param,
            per_page_param=endpoint_config.per_page_param
            or self.default_per_page_param,
            data_path=endpoint_config.data_path or self.default_data_path,
            total_path=endpoint_config.total_path or self.default_total_path,
            next_cursor_path=endpoint_config.next_cursor_path,
            total_pages_path=endpoint_config.total_pages_path,
            default_page_size=endpoint_config.default_page_size
            or self.default_page_size,
            max_page_size=endpoint_config.max_page_size,
        )

        return True, resolved

    def _detect_pagination_style(self, parameters: list) -> PaginationStyle | None:
        """Detect pagination style based on endpoint parameters.

        Args:
            parameters: List of endpoint parameter objects.

        Returns:
            Detected PaginationStyle or None if no pagination detected.
        """
        param_names = {p.name for p in parameters if hasattr(p, 'name')}

        # Check for offset-based pagination (offset + limit)
        if (
            self.default_offset_param in param_names
            and self.default_limit_param in param_names
        ):
            return PaginationStyle.OFFSET

        # Check for cursor-based pagination (cursor + limit)
        if (
            self.default_cursor_param in param_names
            and self.default_limit_param in param_names
        ):
            return PaginationStyle.CURSOR

        # Check for page-based pagination (page + per_page)
        if (
            self.default_page_param in param_names
            and self.default_per_page_param in param_names
        ):
            return PaginationStyle.PAGE

        return None


class ResolvedPaginationConfig(BaseModel):
    """Resolved pagination configuration with all defaults applied.

    This is the configuration used during code generation after
    merging endpoint-specific config with global defaults.
    """

    style: PaginationStyle
    offset_param: str
    limit_param: str
    cursor_param: str
    page_param: str
    per_page_param: str
    data_path: str | None
    total_path: str | None
    next_cursor_path: str | None
    total_pages_path: str | None
    default_page_size: int
    max_page_size: int | None

    model_config = {'extra': 'forbid'}


class DataFrameConfig(BaseModel):
    """Configuration for DataFrame conversion methods.

    When enabled, generates additional endpoint methods that return
    pandas DataFrames (_df suffix) and/or polars DataFrames (_pl suffix).

    Attributes:
        enabled: Enable DataFrame method generation.
        pandas: Generate _df methods returning pandas DataFrames.
        polars: Generate _pl methods returning polars DataFrames.
        default_path: Default JSON path for extracting list data from responses.
        include_all: Generate DataFrame methods for all list-returning endpoints.
        endpoints: Per-endpoint configuration overrides.
    """

    enabled: bool = Field(
        default=False,
        description='Enable DataFrame method generation.',
    )

    pandas: bool = Field(
        default=True,
        description='Generate _df methods (pandas DataFrames).',
    )

    polars: bool = Field(
        default=False,
        description='Generate _pl methods (polars DataFrames).',
    )

    default_path: str | None = Field(
        default=None,
        description='Default JSON path for extracting list data.',
    )

    include_all: bool = Field(
        default=True,
        description='Generate DataFrame methods for all list-returning endpoints.',
    )

    endpoints: dict[str, EndpointDataFrameConfig] = Field(
        default_factory=dict,
        description='Per-endpoint configuration overrides.',
    )

    model_config = {'extra': 'forbid'}

    def should_generate_for_endpoint(
        self,
        endpoint_name: str,
        returns_list: bool = True,
    ) -> tuple[bool, bool, str | None]:
        """Determine if DataFrame methods should be generated for an endpoint.

        Args:
            endpoint_name: The name of the endpoint function.
            returns_list: Whether the endpoint returns a list type.

        Returns:
            A tuple of (generate_pandas, generate_polars, path) indicating
            which methods to generate and what path to use.
        """
        if not self.enabled:
            return False, False, None

        # Check for endpoint-specific override
        endpoint_config = self.endpoints.get(endpoint_name)

        if endpoint_config is not None:
            # Endpoint has specific config
            if endpoint_config.enabled is False:
                return False, False, None

            # Determine pandas generation
            gen_pandas = (
                endpoint_config.pandas
                if endpoint_config.pandas is not None
                else self.pandas
            )

            # Determine polars generation
            gen_polars = (
                endpoint_config.polars
                if endpoint_config.polars is not None
                else self.polars
            )

            # Determine path
            path = (
                endpoint_config.path
                if endpoint_config.path is not None
                else self.default_path
            )

            # If endpoint is explicitly enabled, generate regardless of return type
            if endpoint_config.enabled is True:
                return gen_pandas, gen_polars, path

            # Otherwise, respect include_all and returns_list
            if self.include_all and returns_list:
                return gen_pandas, gen_polars, path

            return False, False, None

        # No endpoint-specific config - use defaults
        if not self.include_all:
            return False, False, None

        if not returns_list:
            return False, False, None

        return self.pandas, self.polars, self.default_path


class EndpointResponseUnwrapConfig(BaseModel):
    """Per-endpoint response unwrap configuration.

    Allows overriding the default response unwrap settings for specific endpoints.

    Attributes:
        enabled: Override whether to unwrap response for this endpoint.
        data_path: JSON path to extract data from response.
    """

    enabled: bool | None = Field(
        default=None,
        description='Override whether to unwrap response for this endpoint.',
    )

    data_path: str | None = Field(
        default=None,
        description='JSON path to extract data from response.',
    )

    model_config = {'extra': 'forbid'}


class ResponseUnwrapConfig(BaseModel):
    """Configuration for response unwrapping.

    When enabled, generates endpoint functions that extract and return
    just the data portion of envelope-style responses, making non-paginated
    endpoints consistent with paginated ones.

    Attributes:
        enabled: Enable response unwrapping.
        data_path: Default JSON path to extract data from responses.
        endpoints: Per-endpoint configuration overrides.
    """

    enabled: bool = Field(
        default=False,
        description='Enable response unwrapping.',
    )

    data_path: str = Field(
        default='data',
        description='Default JSON path to extract data from responses.',
    )

    endpoints: dict[str, EndpointResponseUnwrapConfig] = Field(
        default_factory=dict,
        description='Per-endpoint configuration overrides.',
    )

    model_config = {'extra': 'forbid'}

    def get_unwrap_config_for_endpoint(
        self,
        endpoint_name: str,
    ) -> tuple[bool, str | None]:
        """Determine if response should be unwrapped for an endpoint.

        Args:
            endpoint_name: The name of the endpoint function.

        Returns:
            A tuple of (should_unwrap, data_path).
        """
        if not self.enabled:
            return False, None

        # Check for endpoint-specific override
        endpoint_config = self.endpoints.get(endpoint_name)

        if endpoint_config is not None:
            if endpoint_config.enabled is False:
                return False, None

            path = endpoint_config.data_path or self.data_path
            return True, path

        # Use defaults
        return True, self.data_path


class ModuleDefinition(BaseModel):
    """Definition for a single module or module group.

    Supports both flat modules (with just paths) and nested module hierarchies.

    Attributes:
        paths: List of glob patterns to match endpoint paths.
        modules: Nested submodules (recursive structure).
        strip_prefix: Prefix to strip from paths in this group.
        package_prefix: Prefix for generated imports.
        file_name: Override for the generated file name.
        description: Module docstring.
    """

    paths: list[str] = Field(default_factory=list)
    modules: dict[str, ModuleDefinition] = Field(default_factory=dict)
    strip_prefix: str | None = None
    package_prefix: str | None = None
    file_name: str | None = None
    description: str | None = None

    model_config = {'extra': 'forbid'}


# Rebuild model to resolve forward references
ModuleDefinition.model_rebuild()


# Type alias for module_map values which can be:
# - A list of path patterns (shorthand)
# - A single path pattern string (shorthand)
# - A full ModuleDefinition
ModuleMapValue = ModuleDefinition | list[str] | str | dict


class ModuleSplitConfig(BaseModel):
    """Configuration for splitting endpoints into submodules.

    Attributes:
        enabled: Whether module splitting is enabled.
        strategy: The splitting strategy to use.
        global_strip_prefixes: Prefixes to strip from all paths before matching.
        path_depth: Number of path segments to use for path-based strategy.
        min_endpoints: Minimum endpoints required per module before consolidating.
        fallback_module: Module name for endpoints that don't match any rule.
        module_map: Custom mapping of module names to path patterns or definitions.
        flat_structure: If True, generate flat file structure instead of directories.
        split_models: Whether to split models per module (advanced).
        shared_models_module: Module name for shared models when split_models is True.
    """

    enabled: bool = False
    strategy: SplitStrategy | Literal['none', 'path', 'tag', 'hybrid', 'custom'] = (
        SplitStrategy.HYBRID
    )
    global_strip_prefixes: list[str] = Field(
        default_factory=lambda: ['/api', '/api/v1', '/api/v2', '/api/v3']
    )
    path_depth: int = Field(default=1, ge=1, le=5)
    min_endpoints: int = Field(default=2, ge=1)
    fallback_module: str = 'common'
    module_map: dict[str, ModuleMapValue] = Field(default_factory=dict)
    flat_structure: bool = False
    split_models: bool = False
    shared_models_module: str = '_models'

    model_config = {'extra': 'forbid'}

    @field_validator('strategy', mode='before')
    @classmethod
    def normalize_strategy(cls, v: Any) -> SplitStrategy:
        """Convert string strategy to enum."""
        if isinstance(v, str):
            return SplitStrategy(v.lower())
        return v

    @field_validator('module_map', mode='before')
    @classmethod
    def normalize_module_map_before(cls, v: Any) -> dict[str, ModuleDefinition]:
        """Normalize shorthand module_map syntax to full ModuleDefinition objects.

        Handles:
        - {"users": ["/user/*"]} → ModuleDefinition(paths=["/user/*"])
        - {"users": "/user/*"} → ModuleDefinition(paths=["/user/*"])
        - Nested dicts without paths/modules → nested ModuleDefinition
        """
        if isinstance(v, dict):
            return _normalize_module_map(v)
        return v


def _is_module_definition_dict(value: dict) -> bool:
    """Check if a dict looks like a ModuleDefinition (has known keys)."""
    known_keys = {
        'paths',
        'modules',
        'strip_prefix',
        'package_prefix',
        'file_name',
        'description',
    }
    return bool(set(value.keys()) & known_keys)


def _normalize_module_map(
    module_map: dict[str, ModuleMapValue],
) -> dict[str, ModuleDefinition]:
    """Recursively normalize module_map values to ModuleDefinition objects."""
    normalized: dict[str, ModuleDefinition] = {}

    for key, value in module_map.items():
        if isinstance(value, ModuleDefinition):
            # Already a ModuleDefinition, but recursively normalize its modules
            if value.modules:
                value = value.model_copy(
                    update={'modules': _normalize_module_map(value.modules)}
                )
            normalized[key] = value
        elif isinstance(value, str):
            # Single path pattern string → ModuleDefinition with one path
            normalized[key] = ModuleDefinition(paths=[value])
        elif isinstance(value, list):
            # List of path patterns → ModuleDefinition with paths
            normalized[key] = ModuleDefinition(paths=value)
        elif isinstance(value, dict):
            # Dict that's not yet a ModuleDefinition - could be:
            # 1. A dict that should be parsed as ModuleDefinition (has known keys)
            # 2. A dict of nested modules (shorthand for modules={...})
            if _is_module_definition_dict(value):
                # Try to parse as ModuleDefinition
                definition = ModuleDefinition.model_validate(value)
                if definition.modules:
                    definition = definition.model_copy(
                        update={'modules': _normalize_module_map(definition.modules)}
                    )
                normalized[key] = definition
            else:
                # Treat the dict as a nested module structure
                # This handles cases like: {"identity": {"users": [...], "auth": [...]}}
                nested_modules = _normalize_module_map(value)
                normalized[key] = ModuleDefinition(modules=nested_modules)
        else:
            raise ValueError(
                f"Invalid module_map value for '{key}': expected str, list, dict, "
                f'or ModuleDefinition, got {type(value).__name__}'
            )

    return normalized


class DocumentConfig(BaseModel):
    """Configuration for a single OpenAPI document to be processed.

    Attributes:
        source: Path or URL to the OpenAPI document.
        output: Output directory for the generated code.
        base_url: Optional base URL override for the API.
        models_file: Name of the generated models file.
        models_import_path: Optional import path for models in endpoints.
        endpoints_file: Name of the generated endpoints file.
        generate_async: Whether to generate async endpoint functions.
        generate_sync: Whether to generate sync endpoint functions.
        client_class_name: Optional name for a generated client class.
        module_split: Configuration for splitting endpoints into submodules.
    """

    source: str = Field(..., description='Path or URL to the OpenAPI document.')

    base_url: str | None = Field(
        None,
        description='Optional base URL to override servers defined in the OpenAPI document.',
    )

    output: str = Field(..., description='Output directory for the generated code.')

    include_paths: list[str] | None = Field(
        default=None,
        description=(
            'List of path patterns to include. Only endpoints matching these patterns '
            'will be generated. Supports glob patterns (e.g., "/api/v1/users/*"). '
            'If None, all paths are included.'
        ),
    )

    exclude_paths: list[str] | None = Field(
        default=None,
        description=(
            'List of path patterns to exclude. Endpoints matching these patterns '
            'will be skipped. Supports glob patterns (e.g., "/internal/*"). '
            'Applied after include_paths filtering.'
        ),
    )

    models_file: str = Field('models.py', description='File name for generated models.')

    models_import_path: str | None = Field(
        None, description='Optional import path for generated models.'
    )

    endpoints_file: str = Field(
        'endpoints.py', description='File name for generated endpoints.'
    )

    generate_async: bool = Field(
        True, description='Whether to generate async endpoint functions.'
    )

    generate_sync: bool = Field(
        True, description='Whether to generate sync endpoint functions.'
    )

    client_class_name: str | None = Field(
        None, description='Optional name for a generated client class.'
    )

    module_split: ModuleSplitConfig = Field(
        default_factory=ModuleSplitConfig,
        description='Configuration for splitting endpoints into submodules.',
    )

    dataframe: DataFrameConfig = Field(
        default_factory=DataFrameConfig,
        description='Configuration for DataFrame conversion methods.',
    )

    pagination: PaginationConfig = Field(
        default_factory=PaginationConfig,
        description='Configuration for automatic pagination.',
    )

    response_unwrap: ResponseUnwrapConfig = Field(
        default_factory=ResponseUnwrapConfig,
        description='Configuration for response unwrapping.',
    )

    @field_validator('source')
    @classmethod
    def validate_source(cls, v: str) -> str:
        """Validate that source is a non-empty string."""
        if not v or not v.strip():
            raise ValueError('source cannot be empty')
        return v.strip()

    @field_validator('output')
    @classmethod
    def validate_output(cls, v: str) -> str:
        """Validate that output is a non-empty string."""
        if not v or not v.strip():
            raise ValueError('output cannot be empty')
        return v.strip()

    @field_validator('models_file', 'endpoints_file')
    @classmethod
    def validate_filename(cls, v: str) -> str:
        """Validate that file names end with .py."""
        if not v.endswith('.py'):
            raise ValueError(f'File name must end with .py, got: {v}')
        return v


class CodegenConfig(BaseSettings):
    """Main configuration for OtterAPI code generation.

    Attributes:
        documents: List of OpenAPI documents to process.
        generate_endpoints: Whether to generate endpoint functions.
        format_output: Whether to format generated code with black/ruff.
        validate_output: Whether to validate generated code syntax.
        create_py_typed: Whether to create py.typed marker files.
    """

    documents: list[DocumentConfig] = Field(
        ..., description='List of OpenAPI documents to process.'
    )

    generate_endpoints: bool = Field(
        True, description='Whether to generate endpoint functions.'
    )

    format_output: bool = Field(
        True, description='Whether to format generated code with black/ruff.'
    )

    validate_output: bool = Field(
        True, description='Whether to validate generated code syntax.'
    )

    create_py_typed: bool = Field(
        True, description='Whether to create py.typed marker files.'
    )

    @field_validator('documents')
    @classmethod
    def validate_documents(cls, v: list[DocumentConfig]) -> list[DocumentConfig]:
        """Validate that at least one document is configured."""
        if not v:
            raise ValueError('At least one document must be configured')
        return v

    model_config = {
        'env_prefix': 'OTTER_',
        'env_nested_delimiter': '__',
    }


def load_yaml(path: str | Path) -> dict:
    """Load configuration from a YAML file.

    Args:
        path: Path to the YAML file.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        yaml.YAMLError: If the file is not valid YAML.
    """
    import yaml

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'Configuration file not found: {path}')

    content = path.read_text(encoding='utf-8')
    data = yaml.safe_load(content)

    # Expand environment variables
    return _expand_env_vars_recursive(data)


def load_json(path: str | Path) -> dict:
    """Load configuration from a JSON file.

    Args:
        path: Path to the JSON file.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'Configuration file not found: {path}')

    content = path.read_text(encoding='utf-8')
    data = json.loads(content)

    # Expand environment variables
    return _expand_env_vars_recursive(data)


def load_toml(path: str | Path) -> dict:
    """Load configuration from a TOML file (pyproject.toml).

    Args:
        path: Path to the TOML file.

    Returns:
        Parsed configuration dictionary for the otterapi tool section.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        KeyError: If the file doesn't contain otterapi configuration.
    """
    import tomllib

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'Configuration file not found: {path}')

    content = path.read_text(encoding='utf-8')
    data = tomllib.loads(content)

    # Extract tool.otterapi section
    tools = data.get('tool', {})
    if 'otterapi' not in tools:
        raise KeyError(f'No [tool.otterapi] section found in {path}')

    # Expand environment variables
    return _expand_env_vars_recursive(tools['otterapi'])


def load_config_file(path: str | Path) -> dict:
    """Load configuration from a file, auto-detecting the format.

    Args:
        path: Path to the configuration file.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If the file format is not supported.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f'Configuration file not found: {path}')

    suffix = path.suffix.lower()

    if suffix in ('.yaml', '.yml'):
        return load_yaml(path)
    elif suffix == '.json':
        return load_json(path)
    elif suffix == '.toml':
        return load_toml(path)
    else:
        # Try to auto-detect based on content
        content = path.read_text(encoding='utf-8').strip()
        if content.startswith('{'):
            return load_json(path)
        elif content.startswith('['):
            return load_toml(path)
        else:
            return load_yaml(path)


def get_config(path: str | None = None) -> CodegenConfig:
    """Load OtterAPI configuration from a file or environment.

    This function attempts to load configuration in the following order:
    1. From the specified path (if provided)
    2. From default config files in the current directory
    3. From pyproject.toml [tool.otterapi] section
    4. From environment variables

    Args:
        path: Optional path to a configuration file.

    Returns:
        Validated CodegenConfig object.

    Raises:
        FileNotFoundError: If no configuration can be found.
        pydantic.ValidationError: If the configuration is invalid.
    """
    # If path is specified, use it directly
    if path:
        data = load_config_file(path)
        return CodegenConfig.model_validate(data)

    cwd = Path.cwd()

    # Try default config files
    for filename in DEFAULT_FILENAMES:
        config_path = cwd / filename
        if config_path.exists():
            data = load_config_file(config_path)
            return CodegenConfig.model_validate(data)

    # Try pyproject.toml
    pyproject_path = cwd / 'pyproject.toml'
    if pyproject_path.exists():
        try:
            data = load_toml(pyproject_path)
            return CodegenConfig.model_validate(data)
        except KeyError:
            pass  # No otterapi section, continue looking

    # Try to build from environment variables
    env_source = os.environ.get('OTTER_SOURCE')
    env_output = os.environ.get('OTTER_OUTPUT')

    if env_source and env_output:
        return CodegenConfig(
            documents=[
                DocumentConfig(
                    source=env_source,
                    output=env_output,
                    base_url=os.environ.get('OTTER_BASE_URL'),
                    models_file=os.environ.get('OTTER_MODELS_FILE', 'models.py'),
                    endpoints_file=os.environ.get(
                        'OTTER_ENDPOINTS_FILE', 'endpoints.py'
                    ),
                )
            ]
        )

    raise FileNotFoundError(
        'No configuration found. Create an otter.yml file, add [tool.otterapi] '
        'to pyproject.toml, or set OTTER_SOURCE and OTTER_OUTPUT environment variables.'
    )


def create_default_config() -> dict:
    """Create a default configuration dictionary.

    Returns:
        Dictionary with default configuration values.
    """
    return {
        'documents': [
            {
                'source': 'https://petstore3.swagger.io/api/v3/openapi.json',
                'output': './client',
                'models_file': 'models.py',
                'endpoints_file': 'endpoints.py',
                'generate_async': True,
                'generate_sync': True,
            }
        ],
        'format_output': True,
        'validate_output': True,
        'create_py_typed': True,
    }
