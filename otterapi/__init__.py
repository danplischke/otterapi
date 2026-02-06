"""OtterAPI - Generate type-safe Python clients from OpenAPI specifications.

OtterAPI is a code generation library that creates fully-typed Python client
code from OpenAPI 3.x specifications. It generates Pydantic models for data
validation and httpx-based endpoint functions for making API requests.

Quick Start:
    >>> from otterapi import Codegen, DocumentConfig
    >>>
    >>> config = DocumentConfig(
    ...     source="https://api.example.com/openapi.json",
    ...     output="./client"
    ... )
    >>> codegen = Codegen(config)
    >>> codegen.generate()

CLI Usage:
    $ otterapi generate --source ./api.yaml --output ./client
    $ otterapi init  # Create a configuration file interactively
    $ otterapi validate ./api.yaml  # Validate an OpenAPI spec

For more information, see the documentation at:
https://github.com/yourusername/otterapi
"""

from otterapi.codegen.codegen import Codegen
from otterapi.codegen.schema import SchemaLoader, SchemaResolver
from otterapi.codegen.types import TypeGenerator, TypeRegistry
from otterapi.config import CodegenConfig, DocumentConfig, get_config
from otterapi.exceptions import (
    CodeGenerationError,
    ConfigurationError,
    EndpointGenerationError,
    OtterAPIError,
    OutputError,
    SchemaError,
    SchemaLoadError,
    SchemaReferenceError,
    SchemaValidationError,
    TypeGenerationError,
    UnsupportedFeatureError,
)

__all__ = [
    # Main classes
    'Codegen',
    'SchemaLoader',
    'SchemaResolver',
    'TypeRegistry',
    'TypeGenerator',
    # Configuration
    'CodegenConfig',
    'DocumentConfig',
    'get_config',
    # Exceptions
    'OtterAPIError',
    'SchemaError',
    'SchemaLoadError',
    'SchemaValidationError',
    'SchemaReferenceError',
    'CodeGenerationError',
    'TypeGenerationError',
    'EndpointGenerationError',
    'ConfigurationError',
    'OutputError',
    'UnsupportedFeatureError',
]

# Version is dynamically set by setuptools-scm or hatch-vcs
try:
    from otterapi._version import version as __version__
except ImportError:
    __version__ = 'unknown'
