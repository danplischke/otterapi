"""Code generation module for OtterAPI.

This module provides the core code generation functionality for creating
Python client code from OpenAPI specifications.

Main Components:
    - Codegen: The main orchestrator for code generation
    - TypeGenerator: Generates Pydantic models from OpenAPI schemas
    - SchemaLoader: Loads OpenAPI schemas from URLs or files
    - SchemaResolver: Resolves $ref references in schemas
    - TypeRegistry: Manages generated types and their dependencies
    - CodeEmitter: Handles output of generated code

Example:
    >>> from otterapi.codegen import Codegen
    >>> from otterapi.config import DocumentConfig
    >>>
    >>> config = DocumentConfig(
    ...     source="./openapi.json",
    ...     output="./client"
    ... )
    >>> codegen = Codegen(config)
    >>> codegen.generate()
"""

from otterapi.codegen.codegen import Codegen
from otterapi.codegen.schema import SchemaLoader, SchemaResolver
from otterapi.codegen.types import (
    Endpoint,
    Parameter,
    RequestBodyInfo,
    ResponseInfo,
    Type,
    TypeGenerator,
)

__all__ = [
    'Codegen',
    'TypeGenerator',
    'Type',
    'Endpoint',
    'Parameter',
    'RequestBodyInfo',
    'ResponseInfo',
    'SchemaLoader',
    'SchemaResolver',
]
