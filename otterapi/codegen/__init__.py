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
from otterapi.codegen.emitter import CodeEmitter, FileEmitter, StringEmitter
from otterapi.codegen.import_collector import ImportCollector
from otterapi.codegen.schema_loader import SchemaLoader
from otterapi.codegen.schema_resolver import SchemaResolver
from otterapi.codegen.type_registry import TypeInfo, TypeRegistry
from otterapi.codegen.types import (
    Endpoint,
    Parameter,
    RequestBodyInfo,
    ResponseInfo,
    Type,
    TypeGenerator,
)

__all__ = [
    # Main codegen class
    'Codegen',
    # Type generation
    'TypeGenerator',
    'Type',
    'TypeRegistry',
    'TypeInfo',
    # Schema handling
    'SchemaLoader',
    'SchemaResolver',
    # Endpoint types
    'Endpoint',
    'Parameter',
    'RequestBodyInfo',
    'ResponseInfo',
    # Code emission
    'CodeEmitter',
    'FileEmitter',
    'StringEmitter',
    'ImportCollector',
]
