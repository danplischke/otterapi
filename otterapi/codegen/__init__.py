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

from otterapi.codegen.ast_utils import ImportCollector
from otterapi.codegen.codegen import Codegen

# Re-export from dataframes module
from otterapi.codegen.dataframes import (
    DataFrameMethodConfig,
    generate_dataframe_module,
    get_dataframe_config_for_endpoint,
)
from otterapi.codegen.emitter import CodeEmitter, FileEmitter, StringEmitter

# Re-export from endpoints module
from otterapi.codegen.endpoints import (
    DataFrameLibrary,
    EndpointFunctionConfig,
    EndpointFunctionFactory,
    EndpointMode,
    FunctionSignature,
    FunctionSignatureBuilder,
    ParameterASTBuilder,
)
from otterapi.codegen.schema import SchemaLoader, SchemaResolver

# Re-export from splitting module
from otterapi.codegen.splitting import (
    EmittedModule,
    ModuleMapResolver,
    ModuleTree,
    ModuleTreeBuilder,
    ResolvedModule,
    SplitModuleEmitter,
    build_module_tree,
)
from otterapi.codegen.types import (
    Endpoint,
    ModelNameCollector,
    Parameter,
    RequestBodyInfo,
    ResponseInfo,
    Type,
    TypeGenerator,
    TypeInfo,
    TypeRegistry,
    collect_used_model_names,
)

__all__ = [
    # Main codegen class
    'Codegen',
    # Type generation
    'TypeGenerator',
    'Type',
    'TypeRegistry',
    'TypeInfo',
    'ModelNameCollector',
    'collect_used_model_names',
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
    # Endpoint building
    'EndpointFunctionConfig',
    'EndpointFunctionFactory',
    'EndpointMode',
    'DataFrameLibrary',
    'FunctionSignature',
    'FunctionSignatureBuilder',
    'ParameterASTBuilder',
    # DataFrame utilities
    'DataFrameMethodConfig',
    'generate_dataframe_module',
    'get_dataframe_config_for_endpoint',
    # Module splitting
    'ModuleTree',
    'ModuleTreeBuilder',
    'ModuleMapResolver',
    'ResolvedModule',
    'EmittedModule',
    'SplitModuleEmitter',
    'build_module_tree',
]
