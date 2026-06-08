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

# ``X as X`` is the PEP 484 / ruff signal for "this is a deliberate
# re-export". Wave 3.15 (issue #3 item 15) keeps these symbols reachable
# via ``otterapi.codegen.X`` for backwards compatibility while removing
# them from ``__all__``.
from otterapi.codegen.ast_utils import ImportCollector as ImportCollector
from otterapi.codegen.codegen import Codegen as Codegen
from otterapi.codegen.dataframes import (
    DataFrameMethodConfig as DataFrameMethodConfig,
    generate_dataframe_module as generate_dataframe_module,
    get_dataframe_config_for_endpoint as get_dataframe_config_for_endpoint,
)
from otterapi.codegen.emitter import (
    CodeEmitter as CodeEmitter,
    FileEmitter as FileEmitter,
    StringEmitter as StringEmitter,
)
from otterapi.codegen.endpoints import (
    DataFrameLibrary as DataFrameLibrary,
    EndpointFunctionConfig as EndpointFunctionConfig,
    EndpointFunctionFactory as EndpointFunctionFactory,
    EndpointMode as EndpointMode,
    FunctionSignature as FunctionSignature,
    FunctionSignatureBuilder as FunctionSignatureBuilder,
    ParameterASTBuilder as ParameterASTBuilder,
)
from otterapi.codegen.schema import (
    SchemaLoader as SchemaLoader,
    SchemaResolver as SchemaResolver,
)
from otterapi.codegen.splitting import (
    EmittedModule as EmittedModule,
    ModuleMapResolver as ModuleMapResolver,
    ModuleTree as ModuleTree,
    ModuleTreeBuilder as ModuleTreeBuilder,
    ResolvedModule as ResolvedModule,
    SplitModuleEmitter as SplitModuleEmitter,
    build_module_tree as build_module_tree,
)
from otterapi.codegen.types import (
    Endpoint as Endpoint,
    ModelNameCollector as ModelNameCollector,
    Parameter as Parameter,
    RequestBodyInfo as RequestBodyInfo,
    ResponseInfo as ResponseInfo,
    Type as Type,
    TypeGenerator as TypeGenerator,
    TypeInfo as TypeInfo,
    TypeRegistry as TypeRegistry,
    collect_used_model_names as collect_used_model_names,
)

# ---------------------------------------------------------------------------
# Stable public surface (Wave 3.15, issue #3 item 15)
# ---------------------------------------------------------------------------
# Only the names listed in ``__all__`` are part of the supported public API.
# Other names imported above (``ImportCollector``, ``EndpointFunctionFactory``,
# ``ParameterASTBuilder``, the splitting internals, etc.) remain accessible
# via direct attribute access for backwards compatibility but are considered
# internal -- they may move or change shape between minor releases. New
# integrations should import only from the list below.
__all__ = [
    # Main codegen class
    'Codegen',
    # Type generation -- describing operations, parameters, response shapes.
    'TypeGenerator',
    'Type',
    'Endpoint',
    'Parameter',
    'RequestBodyInfo',
    'ResponseInfo',
    # Schema handling -- loading + resolving an OpenAPI document.
    'SchemaLoader',
    'SchemaResolver',
]
