"""Builders package for unified AST construction.

This package contains builder classes that provide a single source of truth
for constructing AST nodes used across the codegen module.
"""

from otterapi.codegen.builders.endpoint_factory import (
    DataFrameLibrary,
    EndpointFunctionConfig,
    EndpointFunctionFactory,
    EndpointMode,
    build_delegating_dataframe_fn,
    build_delegating_endpoint_fn,
    build_standalone_dataframe_fn,
    build_standalone_endpoint_fn,
)
from otterapi.codegen.builders.model_collector import ModelNameCollector
from otterapi.codegen.builders.parameter_builder import ParameterASTBuilder
from otterapi.codegen.builders.signature_builder import (
    FunctionSignature,
    FunctionSignatureBuilder,
)

__all__ = [
    # Model collection
    'ModelNameCollector',
    # Parameter building
    'ParameterASTBuilder',
    # Signature building
    'FunctionSignature',
    'FunctionSignatureBuilder',
    # Endpoint factory
    'EndpointFunctionConfig',
    'EndpointFunctionFactory',
    'EndpointMode',
    'DataFrameLibrary',
    # Convenience functions
    'build_standalone_endpoint_fn',
    'build_delegating_endpoint_fn',
    'build_standalone_dataframe_fn',
    'build_delegating_dataframe_fn',
]
