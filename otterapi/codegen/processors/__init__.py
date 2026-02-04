"""Processors package for OpenAPI extraction logic.

This package contains processor classes that handle extraction and
transformation of OpenAPI specification elements into internal types.
"""

from otterapi.codegen.processors.parameter_processor import ParameterProcessor
from otterapi.codegen.processors.response_processor import ResponseProcessor

__all__ = [
    'ParameterProcessor',
    'ResponseProcessor',
]
