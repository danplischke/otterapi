"""Parameter processing utilities for OpenAPI operations.

This module provides the ParameterProcessor class that handles extraction
and processing of OpenAPI parameter definitions. It consolidates logic
that was previously in codegen.py.
"""

from typing import TYPE_CHECKING

from otterapi.codegen.types import Parameter, RequestBodyInfo
from otterapi.codegen.utils import sanitize_identifier, sanitize_parameter_field_name

if TYPE_CHECKING:
    from otterapi.codegen.types import TypeGenerator
    from otterapi.codegen.utils import OpenAPIProcessor
    from otterapi.openapi.v3_2.v3_2 import Operation

__all__ = ['ParameterProcessor']

# Content types that should be treated as JSON
JSON_CONTENT_TYPES = {'application/json', 'text/json'}


class ParameterProcessor:
    """Handles extraction of OpenAPI parameters.

    This class extracts operation parameters (path, query, header, cookie)
    and request body information from OpenAPI operations.

    Example:
        >>> processor = ParameterProcessor(typegen, openapi_processor)
        >>> parameters = processor.extract_operation_parameters(operation)
        >>> body_info = processor.extract_request_body(operation)
    """

    def __init__(self, typegen: 'TypeGenerator', openapi_processor: 'OpenAPIProcessor'):
        """Initialize the parameter processor.

        Args:
            typegen: The TypeGenerator for creating Pydantic model types.
            openapi_processor: The OpenAPI processor for resolving references.
        """
        self.typegen = typegen
        self._openapi_processor = openapi_processor

    def extract_operation_parameters(self, operation: 'Operation') -> list[Parameter]:
        """Extract path, query, header, and cookie parameters from an operation.

        Args:
            operation: The OpenAPI operation to extract parameters from.

        Returns:
            List of Parameter objects for path/query/header/cookie parameters.
        """
        params = []
        for param in operation.parameters or []:
            param_type = None
            if param.schema_:
                param_type = self.typegen.schema_to_type(param.schema_)

            params.append(
                Parameter(
                    name=param.name,
                    name_sanitized=sanitize_parameter_field_name(param.name),
                    location=param.in_,  # query, path, header, cookie
                    required=param.required or False,
                    type=param_type,
                    description=param.description,
                )
            )
        return params

    def extract_request_body(self, operation: 'Operation') -> RequestBodyInfo | None:
        """Extract request body information from an operation.

        Handles different content types including:
        - application/json: JSON body with schema validation
        - multipart/form-data: File uploads and form data
        - application/x-www-form-urlencoded: URL-encoded form data
        - application/octet-stream: Binary data

        Args:
            operation: The OpenAPI operation to extract request body from.

        Returns:
            RequestBodyInfo object with content type and schema, or None if no body exists.
        """
        if not operation.requestBody:
            return None

        body, _ = self._openapi_processor._resolve_reference(operation.requestBody)
        if not body.content:
            return None

        # Select the best content type
        selected_content_type, selected_media_type = self._select_content_type(
            body.content
        )

        # Create type from schema if available
        body_type = None
        if selected_media_type.schema_:
            body_type = self.typegen.schema_to_type(
                selected_media_type.schema_,
                base_name=f'{sanitize_identifier(operation.operationId)}RequestBody',
            )

        return RequestBodyInfo(
            content_type=selected_content_type,
            type=body_type,
            required=body.required or False,
            description=body.description,
        )

    def _select_content_type(self, content: dict) -> tuple[str, any]:
        """Select the best content type from available options.

        Prefers JSON content types for better type safety.

        Args:
            content: Dictionary mapping content types to media type objects.

        Returns:
            Tuple of (selected_content_type, selected_media_type).
        """
        # First, try to find a JSON content type
        for content_type, media_type in content.items():
            if content_type in JSON_CONTENT_TYPES or content_type.endswith('+json'):
                return content_type, media_type

        # If no JSON found, use the first available content type
        return next(iter(content.items()))

    def get_param_model(
        self, operation: 'Operation'
    ) -> tuple[list[Parameter], RequestBodyInfo | None]:
        """Get all parameters and request body info for an operation.

        This is a convenience method that combines extract_operation_parameters
        and extract_request_body.

        Args:
            operation: The OpenAPI operation to extract parameters from.

        Returns:
            A tuple of (parameters, request_body_info) where:
            - parameters: List of Parameter objects (path, query, header)
            - request_body_info: RequestBodyInfo object or None
        """
        params = self.extract_operation_parameters(operation)
        body_info = self.extract_request_body(operation)

        return params, body_info
