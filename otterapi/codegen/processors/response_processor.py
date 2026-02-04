"""Response processing utilities for OpenAPI operations.

This module provides the ResponseProcessor class that handles extraction
and processing of OpenAPI response definitions. It consolidates logic
that was previously in codegen.py.
"""

import logging
from typing import TYPE_CHECKING

from otterapi.codegen.ast_utils import _name, _union_expr
from otterapi.codegen.types import ResponseInfo, Type
from otterapi.codegen.utils import sanitize_identifier

if TYPE_CHECKING:
    from otterapi.codegen.types import TypeGenerator
    from otterapi.openapi.v3_2.v3_2 import Operation

__all__ = ['ResponseProcessor']

# Content types that should be treated as JSON
JSON_CONTENT_TYPES = {'application/json', 'text/json'}


class ResponseProcessor:
    """Handles extraction and processing of OpenAPI response definitions.

    This class extracts response information from OpenAPI operations,
    including status codes, content types, and response types. It also
    handles creating union types when multiple response types are possible.

    Example:
        >>> processor = ResponseProcessor(typegen)
        >>> response_infos = processor.extract_response_info(operation)
        >>> response_infos, response_type = processor.get_response_models(operation)
    """

    def __init__(self, typegen: 'TypeGenerator'):
        """Initialize the response processor.

        Args:
            typegen: The TypeGenerator for creating Pydantic model types.
        """
        self.typegen = typegen

    def extract_response_info(self, operation: 'Operation') -> dict[int, ResponseInfo]:
        """Extract response information including content type from an operation.

        This method extracts response schemas and content types for each status code.
        When multiple content types are available for a response, it prefers JSON
        content types for better type safety.

        Args:
            operation: The OpenAPI operation to extract responses from.

        Returns:
            Dictionary mapping status codes to ResponseInfo objects.
        """
        responses: dict[int, ResponseInfo] = {}

        if not operation.responses:
            return responses

        for status_code_str, response in operation.responses.root.items():
            try:
                status_code = int(status_code_str)
            except ValueError:
                # Skip non-numeric status codes like 'default'
                logging.debug(f'Skipping non-numeric status code: {status_code_str}')
                continue

            if not response.content:
                continue

            # Find the best content type to use
            selected_content_type, selected_media_type = self._select_content_type(
                response.content
            )

            # Create ResponseInfo with type if schema exists
            response_type = None
            if selected_media_type.schema_:
                response_type = self.typegen.schema_to_type(
                    selected_media_type.schema_,
                    base_name=f'{sanitize_identifier(operation.operationId)}Response',
                )

            responses[status_code] = ResponseInfo(
                status_code=status_code,
                content_type=selected_content_type,
                type=response_type,
            )

        return responses

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

    def create_response_union(self, types: list[Type]) -> Type:
        """Create a union type from multiple response types.

        Args:
            types: List of response types to combine into a union.

        Returns:
            A union Type combining all the input types.
        """
        union_type = Type(
            None,
            None,
            annotation_ast=_union_expr([t.annotation_ast for t in types]),
            implementation_ast=None,
            type='primitive',
        )
        # Aggregate imports from all types
        union_type.copy_imports_from_sub_types(types)
        union_type.add_annotation_import('typing', 'Union')
        return union_type

    def get_response_models(
        self, operation: 'Operation'
    ) -> tuple[list[ResponseInfo], Type | None]:
        """Get response models and info from an operation.

        Args:
            operation: The OpenAPI operation to extract response models from.

        Returns:
            A tuple of (response_infos, response_type) where:
            - response_infos: List of ResponseInfo objects for all status codes
            - response_type: The unified response type (single or union), or None
        """
        responses = self.extract_response_info(operation)

        if not responses:
            return [], None

        response_list = list(responses.values())

        # Collect all types that have JSON responses
        json_types = [r.type for r in response_list if r.is_json and r.type]

        # Also include non-JSON response types in the union (bytes, str, Response)
        non_json_types = self._collect_non_json_types(response_list)

        all_types = json_types + non_json_types

        if len(all_types) == 0:
            return response_list, None
        elif len(all_types) == 1:
            return response_list, all_types[0]
        else:
            return response_list, self.create_response_union(all_types)

    def _collect_non_json_types(self, response_list: list[ResponseInfo]) -> list[Type]:
        """Collect non-JSON response types (bytes, str, Response).

        Args:
            response_list: List of ResponseInfo objects.

        Returns:
            List of Type objects for non-JSON responses.
        """
        non_json_types = []
        has_binary = False
        has_text = False
        has_raw = False

        for r in response_list:
            if r.is_binary and not has_binary:
                bytes_type = Type(
                    reference=None,
                    name=None,
                    type='primitive',
                    annotation_ast=_name('bytes'),
                )
                non_json_types.append(bytes_type)
                has_binary = True
            elif r.is_text and not has_text:
                str_type = Type(
                    reference=None,
                    name=None,
                    type='primitive',
                    annotation_ast=_name('str'),
                )
                non_json_types.append(str_type)
                has_text = True
            elif r.is_raw and not has_raw:
                # For unknown content types, return the raw httpx.Response
                response_type = Type(
                    reference=None,
                    name=None,
                    type='primitive',
                    annotation_ast=_name('Response'),
                )
                response_type.add_annotation_import('httpx', 'Response')
                non_json_types.append(response_type)
                has_raw = True

        return non_json_types
