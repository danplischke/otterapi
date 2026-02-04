"""Unified parameter AST building utilities.

This module provides a single source of truth for building AST nodes
related to request parameters (query, header, path, body).

These builders are used by both endpoints.py and client_generator.py
to ensure consistent parameter handling across the codebase.
"""

import ast
import re
from typing import TYPE_CHECKING

from otterapi.codegen.ast_utils import _attr, _call, _name

if TYPE_CHECKING:
    from otterapi.codegen.types import Parameter, RequestBodyInfo

__all__ = ['ParameterASTBuilder']


class ParameterASTBuilder:
    """Unified builder for parameter-related AST nodes.

    This class provides static methods for building AST nodes that represent
    various types of request parameters. It consolidates logic that was
    previously duplicated across endpoints.py and client_generator.py.

    Example:
        >>> from otterapi.codegen.builders import ParameterASTBuilder
        >>> query_dict = ParameterASTBuilder.build_query_params(parameters)
        >>> path_expr = ParameterASTBuilder.build_path_expr('/pet/{petId}', parameters)
    """

    @staticmethod
    def build_query_params(parameters: list['Parameter']) -> ast.Dict | None:
        """Build a dictionary AST node for query parameters.

        Creates an AST Dict node mapping query parameter names to their values.
        Only includes parameters where location == 'query'.

        Args:
            parameters: List of Parameter objects to filter for query params.

        Returns:
            An AST Dict node for query parameters, or None if no query params.

        Example:
            Input: [Parameter(name='status', location='query', name_sanitized='status')]
            Output: AST for {'status': status}
        """
        query_params = [p for p in parameters if p.location == 'query']
        if not query_params:
            return None

        return ast.Dict(
            keys=[ast.Constant(value=param.name) for param in query_params],
            values=[_name(param.name_sanitized) for param in query_params],
        )

    @staticmethod
    def build_header_params(parameters: list['Parameter']) -> ast.Dict | None:
        """Build a dictionary AST node for header parameters.

        Creates an AST Dict node mapping header parameter names to their values.
        Only includes parameters where location == 'header'.

        Args:
            parameters: List of Parameter objects to filter for headers.

        Returns:
            An AST Dict node for header parameters, or None if no header params.

        Example:
            Input: [Parameter(name='X-Api-Key', location='header', name_sanitized='x_api_key')]
            Output: AST for {'X-Api-Key': x_api_key}
        """
        header_params = [p for p in parameters if p.location == 'header']
        if not header_params:
            return None

        return ast.Dict(
            keys=[ast.Constant(value=param.name) for param in header_params],
            values=[_name(param.name_sanitized) for param in header_params],
        )

    @staticmethod
    def build_path_expr(path: str, parameters: list['Parameter']) -> ast.expr:
        """Build an f-string or constant for the request path.

        Replaces OpenAPI path parameters like {petId} with Python f-string
        expressions using the sanitized parameter names.

        Args:
            path: The OpenAPI path string with placeholders (e.g., '/pet/{petId}').
            parameters: List of Parameter objects to map placeholders to variables.

        Returns:
            An AST expression:
            - ast.Constant for static paths without parameters
            - ast.JoinedStr (f-string) for paths with parameter substitution

        Example:
            Input: path='/pet/{petId}', parameters with name='petId', name_sanitized='pet_id'
            Output: AST for f'/pet/{pet_id}'
        """
        path_params = {
            p.name: p.name_sanitized for p in parameters if p.location == 'path'
        }

        # Check if there are any path parameters
        if not path_params:
            return ast.Constant(value=path)

        # Build an f-string with interpolated path parameters
        pattern = r'\{([^}]+)\}'
        parts = re.split(pattern, path)
        values: list[ast.expr] = []

        for i, part in enumerate(parts):
            if i % 2 == 0:
                # Static part
                if part:
                    values.append(ast.Constant(value=part))
            else:
                # Parameter placeholder
                sanitized = path_params.get(part, part)
                values.append(ast.FormattedValue(value=_name(sanitized), conversion=-1))

        if len(values) == 1 and isinstance(values[0], ast.Constant):
            return values[0]

        return ast.JoinedStr(values=values)

    @staticmethod
    def build_body_expr(
        body: 'RequestBodyInfo | None',
    ) -> tuple[ast.expr | None, str | None]:
        """Build an AST expression for the request body parameter.

        Handles different content types and determines the appropriate
        httpx parameter name to use:
        - JSON content with Pydantic models: uses model_dump()
        - Multipart form data: passes dict directly
        - URL-encoded form data: uses model_dump() for models
        - Other content types: passes value directly

        Args:
            body: The RequestBodyInfo object, or None if no body.

        Returns:
            A tuple of (body_expr, httpx_param_name) where:
            - body_expr: AST expression for the body value, or None
            - httpx_param_name: The httpx keyword to use ('json', 'data', 'files', 'content'),
                               or None if no body

        Example:
            Input: RequestBodyInfo(is_json=True, type=Type(type='model'))
            Output: (AST for body.model_dump(), 'json')
        """
        if not body:
            return None, None

        body_name = 'body'

        # For JSON content with model types, use model_dump()
        if body.is_json and body.type and body.type.type in ('model', 'root'):
            body_expr = _call(
                func=_attr(_name(body_name), 'model_dump'),
                args=[],
            )
        elif body.is_multipart:
            # For multipart, the body should be a dict of file tuples
            # The user passes files as {'field': (filename, content, content_type)}
            body_expr = _name(body_name)
        elif body.is_form:
            # For form data, pass as dict
            if body.type and body.type.type in ('model', 'root'):
                body_expr = _call(
                    func=_attr(_name(body_name), 'model_dump'),
                    args=[],
                )
            else:
                body_expr = _name(body_name)
        else:
            # For other content types or primitive types, use the value directly
            body_expr = _name(body_name)

        return body_expr, body.httpx_param_name

    @staticmethod
    def prepare_all_params(
        parameters: list['Parameter'] | None,
        path: str,
        request_body_info: 'RequestBodyInfo | None' = None,
    ) -> tuple[ast.expr | None, ast.expr | None, ast.expr | None, str | None, ast.expr]:
        """Prepare all parameter AST nodes for a request function call.

        This is a convenience method that calls all the individual builders
        and returns everything needed for a request call.

        Args:
            parameters: List of Parameter objects, or None.
            path: The API path with optional placeholders.
            request_body_info: Request body info, or None.

        Returns:
            A tuple of (query_params, header_params, body_expr, body_param_name, path_expr):
            - query_params: AST Dict for query params or None
            - header_params: AST Dict for header params or None
            - body_expr: AST expression for body or None
            - body_param_name: httpx param name for body or None
            - path_expr: AST expression for the path (Constant or JoinedStr)
        """
        if not parameters:
            parameters = []

        query_params = ParameterASTBuilder.build_query_params(parameters)
        header_params = ParameterASTBuilder.build_header_params(parameters)
        body_expr, body_param_name = ParameterASTBuilder.build_body_expr(
            request_body_info
        )
        path_expr = ParameterASTBuilder.build_path_expr(path, parameters)

        return query_params, header_params, body_expr, body_param_name, path_expr
