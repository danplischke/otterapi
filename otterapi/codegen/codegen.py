"""Code generation module for OtterAPI.

This module provides the main Codegen class that orchestrates the generation
of Python client code from OpenAPI specifications.
"""

import ast
import http
import logging
import os
from urllib.parse import urljoin, urlparse

from upath import UPath

from otterapi.codegen.ast_utils import (
    ImportCollector,
    _all,
    _assign,
    _call,
    _name,
    _union_expr,
)
from otterapi.codegen.client import (
    EndpointInfo,
    generate_api_error_class,
    generate_base_client_class,
    generate_client_stub,
)
from otterapi.codegen.dataframes import (
    DataFrameMethodConfig,
    endpoint_returns_list,
    generate_dataframe_module,
    get_dataframe_config_for_endpoint,
)
from otterapi.codegen.endpoints import async_request_fn, request_fn
from otterapi.codegen.pagination import (
    PaginationMethodConfig,
    generate_pagination_module,
    get_pagination_config_for_endpoint,
)
from otterapi.codegen.schema import SchemaLoader
from otterapi.codegen.types import (
    Endpoint,
    Parameter,
    RequestBodyInfo,
    ResponseInfo,
    Type,
    TypeGenerator,
    collect_used_model_names,
)
from otterapi.codegen.utils import (
    OpenAPIProcessor,
    sanitize_identifier,
    sanitize_parameter_field_name,
    to_snake_case,
    write_mod,
)
from otterapi.config import DocumentConfig
from otterapi.openapi.v3_2.v3_2 import (
    OpenAPI as OpenAPIv3_2,
    Operation,
    Parameter as OpenAPIParameter,
    Reference,
    RequestBody as OpenAPIRequestBody,
    Response as OpenAPIResponse,
)

# Content types that should be treated as JSON
JSON_CONTENT_TYPES = {'application/json', 'text/json'}

HTTP_METHODS = [method.value.lower() for method in http.HTTPMethod]


class Codegen(OpenAPIProcessor):
    """Main code generator for creating Python clients from OpenAPI specifications.

    This class orchestrates the entire code generation process, including:
    - Loading and validating OpenAPI schemas
    - Generating Pydantic models from schema definitions
    - Creating typed endpoint functions for API operations
    - Writing output files with proper imports and structure

    The generator supports OpenAPI 3.x specifications and produces
    fully typed Python code compatible with httpx for HTTP requests
    and Pydantic for data validation.

    Attributes:
        config: The DocumentConfig containing source and output settings.
        openapi: The loaded OpenAPI schema (populated after _load_schema).
        typegen: The TypeGenerator for creating Pydantic models.

    Example:
        >>> from otterapi.config import DocumentConfig
        >>> from otterapi.codegen.codegen import Codegen
        >>>
        >>> config = DocumentConfig(
        ...     source="https://api.example.com/openapi.json",
        ...     output="./client"
        ... )
        >>> codegen = Codegen(config)
        >>> codegen.generate()
        # Creates models.py and endpoints.py in ./client/

    Note:
        The schema is not loaded until generate() is called or
        _load_schema() is explicitly invoked.
    """

    def __init__(
        self, config: DocumentConfig, schema_loader: SchemaLoader | None = None
    ):
        """Initialize the code generator.

        Args:
            config: Configuration specifying source schema and output location.
            schema_loader: Optional custom schema loader. If not provided,
                          a default SchemaLoader will be created.
        """
        super().__init__(None)
        self.config = config
        self.openapi: OpenAPIv3_2 | None = None
        self._schema_loader = schema_loader or SchemaLoader()

    def _load_schema(self) -> None:
        """Load and parse the OpenAPI schema from the configured source.

        This method loads the schema from a URL or file path, validates it
        against the OpenAPI specification, and initializes the type generator.

        After calling this method, self.openapi and self.typegen will be
        populated and ready for code generation.

        Raises:
            SchemaLoadError: If the schema cannot be loaded from the source.
            SchemaValidationError: If the schema is not valid OpenAPI.
        """
        self.openapi = self._schema_loader.load(self.config.source)
        self.typegen = TypeGenerator(self.openapi)

    def _extract_response_info(self, operation: Operation) -> dict[int, ResponseInfo]:
        """Extract response information including content type from an operation.

        This method extracts response schemas and content types for each status code.
        When multiple content types are available for a response, it prefers JSON
        content types for better type safety.

        For non-JSON content types (XML, binary, text, etc.), no response type is
        generated and the endpoint will return the raw httpx.Response object.

        Args:
            operation: The OpenAPI operation to extract responses from.

        Returns:
            Dictionary mapping status codes to ResponseInfo objects.
        """
        responses: dict[int, ResponseInfo] = {}

        if not operation.responses:
            return responses

        for status_code_str, response_or_ref in operation.responses.root.items():
            try:
                status_code = int(status_code_str)
            except ValueError:
                logging.debug(f'Skipping non-numeric status code: {status_code_str}')
                continue

            # Resolve reference if needed
            response = self._resolve_response_reference(response_or_ref)
            if response is None or not response.content:
                continue

            selected_content_type, selected_media_type = self._select_content_type(
                response.content
            )

            # Only generate typed response for JSON content types
            # For other content types (XML, binary, etc.), return raw httpx.Response
            response_type = None
            is_json_content = (
                selected_content_type in JSON_CONTENT_TYPES
                or selected_content_type.endswith('+json')
            )

            if is_json_content and selected_media_type.schema_:
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
        for content_type, media_type in content.items():
            if content_type in JSON_CONTENT_TYPES or content_type.endswith('+json'):
                return content_type, media_type

        return next(iter(content.items()))

    def _create_response_union(self, types: list[Type]) -> Type:
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
        union_type.copy_imports_from_sub_types(types)
        return union_type

    def _collect_non_json_types(self, response_list: list[ResponseInfo]) -> list[Type]:
        """Collect non-JSON response types.

        For all non-JSON content types (binary, text, XML, etc.), we return
        the raw httpx.Response object. This gives users full control over
        how to handle the response (.content, .text, .json(), etc.).
        """
        # Check if there are any non-JSON responses
        has_non_json = any(not r.is_json for r in response_list)

        if has_non_json:
            response_type = Type(
                reference=None,
                name=None,
                type='primitive',
                annotation_ast=_name('Response'),
            )
            response_type.add_annotation_import('httpx', 'Response')
            return [response_type]

        return []

    def _get_response_models(
        self, operation: Operation
    ) -> tuple[list[ResponseInfo], Type | None]:
        """Get response models and info from an operation.

        Args:
            operation: The OpenAPI operation to extract response models from.

        Returns:
            A tuple of (response_infos, response_type) where:
            - response_infos: List of ResponseInfo objects for all status codes
            - response_type: The unified response type (single or union), or None
        """
        responses = self._extract_response_info(operation)

        if not responses:
            return [], None

        response_list = list(responses.values())

        json_types = [r.type for r in response_list if r.is_json and r.type]
        non_json_types = self._collect_non_json_types(response_list)

        all_types = json_types + non_json_types

        if len(all_types) == 0:
            return response_list, None
        elif len(all_types) == 1:
            return response_list, all_types[0]
        else:
            return response_list, self._create_response_union(all_types)

    def _extract_operation_parameters(
        self, operation: Operation, path_item_parameters: list | None = None
    ) -> list[Parameter]:
        """Extract path, query, header, and cookie parameters from an operation.

        Merges path-level parameters with operation-level parameters.
        Operation parameters override path-level parameters with the same name and location.
        Handles $ref references to #/components/parameters/.

        Args:
            operation: The OpenAPI operation to extract parameters from.
            path_item_parameters: Optional path-level parameters to inherit.

        Returns:
            List of Parameter objects for path/query/header/cookie parameters.
        """
        # Merge path-level and operation-level parameters
        # Operation parameters override path-level parameters with same name+location
        all_params = []
        param_keys_seen = set()  # Track (name, location) to handle overrides

        # First, add operation-level parameters (they take precedence)
        for param_or_ref in operation.parameters or []:
            # Resolve reference if needed
            param = self._resolve_parameter_reference(param_or_ref)
            if param is None:
                continue

            param_type = None
            if param.schema_:
                param_type = self.typegen.schema_to_type(param.schema_)

            all_params.append(
                Parameter(
                    name=param.name,
                    name_sanitized=sanitize_parameter_field_name(param.name),
                    location=param.in_,
                    required=param.required or False,
                    type=param_type,
                    description=param.description,
                )
            )
            param_keys_seen.add((param.name, param.in_))

        # Then, add path-level parameters that weren't overridden
        for param_or_ref in path_item_parameters or []:
            # Resolve reference if needed
            param = self._resolve_parameter_reference(param_or_ref)
            if param is None:
                continue

            if (param.name, param.in_) not in param_keys_seen:
                param_type = None
                if param.schema_:
                    param_type = self.typegen.schema_to_type(param.schema_)

                all_params.append(
                    Parameter(
                        name=param.name,
                        name_sanitized=sanitize_parameter_field_name(param.name),
                        location=param.in_,
                        required=param.required or False,
                        type=param_type,
                        description=param.description,
                    )
                )

        return all_params

    def _resolve_parameter_reference(
        self, param_or_ref: OpenAPIParameter | Reference
    ) -> OpenAPIParameter | None:
        """Resolve a parameter reference to the actual Parameter object.

        Args:
            param_or_ref: Either a Parameter object or a Reference to one.

        Returns:
            The resolved Parameter object, or None if reference cannot be resolved.
        """
        if isinstance(param_or_ref, Reference):
            if not param_or_ref.ref.startswith('#/components/parameters/'):
                logging.warning(
                    f'Unsupported parameter reference format: {param_or_ref.ref}'
                )
                return None

            param_name = param_or_ref.ref.split('/')[-1]
            if (
                not self.openapi.components
                or not self.openapi.components.parameters
                or param_name not in self.openapi.components.parameters
            ):
                logging.warning(
                    f"Referenced parameter '{param_name}' not found in components.parameters"
                )
                return None

            resolved = self.openapi.components.parameters[param_name]
            # Handle nested references
            if isinstance(resolved, Reference):
                return self._resolve_parameter_reference(resolved)
            return resolved

        return param_or_ref

    def _resolve_response_reference(
        self, response_or_ref: OpenAPIResponse | Reference
    ) -> OpenAPIResponse | None:
        """Resolve a response reference to the actual Response object.

        Args:
            response_or_ref: Either a Response object or a Reference to one.

        Returns:
            The resolved Response object, or None if reference cannot be resolved.
        """
        if isinstance(response_or_ref, Reference):
            if not response_or_ref.ref.startswith('#/components/responses/'):
                logging.warning(
                    f'Unsupported response reference format: {response_or_ref.ref}'
                )
                return None

            response_name = response_or_ref.ref.split('/')[-1]
            if (
                not self.openapi.components
                or not self.openapi.components.responses
                or response_name not in self.openapi.components.responses
            ):
                logging.warning(
                    f"Referenced response '{response_name}' not found in components.responses"
                )
                return None

            resolved = self.openapi.components.responses[response_name]
            # Handle nested references
            if isinstance(resolved, Reference):
                return self._resolve_response_reference(resolved)
            return resolved

        return response_or_ref

    def _resolve_request_body_reference(
        self, body_or_ref: OpenAPIRequestBody | Reference
    ) -> OpenAPIRequestBody | None:
        """Resolve a request body reference to the actual RequestBody object.

        Args:
            body_or_ref: Either a RequestBody object or a Reference to one.

        Returns:
            The resolved RequestBody object, or None if reference cannot be resolved.
        """
        if isinstance(body_or_ref, Reference):
            if not body_or_ref.ref.startswith('#/components/requestBodies/'):
                logging.warning(
                    f'Unsupported request body reference format: {body_or_ref.ref}'
                )
                return None

            body_name = body_or_ref.ref.split('/')[-1]
            if (
                not self.openapi.components
                or not self.openapi.components.requestBodies
                or body_name not in self.openapi.components.requestBodies
            ):
                logging.warning(
                    f"Referenced request body '{body_name}' not found in components.requestBodies"
                )
                return None

            resolved = self.openapi.components.requestBodies[body_name]
            # Handle nested references
            if isinstance(resolved, Reference):
                return self._resolve_request_body_reference(resolved)
            return resolved

        return body_or_ref

    def _extract_request_body(self, operation: Operation) -> RequestBodyInfo | None:
        """Extract request body information from an operation.

        Args:
            operation: The OpenAPI operation to extract request body from.

        Returns:
            RequestBodyInfo object with content type and schema, or None if no body exists.
        """
        if not operation.requestBody:
            return None

        body = self._resolve_request_body_reference(operation.requestBody)
        if body is None or not body.content:
            return None

        selected_content_type, selected_media_type = self._select_content_type(
            body.content
        )

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

    def _get_param_model(
        self, operation: Operation, path_item_parameters: list | None = None
    ) -> tuple[list[Parameter], RequestBodyInfo | None]:
        """Get all parameters and request body info for an operation.

        Args:
            operation: The OpenAPI operation to extract parameters from.
            path_item_parameters: Optional path-level parameters to inherit.

        Returns:
            A tuple of (parameters, request_body_info) where:
            - parameters: List of Parameter objects (path, query, header)
            - request_body_info: RequestBodyInfo object or None
        """
        params = self._extract_operation_parameters(operation, path_item_parameters)
        body_info = self._extract_request_body(operation)

        return params, body_info

    def _generate_endpoint(
        self,
        path: str,
        method: str,
        operation: Operation,
        path_item_parameters: list | None = None,
    ) -> Endpoint:
        """Generate an endpoint with sync and async functions.

        Args:
            path: The API path for the endpoint.
            method: The HTTP method (get, post, etc.).
            operation: The OpenAPI operation definition.
            path_item_parameters: Optional list of path-level parameters to inherit.

        Returns:
            An Endpoint object containing the generated functions and imports.
        """
        # Convert operationId to snake_case for Pythonic function names
        raw_name = (
            operation.operationId
            or f'{method}_{path.replace("/", "_").replace("{", "").replace("}", "")}'
        )
        fn_name = to_snake_case(raw_name)
        async_fn_name = f'a{fn_name}'

        parameters, request_body_info = self._get_param_model(
            operation, path_item_parameters
        )
        response_infos, response_model = self._get_response_models(operation)

        # Build docstring with deprecation warning if needed
        docs = operation.description or ''
        if operation.deprecated:
            deprecation_notice = '.. deprecated::\n    This endpoint is deprecated.'
            if docs:
                docs = f'{docs}\n\n{deprecation_notice}'
            else:
                docs = deprecation_notice

        async_fn, async_imports = async_request_fn(
            name=async_fn_name,
            method=method,
            path=path,
            response_model=response_model,
            docs=docs if docs else None,
            parameters=parameters,
            response_infos=response_infos,
            request_body_info=request_body_info,
        )

        sync_fn, imports = request_fn(
            name=fn_name,
            method=method,
            path=path,
            response_model=response_model,
            docs=docs if docs else None,
            parameters=parameters,
            response_infos=response_infos,
            request_body_info=request_body_info,
        )

        # Extract tags from operation
        tags = list(operation.tags) if operation.tags else None

        ep = Endpoint(
            sync_ast=sync_fn,
            sync_fn_name=fn_name,
            async_fn_name=async_fn_name,
            async_ast=async_fn,
            name=fn_name,
            method=method,
            path=path,
            description=operation.description,
            tags=tags,
            parameters=parameters,
            request_body=request_body_info,
            response_type=response_model,
            response_infos=response_infos,
        )

        ep.add_imports([imports, async_imports])

        for param in parameters:
            if param.type and param.type.annotation_imports:
                ep.add_imports([param.type.annotation_imports])

        # Add imports for request body type if present
        if request_body_info and request_body_info.type:
            if request_body_info.type.annotation_imports:
                ep.add_imports([request_body_info.type.annotation_imports])

        # Add imports for response model type if present
        if response_model and response_model.annotation_imports:
            ep.add_imports([response_model.annotation_imports])

        return ep

    def _generate_endpoints(self) -> list[Endpoint]:
        """Generate all endpoints from the OpenAPI paths.

        Returns:
            List of generated Endpoint objects.
        """
        endpoints: list[Endpoint] = []
        # Paths is a RootModel, access .root to get the underlying dict
        paths_dict = (
            self.openapi.paths.root
            if hasattr(self.openapi.paths, 'root')
            else self.openapi.paths
        )
        for path, path_item in paths_dict.items():
            # Apply path filtering
            if not self._should_include_path(path):
                continue

            # Get path-level parameters to pass to each operation
            path_item_parameters = (
                path_item.parameters if hasattr(path_item, 'parameters') else None
            )
            for method in HTTP_METHODS:
                operation = getattr(path_item, method, None)
                if operation:
                    ep = self._generate_endpoint(
                        path, method, operation, path_item_parameters
                    )
                    endpoints.append(ep)
        return endpoints

    def _should_include_path(self, path: str) -> bool:
        """Check if a path should be included based on include_paths and exclude_paths.

        Args:
            path: The API path to check.

        Returns:
            True if the path should be included, False otherwise.
        """
        import fnmatch

        # Check include_paths first (if specified, path must match at least one)
        if self.config.include_paths:
            included = any(
                fnmatch.fnmatch(path, pattern) for pattern in self.config.include_paths
            )
            if not included:
                return False

        # Check exclude_paths (if path matches any, exclude it)
        if self.config.exclude_paths:
            excluded = any(
                fnmatch.fnmatch(path, pattern) for pattern in self.config.exclude_paths
            )
            if excluded:
                return False

        return True

    def _is_absolute_url(self, url: str) -> bool:
        """Check if a URL is absolute (has scheme and netloc).

        Args:
            url: The URL to check.

        Returns:
            True if the URL is absolute, False otherwise.
        """
        parsed = urlparse(url)
        return bool(parsed.scheme and parsed.netloc)

    def _resolve_base_url(self) -> str:
        """Resolve the base URL from config or OpenAPI spec.

        If the server URL in the spec is relative, attempts to resolve it
        against the source URL (if the spec was loaded from a URL).

        Returns:
            The base URL to use for API requests.

        Raises:
            ValueError: If no base URL can be determined, multiple servers are defined,
                       or a relative server URL cannot be resolved.
        """
        # Config base_url takes precedence
        if self.config.base_url:
            return self.config.base_url

        # If no servers in spec, config must provide base_url
        if not self.openapi.servers:
            raise ValueError(
                'No base url provided. Make sure you specify the base_url in the otterapi config or the OpenAPI document contains a valid servers section'
            )

        # Only support single server
        if len(self.openapi.servers) > 1:
            raise ValueError(
                'Multiple servers are not supported. Set the base_url in the config.'
            )

        # TODO: handle server variables
        baseurl = self.openapi.servers[0].url

        if not baseurl:
            raise ValueError(
                'No base url provided. Make sure you specify the base_url in the otterapi config or the OpenAPI document contains a valid servers section'
            )

        # Check if the server URL is relative
        if not self._is_absolute_url(baseurl):
            # Try to resolve against the source URL if it's a URL
            source = self.config.source
            if self._is_absolute_url(source):
                # Resolve relative server URL against the source URL
                resolved_url = urljoin(source, baseurl)
                logging.info(
                    f"Resolved relative server URL '{baseurl}' to '{resolved_url}' "
                    f"using source URL '{source}'"
                )
                return resolved_url
            else:
                # Source is a file path, can't resolve relative URL
                raise ValueError(
                    f"Server URL '{baseurl}' is relative and cannot be resolved. "
                    f'The OpenAPI spec was loaded from a file, not a URL. '
                    f'Please specify an absolute base_url in the otterapi config.'
                )

        return baseurl

    def _collect_used_model_names(self, endpoints: list[Endpoint]) -> set[str]:
        """Collect model names that are actually used in endpoint signatures.

        Only collects models that have implementations (defined in models.py)
        and are referenced in endpoint parameters, request bodies, or responses.

        Args:
            endpoints: List of Endpoint objects to check for model usage.

        Returns:
            Set of model names actually used in endpoints.

        Note:
            This method delegates to collect_used_model_names() from builders.model_collector.
        """
        return collect_used_model_names(endpoints, self.typegen.types)

    def _create_model_import(
        self, models_file: UPath, model_names: set[str]
    ) -> ast.ImportFrom:
        """Create an import statement for models.

        Args:
            models_file: Path to the models file.
            model_names: Set of model names to import.

        Returns:
            AST ImportFrom statement for the models.
        """
        return ast.ImportFrom(
            module=self.config.models_import_path or models_file.stem,
            names=[ast.alias(name=name, asname=None) for name in sorted(model_names)],
            level=1 if not self.config.models_import_path else 0,  # relative import
        )

    def _build_endpoint_file_body(
        self, baseurl: str, endpoints: list[Endpoint]
    ) -> tuple[list[ast.stmt], ImportCollector, set[str]]:
        """Build the body of the endpoints file with standalone functions.

        Generates standalone functions with full implementations that use the Client.

        Args:
            baseurl: The base URL for API requests (unused, kept for API compat).
            endpoints: List of Endpoint objects to include.

        Returns:
            Tuple of (body statements, import collector, endpoint names).
        """
        from otterapi.codegen.endpoints import (
            build_default_client_code,
            build_standalone_dataframe_fn,
            build_standalone_endpoint_fn,
            build_standalone_paginated_dataframe_fn,
            build_standalone_paginated_fn,
            build_standalone_paginated_iter_fn,
        )

        body: list[ast.stmt] = []
        import_collector = ImportCollector()

        # Add default client variable and _get_client() function
        client_stmts, client_imports = build_default_client_code()
        body.extend(client_stmts)
        import_collector.add_imports(client_imports)

        # Track if we need DataFrame type hints
        has_dataframe_methods = False

        # Track if we need pagination imports
        has_pagination_methods = False

        # Add standalone endpoint functions
        endpoint_names = set()
        for endpoint in endpoints:
            # Track whether paginated DataFrame methods were generated for this endpoint
            generated_paginated_df = False

            # Check if this endpoint has pagination configured
            pag_config = None
            if self.config.pagination.enabled:
                pag_config = self._get_pagination_config(endpoint)

            # Generate pagination methods if configured, otherwise regular functions
            if pag_config:
                has_pagination_methods = True

                # Get item type from response type if it's a list
                item_type_ast = self._get_item_type_ast(endpoint)

                # Build pagination config dict
                pag_dict = {
                    'offset_param': pag_config.offset_param,
                    'limit_param': pag_config.limit_param,
                    'cursor_param': pag_config.cursor_param,
                    'page_param': pag_config.page_param,
                    'per_page_param': pag_config.per_page_param,
                    'data_path': pag_config.data_path,
                    'total_path': pag_config.total_path,
                    'next_cursor_path': pag_config.next_cursor_path,
                    'total_pages_path': pag_config.total_pages_path,
                    'default_page_size': pag_config.default_page_size,
                }

                # Sync paginated function (replaces regular sync function)
                pag_fn, pag_imports = build_standalone_paginated_fn(
                    fn_name=endpoint.sync_fn_name,
                    method=endpoint.method,
                    path=endpoint.path,
                    parameters=endpoint.parameters,
                    request_body_info=endpoint.request_body,
                    response_type=endpoint.response_type,
                    pagination_style=pag_config.style,
                    pagination_config=pag_dict,
                    item_type_ast=item_type_ast,
                    docs=endpoint.description,
                    is_async=False,
                )
                endpoint_names.add(endpoint.sync_fn_name)
                body.append(pag_fn)
                import_collector.add_imports(pag_imports)

                # Async paginated function (replaces regular async function)
                async_pag_fn, async_pag_imports = build_standalone_paginated_fn(
                    fn_name=endpoint.async_fn_name,
                    method=endpoint.method,
                    path=endpoint.path,
                    parameters=endpoint.parameters,
                    request_body_info=endpoint.request_body,
                    response_type=endpoint.response_type,
                    pagination_style=pag_config.style,
                    pagination_config=pag_dict,
                    item_type_ast=item_type_ast,
                    docs=endpoint.description,
                    is_async=True,
                )
                endpoint_names.add(endpoint.async_fn_name)
                body.append(async_pag_fn)
                import_collector.add_imports(async_pag_imports)

                # Sync iterator function
                iter_fn_name = f'{endpoint.sync_fn_name}_iter'
                iter_fn, iter_imports = build_standalone_paginated_iter_fn(
                    fn_name=iter_fn_name,
                    method=endpoint.method,
                    path=endpoint.path,
                    parameters=endpoint.parameters,
                    request_body_info=endpoint.request_body,
                    response_type=endpoint.response_type,
                    pagination_style=pag_config.style,
                    pagination_config=pag_dict,
                    item_type_ast=item_type_ast,
                    docs=endpoint.description,
                    is_async=False,
                )
                endpoint_names.add(iter_fn_name)
                body.append(iter_fn)
                import_collector.add_imports(iter_imports)

                # Async iterator function
                async_iter_fn_name = f'{endpoint.async_fn_name}_iter'
                async_iter_fn, async_iter_imports = build_standalone_paginated_iter_fn(
                    fn_name=async_iter_fn_name,
                    method=endpoint.method,
                    path=endpoint.path,
                    parameters=endpoint.parameters,
                    request_body_info=endpoint.request_body,
                    response_type=endpoint.response_type,
                    pagination_style=pag_config.style,
                    pagination_config=pag_dict,
                    item_type_ast=item_type_ast,
                    docs=endpoint.description,
                    is_async=True,
                )
                endpoint_names.add(async_iter_fn_name)
                body.append(async_iter_fn)
                import_collector.add_imports(async_iter_imports)

                # Generate paginated DataFrame methods if dataframe is enabled
                # For paginated endpoints, we know they return lists, so check config directly
                if self.config.dataframe.enabled:
                    # Check if endpoint is explicitly disabled
                    endpoint_df_config = self.config.dataframe.endpoints.get(
                        endpoint.sync_fn_name
                    )
                    if endpoint_df_config and endpoint_df_config.enabled is False:
                        pass  # Skip DataFrame generation for this endpoint
                    elif self.config.dataframe.pandas:
                        generated_paginated_df = True
                        has_dataframe_methods = True
                        has_pagination_methods = True
                        # Sync pandas paginated method
                        pandas_fn_name = f'{endpoint.sync_fn_name}_df'
                        pandas_fn, pandas_imports = (
                            build_standalone_paginated_dataframe_fn(
                                fn_name=pandas_fn_name,
                                method=endpoint.method,
                                path=endpoint.path,
                                parameters=endpoint.parameters,
                                request_body_info=endpoint.request_body,
                                response_type=endpoint.response_type,
                                pagination_style=pag_config.style,
                                pagination_config=pag_dict,
                                library='pandas',
                                item_type_ast=item_type_ast,
                                docs=endpoint.description,
                                is_async=False,
                            )
                        )
                        endpoint_names.add(pandas_fn_name)
                        body.append(pandas_fn)
                        import_collector.add_imports(pandas_imports)

                        # Async pandas paginated method
                        async_pandas_fn_name = f'{endpoint.async_fn_name}_df'
                        async_pandas_fn, async_pandas_imports = (
                            build_standalone_paginated_dataframe_fn(
                                fn_name=async_pandas_fn_name,
                                method=endpoint.method,
                                path=endpoint.path,
                                parameters=endpoint.parameters,
                                request_body_info=endpoint.request_body,
                                response_type=endpoint.response_type,
                                pagination_style=pag_config.style,
                                pagination_config=pag_dict,
                                library='pandas',
                                item_type_ast=item_type_ast,
                                docs=endpoint.description,
                                is_async=True,
                            )
                        )
                        endpoint_names.add(async_pandas_fn_name)
                        body.append(async_pandas_fn)
                        import_collector.add_imports(async_pandas_imports)

                    if self.config.dataframe.polars:
                        generated_paginated_df = True
                        has_dataframe_methods = True
                        has_pagination_methods = True
                        # Sync polars paginated method
                        polars_fn_name = f'{endpoint.sync_fn_name}_pl'
                        polars_fn, polars_imports = (
                            build_standalone_paginated_dataframe_fn(
                                fn_name=polars_fn_name,
                                method=endpoint.method,
                                path=endpoint.path,
                                parameters=endpoint.parameters,
                                request_body_info=endpoint.request_body,
                                response_type=endpoint.response_type,
                                pagination_style=pag_config.style,
                                pagination_config=pag_dict,
                                library='polars',
                                item_type_ast=item_type_ast,
                                docs=endpoint.description,
                                is_async=False,
                            )
                        )
                        endpoint_names.add(polars_fn_name)
                        body.append(polars_fn)
                        import_collector.add_imports(polars_imports)

                        # Async polars paginated method
                        async_polars_fn_name = f'{endpoint.async_fn_name}_pl'
                        async_polars_fn, async_polars_imports = (
                            build_standalone_paginated_dataframe_fn(
                                fn_name=async_polars_fn_name,
                                method=endpoint.method,
                                path=endpoint.path,
                                parameters=endpoint.parameters,
                                request_body_info=endpoint.request_body,
                                response_type=endpoint.response_type,
                                pagination_style=pag_config.style,
                                pagination_config=pag_dict,
                                library='polars',
                                item_type_ast=item_type_ast,
                                docs=endpoint.description,
                                is_async=True,
                            )
                        )
                        endpoint_names.add(async_polars_fn_name)
                        body.append(async_polars_fn)
                        import_collector.add_imports(async_polars_imports)
            else:
                # Build regular sync standalone function
                sync_fn, sync_imports = build_standalone_endpoint_fn(
                    fn_name=endpoint.sync_fn_name,
                    method=endpoint.method,
                    path=endpoint.path,
                    parameters=endpoint.parameters,
                    request_body_info=endpoint.request_body,
                    response_type=endpoint.response_type,
                    response_infos=endpoint.response_infos,
                    docs=endpoint.description,
                    is_async=False,
                )
                endpoint_names.add(endpoint.sync_fn_name)
                body.append(sync_fn)
                import_collector.add_imports(sync_imports)

                # Build regular async standalone function
                async_fn, async_imports = build_standalone_endpoint_fn(
                    fn_name=endpoint.async_fn_name,
                    method=endpoint.method,
                    path=endpoint.path,
                    parameters=endpoint.parameters,
                    request_body_info=endpoint.request_body,
                    response_type=endpoint.response_type,
                    response_infos=endpoint.response_infos,
                    docs=endpoint.description,
                    is_async=True,
                )
                endpoint_names.add(endpoint.async_fn_name)
                body.append(async_fn)
                import_collector.add_imports(async_imports)

                # Note: Pagination methods already handled above if pag_config exists
                # Skip to next endpoint since pagination methods already generated
                pass

            # Generate DataFrame methods if configured
            # Skip if paginated DataFrame methods were already generated for this endpoint
            if self.config.dataframe.enabled and not generated_paginated_df:
                df_config = self._get_dataframe_config(endpoint)

                if df_config.generate_pandas:
                    has_dataframe_methods = True
                    # Sync pandas method
                    pandas_fn_name = f'{endpoint.sync_fn_name}_df'
                    pandas_fn, pandas_imports = build_standalone_dataframe_fn(
                        fn_name=pandas_fn_name,
                        method=endpoint.method,
                        path=endpoint.path,
                        parameters=endpoint.parameters,
                        request_body_info=endpoint.request_body,
                        library='pandas',
                        default_path=df_config.path,
                        docs=endpoint.description,
                        is_async=False,
                    )
                    endpoint_names.add(pandas_fn_name)
                    body.append(pandas_fn)
                    import_collector.add_imports(pandas_imports)

                    # Async pandas method
                    async_pandas_fn_name = f'{endpoint.async_fn_name}_df'
                    async_pandas_fn, async_pandas_imports = (
                        build_standalone_dataframe_fn(
                            fn_name=async_pandas_fn_name,
                            method=endpoint.method,
                            path=endpoint.path,
                            parameters=endpoint.parameters,
                            request_body_info=endpoint.request_body,
                            library='pandas',
                            default_path=df_config.path,
                            docs=endpoint.description,
                            is_async=True,
                        )
                    )
                    endpoint_names.add(async_pandas_fn_name)
                    body.append(async_pandas_fn)
                    import_collector.add_imports(async_pandas_imports)

                if df_config.generate_polars:
                    has_dataframe_methods = True
                    # Sync polars method
                    polars_fn_name = f'{endpoint.sync_fn_name}_pl'
                    polars_fn, polars_imports = build_standalone_dataframe_fn(
                        fn_name=polars_fn_name,
                        method=endpoint.method,
                        path=endpoint.path,
                        parameters=endpoint.parameters,
                        request_body_info=endpoint.request_body,
                        library='polars',
                        default_path=df_config.path,
                        docs=endpoint.description,
                        is_async=False,
                    )
                    endpoint_names.add(polars_fn_name)
                    body.append(polars_fn)
                    import_collector.add_imports(polars_imports)

                    # Async polars method
                    async_polars_fn_name = f'{endpoint.async_fn_name}_pl'
                    async_polars_fn, async_polars_imports = (
                        build_standalone_dataframe_fn(
                            fn_name=async_polars_fn_name,
                            method=endpoint.method,
                            path=endpoint.path,
                            parameters=endpoint.parameters,
                            request_body_info=endpoint.request_body,
                            library='polars',
                            default_path=df_config.path,
                            docs=endpoint.description,
                            is_async=True,
                        )
                    )
                    endpoint_names.add(async_polars_fn_name)
                    body.append(async_polars_fn)
                    import_collector.add_imports(async_polars_imports)

        # Add TYPE_CHECKING block for DataFrame type hints if needed
        if has_dataframe_methods:
            import_collector.add_imports({'typing': {'TYPE_CHECKING'}})
            type_checking_block = ast.If(
                test=_name('TYPE_CHECKING'),
                body=[
                    ast.Import(names=[ast.alias(name='pandas', asname='pd')]),
                    ast.Import(names=[ast.alias(name='polars', asname='pl')]),
                ],
                orelse=[],
            )
            body.insert(0, type_checking_block)

            # Add dataframe helper imports
            dataframe_import = ast.ImportFrom(
                module='_dataframe',
                names=[
                    ast.alias(name='to_pandas', asname=None),
                    ast.alias(name='to_polars', asname=None),
                ],
                level=1,
            )
            body.insert(0, dataframe_import)

        # Add pagination imports if needed
        if has_pagination_methods:
            import_collector.add_imports(
                {'collections.abc': {'Iterator', 'AsyncIterator'}}
            )
            pagination_import = ast.ImportFrom(
                module='_pagination',
                names=[
                    ast.alias(name='paginate_offset', asname=None),
                    ast.alias(name='paginate_offset_async', asname=None),
                    ast.alias(name='paginate_cursor', asname=None),
                    ast.alias(name='paginate_cursor_async', asname=None),
                    ast.alias(name='paginate_page', asname=None),
                    ast.alias(name='paginate_page_async', asname=None),
                    ast.alias(name='iterate_offset', asname=None),
                    ast.alias(name='iterate_offset_async', asname=None),
                    ast.alias(name='iterate_cursor', asname=None),
                    ast.alias(name='iterate_cursor_async', asname=None),
                    ast.alias(name='iterate_page', asname=None),
                    ast.alias(name='iterate_page_async', asname=None),
                    ast.alias(name='extract_path', asname=None),
                ],
                level=1,
            )
            body.insert(0, pagination_import)

        return body, import_collector, endpoint_names

    def _generate_endpoint_file(
        self, path: UPath, models_file: UPath, endpoints: list[Endpoint]
    ) -> None:
        """Generate the endpoints Python file with delegating functions.

        Args:
            path: Path where the endpoints file should be written.
            models_file: Path to the models file for import generation.
            endpoints: List of Endpoint objects to include.
        """
        baseurl = self._resolve_base_url()

        # Build file body and collect imports
        body, import_collector, endpoint_names = self._build_endpoint_file_body(
            baseurl, endpoints
        )

        # Add __all__ export
        body.insert(0, _all(sorted(endpoint_names)))

        # Add model imports only for models actually used in endpoints
        model_names = self._collect_used_model_names(endpoints)
        if model_names:
            model_import = self._create_model_import(models_file, model_names)
            body.insert(0, model_import)

        # Add Client import (relative import from same directory)
        client_import = ast.ImportFrom(
            module='client',
            names=[ast.alias(name='Client', asname=None)],
            level=1,
        )
        body.insert(0, client_import)

        # Add all other imports at the beginning
        for import_stmt in import_collector.to_ast():
            body.insert(0, import_stmt)

        write_mod(body, path)

    def _generate_models_file(self, path: UPath) -> None:
        """Generate the models Python file with Pydantic models.

        Args:
            path: Path where the models file should be written.
        """
        assert self.typegen is not None

        body: list[ast.stmt] = []
        import_collector = ImportCollector()
        all_names = set()

        for type_name, type_ in self.typegen.types.items():
            if type_.implementation_ast:
                body.append(type_.implementation_ast)
                if type_.name:
                    all_names.add(type_.name)

                # Collect imports from implementation and annotations
                import_collector.add_imports(type_.implementation_imports)
                import_collector.add_imports(type_.annotation_imports)

        # Add __all__ export
        body.insert(0, _all(sorted(all_names)))

        # Add all imports at the beginning
        for import_stmt in import_collector.to_ast():
            body.insert(0, import_stmt)

        write_mod(body, path)

    def generate(self):
        self._load_schema()

        assert self.openapi is not None

        if not self.openapi.paths:
            raise ValueError('OpenAPI spec has no paths to generate endpoints from')

        directory = UPath(self.config.output)
        directory.mkdir(parents=True, exist_ok=True)

        if not os.access(str(directory), os.W_OK):
            raise RuntimeError(f'Directory {directory} is not writable')

        generated_files: list[str] = []
        output_name = self.config.output

        endpoints = self._generate_endpoints()

        models_file = directory / self.config.models_file
        self._generate_models_file(models_file)
        generated_files.append(f'{output_name}/{self.config.models_file}')

        base_url = self._resolve_base_url()

        # Generate pagination module if enabled
        if self.config.pagination.enabled:
            generate_pagination_module(directory)
            generated_files.append(f'{output_name}/_pagination.py')

        # Generate client class
        client_name = self._get_client_class_name()

        # Check if module splitting is enabled
        if self.config.module_split.enabled:
            split_files = self._generate_split_endpoints(
                directory, models_file, endpoints, base_url, client_name
            )
            generated_files.extend(split_files)
        else:
            # Original single-file generation
            endpoints_file = directory / self.config.endpoints_file
            self._generate_endpoint_file(endpoints_file, models_file, endpoints)
            generated_files.append(f'{output_name}/{self.config.endpoints_file}')

        client_files = self._generate_client_file(
            directory, endpoints, base_url, client_name
        )
        generated_files.extend(client_files)

        # Write __init__.py only if not using module splitting (splitting handles its own __init__.py)
        if not self.config.module_split.enabled:
            self._generate_init_file(directory, endpoints, client_name)
            generated_files.append(f'{output_name}/__init__.py')

        return generated_files

    def _generate_init_file(
        self,
        directory: UPath,
        endpoints: list[Endpoint],
        client_class_name: str,
    ) -> None:
        """Generate __init__.py with all exports for non-split mode.

        Args:
            directory: Output directory.
            endpoints: List of Endpoint objects.
            client_class_name: Name of the client class.
        """
        body: list[ast.stmt] = []
        all_names: list[str] = []

        # Get endpoint names (including DataFrame methods if configured)
        endpoint_names = []
        for endpoint in endpoints:
            endpoint_names.append(endpoint.sync_fn_name)
            endpoint_names.append(endpoint.async_fn_name)

            # Check pagination config for this endpoint
            pag_config = None
            if self.config.pagination.enabled:
                pag_config = self._get_pagination_config(endpoint)

            # Add pagination method names if configured
            if pag_config:
                endpoint_names.append(f'{endpoint.sync_fn_name}_iter')
                endpoint_names.append(f'{endpoint.async_fn_name}_iter')

            # Add DataFrame method names if configured
            if self.config.dataframe.enabled:
                # For paginated endpoints, DataFrame methods are generated regardless
                # of whether the original response type is a list
                is_paginated = pag_config is not None

                if is_paginated:
                    # Check if endpoint is explicitly disabled
                    endpoint_df_config = self.config.dataframe.endpoints.get(
                        endpoint.sync_fn_name
                    )
                    if endpoint_df_config and endpoint_df_config.enabled is False:
                        pass  # Skip DataFrame exports for this endpoint
                    else:
                        if self.config.dataframe.pandas:
                            endpoint_names.append(f'{endpoint.sync_fn_name}_df')
                            endpoint_names.append(f'{endpoint.async_fn_name}_df')
                        if self.config.dataframe.polars:
                            endpoint_names.append(f'{endpoint.sync_fn_name}_pl')
                            endpoint_names.append(f'{endpoint.async_fn_name}_pl')
                else:
                    # For non-paginated endpoints, use the standard config check
                    df_config = self._get_dataframe_config(endpoint)
                    if df_config.generate_pandas:
                        endpoint_names.append(f'{endpoint.sync_fn_name}_df')
                        endpoint_names.append(f'{endpoint.async_fn_name}_df')
                    if df_config.generate_polars:
                        endpoint_names.append(f'{endpoint.sync_fn_name}_pl')
                        endpoint_names.append(f'{endpoint.async_fn_name}_pl')

        # Import endpoints from endpoints.py
        endpoints_file_stem = self.config.endpoints_file.replace('.py', '')
        if endpoint_names:
            body.append(
                ast.ImportFrom(
                    module=endpoints_file_stem,
                    names=[
                        ast.alias(name=name, asname=None)
                        for name in sorted(endpoint_names)
                    ],
                    level=1,
                )
            )
            all_names.extend(endpoint_names)

        # Import Client from client.py
        body.append(
            ast.ImportFrom(
                module='client',
                names=[ast.alias(name='Client', asname=None)],
                level=1,
            )
        )
        all_names.append('Client')

        # Import BaseClient from _client.py
        base_client_name = f'Base{client_class_name}'
        body.append(
            ast.ImportFrom(
                module='_client',
                names=[ast.alias(name=base_client_name, asname=None)],
                level=1,
            )
        )
        all_names.append(base_client_name)

        # Also get all model names from typegen
        all_model_names = {
            type_.name
            for type_ in self.typegen.types.values()
            if type_.name and type_.implementation_ast
        }
        if all_model_names:
            body.append(
                ast.ImportFrom(
                    module=self.config.models_file.replace('.py', ''),
                    names=[
                        ast.alias(name=name, asname=None)
                        for name in sorted(all_model_names)
                    ],
                    level=1,
                )
            )
            all_names.extend(all_model_names)

        # Add __all__ at the beginning
        body.insert(0, _all(sorted(set(all_names))))

        # Write __init__.py
        init_path = directory / '__init__.py'
        write_mod(body, init_path)

    def _generate_split_endpoints(
        self,
        directory: UPath,
        models_file: UPath,
        endpoints: list[Endpoint],
        base_url: str,
        client_class_name: str,
    ) -> list[str]:
        """Generate split endpoint modules based on configuration.

        Args:
            directory: Output directory.
            models_file: Path to the models file.
            endpoints: List of Endpoint objects.
            base_url: The base URL for API requests.
            client_class_name: Name of the client class (e.g., 'SwaggerPetstoreOpenAPI30Client').

        Returns:
            List of relative paths to generated files.
        """
        from otterapi.codegen.splitting import (
            ModuleTreeBuilder,
            SplitModuleEmitter,
        )

        # Build the module tree
        builder = ModuleTreeBuilder(self.config.module_split)
        tree = builder.build(endpoints)

        # Emit the split modules
        emitter = SplitModuleEmitter(
            config=self.config.module_split,
            output_dir=directory,
            models_file=models_file,
            models_import_path=self.config.models_import_path,
            client_class_name=client_class_name,
            dataframe_config=self.config.dataframe,
            pagination_config=self.config.pagination,
        )

        emitted = emitter.emit(
            tree=tree,
            base_url=base_url,
            typegen_types=self.typegen.types,
        )

        # Collect generated file paths
        output_name = self.config.output
        generated_files = []
        for module in emitted:
            rel_path = str(module.path.relative_to(directory))
            generated_files.append(f'{output_name}/{rel_path}')

        # Add __init__.py files
        generated_files.append(f'{output_name}/__init__.py')

        return generated_files

    def _get_client_class_name(self) -> str:
        """Get the client class name from config or derive from API title."""
        if self.config.client_class_name:
            return self.config.client_class_name

        # Derive from API title
        if self.openapi and self.openapi.info and self.openapi.info.title:
            title = self.openapi.info.title
            # Convert to PascalCase and add Client suffix
            name = sanitize_identifier(title)
            if not name.endswith('Client'):
                name = f'{name}Client'
            return name

        return 'APIClient'

    def _generate_client_file(
        self,
        directory: UPath,
        endpoints: list[Endpoint],
        base_url: str,
        client_name: str,
    ) -> list[str]:
        """Generate the client class files.

        Generates:
        - _client.py: Always regenerated, contains BaseClient class
        - client.py: Created once if missing, user can customize
        - _dataframe.py: Generated if DataFrame methods are enabled

        Args:
            directory: Output directory.
            endpoints: List of Endpoint objects.
            base_url: Default base URL from spec.
            client_name: Name for the client class.

        Returns:
            List of relative paths to generated files.
        """
        base_client_name = f'Base{client_name}'

        # Convert endpoints to EndpointInfo for client generation
        endpoint_infos = self._endpoints_to_info(endpoints)

        # Check if any endpoint has DataFrame methods
        has_dataframe_methods = any(
            ep.dataframe_config.generate_pandas or ep.dataframe_config.generate_polars
            for ep in endpoint_infos
        )

        # Also check if pagination + dataframe is enabled
        # Paginated endpoints always get dataframe methods if dataframe is enabled,
        # regardless of whether the original endpoint returns a list
        if (
            not has_dataframe_methods
            and self.config.pagination.enabled
            and self.config.dataframe.enabled
            and (self.config.dataframe.pandas or self.config.dataframe.polars)
        ):
            for endpoint in endpoints:
                pag_config = self._get_pagination_config(endpoint)
                if pag_config:
                    # Check if endpoint is not explicitly disabled for dataframe
                    endpoint_df_config = self.config.dataframe.endpoints.get(
                        endpoint.sync_fn_name
                    )
                    if not (endpoint_df_config and endpoint_df_config.enabled is False):
                        has_dataframe_methods = True
                        break

        output_name = self.config.output
        generated_files = []

        # Generate _dataframe.py if needed
        if has_dataframe_methods:
            generate_dataframe_module(directory)
            generated_files.append(f'{output_name}/_dataframe.py')

        # Generate base client class (infrastructure only, no endpoint methods)
        class_ast, client_imports = generate_base_client_class(
            class_name=base_client_name,
            default_base_url=base_url,
            default_timeout=30.0,
        )

        # Build the _client.py file
        body: list[ast.stmt] = []
        import_collector = ImportCollector()
        import_collector.add_imports(client_imports)

        # No model imports needed in _client.py anymore - models are imported in module files

        # Add other imports
        for import_stmt in import_collector.to_ast():
            body.insert(0, import_stmt)

        # Add TypeVar definition: T = TypeVar('T')
        typevar_def = _assign(
            _name('T'),
            _call(
                func=_name('TypeVar'),
                args=[ast.Constant(value='T')],
            ),
        )
        body.append(typevar_def)

        # Add __all__ export (include APIError)
        body.append(_all([base_client_name, 'APIError']))

        # Add APIError class
        api_error_class = generate_api_error_class()
        body.append(api_error_class)

        # Add the client class
        body.append(class_ast)

        # Write _client.py (always regenerated)
        client_file = directory / '_client.py'
        write_mod(body, client_file)
        generated_files.append(f'{output_name}/_client.py')

        # Generate client.py stub (only if it doesn't exist)
        user_client_file = directory / 'client.py'
        if not user_client_file.exists():
            stub_content = generate_client_stub(
                class_name=client_name,
                base_class_name=base_client_name,
                module_name='_client',
            )
            user_client_file.write_text(stub_content)
            generated_files.append(f'{output_name}/client.py')

        return generated_files

    def _endpoints_to_info(self, endpoints: list[Endpoint]) -> list[EndpointInfo]:
        """Convert Endpoint objects to EndpointInfo for client generation."""
        infos = []
        for ep in endpoints:
            # Determine DataFrame configuration for this endpoint
            dataframe_config = self._get_dataframe_config(ep)

            info = EndpointInfo(
                name=ep.fn.name,
                async_name=ep.async_fn.name,
                method=ep.method,
                path=ep.path,
                parameters=ep.parameters,
                request_body=ep.request_body,
                response_type=ep.response_type,
                response_infos=ep.response_infos,
                description=ep.description,
                dataframe_config=dataframe_config,
            )
            infos.append(info)
        return infos

    def _get_dataframe_config(self, endpoint: Endpoint) -> DataFrameMethodConfig:
        """Get the DataFrame method configuration for an endpoint.

        Args:
            endpoint: The endpoint to check.

        Returns:
            DataFrameMethodConfig with generation flags and path.

        Note:
            This method delegates to get_dataframe_config_for_endpoint() from dataframe_utils.
        """
        return get_dataframe_config_for_endpoint(endpoint, self.config.dataframe)

    def _endpoint_returns_list(self, endpoint: Endpoint) -> bool:
        """Check if an endpoint returns a list type.

        Args:
            endpoint: The endpoint to check.

        Returns:
            True if the endpoint returns a list, False otherwise.

        Note:
            This method delegates to endpoint_returns_list() from dataframe_utils.
        """
        return endpoint_returns_list(endpoint)

    def _get_pagination_config(
        self, endpoint: Endpoint
    ) -> PaginationMethodConfig | None:
        """Get the pagination method configuration for an endpoint.

        Args:
            endpoint: The endpoint to check.

        Returns:
            PaginationMethodConfig if pagination is configured, None otherwise.
        """
        return get_pagination_config_for_endpoint(
            endpoint.sync_fn_name,
            self.config.pagination,
            endpoint.parameters,
        )

    def _get_item_type_ast(self, endpoint: Endpoint) -> ast.expr | None:
        """Extract the item type AST from a list response type.

        For example, if response_type is list[User], returns the AST for User.

        Args:
            endpoint: The endpoint to check.

        Returns:
            The AST expression for the item type, or None if not a list type.
        """
        if not endpoint.response_type or not endpoint.response_type.annotation_ast:
            return None

        ann = endpoint.response_type.annotation_ast
        if isinstance(ann, ast.Subscript):
            if isinstance(ann.value, ast.Name) and ann.value.id == 'list':
                return ann.slice

        return None
