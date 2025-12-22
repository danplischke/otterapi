import ast
import http
import logging
import os

from upath import UPath

from otterapi.codegen.ast_utils import _all, _union_expr
from otterapi.codegen_v2.ast_utils import _assign, _name, _call
from otterapi.codegen_v2.endpoints import request_fn, async_request_fn
from otterapi.codegen_v2.import_collector import ImportCollector
from otterapi.codegen_v2.schema_loader import SchemaLoader
from otterapi.codegen_v2.types import TypeGenerator, Endpoint, Parameter, Type, ResponseInfo, RequestBodyInfo
from otterapi.codegen_v2.utils import OpenAPIProcessor, write_mod, sanitize_identifier, sanitize_parameter_field_name, \
    write_init_file
from otterapi.config import DocumentConfig
from otterapi.openapi.v3_2.v3_2 import OpenAPI as OpenAPIv3_2, Operation


HTTP_METHODS = [method.value.lower() for method in http.HTTPMethod]


class Codegen(OpenAPIProcessor):

    def __init__(self, config: DocumentConfig, schema_loader: SchemaLoader | None = None):
        super().__init__(None)
        self.config = config
        self.openapi: OpenAPIv3_2 | None = None
        self._schema_loader = schema_loader or SchemaLoader()

    def _load_schema(self):
        self.openapi = self._schema_loader.load(self.config.source)
        self.typegen = TypeGenerator(self.openapi)

    def _extract_response_info(self, operation: Operation) -> dict[int, ResponseInfo]:
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

        # Content type priority - prefer JSON for type safety
        json_content_types = {'application/json', 'text/json'}

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
            selected_content_type = None
            selected_media_type = None

            # First, try to find a JSON content type
            for content_type, media_type in response.content.items():
                if content_type in json_content_types or content_type.endswith('+json'):
                    selected_content_type = content_type
                    selected_media_type = media_type
                    break

            # If no JSON found, use the first available content type
            if selected_content_type is None:
                selected_content_type, selected_media_type = next(iter(response.content.items()))

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
        # Aggregate imports from all types
        union_type.copy_imports_from_sub_types(types)
        union_type.add_annotation_import('typing', 'Union')
        return union_type

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

        # Collect all types that have JSON responses
        json_types = [r.type for r in response_list if r.is_json and r.type]

        # Also include non-JSON response types in the union (bytes, str, Response)
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

        all_types = json_types + non_json_types

        if len(all_types) == 0:
            return response_list, None
        elif len(all_types) == 1:
            return response_list, all_types[0]
        else:
            return response_list, self._create_response_union(all_types)

    def _extract_operation_parameters(self, operation: Operation) -> list[Parameter]:
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

    def _extract_request_body(self, operation: Operation) -> RequestBodyInfo | None:
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

        body, _ = self._resolve_reference(operation.requestBody)
        if not body.content:
            return None

        # Content type priority - prefer JSON for type safety
        json_content_types = {'application/json', 'text/json'}

        selected_content_type = None
        selected_media_type = None

        # First, try to find a JSON content type
        for content_type, media_type in body.content.items():
            if content_type in json_content_types or content_type.endswith('+json'):
                selected_content_type = content_type
                selected_media_type = media_type
                break

        # If no JSON found, use the first available content type
        if selected_content_type is None:
            selected_content_type, selected_media_type = next(iter(body.content.items()))

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

    def _get_param_model(self, operation: Operation) -> tuple[list[Parameter], RequestBodyInfo | None]:
        """Get all parameters and request body info for an operation.

        Args:
            operation: The OpenAPI operation to extract parameters from.

        Returns:
            A tuple of (parameters, request_body_info) where:
            - parameters: List of Parameter objects (path, query, header)
            - request_body_info: RequestBodyInfo object or None
        """
        params = self._extract_operation_parameters(operation)
        body_info = self._extract_request_body(operation)

        return params, body_info

    def _generate_endpoint(self, path: str, method: str, operation: Operation) -> Endpoint:
        """Generate an endpoint with sync and async functions.

        Args:
            path: The API path for the endpoint.
            method: The HTTP method (get, post, etc.).
            operation: The OpenAPI operation definition.

        Returns:
            An Endpoint object containing the generated functions and imports.
        """
        fn_name = (
                operation.operationId
                or f'{method}_{path.replace("/", "_").replace("{", "").replace("}", "")}'
        )
        async_fn_name = f'a{fn_name}'

        parameters, request_body_info = self._get_param_model(operation)
        response_infos, response_model = self._get_response_models(operation)

        async_fn, async_imports = async_request_fn(
            name=async_fn_name,
            method=method,
            path=path,
            response_model=response_model,
            docs=operation.description,
            parameters=parameters,
            response_infos=response_infos,
            request_body_info=request_body_info,
        )

        sync_fn, imports = request_fn(
            name=fn_name,
            method=method,
            path=path,
            response_model=response_model,
            docs=operation.description,
            parameters=parameters,
            response_infos=response_infos,
            request_body_info=request_body_info,
        )

        ep = Endpoint(
            sync_ast=sync_fn,
            sync_fn_name=fn_name,
            async_fn_name=async_fn_name,
            async_ast=async_fn,
            name=fn_name,
        )

        ep.add_imports([imports, async_imports])

        for param in parameters:
            if param.type and param.type.annotation_imports:
                ep.add_imports([param.type.annotation_imports])

        # Add imports for request body type if present
        if request_body_info and request_body_info.type:
            if request_body_info.type.annotation_imports:
                ep.add_imports([request_body_info.type.annotation_imports])

        return ep

    def _generate_endpoints(self) -> list[Endpoint]:
        """Generate all endpoints from the OpenAPI paths.

        Returns:
            List of generated Endpoint objects.
        """
        endpoints: list[Endpoint] = []
        # Paths is a RootModel, access .root to get the underlying dict
        paths_dict = self.openapi.paths.root if hasattr(self.openapi.paths, 'root') else self.openapi.paths
        for path, path_item in paths_dict.items():
            for method in HTTP_METHODS:
                operation = getattr(path_item, method, None)
                if operation:
                    ep = self._generate_endpoint(path, method, operation)
                    endpoints.append(ep)
        return endpoints

    def _resolve_base_url(self) -> str:
        """Resolve the base URL from config or OpenAPI spec.

        Returns:
            The base URL to use for API requests.

        Raises:
            ValueError: If no base URL can be determined or multiple servers are defined.
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

        return baseurl

    def _collect_model_names(self) -> set[str]:
        """Collect all model names that have implementations.

        Returns:
            Set of model names to be imported in the endpoints file.
        """
        model_names = set()
        for type_name, type_ in self.typegen.types.items():
            if type_.name and type_.implementation_ast:
                model_names.add(type_.name)
        return model_names

    def _create_model_import(self, models_file: UPath, model_names: set[str]) -> ast.ImportFrom:
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
        """Build the body of the endpoints file with functions and collect imports.

        Args:
            baseurl: The base URL for API requests.
            endpoints: List of Endpoint objects to include.

        Returns:
            Tuple of (body statements, import collector, endpoint names).
        """
        from otterapi.codegen_v2.endpoints import base_async_request_fn, base_request_fn

        # Initialize body with constants
        body: list[ast.stmt] = [
            _assign(_name('BASE_URL'), ast.Constant(baseurl)),
            _assign(_name('T'), _call(_name('TypeVar'), [ast.Constant('T')])),
        ]

        import_collector = ImportCollector()

        # Add base request functions
        sync_base_fn, sync_base_imports = base_request_fn()
        async_base_fn, async_base_imports = base_async_request_fn()

        body.append(sync_base_fn)
        body.append(async_base_fn)

        import_collector.add_imports(sync_base_imports)
        import_collector.add_imports(async_base_imports)

        # Add endpoint functions and collect imports
        endpoint_names = set()
        for endpoint in endpoints:
            endpoint_names.update([endpoint.sync_fn_name, endpoint.async_fn_name])
            body.append(endpoint.sync_ast)
            body.append(endpoint.async_ast)
            import_collector.add_imports(endpoint.imports)

        return body, import_collector, endpoint_names

    def _generate_endpoint_file(
            self, path: UPath, models_file: UPath, endpoints: list[Endpoint]
    ) -> None:
        """Generate the endpoints Python file.

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

        # Add model imports if any models exist
        model_names = self._collect_model_names()
        if model_names:
            model_import = self._create_model_import(models_file, model_names)
            body.insert(0, model_import)

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

        endpoints = self._generate_endpoints()

        models_file = directory / self.config.models_file
        self._generate_models_file(models_file)

        endpoints_file = directory / self.config.endpoints_file
        self._generate_endpoint_file(endpoints_file, models_file, endpoints)

        write_init_file(directory)

