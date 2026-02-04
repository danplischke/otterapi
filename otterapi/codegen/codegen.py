"""Code generation module for OtterAPI.

This module provides the main Codegen class that orchestrates the generation
of Python client code from OpenAPI specifications.
"""

import ast
import http
import os

from upath import UPath

from otterapi.codegen.ast_utils import _all, _assign, _call, _name
from otterapi.codegen.builders.model_collector import collect_used_model_names
from otterapi.codegen.client_generator import (
    EndpointInfo,
    generate_base_client_class,
    generate_client_stub,
)
from otterapi.codegen.dataframe_generator import generate_dataframe_module
from otterapi.codegen.dataframe_utils import (
    DataFrameMethodConfig,
    endpoint_returns_list,
    get_dataframe_config_for_endpoint,
)
from otterapi.codegen.endpoints import async_request_fn, request_fn
from otterapi.codegen.import_collector import ImportCollector
from otterapi.codegen.processors.parameter_processor import ParameterProcessor
from otterapi.codegen.processors.response_processor import ResponseProcessor
from otterapi.codegen.schema_loader import SchemaLoader
from otterapi.codegen.types import (
    Endpoint,
    Parameter,
    RequestBodyInfo,
    ResponseInfo,
    Type,
    TypeGenerator,
)
from otterapi.codegen.utils import (
    OpenAPIProcessor,
    sanitize_identifier,
    to_snake_case,
    write_mod,
)
from otterapi.config import DocumentConfig
from otterapi.openapi.v3_2.v3_2 import OpenAPI as OpenAPIv3_2, Operation

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

        # Initialize processors
        self._response_processor = ResponseProcessor(self.typegen)
        self._parameter_processor = ParameterProcessor(self.typegen, self)

    def _extract_response_info(self, operation: Operation) -> dict[int, ResponseInfo]:
        """Extract response information including content type from an operation.

        This method extracts response schemas and content types for each status code.
        When multiple content types are available for a response, it prefers JSON
        content types for better type safety.

        Args:
            operation: The OpenAPI operation to extract responses from.

        Returns:
            Dictionary mapping status codes to ResponseInfo objects.

        Note:
            This method delegates to ResponseProcessor.extract_response_info().
        """
        return self._response_processor.extract_response_info(operation)

    def _create_response_union(self, types: list[Type]) -> Type:
        """Create a union type from multiple response types.

        Args:
            types: List of response types to combine into a union.

        Returns:
            A union Type combining all the input types.

        Note:
            This method delegates to ResponseProcessor.create_response_union().
        """
        return self._response_processor.create_response_union(types)

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

        Note:
            This method delegates to ResponseProcessor.get_response_models().
        """
        return self._response_processor.get_response_models(operation)

    def _extract_operation_parameters(self, operation: Operation) -> list[Parameter]:
        """Extract path, query, header, and cookie parameters from an operation.

        Args:
            operation: The OpenAPI operation to extract parameters from.

        Returns:
            List of Parameter objects for path/query/header/cookie parameters.

        Note:
            This method delegates to ParameterProcessor.extract_operation_parameters().
        """
        return self._parameter_processor.extract_operation_parameters(operation)

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

        Note:
            This method delegates to ParameterProcessor.extract_request_body().
        """
        return self._parameter_processor.extract_request_body(operation)

    def _get_param_model(
        self, operation: Operation
    ) -> tuple[list[Parameter], RequestBodyInfo | None]:
        """Get all parameters and request body info for an operation.

        Args:
            operation: The OpenAPI operation to extract parameters from.

        Returns:
            A tuple of (parameters, request_body_info) where:
            - parameters: List of Parameter objects (path, query, header)
            - request_body_info: RequestBodyInfo object or None

        Note:
            This method delegates to ParameterProcessor.get_param_model().
        """
        return self._parameter_processor.get_param_model(operation)

    def _generate_endpoint(
        self, path: str, method: str, operation: Operation
    ) -> Endpoint:
        """Generate an endpoint with sync and async functions.

        Args:
            path: The API path for the endpoint.
            method: The HTTP method (get, post, etc.).
            operation: The OpenAPI operation definition.

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
        )

        body: list[ast.stmt] = []
        import_collector = ImportCollector()

        # Add default client variable and _get_client() function
        client_stmts, client_imports = build_default_client_code()
        body.extend(client_stmts)
        import_collector.add_imports(client_imports)

        # Track if we need DataFrame type hints
        has_dataframe_methods = False

        # Add standalone endpoint functions
        endpoint_names = set()
        for endpoint in endpoints:
            # Build sync standalone function
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

            # Build async standalone function
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

            # Generate DataFrame methods if configured
            if self.config.dataframe.enabled:
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
                module='._dataframe',
                names=[
                    ast.alias(name='to_pandas', asname=None),
                    ast.alias(name='to_polars', asname=None),
                ],
                level=0,
            )
            body.insert(0, dataframe_import)

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

        endpoints = self._generate_endpoints()

        models_file = directory / self.config.models_file
        self._generate_models_file(models_file)

        base_url = self._resolve_base_url()

        # Generate client class
        client_name = self._get_client_class_name()

        # Check if module splitting is enabled
        if self.config.module_split.enabled:
            self._generate_split_endpoints(
                directory, models_file, endpoints, base_url, client_name
            )
        else:
            # Original single-file generation
            endpoints_file = directory / self.config.endpoints_file
            self._generate_endpoint_file(endpoints_file, models_file, endpoints)

        self._generate_client_file(directory, endpoints, base_url, client_name)

        # Write __init__.py only if not using module splitting (splitting handles its own __init__.py)
        if not self.config.module_split.enabled:
            self._generate_init_file(directory, endpoints, client_name)

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

            # Add DataFrame method names if configured
            if self.config.dataframe.enabled:
                df_config = self._get_dataframe_config(endpoint)
                if df_config.generate_pandas:
                    endpoint_names.append(f'{endpoint.sync_fn_name}_df')
                    endpoint_names.append(f'{endpoint.async_fn_name}_df')
                if df_config.generate_polars:
                    endpoint_names.append(f'{endpoint.sync_fn_name}_pl')
                    endpoint_names.append(f'{endpoint.async_fn_name}_pl')

        # Import endpoints from endpoints.py
        endpoints_file_stem = self.config.endpoints_file.replace('.py', '')
        body.append(
            ast.ImportFrom(
                module=endpoints_file_stem,
                names=[
                    ast.alias(name=name, asname=None) for name in sorted(endpoint_names)
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
    ) -> None:
        """Generate split endpoint modules based on configuration.

        Args:
            directory: Output directory.
            models_file: Path to the models file.
            endpoints: List of Endpoint objects.
            base_url: The base URL for API requests.
            client_class_name: Name of the client class (e.g., 'SwaggerPetstoreOpenAPI30Client').
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
        )

        emitter.emit(
            tree=tree,
            base_url=base_url,
            typegen_types=self.typegen.types,
        )

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
    ) -> None:
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
        """
        base_client_name = f'Base{client_name}'

        # Convert endpoints to EndpointInfo for client generation
        endpoint_infos = self._endpoints_to_info(endpoints)

        # Check if any endpoint has DataFrame methods
        has_dataframe_methods = any(
            ep.dataframe_config.generate_pandas or ep.dataframe_config.generate_polars
            for ep in endpoint_infos
        )

        # Generate _dataframe.py if needed
        if has_dataframe_methods:
            generate_dataframe_module(directory)

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

        # Add __all__ export
        body.append(_all([base_client_name]))

        # Add the class
        body.append(class_ast)

        # Write _client.py (always regenerated)
        client_file = directory / '_client.py'
        write_mod(body, client_file)

        # Generate client.py stub (only if it doesn't exist)
        user_client_file = directory / 'client.py'
        if not user_client_file.exists():
            stub_content = generate_client_stub(
                class_name=client_name,
                base_class_name=base_client_name,
                module_name='_client',
            )
            user_client_file.write_text(stub_content)

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
