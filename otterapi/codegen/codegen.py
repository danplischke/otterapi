"""Code generation module for OtterAPI.

This module provides the main Codegen class that orchestrates the generation
of Python client code from OpenAPI specifications.
"""

import ast
import fnmatch
import logging
import os
from importlib.resources import files
from typing import Any, Literal, cast
from urllib.parse import urljoin, urlparse

from upath import UPath

from otterapi.codegen._features import all_features, write_enabled_features
from otterapi.codegen.ast_utils import (
    ImportCollector,
    _all,
    _assign,
    _name,
    _union_expr,
)
from otterapi.codegen.client import (
    _exported_error_names,
    generate_api_error_hierarchy,
    generate_base_client_class,
    generate_client_stub,
)
from otterapi.codegen.emit import (
    EmitConfig,
    TypeResolver,
    build_endpoints_module_body,
)
from otterapi.codegen.endpoints import async_request_fn, request_fn
from otterapi.codegen.schema import SchemaLoader
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
    sanitize_parameter_field_name,
    to_snake_case,
    write_mod,
)
from otterapi.config import DocumentConfig
from otterapi.openapi.constants import HTTP_METHODS, MediaType
from otterapi.openapi.v3_2.v3_2 import (
    OpenAPI as OpenAPIv3_2,
    Operation,
    Parameter as OpenAPIParameter,
    Reference,
    RequestBody as OpenAPIRequestBody,
    Response as OpenAPIResponse,
)

logger = logging.getLogger(__name__)


def _load_models_mixin_source() -> str:
    return (
        files('otterapi.codegen.runtime')
        .joinpath('_models_mixin.py')
        .read_text('utf-8')
    )


def _topo_sort_by_base_classes(
    types_list: list,
    model_names: set[str],
) -> list:
    """Return *types_list* re-ordered so every base class precedes its subclasses.

    ``get_sorted_types()`` uses field-level dependency edges.  When it breaks a
    field-level cycle it may place a subclass before its base class.
    ``from __future__ import annotations`` defers annotation evaluation, so
    field-level cycles are fine — but ``class Foo(Bar):`` base expressions are
    evaluated *eagerly* at class-definition time.  Inheritance always forms a
    DAG in Python (cycles are a ``TypeError``), so a simple DFS topological sort
    on base-class edges always terminates and produces a valid ordering.

    Types with no name (primitive aliases, etc.) are appended unchanged after
    the named types.
    """
    name_to_type: dict[str, object] = {t.name: t for t in types_list if t.name}

    def base_deps(t: object) -> set[str]:
        impl = t.implementation_ast  # type: ignore[attr-defined]
        if not isinstance(impl, ast.ClassDef):
            return set()
        return {
            b.id for b in impl.bases if isinstance(b, ast.Name) and b.id in model_names
        }

    result: list = []
    visited: set[str] = set()

    def dfs(name: str) -> None:
        if name in visited:
            return
        visited.add(name)
        t = name_to_type.get(name)
        if t is None:
            return
        for base_name in base_deps(t):
            dfs(base_name)
        result.append(t)

    for t in types_list:
        if t.name:
            dfs(t.name)

    # Preserve unnamed entries (should be rare / none for ClassDef models)
    for t in types_list:
        if not t.name:
            result.append(t)

    return result


# Content types that should be treated as JSON
JSON_CONTENT_TYPES = {MediaType.JSON, MediaType.TEXT_JSON}


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
        self,
        config: DocumentConfig,
        schema_loader: SchemaLoader | None = None,
        *,
        format_output: bool = True,
        validate_output: bool = True,
        generate_endpoints: bool = True,
        create_py_typed: bool = True,
        lenient: bool = False,
    ):
        """Initialize the code generator.

        Args:
            config: Configuration specifying source schema and output location.
            schema_loader: Optional custom schema loader. If not provided,
                          a default SchemaLoader will be created.
            format_output: Whether to format generated code with ruff/black.
            validate_output: Whether to validate generated code syntax.
            generate_endpoints: Whether to generate endpoint functions.
            create_py_typed: Whether to create a py.typed marker file.
            lenient: Whether to drop unrecognized, structurally-invalid fields
                from the input specification instead of failing validation.
                Ignored when a custom ``schema_loader`` is supplied.
        """
        super().__init__(None)
        self.config = config
        self.openapi: OpenAPIv3_2 | None = None
        self.typegen: TypeGenerator | None = None
        self._schema_loader = schema_loader or SchemaLoader(lenient=lenient)
        self.format_output = format_output
        self.validate_output = validate_output
        self.generate_endpoints = generate_endpoints
        self.create_py_typed = create_py_typed

    @property
    def _adapter(self):
        """Facade over ``self.openapi`` that hides the parser's attribute shape.

        Constructed lazily on first use so subclasses / tests that swap in
        a different ``openapi`` after construction still work.
        """
        from otterapi.codegen._openapi_adapter import OpenAPIAdapter

        if self.openapi is None:
            raise RuntimeError(
                'Codegen.openapi is not loaded yet; call _load_schema() first.'
            )
        return OpenAPIAdapter(self.openapi)

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
        self.typegen = TypeGenerator(
            self.openapi, pydantic_version=self.config.pydantic_version
        )

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
                logger.debug(f'Skipping non-numeric status code: {status_code_str}')
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
                assert self.typegen is not None
                response_type = self.typegen.schema_to_type(
                    selected_media_type.schema_,
                    base_name=f'{sanitize_identifier(operation.operationId or "")}Response',
                )

            responses[status_code] = ResponseInfo(
                status_code=status_code,
                content_type=selected_content_type,
                type=response_type,
            )

        return responses

    def _select_content_type(self, content: dict) -> tuple[str, Any]:
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
            annotation_ast=_union_expr(
                [t.annotation_ast for t in types if t.annotation_ast is not None]
            ),
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

    @staticmethod
    def _dedupe_types(types: list[Type]) -> list[Type]:
        """Drop duplicate types (by rendered annotation), preserving order.

        An operation often maps several status codes (e.g. ``200`` and
        ``default``, or ``200`` and ``201``) to the same response model. Without
        deduplication the response union collapses to ``X | X``.
        """
        seen: set[str] = set()
        unique: list[Type] = []
        for t in types:
            key = (
                ast.dump(t.annotation_ast) if t.annotation_ast is not None else repr(t)
            )
            if key not in seen:
                seen.add(key)
                unique.append(t)
        return unique

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

        all_types = self._dedupe_types(json_types + non_json_types)

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
            param = self._resolve_parameter_reference(param_or_ref)
            if param is None:
                continue

            all_params.append(self._build_parameter(param))
            param_keys_seen.add((param.name, param.in_))

        # Then, add path-level parameters that weren't overridden
        for param_or_ref in path_item_parameters or []:
            param = self._resolve_parameter_reference(param_or_ref)
            if param is None:
                continue

            if (param.name, param.in_) not in param_keys_seen:
                all_params.append(self._build_parameter(param))

        # Distinct OpenAPI parameters can map to the same Python identifier
        # (e.g. a query and header of the same name, or "user id" and
        # "user-id" which both sanitize to "user_id"), which would emit a
        # function with a duplicate argument. Disambiguate the generated
        # identifier while preserving the wire name (param.name).
        used_names: set[str] = set()
        for built_param in all_params:
            candidate = built_param.name_sanitized
            if candidate in used_names:
                suffix = 2
                while f'{candidate}_{suffix}' in used_names:
                    suffix += 1
                candidate = f'{candidate}_{suffix}'
                built_param.name_sanitized = candidate
            used_names.add(candidate)

        return all_params

    def _build_parameter(self, param: OpenAPIParameter) -> Parameter:
        """Build a Parameter from a resolved OpenAPI parameter object."""
        param_type = None
        if param.schema_:
            assert self.typegen is not None
            param_type = self.typegen.schema_to_type(param.schema_)

        return Parameter(
            name=param.name,
            name_sanitized=sanitize_parameter_field_name(param.name),
            location=cast(
                "Literal['query', 'path', 'header', 'cookie', 'body']", param.in_
            ),
            required=param.required or False,
            type=param_type,
            description=param.description,
        )

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
                logger.warning(
                    f'Unsupported parameter reference format: {param_or_ref.ref}'
                )
                return None

            param_name = param_or_ref.ref.split('/')[-1]
            # The adapter encapsulates "components/parameters/<name>" lookup so
            # this call site stays agnostic of the v3.2 attribute path.
            resolved = self._adapter.components_parameter(param_name)
            if resolved is None:
                logger.warning(
                    f"Referenced parameter '{param_name}' not found in components.parameters"
                )
                return None
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
                logger.warning(
                    f'Unsupported response reference format: {response_or_ref.ref}'
                )
                return None

            response_name = response_or_ref.ref.split('/')[-1]
            resolved = self._adapter.components_response(response_name)
            if resolved is None:
                logger.warning(
                    f"Referenced response '{response_name}' not found in components.responses"
                )
                return None
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
                logger.warning(
                    f'Unsupported request body reference format: {body_or_ref.ref}'
                )
                return None

            body_name = body_or_ref.ref.split('/')[-1]
            resolved = self._adapter.components_request_body(body_name)
            if resolved is None:
                logger.warning(
                    f"Referenced request body '{body_name}' not found in components.requestBodies"
                )
                return None
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
            assert self.typegen is not None
            body_type = self.typegen.schema_to_type(
                selected_media_type.schema_,
                base_name=f'{sanitize_identifier(operation.operationId or "")}RequestBody',
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

    @staticmethod
    def _path_based_name(method: str, path: str) -> str:
        """Derive a raw function name from an HTTP method and URL path.

        Example: ``('get', '/v1/feed/trial/changes')`` ->
        ``'get__v1_feed_trial_changes'`` (``to_snake_case`` collapses the stray
        underscores afterwards). The method is always included so that different
        verbs on the same path (e.g. GET vs POST ``/search/count``) stay distinct.
        """
        cleaned = path.replace('/', '_').replace('{', '').replace('}', '')
        return f'{method}_{cleaned}'

    def _raw_operation_name(self, method: str, path: str, operation: Operation) -> str:
        """Return the pre-sanitized function name for an operation.

        Honors the ``function_naming`` config option: ``'path'`` always derives
        the name from the method and path (useful for specs that reuse
        ``operationId`` across many paths, which otherwise collapse to a single
        name with numeric suffixes); otherwise the ``operationId`` is used,
        falling back to the path when it is absent.
        """
        if self.config.function_naming == 'path':
            return self._path_based_name(method, path)
        return operation.operationId or self._path_based_name(method, path)

    @staticmethod
    def _unique_fn_name(raw_name: str, used_names: set[str]) -> str:
        """Snake-case ``raw_name`` and disambiguate it against ``used_names``.

        Guarantees no two endpoints share a function name even when distinct
        raw names sanitize to the same identifier (this can happen with
        path-based naming, e.g. ``/feed/changes`` and ``/feed-changes``).
        """
        fn_name = to_snake_case(raw_name)
        if fn_name in used_names:
            suffix = 2
            while f'{fn_name}_{suffix}' in used_names:
                suffix += 1
            fn_name = f'{fn_name}_{suffix}'
        used_names.add(fn_name)
        return fn_name

    def _generate_endpoint(
        self,
        path: str,
        method: str,
        operation: Operation,
        fn_name: str,
        path_item_parameters: list | None = None,
    ) -> Endpoint:
        """Generate an endpoint with sync and async functions.

        Args:
            path: The API path for the endpoint.
            method: The HTTP method (get, post, etc.).
            operation: The OpenAPI operation definition.
            fn_name: The already sanitized and de-duplicated sync function name.
            path_item_parameters: Optional list of path-level parameters to inherit.

        Returns:
            An Endpoint object containing the generated functions and imports.
        """
        async_fn_name = f'async_{fn_name}'

        parameters, request_body_info = self._get_param_model(
            operation, path_item_parameters
        )
        response_infos, response_model = self._get_response_models(operation)
        docs = self._build_operation_docs(operation)

        async_fn, async_imports = async_request_fn(
            name=async_fn_name,
            method=method,
            path=path,
            response_model=response_model,
            docs=docs,
            parameters=parameters,
            response_infos=response_infos,
            request_body_info=request_body_info,
        )

        sync_fn, imports = request_fn(
            name=fn_name,
            method=method,
            path=path,
            response_model=response_model,
            docs=docs,
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
        self._add_endpoint_type_imports(
            ep, parameters, request_body_info, response_model
        )

        return ep

    def _build_operation_docs(self, operation: Operation) -> str | None:
        """Build the docstring for an operation, including a deprecation notice."""
        docs = operation.description or ''
        if operation.deprecated:
            deprecation_notice = '.. deprecated::\n    This endpoint is deprecated.'
            docs = f'{docs}\n\n{deprecation_notice}' if docs else deprecation_notice
        return docs or None

    def _add_endpoint_type_imports(
        self,
        ep: Endpoint,
        parameters: list[Parameter],
        request_body_info: RequestBodyInfo | None,
        response_model: Type | None,
    ) -> None:
        """Collect annotation imports from parameter, request body, and response types."""
        for param in parameters:
            if param.type and param.type.annotation_imports:
                ep.add_imports([param.type.annotation_imports])

        if (
            request_body_info
            and request_body_info.type
            and request_body_info.type.annotation_imports
        ):
            ep.add_imports([request_body_info.type.annotation_imports])

        if response_model and response_model.annotation_imports:
            ep.add_imports([response_model.annotation_imports])

    def _generate_endpoints(self) -> list[Endpoint]:
        """Generate all endpoints from the OpenAPI paths.

        Returns:
            List of generated Endpoint objects.
        """
        endpoints: list[Endpoint] = []
        used_fn_names: set[str] = set()
        # Use the adapter for path access -- hides the "RootModel vs dict"
        # wrinkle from the rest of codegen.
        for path, path_item in self._adapter.paths().items():
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
                    fn_name = self._unique_fn_name(
                        self._raw_operation_name(method, path, operation),
                        used_fn_names,
                    )
                    ep = self._generate_endpoint(
                        path, method, operation, fn_name, path_item_parameters
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

        # Fetch declared servers through the adapter.
        servers = self._adapter.servers()
        if not servers:
            raise ValueError(
                'No base url provided. Make sure you specify the base_url in the otterapi config or the OpenAPI document contains a valid servers section'
            )

        # Only support single server
        if len(servers) > 1:
            raise ValueError(
                'Multiple servers are not supported. Set the base_url in the config.'
            )

        server = servers[0]
        baseurl = server.url
        if server.variables:
            for name, variable in server.variables.items():
                baseurl = baseurl.replace(f'{{{name}}}', variable.default)

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
                logger.info(
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

    def _generate_endpoint_file(
        self, path: UPath, endpoints: list[Endpoint]
    ) -> set[str]:
        """Generate the endpoints Python file with standalone functions.

        Returns the set of exported function names so callers can pass them
        directly to ``_generate_init_file`` without recomputing.

        The per-endpoint emission is delegated to the shared, layout-agnostic
        core in :mod:`otterapi.codegen.emit`, which the split-module writer uses
        too.
        """
        assert self.typegen is not None
        body, names = build_endpoints_module_body(
            endpoints,
            EmitConfig.from_document(self.config),
            TypeResolver(self.typegen.types),
            reexport_models=self.config.reexport_models,
            reexport_model_exclude_patterns=self.config.reexport_model_exclude_patterns,
        )
        write_mod(
            body,
            path,
            format_code=self.format_output,
            validate_code=self.validate_output,
        )
        return set(names)

    @staticmethod
    def _inject_html_repr_mixin(impl: ast.stmt, all_model_names: set[str]) -> ast.stmt:
        """Prepend ``_HtmlReprMixin`` as the first base of a model ClassDef.

        This makes each model render as an HTML table in Jupyter without
        touching the per-model implementation_ast (which is exec'd in unit
        tests).  Injection is skipped when a base is itself a generated model:
        that base already gets ``_HtmlReprMixin``, so adding it again would
        produce an unresolvable MRO (e.g. ``Child(_HtmlReprMixin, Parent)``
        where ``Parent`` is already ``Child(_HtmlReprMixin, BaseModel)``).
        """
        if not isinstance(impl, ast.ClassDef):
            return impl
        has_model_base = any(
            isinstance(b, ast.Name) and b.id in all_model_names for b in impl.bases
        )
        if has_model_base:
            return impl
        return ast.ClassDef(
            name=impl.name,
            bases=[
                ast.Name(id='_HtmlReprMixin', ctx=ast.Load()),
                *impl.bases,
            ],
            keywords=impl.keywords,
            body=impl.body,
            decorator_list=impl.decorator_list,
        )

    def _forward_ref_rebuild_stmts(self, all_names: set[str]) -> list[ast.stmt]:
        """Emit ``model_rebuild()`` / ``update_forward_refs()`` for cyclic models.

        Cyclic models need an explicit rebuild so Pydantic can resolve the
        forward references introduced by ``from __future__ import annotations``
        and self-referential schemas.
        """
        assert self.typegen is not None
        method = (
            'update_forward_refs'
            if self.typegen.pydantic_version == 1
            else 'model_rebuild'
        )
        cyclic: set[str] = getattr(self.typegen, '_cyclic', set())
        stmts: list[ast.stmt] = []
        for model_name in sorted(cyclic):
            if model_name in all_names:
                stmts.append(
                    ast.Expr(
                        ast.Call(
                            func=ast.Attribute(
                                value=ast.Name(id=model_name, ctx=ast.Load()),
                                attr=method,
                                ctx=ast.Load(),
                            ),
                            args=[],
                            keywords=[],
                        )
                    )
                )
        return stmts

    def _generate_models_file(self, path: UPath) -> None:
        """Generate the models Python file with Pydantic models.

        Args:
            path: Path where the models file should be written.
        """
        if self.typegen is None:
            raise RuntimeError(
                'TypeGenerator is not initialized; call _load_schema() first.'
            )

        body: list[ast.stmt] = []
        import_collector = ImportCollector()
        all_names = set()

        # Names of all types that will get _HtmlReprMixin injected.  Used
        # below to skip re-injection for classes whose base is already a
        # generated model (which would cause an MRO conflict).
        all_model_names: set[str] = {
            type_.name
            for type_ in self.typegen.types.values()
            if type_.name and isinstance(type_.implementation_ast, ast.ClassDef)
        }

        # Emit types in dependency order (dependencies before dependents) so
        # that forward references in field annotations resolve at class-creation
        # time.  get_sorted_types() returns dependents-first; reversing gives
        # the dependencies-first order needed for correct emission.
        # Types not reachable from get_sorted_types (e.g. orphaned names) are
        # appended at the end via the seen-set guard.
        sorted_types = list(reversed(self.typegen.get_sorted_types()))
        seen: set[str] = {t.name for t in sorted_types if t.name}
        for extra in self.typegen.types.values():
            if extra.name not in seen:
                sorted_types.append(extra)

        # Re-sort to guarantee every base class precedes every subclass.
        # get_sorted_types() uses field-level dependencies; when it breaks a
        # field-level cycle it may place a subclass before its base class.
        # from __future__ import annotations makes field-type annotations lazy,
        # but base classes in `class Foo(Base):` are evaluated EAGERLY, so the
        # ordering of ClassDef nodes in the emitted file must respect the
        # inheritance DAG.  Python forbids inheritance cycles, so this sort
        # always terminates.
        sorted_types = _topo_sort_by_base_classes(sorted_types, all_model_names)

        for type_ in sorted_types:
            if type_.implementation_ast:
                impl = type_.implementation_ast
                # Inject _HtmlReprMixin as the first base of every model class so
                # it renders as an HTML table in Jupyter.
                impl = self._inject_html_repr_mixin(impl, all_model_names)
                body.append(impl)
                if type_.name:
                    all_names.add(type_.name)

                # Collect imports from implementation and annotations
                import_collector.add_imports(type_.implementation_imports)
                import_collector.add_imports(type_.annotation_imports)

        # Emit model_rebuild() / update_forward_refs() for cyclic models so
        # that Pydantic can resolve the forward references introduced by
        # from __future__ import annotations and self-referential schemas.
        body.extend(self._forward_ref_rebuild_stmts(all_names))

        # Prepend the _HtmlReprMixin helper (defines _repr_html_ for Jupyter)
        body[0:0] = ast.parse(_load_models_mixin_source()).body

        # Add __all__ export
        body.insert(0, _all(sorted(all_names)))

        # Add all imports at the beginning
        for import_stmt in import_collector.to_ast():
            body.insert(0, import_stmt)

        # from __future__ import annotations must be the very first statement
        # so forward references in type annotations resolve lazily.  This is
        # required when cyclic models refer to each other before their class
        # bodies are fully defined.
        body.insert(
            0,
            ast.ImportFrom(
                module='__future__',
                names=[ast.alias(name='annotations')],
                level=0,
            ),
        )

        write_mod(
            body,
            path,
            format_code=self.format_output,
            validate_code=self.validate_output,
        )

    def generate(self):
        self._load_schema()

        if self.openapi is None:
            raise RuntimeError(
                'OpenAPI document was not loaded; _load_schema() failed silently.'
            )

        if not self._adapter.has_paths():
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

        for written in write_enabled_features(self.config, directory, all_features()):
            generated_files.append(f'{output_name}/{written.name}')

        # Generate client class
        client_name = self._get_client_class_name()

        # Check if module splitting is enabled
        if self.generate_endpoints:
            if self.config.module_split.enabled:
                split_files = self._generate_split_endpoints(
                    directory, models_file, endpoints, client_name
                )
                generated_files.extend(split_files)
            else:
                endpoints_file = directory / self.config.endpoints_file
                endpoint_names = self._generate_endpoint_file(endpoints_file, endpoints)
                generated_files.append(f'{output_name}/{self.config.endpoints_file}')
        else:
            endpoint_names = set()

        client_files = self._generate_client_file(
            directory, endpoints, base_url, client_name
        )
        generated_files.extend(client_files)

        # Write __init__.py only if not using module splitting (splitting handles its own __init__.py)
        if not self.config.module_split.enabled:
            self._generate_init_file(directory, endpoint_names, client_name)
            generated_files.append(f'{output_name}/__init__.py')

        if self.create_py_typed:
            py_typed = directory / 'py.typed'
            py_typed.touch()
            generated_files.append(f'{output_name}/py.typed')

        return generated_files

    @staticmethod
    def _add_reexport(
        body: list[ast.stmt],
        all_names: list[str],
        module: str,
        names: list[str],
    ) -> None:
        """Append ``from .<module> import <names>`` and record them for ``__all__``.

        No-op when ``names`` is empty so callers can pass conditionally-built
        export lists without guarding each one.
        """
        if not names:
            return
        body.append(
            ast.ImportFrom(
                module=module,
                names=[ast.alias(name=n, asname=None) for n in names],
                level=1,
            )
        )
        all_names.extend(names)

    def _add_feature_reexports(
        self, body: list[ast.stmt], all_names: list[str]
    ) -> None:
        """Re-export runtime helpers from any enabled feature modules.

        Keeps the package's public surface entirely in __init__.py; the
        underscore-prefixed feature files remain the "regenerated, do not edit"
        source of truth that users never import from directly.
        """
        if self.config.pagination.enabled:
            self._add_reexport(
                body,
                all_names,
                '_pagination',
                [
                    'paginate_offset',
                    'paginate_offset_async',
                    'paginate_cursor',
                    'paginate_cursor_async',
                    'paginate_page',
                    'paginate_page_async',
                    'iterate_offset',
                    'iterate_offset_async',
                    'iterate_cursor',
                    'iterate_cursor_async',
                    'iterate_page',
                    'iterate_page_async',
                ],
            )
        if self.config.export.enabled:
            self._add_reexport(
                body,
                all_names,
                '_export',
                [
                    'export',
                    'export_async',
                    'to_csv',
                    'to_csv_async',
                    'to_tsv',
                    'to_tsv_async',
                    'to_jsonl',
                    'to_jsonl_async',
                    'to_parquet',
                    'to_parquet_async',
                ],
            )
        if self.config.dataframe.enabled:
            df_exports: list[str] = []
            if self.config.dataframe.pandas:
                df_exports.append('to_pandas')
            if self.config.dataframe.polars:
                df_exports.append('to_polars')
            self._add_reexport(body, all_names, '_dataframe', df_exports)
        self._add_reexport(
            body,
            all_names,
            '_concurrency',
            ['run_concurrently', 'run_concurrently_async', 'run_sync'],
        )

    def _generate_init_file(
        self,
        directory: UPath,
        endpoint_names: set[str],
        client_class_name: str,
    ) -> None:
        """Generate __init__.py with all exports for non-split mode."""
        body: list[ast.stmt] = []
        all_names: list[str] = []

        # Re-export endpoints, the Client, and the error hierarchy.
        if endpoint_names:
            self._add_reexport(
                body,
                all_names,
                self.config.endpoints_file.replace('.py', ''),
                sorted(endpoint_names),
            )
        self._add_reexport(body, all_names, 'client', ['Client'])
        # BaseClient + the full error hierarchy from _client.py so users can
        # ``from my_pkg import NotFoundError`` without reaching into the
        # underscore-prefixed runtime module.
        self._add_reexport(
            body,
            all_names,
            '_client',
            [f'Base{client_class_name}', *_exported_error_names()],
        )

        # Re-export runtime helpers from enabled feature modules.
        self._add_feature_reexports(body, all_names)

        # Re-export model classes from the models module.
        assert self.typegen is not None
        all_model_names = {
            type_.name
            for type_ in self.typegen.types.values()
            if type_.name and type_.implementation_ast
        }
        self._add_reexport(
            body,
            all_names,
            self.config.models_file.replace('.py', ''),
            sorted(all_model_names),
        )

        # Add __all__ at the beginning
        body.insert(0, _all(sorted(set(all_names))))

        # Write __init__.py
        init_path = directory / '__init__.py'
        write_mod(
            body,
            init_path,
            format_code=self.format_output,
            validate_code=self.validate_output,
        )

    def _generate_split_endpoints(
        self,
        directory: UPath,
        models_file: UPath,
        endpoints: list[Endpoint],
        client_class_name: str,
    ) -> list[str]:
        """Generate split endpoint modules based on configuration.

        Args:
            directory: Output directory.
            models_file: Path to the models file.
            endpoints: List of Endpoint objects.
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
            response_unwrap_config=self.config.response_unwrap,
            export_config=self.config.export,
            generate_sync=self.config.generate_sync,
            generate_async=self.config.generate_async,
            reexport_models=self.config.reexport_models,
            reexport_model_exclude_patterns=self.config.reexport_model_exclude_patterns,
            format_output=self.format_output,
            validate_output=self.validate_output,
        )

        assert self.typegen is not None
        emitted = emitter.emit(
            tree=tree,
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

        # Derive from API title (via the adapter so the parser's attribute
        # shape stays encapsulated).
        if self.openapi is not None and self._adapter.title():
            title = self._adapter.title()
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

        output_name = self.config.output
        generated_files = []

        # Generate base client class (infrastructure only, no endpoint methods)
        class_ast, client_imports = generate_base_client_class(
            class_name=base_client_name,
            default_base_url=base_url,
            default_timeout=30.0,
            pydantic_version=self.config.pydantic_version,
        )

        # Build the _client.py file
        body: list[ast.stmt] = []
        import_collector = ImportCollector()
        import_collector.add_imports(client_imports)

        # No model imports needed in _client.py anymore - models are imported in module files

        # Add other imports
        for import_stmt in import_collector.to_ast():
            body.insert(0, import_stmt)

        # Add __all__ export (include the full per-status error hierarchy
        # so users can ``from .<pkg>._client import NotFoundError`` etc.).
        body.append(_all(sorted([base_client_name, *_exported_error_names()])))

        # Add the BaseAPIError + per-status subclass hierarchy.
        body.extend(generate_api_error_hierarchy())

        # Add APIError = BaseAPIError alias for internal use
        body.append(_assign(_name('APIError'), _name('BaseAPIError')))

        # Add the client class
        body.append(class_ast)

        # Write _client.py (always regenerated)
        client_file = directory / '_client.py'
        write_mod(
            body,
            client_file,
            format_code=self.format_output,
            validate_code=self.validate_output,
        )
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
