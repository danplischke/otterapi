"""
Pydantic V2 models for Swagger/OpenAPI 2.0 specification.

Based on the JSON Schema at: http://swagger.io/v2/schema.json

Usage Example:
-------------

    from otterapi.openapi.v2 import Swagger
    import json

    # Load a Swagger 2.0 document
    with open('swagger.json') as f:
        swagger_dict = json.load(f)

    # Parse and validate
    spec = Swagger(**swagger_dict)

    # Access the parsed data
    print(f"API: {spec.info.title} v{spec.info.version}")
    print(f"Host: {spec.host}")

    # Iterate over paths
    for path_name in spec.paths.root:
        path_item = spec.paths.root[path_name]
        if path_item.get:
            print(f"GET {path_name}: {path_item.get.summary}")

    # Export back to dict
    output = spec.model_dump(by_alias=True, exclude_none=True)

    # Create a new spec from scratch
    new_spec = Swagger(
        swagger="2.0",
        info={"title": "My API", "version": "1.0.0"},
        paths={
            "/users": {
                "get": {
                    "summary": "List users",
                    "responses": {
                        "200": {"description": "Success"}
                    }
                }
            }
        }
    )

Features:
---------
- Full Swagger 2.0 specification support
- Pydantic V2 validation
- Type hints for IDE support
- Vendor extensions (x-*) support
- Proper validation of path parameters (must be required)
- JSON reference ($ref) support
- All OAuth2 flows supported
"""

from enum import Enum
from typing import Any, Literal, Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    RootModel,
    model_validator,
)

# Import OpenAPI 3.0 models for upgrade functionality
from otterapi.openapi.constants import MediaType
from otterapi.openapi.v3 import OpenAPI, v3 as openapi_v3
from otterapi.openapi.warnings import WarningCollector

# ============================================================================
# Enums
# ============================================================================


class SchemeType(str, Enum):
    """Transfer protocol schemes."""

    HTTP = 'http'
    HTTPS = 'https'
    WS = 'ws'
    WSS = 'wss'


class ParameterLocation(str, Enum):
    """Parameter location types."""

    QUERY = 'query'
    HEADER = 'header'
    PATH = 'path'
    FORM_DATA = 'formData'
    BODY = 'body'


class PrimitiveType(str, Enum):
    """Primitive types for non-body parameters."""

    STRING = 'string'
    NUMBER = 'number'
    INTEGER = 'integer'
    BOOLEAN = 'boolean'
    ARRAY = 'array'
    FILE = 'file'


class CollectionFormat(str, Enum):
    """Collection format for array parameters."""

    CSV = 'csv'
    SSV = 'ssv'
    TSV = 'tsv'
    PIPES = 'pipes'


class CollectionFormatWithMulti(str, Enum):
    """Collection format including multi for query/formData parameters."""

    CSV = 'csv'
    SSV = 'ssv'
    TSV = 'tsv'
    PIPES = 'pipes'
    MULTI = 'multi'


class SecuritySchemeType(str, Enum):
    """Security scheme types."""

    BASIC = 'basic'
    API_KEY = 'apiKey'
    OAUTH2 = 'oauth2'


class OAuth2Flow(str, Enum):
    """OAuth2 flow types."""

    IMPLICIT = 'implicit'
    PASSWORD = 'password'
    APPLICATION = 'application'
    ACCESS_CODE = 'accessCode'


class ApiKeyLocation(str, Enum):
    """API key location."""

    HEADER = 'header'
    QUERY = 'query'


# ============================================================================
# Helper Classes
# ============================================================================


_WARNING_TEMPLATES: dict[str, str] = {
    'oauth2_implicit': 'OAuth2 implicit flow restructured for OpenAPI 3.0',
    'oauth2_password': 'OAuth2 password flow restructured for OpenAPI 3.0',
    'oauth2_client_credentials': 'OAuth2 application flow converted to clientCredentials for OpenAPI 3.0',
    'oauth2_authorization_code': 'OAuth2 accessCode flow converted to authorizationCode for OpenAPI 3.0',
    'collection_format_multi': "collectionFormat 'multi' converted to style=form with explode=true",
    'collection_format_tsv': "collectionFormat 'tsv' has no direct equivalent in OpenAPI 3.0, using pipeDelimited",
    'file_upload_consumes': 'File upload parameter requires multipart/form-data content type',
    'inferred_response': 'Inferred 200 response schema from request body',
    'body_param_at_path': 'Body parameter found at path level, converting to inline requestBody',
}


# ============================================================================
# Base Models
# ============================================================================


class BaseModelWithVendorExtensions(BaseModel):
    """Base model that allows vendor extensions (x- fields)."""

    model_config = ConfigDict(extra='allow', populate_by_name=True)


class JsonReference(BaseModel):
    """JSON Reference object."""

    ref: str = Field(..., alias='$ref')

    model_config = ConfigDict(populate_by_name=True)


# ============================================================================
# Info Models
# ============================================================================


class Contact(BaseModelWithVendorExtensions):
    """Contact information for the API."""

    name: str | None = None
    url: HttpUrl | None = None
    email: str | None = None


class License(BaseModelWithVendorExtensions):
    """License information for the API."""

    name: str
    url: HttpUrl | None = None


class Info(BaseModelWithVendorExtensions):
    """General information about the API."""

    title: str
    version: str
    description: str | None = None
    terms_of_service: str | None = Field(None, alias='termsOfService')
    contact: Contact | None = None
    license: License | None = None


class ExternalDocs(BaseModelWithVendorExtensions):
    """External documentation reference."""

    url: HttpUrl
    description: str | None = None


class Tag(BaseModelWithVendorExtensions):
    """API tag for grouping operations."""

    name: str
    description: str | None = None
    external_docs: ExternalDocs | None = Field(None, alias='externalDocs')


# ============================================================================
# XML Model
# ============================================================================


class XML(BaseModelWithVendorExtensions):
    """XML representation metadata."""

    name: str | None = None
    namespace: str | None = None
    prefix: str | None = None
    attribute: bool = False
    wrapped: bool = False


# ============================================================================
# Schema Models
# ============================================================================


class Schema(BaseModelWithVendorExtensions):
    """
    JSON Schema object for Swagger 2.0.

    Note: This is a simplified version. Full schema validation is complex
    and may require recursive type definitions.
    """

    ref: str | None = Field(None, alias='$ref')
    format: str | None = None
    title: str | None = None
    description: str | None = None
    default: Any | None = None
    multiple_of: float | None = Field(None, alias='multipleOf', gt=0)
    maximum: float | None = None
    exclusive_maximum: bool | None = Field(None, alias='exclusiveMaximum')
    minimum: float | None = None
    exclusive_minimum: bool | None = Field(None, alias='exclusiveMinimum')
    max_length: int | None = Field(None, alias='maxLength', ge=0)
    min_length: int | None = Field(None, alias='minLength', ge=0)
    pattern: str | None = None
    max_items: int | None = Field(None, alias='maxItems', ge=0)
    min_items: int | None = Field(None, alias='minItems', ge=0)
    unique_items: bool | None = Field(None, alias='uniqueItems')
    max_properties: int | None = Field(None, alias='maxProperties', ge=0)
    min_properties: int | None = Field(None, alias='minProperties', ge=0)
    required: list[str] | None = None
    enum: list[Any] | None = None
    type: str | list[str] | None = None
    items: 'Schema | list[Schema] | None' = None
    all_of: list['Schema'] | None = Field(None, alias='allOf')
    properties: dict[str, 'Schema'] | None = None
    additional_properties: 'Schema | bool | None' = Field(
        None, alias='additionalProperties'
    )
    discriminator: str | None = None
    read_only: bool = Field(False, alias='readOnly')
    xml: XML | None = None
    external_docs: ExternalDocs | None = Field(None, alias='externalDocs')
    example: Any | None = None


class FileSchema(BaseModelWithVendorExtensions):
    """Schema for file uploads."""

    type: Literal['file']
    format: str | None = None
    title: str | None = None
    description: str | None = None
    default: Any | None = None
    required: list[str] | None = None
    read_only: bool = Field(False, alias='readOnly')
    external_docs: ExternalDocs | None = Field(None, alias='externalDocs')
    example: Any | None = None


# ============================================================================
# Parameter Models
# ============================================================================


class PrimitivesItems(BaseModelWithVendorExtensions):
    """Items object for primitive array parameters."""

    type: PrimitiveType | None = None
    format: str | None = None
    items: Optional['PrimitivesItems'] = None
    collection_format: CollectionFormat | None = Field(
        CollectionFormat.CSV, alias='collectionFormat'
    )
    default: Any | None = None
    maximum: float | None = None
    exclusive_maximum: bool | None = Field(None, alias='exclusiveMaximum')
    minimum: float | None = None
    exclusive_minimum: bool | None = Field(None, alias='exclusiveMinimum')
    max_length: int | None = Field(None, alias='maxLength', ge=0)
    min_length: int | None = Field(None, alias='minLength', ge=0)
    pattern: str | None = None
    max_items: int | None = Field(None, alias='maxItems', ge=0)
    min_items: int | None = Field(None, alias='minItems', ge=0)
    unique_items: bool | None = Field(None, alias='uniqueItems')
    enum: list[Any] | None = None
    multiple_of: float | None = Field(None, alias='multipleOf', gt=0)


class BaseParameterFields(BaseModelWithVendorExtensions):
    """Common fields for all parameter types."""

    name: str
    in_: ParameterLocation = Field(..., alias='in')
    description: str | None = None
    required: bool = False


class BodyParameter(BaseParameterFields):
    """Body parameter definition."""

    in_: Literal[ParameterLocation.BODY] = Field(ParameterLocation.BODY, alias='in')
    schema_: Schema = Field(..., alias='schema')
    required: bool = False

    model_config = ConfigDict(extra='forbid', populate_by_name=True)


class NonBodyParameter(BaseParameterFields):
    """Non-body parameter (query, header, path, formData)."""

    in_: (
        Literal[ParameterLocation.QUERY]
        | Literal[ParameterLocation.HEADER]
        | Literal[ParameterLocation.PATH]
        | Literal[ParameterLocation.FORM_DATA]
    ) = Field(..., alias='in')
    type: PrimitiveType
    format: str | None = None
    allow_empty_value: bool | None = Field(None, alias='allowEmptyValue')
    items: PrimitivesItems | None = None
    collection_format: CollectionFormat | CollectionFormatWithMulti | None = Field(
        CollectionFormat.CSV, alias='collectionFormat'
    )
    default: Any | None = None
    maximum: float | None = None
    exclusive_maximum: bool | None = Field(None, alias='exclusiveMaximum')
    minimum: float | None = None
    exclusive_minimum: bool | None = Field(None, alias='exclusiveMinimum')
    max_length: int | None = Field(None, alias='maxLength', ge=0)
    min_length: int | None = Field(None, alias='minLength', ge=0)
    pattern: str | None = None
    max_items: int | None = Field(None, alias='maxItems', ge=0)
    min_items: int | None = Field(None, alias='minItems', ge=0)
    unique_items: bool | None = Field(None, alias='uniqueItems')
    enum: list[Any] | None = None
    multiple_of: float | None = Field(None, alias='multipleOf', gt=0)

    @model_validator(mode='after')
    def validate_path_required(self) -> 'NonBodyParameter':
        """Path parameters must be required."""
        if self.in_ == ParameterLocation.PATH and not self.required:
            raise ValueError('Path parameters must have required=True')
        return self

    model_config = ConfigDict(extra='forbid', populate_by_name=True)


Parameter = BodyParameter | NonBodyParameter | JsonReference


# ============================================================================
# Response Models
# ============================================================================


class Header(BaseModelWithVendorExtensions):
    """Response header definition."""

    type: PrimitiveType
    format: str | None = None
    items: PrimitivesItems | None = None
    collection_format: CollectionFormat | None = Field(
        CollectionFormat.CSV, alias='collectionFormat'
    )
    default: Any | None = None
    maximum: float | None = None
    exclusive_maximum: bool | None = Field(None, alias='exclusiveMaximum')
    minimum: float | None = None
    exclusive_minimum: bool | None = Field(None, alias='exclusiveMinimum')
    max_length: int | None = Field(None, alias='maxLength', ge=0)
    min_length: int | None = Field(None, alias='minLength', ge=0)
    pattern: str | None = None
    max_items: int | None = Field(None, alias='maxItems', ge=0)
    min_items: int | None = Field(None, alias='minItems', ge=0)
    unique_items: bool | None = Field(None, alias='uniqueItems')
    enum: list[Any] | None = None
    multiple_of: float | None = Field(None, alias='multipleOf', gt=0)
    description: str | None = None


class Response(BaseModelWithVendorExtensions):
    """Response object."""

    description: str
    schema_: Schema | FileSchema | None = Field(None, alias='schema')
    headers: dict[str, Header] | None = None
    examples: dict[str, Any] | None = None

    model_config = ConfigDict(extra='forbid', populate_by_name=True)


ResponseValue = Response | JsonReference


class Responses(BaseModelWithVendorExtensions):
    """
    Response definitions for an operation.

    Keys can be HTTP status codes (as strings) or "default".
    """

    model_config = ConfigDict(extra='allow', populate_by_name=True)

    def __getitem__(self, key: str) -> ResponseValue:
        """Allow dict-like access to response codes."""
        return self.__pydantic_extra__.get(key) or getattr(self, key, None)

    @staticmethod
    def _validate_response_key(key: str) -> None:
        """Ensure a response map key is a 3-digit status code, 'default', or vendor extension."""
        if key.startswith('x-'):
            return
        if key == 'default' or (key.isdigit() and len(key) == 3):
            return
        raise ValueError(
            f"Response key must be a 3-digit status code or 'default', got: {key}"
        )

    @staticmethod
    def _convert_response_value(value: Any) -> ResponseValue | Any:
        """Convert a raw mapping value into a Response or JsonReference object."""
        if not isinstance(value, dict):
            return value
        if '$ref' in value:
            return JsonReference(**value)
        return Response(**value)

    @model_validator(mode='before')
    @classmethod
    def validate_and_convert_responses(cls, data: Any) -> Any:
        """Validate response keys and convert to Response objects."""
        if not isinstance(data, dict):
            return data

        result = {}
        for key, value in data.items():
            cls._validate_response_key(key)
            result[key] = cls._convert_response_value(value)

        return result


# ============================================================================
# Operation Models
# ============================================================================


class Operation(BaseModelWithVendorExtensions):
    """Operation (HTTP method) on a path."""

    tags: list[str] | None = None
    summary: str | None = None
    description: str | None = None
    external_docs: ExternalDocs | None = Field(None, alias='externalDocs')
    operation_id: str | None = Field(None, alias='operationId')
    consumes: list[str] | None = None
    produces: list[str] | None = None
    parameters: list[Parameter] | None = None
    responses: Responses
    schemes: list[SchemeType] | None = None
    deprecated: bool = False
    security: list[dict[str, list[str]]] | None = None

    model_config = ConfigDict(extra='forbid', populate_by_name=True)


class PathItem(BaseModelWithVendorExtensions):
    """Path item with operations."""

    ref: str | None = Field(None, alias='$ref')
    get: Operation | None = None
    put: Operation | None = None
    post: Operation | None = None
    delete: Operation | None = None
    options: Operation | None = None
    head: Operation | None = None
    patch: Operation | None = None
    parameters: list[Parameter] | None = None

    model_config = ConfigDict(extra='forbid', populate_by_name=True)


class Paths(RootModel[dict[str, PathItem | Any]]):
    """
    Paths object containing all API paths.

    Keys must start with "/" (except vendor extensions starting with "x-").
    """

    @model_validator(mode='before')
    @classmethod
    def validate_and_convert_paths(cls, data: Any) -> Any:
        """Validate path keys and convert path items to PathItem objects."""
        if not isinstance(data, dict):
            return data

        result = {}
        for key, value in data.items():
            # Validate path keys
            if not key.startswith('x-') and not key.startswith('/'):
                raise ValueError(f"Path must start with '/', got: {key}")

            # Convert dict values to PathItem objects for paths
            if key.startswith('/') and isinstance(value, dict):
                result[key] = PathItem(**value)
            else:
                result[key] = value

        return result


# ============================================================================
# Security Models
# ============================================================================


class BasicAuthenticationSecurity(BaseModelWithVendorExtensions):
    """Basic authentication security scheme."""

    type: Literal[SecuritySchemeType.BASIC]
    description: str | None = None

    model_config = ConfigDict(extra='forbid', populate_by_name=True)


class ApiKeySecurity(BaseModelWithVendorExtensions):
    """API key security scheme."""

    type: Literal[SecuritySchemeType.API_KEY]
    name: str
    in_: ApiKeyLocation = Field(..., alias='in')
    description: str | None = None

    model_config = ConfigDict(extra='forbid', populate_by_name=True)


class OAuth2ImplicitSecurity(BaseModelWithVendorExtensions):
    """OAuth2 implicit flow security scheme."""

    type: Literal[SecuritySchemeType.OAUTH2]
    flow: Literal[OAuth2Flow.IMPLICIT]
    authorization_url: HttpUrl = Field(..., alias='authorizationUrl')
    scopes: dict[str, str] = {}
    description: str | None = None

    model_config = ConfigDict(extra='forbid', populate_by_name=True)


class OAuth2PasswordSecurity(BaseModelWithVendorExtensions):
    """OAuth2 password flow security scheme."""

    type: Literal[SecuritySchemeType.OAUTH2]
    flow: Literal[OAuth2Flow.PASSWORD]
    token_url: HttpUrl = Field(..., alias='tokenUrl')
    scopes: dict[str, str] = {}
    description: str | None = None

    model_config = ConfigDict(extra='forbid', populate_by_name=True)


class OAuth2ApplicationSecurity(BaseModelWithVendorExtensions):
    """OAuth2 application flow security scheme."""

    type: Literal[SecuritySchemeType.OAUTH2]
    flow: Literal[OAuth2Flow.APPLICATION]
    token_url: HttpUrl = Field(..., alias='tokenUrl')
    scopes: dict[str, str] = {}
    description: str | None = None

    model_config = ConfigDict(extra='forbid', populate_by_name=True)


class OAuth2AccessCodeSecurity(BaseModelWithVendorExtensions):
    """OAuth2 access code flow security scheme."""

    type: Literal[SecuritySchemeType.OAUTH2]
    flow: Literal[OAuth2Flow.ACCESS_CODE]
    authorization_url: HttpUrl = Field(..., alias='authorizationUrl')
    token_url: HttpUrl = Field(..., alias='tokenUrl')
    scopes: dict[str, str] = {}
    description: str | None = None

    model_config = ConfigDict(extra='forbid', populate_by_name=True)


SecurityScheme = (
    BasicAuthenticationSecurity
    | ApiKeySecurity
    | OAuth2ImplicitSecurity
    | OAuth2PasswordSecurity
    | OAuth2ApplicationSecurity
    | OAuth2AccessCodeSecurity
)


# ============================================================================
# Main Swagger Model
# ============================================================================


class Swagger(BaseModelWithVendorExtensions):
    """
    Root Swagger 2.0 specification object.

    This is the main model representing a complete Swagger/OpenAPI 2.0 document.
    """

    swagger: Literal['2.0']
    info: Info
    host: str | None = Field(None, pattern=r'^[^{}/ :\\]+(?::\d+)?$')
    base_path: str | None = Field(None, alias='basePath', pattern=r'^/')
    schemes: list[SchemeType] | None = None
    consumes: list[str] | None = None
    produces: list[str] | None = None
    paths: Paths
    definitions: dict[str, Schema] | None = None
    parameters: dict[str, Parameter] | None = None
    responses: dict[str, Response] | None = None
    security_definitions: dict[str, SecurityScheme] | None = Field(
        None, alias='securityDefinitions'
    )
    security: list[dict[str, list[str]]] | None = None
    tags: list[Tag] | None = None
    external_docs: ExternalDocs | None = Field(None, alias='externalDocs')

    model_config = ConfigDict(extra='forbid', populate_by_name=True)

    def upgrade(self) -> tuple[OpenAPI, list[str]]:
        """
        Upgrade this Swagger 2.0 specification to OpenAPI 3.0.

        Returns:
            A tuple of (OpenAPI 3.0 model, list of warnings)


        Warnings are generated for:
        - Lossy conversions
        - Structural changes
        - Missing data that requires defaults
        - OAuth2 flow restructuring
        - Collection format conversions

        Warnings are deduplicated and summarized to avoid overwhelming output
        for large APIs.
        """
        warning_collector = WarningCollector(_WARNING_TEMPLATES)

        # Convert basic metadata
        info = self._convert_info()

        # Convert servers from host/basePath/schemes
        servers = self._convert_servers(warning_collector)

        # Convert components
        components = self._convert_components(warning_collector)

        # Convert paths - returns dict for Paths RootModel
        paths_dict = self._convert_paths(warning_collector)

        tags = [
            openapi_v3.Tag(
                name=tag.name,
                description=tag.description,
                externalDocs=openapi_v3.ExternalDocumentation(
                    url=str(tag.external_docs.url),
                    description=tag.external_docs.description,
                )
                if tag.external_docs
                else None,
            )
            for tag in self.tags
        ] if self.tags else None

        external_docs = openapi_v3.ExternalDocumentation(
            url=str(self.external_docs.url),
            description=self.external_docs.description,
        ) if self.external_docs else None

        result = OpenAPI(
            openapi='3.0.3',
            info=info,
            paths=openapi_v3.Paths(root=paths_dict),
            servers=servers or None,
            components=components,
            security=self.security or None,
            tags=tags,
            externalDocs=external_docs,
        )
        return result, warning_collector.get_warnings()

    def _convert_info(self) -> openapi_v3.Info:
        """Convert Info object from Swagger 2.0 to OpenAPI 3.0."""
        contact = None
        if self.info.contact:
            contact = openapi_v3.Contact(
                name=self.info.contact.name,
                url=str(self.info.contact.url) if self.info.contact.url else None,
                email=self.info.contact.email,
            )

        license_obj = None
        if self.info.license:
            license_obj = openapi_v3.License(
                name=self.info.license.name,
                url=str(self.info.license.url) if self.info.license.url else None,
            )

        return openapi_v3.Info(
            title=self.info.title,
            version=self.info.version,
            description=self.info.description,
            termsOfService=self.info.terms_of_service,
            contact=contact,
            license=license_obj,
        )

    def _convert_servers(self, warnings: WarningCollector) -> list[openapi_v3.Server]:
        """Convert host, basePath, and schemes to servers array."""
        if not self.host and not self.base_path:
            warnings.add_unique(
                "No host or basePath specified, defaulting to server URL '/'"
            )
            return [openapi_v3.Server(url='/')]

        servers = []
        schemes = self.schemes or [SchemeType.HTTP]
        host = self.host or ''
        base_path = self.base_path or ''

        for scheme in schemes:
            url = f'{scheme.value}://{host}{base_path}' if host else base_path
            servers.append(openapi_v3.Server(url=url))

        return servers

    def _convert_components(
        self, warnings: WarningCollector
    ) -> openapi_v3.Components | None:
        """Convert definitions, parameters, responses, and security to components."""
        schemas = None
        if self.definitions:
            schemas = {
                name: self._convert_schema_to_dict(schema)
                for name, schema in self.definitions.items()
            }

        parameters = None
        if self.parameters:
            parameters = {
                name: self._convert_component_parameter_to_dict(param, warnings)
                for name, param in self.parameters.items()
            }

        responses = None
        if self.responses:
            responses = {
                name: self._convert_response_to_dict(
                    response, self.produces or [MediaType.JSON], warnings
                )
                for name, response in self.responses.items()
            }

        security_schemes = None
        if self.security_definitions:
            security_schemes = {
                name: self._convert_security_scheme_to_dict(scheme, warnings)
                for name, scheme in self.security_definitions.items()
            }

        if not any([schemas, parameters, responses, security_schemes]):
            return None

        return openapi_v3.Components(
            schemas=schemas,
            parameters=parameters,
            responses=responses,
            securitySchemes=security_schemes,
        )

    def _convert_paths(
        self, warnings: WarningCollector
    ) -> dict[str, openapi_v3.PathItem | Any]:
        """Convert paths object."""
        result: dict[str, openapi_v3.PathItem | Any] = {}

        for path, path_item in self.paths.root.items():
            if path.startswith('x-'):
                result[path] = path_item
            elif isinstance(path_item, PathItem):
                result[path] = self._convert_path_item(path_item, warnings)

        return result

    def _convert_path_item(
        self, path_item: PathItem, warnings: WarningCollector
    ) -> openapi_v3.PathItem:
        """Convert a single PathItem."""
        get = (
            self._convert_operation(path_item.get, warnings, method='get')
            if path_item.get
            else None
        )
        put = (
            self._convert_operation(path_item.put, warnings, method='put')
            if path_item.put
            else None
        )
        post = (
            self._convert_operation(path_item.post, warnings, method='post')
            if path_item.post
            else None
        )
        delete = (
            self._convert_operation(path_item.delete, warnings, method='delete')
            if path_item.delete
            else None
        )
        options = (
            self._convert_operation(path_item.options, warnings, method='options')
            if path_item.options
            else None
        )
        head = (
            self._convert_operation(path_item.head, warnings, method='head')
            if path_item.head
            else None
        )
        patch = (
            self._convert_operation(path_item.patch, warnings, method='patch')
            if path_item.patch
            else None
        )

        parameters = (
            [
                self._convert_parameter_item_to_dict(param, warnings)
                for param in path_item.parameters
            ]
            if path_item.parameters
            else None
        )

        obj = openapi_v3.PathItem(
            field_ref=self._update_ref(path_item.ref) if path_item.ref else None,
            get=get,
            put=put,
            post=post,
            delete=delete,
            options=options,
            head=head,
            patch=patch,
            parameters=parameters,
        )
        obj.__pydantic_extra__.update(self._extract_vendor_extensions(path_item))
        return obj

    def _apply_operation_parameters(
        self,
        operation: Operation,
        warnings: list[str],
    ) -> tuple[
        list[openapi_v3.Parameter | openapi_v3.Reference] | None,
        openapi_v3.RequestBody | None,
        openapi_v3.Schema | openapi_v3.Reference | None,
    ]:
        """Convert operation parameters into typed objects.

        Returns a 3-tuple of (parameters, requestBody, body_schema).
        The body_schema is used for response inference.
        """
        if not operation.parameters:
            return None, None, None

        converted = self._convert_parameters(
            operation.parameters,
            operation.consumes or self.consumes,
            warnings,
        )
        params = converted['parameters'] or None
        request_body = converted['requestBody']
        body_schema = converted.get('body_schema')
        return params, request_body, body_schema

    def _build_operation_servers(
        self, schemes: list[SchemeType]
    ) -> list[openapi_v3.Server]:
        """Build OpenAPI 3.0 ``servers`` entries from Swagger 2.0 operation schemes."""
        servers = []
        for scheme in schemes:
            host = self.host or ''
            base_path = self.base_path or ''
            url = f'{scheme.value}://{host}{base_path}' if host else base_path
            servers.append(openapi_v3.Server(url=url))
        return servers

    def _convert_operation(
        self, operation: Operation, warnings: list[str], method: str = None
    ) -> openapi_v3.Operation:
        """Convert an Operation object."""
        external_docs = None
        if operation.external_docs:
            external_docs = openapi_v3.ExternalDocumentation(
                url=str(operation.external_docs.url),
                description=operation.external_docs.description,
            )

        parameters, request_body, body_schema = self._apply_operation_parameters(
            operation, warnings
        )

        responses = self._convert_responses(
            operation.responses,
            operation.produces or self.produces or [MediaType.JSON],
            warnings,
            method=method,
            body_schema=body_schema,
        )

        servers = (
            self._build_operation_servers(operation.schemes)
            if operation.schemes
            else None
        )

        security = (
            [openapi_v3.SecurityRequirement(root=s) for s in operation.security]
            if operation.security
            else None
        )

        obj = openapi_v3.Operation(
            tags=operation.tags if operation.tags else None,
            summary=operation.summary if operation.summary else None,
            description=operation.description if operation.description else None,
            externalDocs=external_docs,
            operationId=operation.operation_id if operation.operation_id else None,
            parameters=parameters,
            requestBody=request_body,
            responses=responses,
            deprecated=True if operation.deprecated else None,
            security=security,
            servers=servers,
        )
        obj.__pydantic_extra__.update(self._extract_vendor_extensions(operation))
        return obj

    def _convert_parameters(
        self,
        parameters: list[Parameter],
        consumes: list[str] | None,
        warnings: list[str],
    ) -> dict[str, Any]:
        """
        Convert parameters list, separating body/formData into requestBody.

        Returns dict with 'parameters', 'requestBody', and 'body_schema' keys.
        The 'body_schema' is used for inferring response schemas when not specified.
        """
        result_params: list[openapi_v3.Parameter | openapi_v3.Reference] = []
        body_param = None
        form_params = []

        for param in parameters:
            if isinstance(param, JsonReference):
                # Handle reference
                result_params.append(
                    openapi_v3.Reference(root={'$ref': self._update_ref(param.ref)})
                )
            elif isinstance(param, BodyParameter):
                body_param = param
            elif isinstance(param, NonBodyParameter):
                if param.in_ == ParameterLocation.FORM_DATA:
                    form_params.append(param)
                else:
                    result_params.append(
                        self._convert_non_body_parameter_to_dict(param, warnings)
                    )

        request_body: openapi_v3.RequestBody | None = None
        body_schema: openapi_v3.Schema | openapi_v3.Reference | None = None
        if body_param:
            request_body = self._convert_body_parameter_to_dict(
                body_param, consumes, warnings
            )
            # Extract the body schema for response inference
            body_schema = self._convert_schema_to_dict(body_param.schema_)
        elif form_params:
            request_body = self._convert_formdata_parameters(
                form_params, consumes, warnings
            )

        return {
            'parameters': result_params,
            'requestBody': request_body,
            'body_schema': body_schema,
        }

    def _convert_body_parameter_to_dict(
        self,
        param: BodyParameter,
        consumes: list[str] | None,
        warnings: list[str],
    ) -> openapi_v3.RequestBody:
        """Convert body parameter to requestBody."""
        media_types = consumes or [MediaType.JSON]
        schema_obj = self._convert_schema_to_dict(param.schema_)

        content = {
            media_type: openapi_v3.MediaType(schema_=schema_obj)
            for media_type in media_types
        }

        return openapi_v3.RequestBody(
            content=content,
            description=param.description if param.description else None,
            required=True if param.required else None,
        )

    def _convert_formdata_parameters(
        self,
        params: list[NonBodyParameter],
        consumes: list[str] | None,
        warnings: list[str],
    ) -> openapi_v3.RequestBody:
        """Convert formData parameters to requestBody."""
        # Determine media type
        has_file = any(p.type == PrimitiveType.FILE for p in params)
        form_media_type = (
            MediaType.MULTIPART if has_file else MediaType.FORM_URLENCODED
        )

        # Check if consumes specifies something different
        if consumes and form_media_type not in consumes:
            if has_file and MediaType.MULTIPART not in consumes:
                warnings.add('file_upload_consumes')

        # Build schema with properties
        properties: dict[str, openapi_v3.Schema | openapi_v3.Reference] = {}
        required = []

        for param in params:
            prop_schema = self._convert_parameter_to_schema(param, warnings)
            properties[param.name] = prop_schema
            if param.required:
                required.append(param.name)

        form_schema = openapi_v3.Schema(
            type=openapi_v3.Type.object,
            properties=properties,
            required=required if required else None,
        )

        content = {
            form_media_type: openapi_v3.MediaType(schema_=form_schema)
        }

        description = params[0].description if params and params[0].description else None

        return openapi_v3.RequestBody(
            content=content,
            description=description,
        )

    def _convert_non_body_parameter_to_dict(
        self, param: NonBodyParameter, warnings: WarningCollector
    ) -> openapi_v3.Parameter:
        """Convert query/header/path parameter to OpenAPI 3.0 format."""
        schema_obj = self._convert_parameter_to_schema(param, warnings)

        style = None
        explode = None
        if param.type == PrimitiveType.ARRAY and param.collection_format:
            style, explode = self._convert_collection_format(
                param.collection_format,
                param.in_,
                warnings,
            )

        obj = openapi_v3.Parameter(
            name=param.name,
            in_=param.in_.value,
            description=param.description if param.description else None,
            required=True if param.required else None,
            allowEmptyValue=True if param.allow_empty_value else None,
            schema_=schema_obj,
            style=style,
            explode=explode,
        )
        obj.__pydantic_extra__.update(self._extract_vendor_extensions(param))
        return obj

    def _convert_parameter_to_schema(
        self, param: NonBodyParameter, warnings: WarningCollector
    ) -> openapi_v3.Schema:
        """Convert parameter properties to a schema object."""
        if param.type == PrimitiveType.FILE:
            schema_type = openapi_v3.Type.string
            schema_format = 'binary'
        else:
            schema_type = openapi_v3.Type(param.type.value)
            schema_format = param.format if param.format else None

        items = self._convert_primitives_items(param.items) if param.items else None

        return openapi_v3.Schema(
            type=schema_type,
            format=schema_format,
            items=items,
            default=param.default,
            maximum=param.maximum,
            exclusiveMaximum=param.exclusive_maximum,
            minimum=param.minimum,
            exclusiveMinimum=param.exclusive_minimum,
            maxLength=param.max_length,
            minLength=param.min_length,
            pattern=param.pattern if param.pattern else None,
            maxItems=param.max_items,
            minItems=param.min_items,
            uniqueItems=param.unique_items,
            enum=param.enum if param.enum else None,
            multipleOf=param.multiple_of,
        )

    def _convert_primitives_items(self, items: PrimitivesItems) -> openapi_v3.Schema:
        """Convert PrimitivesItems to schema."""
        nested_items = (
            self._convert_primitives_items(items.items) if items.items else None
        )

        return openapi_v3.Schema(
            type=openapi_v3.Type(items.type.value) if items.type else None,
            format=items.format if items.format else None,
            items=nested_items,
            default=items.default,
            maximum=items.maximum,
            minimum=items.minimum,
            maxLength=items.max_length,
            minLength=items.min_length,
            pattern=items.pattern if items.pattern else None,
            maxItems=items.max_items,
            minItems=items.min_items,
            uniqueItems=items.unique_items,
            enum=items.enum if items.enum else None,
            multipleOf=items.multiple_of,
        )

    def _convert_collection_format(
        self,
        collection_format: CollectionFormat | CollectionFormatWithMulti | None,
        in_: ParameterLocation,
        warnings: WarningCollector,
    ) -> tuple[str | None, bool | None]:
        """
        Convert collectionFormat to style and explode.

        Returns (style, explode) tuple.
        """
        # Default styles by location
        default_styles = {
            ParameterLocation.QUERY: 'form',
            ParameterLocation.PATH: 'simple',
            ParameterLocation.HEADER: 'simple',
        }

        if isinstance(collection_format, CollectionFormatWithMulti):
            if collection_format == CollectionFormatWithMulti.MULTI:
                # multi -> explode=true with form style
                warnings.add('collection_format_multi')
                return 'form', True
            collection_format = CollectionFormat(collection_format.value)

        # Mapping for other formats
        format_map = {
            CollectionFormat.CSV: (default_styles.get(in_, 'simple'), False),
            CollectionFormat.SSV: ('spaceDelimited', False),
            CollectionFormat.TSV: ('pipeDelimited', False),
            CollectionFormat.PIPES: ('pipeDelimited', False),
        }

        if collection_format == CollectionFormat.TSV:
            warnings.add('collection_format_tsv')

        return format_map.get(collection_format, (None, None))

    def _convert_response_extras(
        self,
        responses: Responses,
        produces: list[str],
        warnings: WarningCollector,
    ) -> dict[str, openapi_v3.Response | openapi_v3.Reference | Any]:
        """Convert the status-code/vendor-extension entries on a Responses object."""
        result: dict[str, openapi_v3.Response | openapi_v3.Reference | Any] = {}
        if not (
            hasattr(responses, '__pydantic_extra__') and responses.__pydantic_extra__
        ):
            return result

        for status_code, response in responses.__pydantic_extra__.items():
            if status_code.startswith('x-'):
                result[status_code] = response
            elif isinstance(response, JsonReference):
                result[status_code] = openapi_v3.Reference(root={'$ref': self._update_ref(response.ref)})
            elif isinstance(response, Response):
                result[status_code] = self._convert_response_to_dict(
                    response, produces, warnings
                )
        return result

    @staticmethod
    def _has_success_response_schema(
        result: dict[str, openapi_v3.Response | openapi_v3.Reference | Any],
    ) -> bool:
        """Check whether ``result`` already has a 2xx response with a content schema."""
        for status_code, response_obj in result.items():
            if not status_code.startswith('2') or status_code.startswith('x-'):
                continue
            if not isinstance(response_obj, openapi_v3.Response):
                continue
            if not response_obj.content:
                continue
            for media_type_obj in response_obj.content.values():
                if isinstance(media_type_obj, openapi_v3.MediaType) and media_type_obj.schema_:
                    return True
        return False

    @staticmethod
    def _build_inferred_success_response(
        body_schema: openapi_v3.Schema | openapi_v3.Reference,
        produces: list[str],
    ) -> openapi_v3.Response:
        """Build a synthetic 200 response that mirrors the request body schema."""
        content = {
            media_type: openapi_v3.MediaType(schema_=body_schema)
            for media_type in produces
        }
        return openapi_v3.Response(
            description='Successful operation',
            content=content,
        )

    def _maybe_infer_success_response(
        self,
        result: dict[str, openapi_v3.Response | openapi_v3.Reference | Any],
        body_schema: openapi_v3.Schema | openapi_v3.Reference | None,
        produces: list[str],
        warnings: WarningCollector,
        method: str,
    ) -> None:
        """Infer a 200 response from the request body schema when one isn't declared.

        For POST/PUT operations with a body parameter that references a model,
        it's common for the response to return the same model — so infer that
        only when the body schema is itself a $ref (model reference) and no
        2xx response with a schema is already present.
        """
        if not (body_schema and method in ('post', 'put')):
            return
        if self._has_success_response_schema(result):
            return
        if not isinstance(body_schema, openapi_v3.Reference):
            return

        warnings.add('inferred_response')
        result['200'] = self._build_inferred_success_response(body_schema, produces)

    def _convert_responses(
        self,
        responses: Responses,
        produces: list[str],
        warnings: WarningCollector,
        body_schema: openapi_v3.Schema | openapi_v3.Reference | None = None,
        method: str = 'get',
    ) -> openapi_v3.Responses:
        """Convert Responses object.

        Args:
            responses: The Swagger 2.0 Responses object
            produces: List of media types the operation produces
            warnings: List to append warnings to
            method: HTTP method (post, put, etc.) for response inference
            body_schema: Body parameter schema for inferring response when not specified
        """
        result = self._convert_response_extras(responses, produces, warnings)
        self._maybe_infer_success_response(
            result, body_schema, produces, warnings, method
        )
        return openapi_v3.Responses(root=result)

    def _convert_header_to_dict(self, header: Header) -> openapi_v3.Header:
        """Convert Header to OpenAPI 3.0 format."""
        schema = openapi_v3.Schema(
            type=openapi_v3.Type(header.type.value),
            format=header.format if header.format else None,
            items=self._convert_primitives_items(header.items) if header.items else None,
            default=header.default,
            maximum=header.maximum,
            minimum=header.minimum,
            maxLength=header.max_length,
            minLength=header.min_length,
            pattern=header.pattern if header.pattern else None,
            maxItems=header.max_items,
            minItems=header.min_items,
            uniqueItems=header.unique_items,
            enum=header.enum if header.enum else None,
            multipleOf=header.multiple_of,
        )

        obj = openapi_v3.Header(
            description=header.description if header.description else None,
            schema_=schema,
        )
        obj.__pydantic_extra__.update(self._extract_vendor_extensions(header))
        return obj

    @staticmethod
    def _build_schema_xml_dict(xml: XML) -> openapi_v3.XML:
        """Build the OpenAPI 3.0 ``xml`` object from a Swagger 2.0 XML definition."""
        return openapi_v3.XML(
            name=xml.name if xml.name else None,
            namespace=xml.namespace if xml.namespace else None,
            prefix=xml.prefix if xml.prefix else None,
            attribute=True if xml.attribute else None,
            wrapped=True if xml.wrapped else None,
        )

    def _convert_schema_to_dict(
        self, schema: Schema | FileSchema
    ) -> openapi_v3.Schema | openapi_v3.Reference:
        """Convert Schema object to OpenAPI 3.0 format."""
        if isinstance(schema, FileSchema):
            return openapi_v3.Schema(
                type=openapi_v3.Type.string,
                format='binary',
            )

        if schema.ref:
            return openapi_v3.Reference(root={'$ref': self._update_ref(schema.ref)})

        # Build items / allOf / properties / additionalProperties
        items = None
        if schema.items:
            if isinstance(schema.items, list):
                # Only use first item for items field (OpenAPI 3.0 doesn't support tuple validation)
                items = self._convert_schema_to_dict(schema.items[0])
            else:
                items = self._convert_schema_to_dict(schema.items)

        all_of = None
        if schema.all_of:
            all_of = [self._convert_schema_to_dict(s) for s in schema.all_of]

        properties = None
        if schema.properties:
            properties = {
                name: self._convert_schema_to_dict(prop)
                for name, prop in schema.properties.items()
            }

        additional_properties = None
        if schema.additional_properties is not None:
            if isinstance(schema.additional_properties, bool):
                additional_properties = schema.additional_properties
            else:
                additional_properties = self._convert_schema_to_dict(
                    schema.additional_properties
                )

        # Discriminator: convert from string to Discriminator object
        discriminator = None
        if schema.discriminator:
            discriminator = openapi_v3.Discriminator(
                propertyName=schema.discriminator
            )

        # XML
        xml = self._build_schema_xml_dict(schema.xml) if schema.xml else None

        # ExternalDocs
        external_docs = None
        if schema.external_docs:
            external_docs = openapi_v3.ExternalDocumentation(
                url=str(schema.external_docs.url),
                description=schema.external_docs.description,
            )

        # Determine type
        schema_type = None
        if schema.type:
            if isinstance(schema.type, list):
                # Take the first non-null type
                non_null = [t for t in schema.type if t != 'null']
                schema_type = openapi_v3.Type(non_null[0]) if non_null else None
            else:
                try:
                    schema_type = openapi_v3.Type(schema.type)
                except ValueError:
                    schema_type = None

        obj = openapi_v3.Schema(
            type=schema_type,
            format=schema.format if schema.format else None,
            title=schema.title if schema.title else None,
            description=schema.description if schema.description else None,
            default=schema.default,
            multipleOf=schema.multiple_of,
            maximum=schema.maximum,
            exclusiveMaximum=schema.exclusive_maximum,
            minimum=schema.minimum,
            exclusiveMinimum=schema.exclusive_minimum,
            maxLength=schema.max_length,
            minLength=schema.min_length,
            pattern=schema.pattern if schema.pattern else None,
            maxItems=schema.max_items,
            minItems=schema.min_items,
            uniqueItems=schema.unique_items,
            maxProperties=schema.max_properties,
            minProperties=schema.min_properties,
            required=schema.required if schema.required else None,
            enum=schema.enum if schema.enum else None,
            items=items,
            allOf=all_of,
            properties=properties,
            additionalProperties=additional_properties,
            discriminator=discriminator,
            readOnly=schema.read_only if schema.read_only else None,
            xml=xml,
            externalDocs=external_docs,
            example=schema.example,
        )
        obj.__pydantic_extra__.update(self._extract_vendor_extensions(schema))
        return obj

    def _convert_component_parameter_to_dict(
        self, param: Parameter, warnings: WarningCollector
    ) -> openapi_v3.Parameter | openapi_v3.Reference:
        """Convert a component-level parameter."""
        if isinstance(param, JsonReference):
            return openapi_v3.Reference(root={'$ref': self._update_ref(param.ref)})
        elif isinstance(param, BodyParameter):
            # Body parameters in components: map to a query parameter placeholder
            return openapi_v3.Parameter(
                name=param.name if param.name else 'body',
                in_='query',
                description=param.description if param.description else None,
                schema_=self._convert_schema_to_dict(param.schema_),
            )
        elif isinstance(param, NonBodyParameter):
            return self._convert_non_body_parameter_to_dict(param, warnings)
        return openapi_v3.Parameter(name='', in_='query')

    def _convert_response_to_dict(
        self, response: Response, produces: list[str], warnings: WarningCollector
    ) -> openapi_v3.Response:
        """Convert a single Response object to OpenAPI 3.0 format."""
        content = None
        if response.schema_:
            schema_obj = self._convert_schema_to_dict(response.schema_)
            content = {}
            for media_type in produces:
                # Build media type with schema and possibly an example
                example_val = (
                    response.examples.get(media_type)
                    if response.examples
                    else None
                )
                content[media_type] = openapi_v3.MediaType(
                    schema_=schema_obj,
                    example=example_val,
                )

        headers = None
        if response.headers:
            headers = {
                name: self._convert_header_to_dict(header)
                for name, header in response.headers.items()
            }

        obj = openapi_v3.Response(
            description=response.description,
            content=content,
            headers=headers,
        )
        return obj

    @staticmethod
    def _build_basic_auth_scheme_dict(
        scheme: BasicAuthenticationSecurity,
    ) -> openapi_v3.SecurityScheme:
        return openapi_v3.SecurityScheme(root=openapi_v3.HTTPSecurityScheme(
            type=openapi_v3.SecuritySchemeType.http,
            scheme='basic',
            description=scheme.description if scheme.description else None,
        ))

    @staticmethod
    def _build_api_key_scheme_dict(scheme: ApiKeySecurity) -> openapi_v3.SecurityScheme:
        inner = openapi_v3.APIKeySecurityScheme(
            type=openapi_v3.SecuritySchemeType.apiKey,
            name=scheme.name,
            in_=scheme.in_.value,
            description=scheme.description if scheme.description else None,
        )
        return openapi_v3.SecurityScheme(root=inner)

    @staticmethod
    def _build_oauth2_implicit_scheme_dict(
        scheme: OAuth2ImplicitSecurity,
    ) -> openapi_v3.SecurityScheme:
        flows = openapi_v3.OAuthFlows(
            implicit=openapi_v3.ImplicitOAuthFlow(
                authorizationUrl=str(scheme.authorization_url),
                scopes=scheme.scopes,
            )
        )
        inner = openapi_v3.OAuth2SecurityScheme(
            type=openapi_v3.SecuritySchemeType.oauth2,
            flows=flows,
            description=scheme.description if scheme.description else None,
        )
        return openapi_v3.SecurityScheme(root=inner)

    @staticmethod
    def _build_oauth2_password_scheme_dict(
        scheme: OAuth2PasswordSecurity,
    ) -> openapi_v3.SecurityScheme:
        flows = openapi_v3.OAuthFlows(
            password=openapi_v3.PasswordOAuthFlow(
                tokenUrl=str(scheme.token_url),
                scopes=scheme.scopes,
            )
        )
        inner = openapi_v3.OAuth2SecurityScheme(
            type=openapi_v3.SecuritySchemeType.oauth2,
            flows=flows,
            description=scheme.description if scheme.description else None,
        )
        return openapi_v3.SecurityScheme(root=inner)

    @staticmethod
    def _build_oauth2_application_scheme_dict(
        scheme: OAuth2ApplicationSecurity,
    ) -> openapi_v3.SecurityScheme:
        flows = openapi_v3.OAuthFlows(
            clientCredentials=openapi_v3.ClientCredentialsFlow(
                tokenUrl=str(scheme.token_url),
                scopes=scheme.scopes,
            )
        )
        inner = openapi_v3.OAuth2SecurityScheme(
            type=openapi_v3.SecuritySchemeType.oauth2,
            flows=flows,
            description=scheme.description if scheme.description else None,
        )
        return openapi_v3.SecurityScheme(root=inner)

    @staticmethod
    def _build_oauth2_access_code_scheme_dict(
        scheme: OAuth2AccessCodeSecurity,
    ) -> openapi_v3.SecurityScheme:
        flows = openapi_v3.OAuthFlows(
            authorizationCode=openapi_v3.AuthorizationCodeOAuthFlow(
                authorizationUrl=str(scheme.authorization_url),
                tokenUrl=str(scheme.token_url),
                scopes=scheme.scopes,
            )
        )
        inner = openapi_v3.OAuth2SecurityScheme(
            type=openapi_v3.SecuritySchemeType.oauth2,
            flows=flows,
            description=scheme.description if scheme.description else None,
        )
        return openapi_v3.SecurityScheme(root=inner)

    def _finalize_security_scheme(
        self,
        obj: openapi_v3.SecurityScheme,
        scheme: SecurityScheme,
        warnings: WarningCollector,
        warning_key: str | None = None,
    ) -> openapi_v3.SecurityScheme:
        """Apply vendor-extension/warning steps shared by every scheme."""
        # Apply vendor extensions to the inner scheme object if it supports extras
        inner = obj.root
        if hasattr(inner, '__pydantic_extra__') and inner.__pydantic_extra__ is not None:
            inner.__pydantic_extra__.update(self._extract_vendor_extensions(scheme))
        if warning_key:
            warnings.add(warning_key)
        return obj

    def _convert_security_scheme_to_dict(
        self, scheme: SecurityScheme, warnings: WarningCollector
    ) -> openapi_v3.SecurityScheme:
        """Convert a single security scheme."""
        if isinstance(scheme, BasicAuthenticationSecurity):
            return self._finalize_security_scheme(
                self._build_basic_auth_scheme_dict(scheme), scheme, warnings
            )

        if isinstance(scheme, ApiKeySecurity):
            return self._finalize_security_scheme(
                self._build_api_key_scheme_dict(scheme), scheme, warnings
            )

        if isinstance(scheme, OAuth2ImplicitSecurity):
            return self._finalize_security_scheme(
                self._build_oauth2_implicit_scheme_dict(scheme),
                scheme,
                warnings,
                'oauth2_implicit',
            )

        if isinstance(scheme, OAuth2PasswordSecurity):
            return self._finalize_security_scheme(
                self._build_oauth2_password_scheme_dict(scheme),
                scheme,
                warnings,
                'oauth2_password',
            )

        if isinstance(scheme, OAuth2ApplicationSecurity):
            return self._finalize_security_scheme(
                self._build_oauth2_application_scheme_dict(scheme),
                scheme,
                warnings,
                'oauth2_client_credentials',
            )

        if isinstance(scheme, OAuth2AccessCodeSecurity):
            return self._finalize_security_scheme(
                self._build_oauth2_access_code_scheme_dict(scheme),
                scheme,
                warnings,
                'oauth2_authorization_code',
            )

        raise ValueError(f'Unknown security scheme type: {type(scheme)}')

    def _update_ref(self, ref: str) -> str:
        """Update $ref paths from Swagger 2.0 to OpenAPI 3.0 format."""
        # Update definitions references
        if ref.startswith('#/definitions/'):
            return ref.replace('#/definitions/', '#/components/schemas/')

        # Update parameters references
        if ref.startswith('#/parameters/'):
            return ref.replace('#/parameters/', '#/components/parameters/')

        # Update responses references
        if ref.startswith('#/responses/'):
            return ref.replace('#/responses/', '#/components/responses/')

        return ref

    def _extract_vendor_extensions(
        self, obj: BaseModelWithVendorExtensions
    ) -> dict[str, Any]:
        """Extract vendor extensions (x-*) from an object."""
        if hasattr(obj, '__pydantic_extra__') and obj.__pydantic_extra__:
            return {
                k: v for k, v in obj.__pydantic_extra__.items() if k.startswith('x-')
            }
        return {}

    def _convert_parameter_item_to_dict(
        self, param: Parameter, warnings: WarningCollector
    ) -> openapi_v3.Parameter | openapi_v3.Reference:
        """Convert a single parameter (handles both body and non-body)."""
        if isinstance(param, JsonReference):
            return openapi_v3.Reference(root={'$ref': self._update_ref(param.ref)})
        elif isinstance(param, BodyParameter):
            # This shouldn't happen at path level in Swagger 2.0, but handle it
            warnings.add('body_param_at_path')
            # Body parameters cannot be expressed as path-level parameters in
            # OpenAPI 3.0; emit the warning and return a minimal placeholder.
            return openapi_v3.Parameter(
                name=param.name if param.name else 'body',
                in_='query',
                description=param.description if param.description else None,
                schema_=self._convert_schema_to_dict(param.schema_),
            )
        elif isinstance(param, NonBodyParameter):
            return self._convert_non_body_parameter_to_dict(param, warnings)
        return openapi_v3.Parameter(name='', in_='query')


# Update forward references for recursive models
Schema.model_rebuild()
PrimitivesItems.model_rebuild()
