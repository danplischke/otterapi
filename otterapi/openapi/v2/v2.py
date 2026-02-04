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
    for path_name in spec.paths.__pydantic_extra__:
        path_item = spec.paths.__pydantic_extra__[path_name]
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
from typing import Any, Literal, Optional, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    model_validator,
)

# Import OpenAPI 3.0 models for upgrade functionality
from otterapi.openapi.v3 import OpenAPI, v3 as openapi_v3

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
    items: Union['Schema', list['Schema']] | None = None
    all_of: list['Schema'] | None = Field(None, alias='allOf')
    properties: dict[str, 'Schema'] | None = None
    additional_properties: Union['Schema', bool] | None = Field(
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

    @model_validator(mode='before')
    @classmethod
    def validate_and_convert_responses(cls, data: Any) -> Any:
        """Validate response keys and convert to Response objects."""
        if not isinstance(data, dict):
            return data

        result = {}
        for key, value in data.items():
            # Validate response keys (except vendor extensions)
            if not key.startswith('x-'):
                if key != 'default' and not (key.isdigit() and len(key) == 3):
                    raise ValueError(
                        f"Response key must be a 3-digit status code or 'default', got: {key}"
                    )

            # Convert dict values to Response objects (or keep JsonReference)
            if isinstance(value, dict):
                if '$ref' in value:
                    result[key] = JsonReference(**value)
                else:
                    result[key] = Response(**value)
            else:
                result[key] = value

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


class Paths(BaseModelWithVendorExtensions):
    """
    Paths object containing all API paths.

    Keys must start with "/" (except vendor extensions starting with "x-").
    """

    model_config = ConfigDict(extra='allow', populate_by_name=True)

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
            A tuple of (OpenAPI 3.0 dict, list of warnings)

        Note: Returns a dict rather than openapi_v3.OpenAPI object because the v3
        models have limitations (e.g., Responses has extra='forbid' but needs to
        accept arbitrary status codes). The dict can be validated with
        openapi_v3.OpenAPI.model_validate() if needed, or used directly.

        Warnings are generated for:
        - Lossy conversions
        - Structural changes
        - Missing data that requires defaults
        - OAuth2 flow restructuring
        - Collection format conversions
        """
        warnings: list[str] = []

        # Convert basic metadata
        info = self._convert_info()

        # Convert servers from host/basePath/schemes
        servers = self._convert_servers(warnings)

        # Convert components
        components = self._convert_components(warnings)

        # Convert paths - returns dict for Paths RootModel
        paths_dict = self._convert_paths(warnings)

        # Build OpenAPI dict
        openapi_dict: dict[str, Any] = {
            'openapi': '3.0.3',
            'info': info.model_dump(by_alias=True, exclude_none=True, mode='json'),
            'paths': paths_dict,
        }

        if servers:
            openapi_dict['servers'] = [
                s.model_dump(by_alias=True, exclude_none=True, mode='json')
                for s in servers
            ]

        if components:
            openapi_dict['components'] = components.model_dump(
                by_alias=True, exclude_none=True, mode='json'
            )

        if self.security:
            openapi_dict['security'] = self.security

        if self.tags:
            openapi_dict['tags'] = [
                openapi_v3.Tag(
                    name=tag.name,
                    description=tag.description,
                    externalDocs=openapi_v3.ExternalDocumentation(
                        url=str(tag.external_docs.url),
                        description=tag.external_docs.description,
                    )
                    if tag.external_docs
                    else None,
                ).model_dump(by_alias=True, exclude_none=True, mode='json')
                for tag in self.tags
            ]

        if self.external_docs:
            openapi_dict['externalDocs'] = openapi_v3.ExternalDocumentation(
                url=str(self.external_docs.url),
                description=self.external_docs.description,
            ).model_dump(by_alias=True, exclude_none=True, mode='json')

        return OpenAPI.model_validate(openapi_dict), warnings

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

    def _convert_servers(self, warnings: list[str]) -> list[openapi_v3.Server]:
        """Convert host, basePath, and schemes to servers array."""
        if not self.host and not self.base_path:
            warnings.append(
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

    def _convert_components(self, warnings: list[str]) -> openapi_v3.Components | None:
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
                    response, self.produces or ['application/json'], warnings
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

        # Use model_validate to create Components from dict
        components_dict = {}
        if schemas:
            components_dict['schemas'] = schemas
        if parameters:
            components_dict['parameters'] = parameters
        if responses:
            components_dict['responses'] = responses
        if security_schemes:
            components_dict['securitySchemes'] = security_schemes

        return openapi_v3.Components.model_validate(components_dict)

    def _convert_paths(self, warnings: list[str]) -> dict[str, Any]:
        """Convert paths object."""
        result = {}

        # Get paths from __pydantic_extra__
        if hasattr(self.paths, '__pydantic_extra__') and self.paths.__pydantic_extra__:
            for path, path_item in self.paths.__pydantic_extra__.items():
                if path.startswith('x-'):
                    # Vendor extension
                    result[path] = path_item
                elif isinstance(path_item, PathItem):
                    result[path] = self._convert_path_item(path_item, warnings)

        return result

    def _convert_path_item(
        self, path_item: PathItem, warnings: list[str]
    ) -> dict[str, Any]:
        """Convert a single PathItem."""
        result: dict[str, Any] = {}

        if path_item.ref:
            result['$ref'] = self._update_ref(path_item.ref)

        # Convert each operation
        for method in ['get', 'put', 'post', 'delete', 'options', 'head', 'patch']:
            operation = getattr(path_item, method, None)
            if operation:
                result[method] = self._convert_operation(
                    operation, warnings, method=method
                )

        # Convert path-level parameters
        if path_item.parameters:
            result['parameters'] = [
                self._convert_parameter_item_to_dict(param, warnings)
                for param in path_item.parameters
            ]

        result.update(self._extract_vendor_extensions(path_item))

        return result

    def _convert_operation(
        self, operation: Operation, warnings: list[str], method: str = None
    ) -> dict[str, Any]:
        """Convert an Operation object."""
        result: dict[str, Any] = {}

        if operation.tags:
            result['tags'] = operation.tags

        if operation.summary:
            result['summary'] = operation.summary

        if operation.description:
            result['description'] = operation.description

        if operation.external_docs:
            result['externalDocs'] = {
                'url': str(operation.external_docs.url),
                'description': operation.external_docs.description,
            }

        if operation.operation_id:
            result['operationId'] = operation.operation_id

        # Convert parameters and extract body/formData
        body_schema = None
        if operation.parameters:
            converted = self._convert_parameters(
                operation.parameters,
                operation.consumes or self.consumes,
                warnings,
            )
            if converted['parameters']:
                result['parameters'] = converted['parameters']
            if converted['requestBody']:
                result['requestBody'] = converted['requestBody']
            # Extract body schema for response inference
            body_schema = converted.get('body_schema')

        # Convert responses
        result['responses'] = self._convert_responses(
            operation.responses,
            operation.produces or self.produces or ['application/json'],
            warnings,
            method=method,
            body_schema=body_schema,
        )

        if operation.deprecated:
            result['deprecated'] = True

        if operation.security:
            result['security'] = operation.security

        if operation.schemes:
            # Convert schemes to servers at operation level
            servers = []
            for scheme in operation.schemes:
                host = self.host or ''
                base_path = self.base_path or ''
                url = f'{scheme.value}://{host}{base_path}' if host else base_path
                servers.append({'url': url})
            result['servers'] = servers

        result.update(self._extract_vendor_extensions(operation))

        return result

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
        result_params = []
        body_param = None
        form_params = []

        for param in parameters:
            if isinstance(param, JsonReference):
                # Handle reference
                result_params.append({'$ref': self._update_ref(param.ref)})
            elif isinstance(param, BodyParameter):
                body_param = param
            elif isinstance(param, NonBodyParameter):
                if param.in_ == ParameterLocation.FORM_DATA:
                    form_params.append(param)
                else:
                    result_params.append(
                        self._convert_non_body_parameter_to_dict(param, warnings)
                    )

        request_body = None
        body_schema = None
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
    ) -> dict[str, Any]:
        """Convert body parameter to requestBody."""
        media_types = consumes or ['application/json']

        content = {}
        for media_type in media_types:
            content[media_type] = {
                'schema': self._convert_schema_to_dict(param.schema_)
            }

        result: dict[str, Any] = {'content': content}

        if param.description:
            result['description'] = param.description

        if param.required:
            result['required'] = True

        return result

    def _convert_formdata_parameters(
        self,
        params: list[NonBodyParameter],
        consumes: list[str] | None,
        warnings: list[str],
    ) -> dict[str, Any]:
        """Convert formData parameters to requestBody."""
        # Determine media type
        has_file = any(p.type == PrimitiveType.FILE for p in params)
        media_type = (
            'multipart/form-data' if has_file else 'application/x-www-form-urlencoded'
        )

        # Check if consumes specifies something different
        if consumes and media_type not in consumes:
            if has_file and 'multipart/form-data' not in consumes:
                warnings.append(
                    f'File upload parameter requires multipart/form-data, '
                    f'but consumes specifies: {consumes}'
                )

        # Build schema with properties
        properties = {}
        required = []

        for param in params:
            prop_schema = self._convert_parameter_to_schema(param, warnings)
            properties[param.name] = prop_schema
            if param.required:
                required.append(param.name)

        schema: dict[str, Any] = {
            'type': 'object',
            'properties': properties,
        }

        if required:
            schema['required'] = required

        result: dict[str, Any] = {'content': {media_type: {'schema': schema}}}

        # Use description from first parameter if any
        if params and params[0].description:
            result['description'] = params[0].description

        return result

    def _convert_non_body_parameter_to_dict(
        self, param: NonBodyParameter, warnings: list[str]
    ) -> dict[str, Any]:
        """Convert query/header/path parameter to OpenAPI 3.0 format."""
        result: dict[str, Any] = {
            'name': param.name,
            'in': param.in_.value,
        }

        if param.description:
            result['description'] = param.description

        if param.required:
            result['required'] = True

        if param.allow_empty_value:
            result['allowEmptyValue'] = True

        # Convert to schema
        result['schema'] = self._convert_parameter_to_schema(param, warnings)

        # Convert collectionFormat to style and explode
        if param.type == PrimitiveType.ARRAY and param.collection_format:
            style, explode = self._convert_collection_format(
                param.collection_format,
                param.in_,
                warnings,
            )
            if style:
                result['style'] = style
            if explode is not None:
                result['explode'] = explode

        result.update(self._extract_vendor_extensions(param))

        return result

    def _convert_parameter_to_schema(
        self, param: NonBodyParameter, warnings: list[str]
    ) -> dict[str, Any]:
        """Convert parameter properties to a schema object."""
        schema: dict[str, Any] = {}

        if param.type == PrimitiveType.FILE:
            schema['type'] = 'string'
            schema['format'] = 'binary'
        else:
            schema['type'] = param.type.value

        if param.format:
            schema['format'] = param.format

        if param.items:
            schema['items'] = self._convert_primitives_items(param.items)

        if param.default is not None:
            schema['default'] = param.default

        if param.maximum is not None:
            schema['maximum'] = param.maximum

        if param.exclusive_maximum is not None:
            schema['exclusiveMaximum'] = param.exclusive_maximum

        if param.minimum is not None:
            schema['minimum'] = param.minimum

        if param.exclusive_minimum is not None:
            schema['exclusiveMinimum'] = param.exclusive_minimum

        if param.max_length is not None:
            schema['maxLength'] = param.max_length

        if param.min_length is not None:
            schema['minLength'] = param.min_length

        if param.pattern:
            schema['pattern'] = param.pattern

        if param.max_items is not None:
            schema['maxItems'] = param.max_items

        if param.min_items is not None:
            schema['minItems'] = param.min_items

        if param.unique_items is not None:
            schema['uniqueItems'] = param.unique_items

        if param.enum:
            schema['enum'] = param.enum

        if param.multiple_of is not None:
            schema['multipleOf'] = param.multiple_of

        return schema

    def _convert_primitives_items(self, items: PrimitivesItems) -> dict[str, Any]:
        """Convert PrimitivesItems to schema."""
        schema: dict[str, Any] = {}

        if items.type:
            schema['type'] = items.type.value

        if items.format:
            schema['format'] = items.format

        if items.items:
            schema['items'] = self._convert_primitives_items(items.items)

        if items.default is not None:
            schema['default'] = items.default

        if items.maximum is not None:
            schema['maximum'] = items.maximum

        if items.minimum is not None:
            schema['minimum'] = items.minimum

        if items.max_length is not None:
            schema['maxLength'] = items.max_length

        if items.min_length is not None:
            schema['minLength'] = items.min_length

        if items.pattern:
            schema['pattern'] = items.pattern

        if items.max_items is not None:
            schema['maxItems'] = items.max_items

        if items.min_items is not None:
            schema['minItems'] = items.min_items

        if items.unique_items is not None:
            schema['uniqueItems'] = items.unique_items

        if items.enum:
            schema['enum'] = items.enum

        if items.multiple_of is not None:
            schema['multipleOf'] = items.multiple_of

        return schema

    def _convert_collection_format(
        self,
        collection_format: CollectionFormat | CollectionFormatWithMulti,
        param_location: ParameterLocation,
        warnings: list[str],
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
                warnings.append(
                    "collectionFormat 'multi' converted to style=form with explode=true"
                )
                return 'form', True
            collection_format = CollectionFormat(collection_format.value)

        # Mapping for other formats
        format_map = {
            CollectionFormat.CSV: (default_styles.get(param_location, 'simple'), False),
            CollectionFormat.SSV: ('spaceDelimited', False),
            CollectionFormat.TSV: ('pipeDelimited', False),
            CollectionFormat.PIPES: ('pipeDelimited', False),
        }

        if collection_format == CollectionFormat.TSV:
            warnings.append(
                "collectionFormat 'tsv' has no direct equivalent in OpenAPI 3.0, "
                'using pipeDelimited'
            )

        return format_map.get(collection_format, (None, None))

    def _convert_responses(
        self,
        responses: Responses,
        produces: list[str],
        warnings: list[str],
        method: str = None,
        body_schema: dict[str, Any] = None,
    ) -> dict[str, Any]:
        """Convert Responses object.

        Args:
            responses: The Swagger 2.0 Responses object
            produces: List of media types the operation produces
            warnings: List to append warnings to
            method: HTTP method (post, put, etc.) for response inference
            body_schema: Body parameter schema for inferring response when not specified
        """
        result = {}

        if hasattr(responses, '__pydantic_extra__') and responses.__pydantic_extra__:
            for status_code, response in responses.__pydantic_extra__.items():
                if status_code.startswith('x-'):
                    result[status_code] = response
                elif isinstance(response, JsonReference):
                    result[status_code] = {'$ref': self._update_ref(response.ref)}
                elif isinstance(response, Response):
                    result[status_code] = self._convert_response_to_dict(
                        response, produces, warnings
                    )

        # Infer success response schema if not present
        # For POST/PUT operations with a body parameter that references a model,
        # it's common for the response to return the same model
        if body_schema and method in ('post', 'put'):
            # Check if there's already a 2xx response with a schema
            has_success_schema = False
            for status_code in result:
                if status_code.startswith('2') and not status_code.startswith('x-'):
                    response_obj = result[status_code]
                    if isinstance(response_obj, dict) and 'content' in response_obj:
                        # Check if any content type has a schema
                        for media_type_obj in response_obj['content'].values():
                            if (
                                isinstance(media_type_obj, dict)
                                and 'schema' in media_type_obj
                            ):
                                has_success_schema = True
                                break
                    if has_success_schema:
                        break

            # Only infer if the body schema is a $ref (model reference)
            if not has_success_schema and body_schema.get('$ref'):
                warnings.append(
                    f'Inferred 200 response schema from request body for {method.upper()} operation'
                )
                # Build the inferred response
                content = {}
                for media_type in produces:
                    content[media_type] = {'schema': body_schema}

                result['200'] = {
                    'description': 'Successful operation',
                    'content': content,
                }

        return result

    def _convert_header_to_dict(self, header: Header) -> dict[str, Any]:
        """Convert Header to OpenAPI 3.0 format."""
        result: dict[str, Any] = {}

        if header.description:
            result['description'] = header.description

        # Convert to schema
        schema: dict[str, Any] = {'type': header.type.value}

        if header.format:
            schema['format'] = header.format

        if header.items:
            schema['items'] = self._convert_primitives_items(header.items)

        if header.default is not None:
            schema['default'] = header.default

        if header.maximum is not None:
            schema['maximum'] = header.maximum

        if header.minimum is not None:
            schema['minimum'] = header.minimum

        if header.max_length is not None:
            schema['maxLength'] = header.max_length

        if header.min_length is not None:
            schema['minLength'] = header.min_length

        if header.pattern:
            schema['pattern'] = header.pattern

        if header.max_items is not None:
            schema['maxItems'] = header.max_items

        if header.min_items is not None:
            schema['minItems'] = header.min_items

        if header.unique_items is not None:
            schema['uniqueItems'] = header.unique_items

        if header.enum:
            schema['enum'] = header.enum

        if header.multiple_of is not None:
            schema['multipleOf'] = header.multiple_of

        result['schema'] = schema

        result.update(self._extract_vendor_extensions(header))

        return result

    def _convert_schema_to_dict(self, schema: Schema | FileSchema) -> dict[str, Any]:
        """Convert Schema object to OpenAPI 3.0 format."""
        if isinstance(schema, FileSchema):
            return {
                'type': 'string',
                'format': 'binary',
            }

        result: dict[str, Any] = {}

        if schema.ref:
            result['$ref'] = self._update_ref(schema.ref)
            return result

        if schema.type:
            result['type'] = schema.type

        if schema.format:
            result['format'] = schema.format

        if schema.title:
            result['title'] = schema.title

        if schema.description:
            result['description'] = schema.description

        if schema.default is not None:
            result['default'] = schema.default

        if schema.multiple_of is not None:
            result['multipleOf'] = schema.multiple_of

        if schema.maximum is not None:
            result['maximum'] = schema.maximum

        if schema.exclusive_maximum is not None:
            result['exclusiveMaximum'] = schema.exclusive_maximum

        if schema.minimum is not None:
            result['minimum'] = schema.minimum

        if schema.exclusive_minimum is not None:
            result['exclusiveMinimum'] = schema.exclusive_minimum

        if schema.max_length is not None:
            result['maxLength'] = schema.max_length

        if schema.min_length is not None:
            result['minLength'] = schema.min_length

        if schema.pattern:
            result['pattern'] = schema.pattern

        if schema.max_items is not None:
            result['maxItems'] = schema.max_items

        if schema.min_items is not None:
            result['minItems'] = schema.min_items

        if schema.unique_items is not None:
            result['uniqueItems'] = schema.unique_items

        if schema.max_properties is not None:
            result['maxProperties'] = schema.max_properties

        if schema.min_properties is not None:
            result['minProperties'] = schema.min_properties

        if schema.required:
            result['required'] = schema.required

        if schema.enum:
            result['enum'] = schema.enum

        if schema.items:
            if isinstance(schema.items, list):
                result['items'] = [
                    self._convert_schema_to_dict(item) for item in schema.items
                ]
            else:
                result['items'] = self._convert_schema_to_dict(schema.items)

        if schema.all_of:
            result['allOf'] = [self._convert_schema_to_dict(s) for s in schema.all_of]

        if schema.properties:
            result['properties'] = {
                name: self._convert_schema_to_dict(prop)
                for name, prop in schema.properties.items()
            }

        if schema.additional_properties is not None:
            if isinstance(schema.additional_properties, bool):
                result['additionalProperties'] = schema.additional_properties
            else:
                result['additionalProperties'] = self._convert_schema_to_dict(
                    schema.additional_properties
                )

        # Convert discriminator from string to object
        if schema.discriminator:
            result['discriminator'] = {'propertyName': schema.discriminator}

        if schema.read_only:
            result['readOnly'] = True

        if schema.xml:
            xml_dict: dict[str, Any] = {}
            if schema.xml.name:
                xml_dict['name'] = schema.xml.name
            if schema.xml.namespace:
                xml_dict['namespace'] = schema.xml.namespace
            if schema.xml.prefix:
                xml_dict['prefix'] = schema.xml.prefix
            if schema.xml.attribute:
                xml_dict['attribute'] = True
            if schema.xml.wrapped:
                xml_dict['wrapped'] = True
            if xml_dict:
                result['xml'] = xml_dict

        if schema.external_docs:
            result['externalDocs'] = {
                'url': str(schema.external_docs.url),
                'description': schema.external_docs.description,
            }

        if schema.example is not None:
            result['example'] = schema.example

        result.update(self._extract_vendor_extensions(schema))

        return result

    def _convert_component_parameter_to_dict(
        self, param: Parameter, warnings: list[str]
    ) -> dict[str, Any]:
        """Convert a component-level parameter."""
        if isinstance(param, JsonReference):
            return {'$ref': self._update_ref(param.ref)}
        elif isinstance(param, BodyParameter):
            # Body parameters in components need special handling
            return self._convert_body_parameter_to_dict(
                param, ['application/json'], warnings
            )
        elif isinstance(param, NonBodyParameter):
            return self._convert_non_body_parameter_to_dict(param, warnings)
        return {}

    def _convert_response_to_dict(
        self, response: Response, produces: list[str], warnings: list[str]
    ) -> dict[str, Any]:
        """Convert a single Response object to dict."""
        result: dict[str, Any] = {'description': response.description}

        # Convert schema to content
        if response.schema_:
            content = {}
            for media_type in produces:
                content[media_type] = {
                    'schema': self._convert_schema_to_dict(response.schema_)
                }
            result['content'] = content

        # Convert headers
        if response.headers:
            result['headers'] = {
                name: self._convert_header_to_dict(header)
                for name, header in response.headers.items()
            }

        # Handle examples
        if response.examples:
            # In OpenAPI 3.0, examples are per media type
            if 'content' in result:
                for media_type in result['content']:
                    if media_type in response.examples:
                        result['content'][media_type]['example'] = response.examples[
                            media_type
                        ]

        result.update(self._extract_vendor_extensions(response))

        return result

    def _convert_security_scheme_to_dict(
        self, scheme: SecurityScheme, warnings: list[str]
    ) -> dict[str, Any]:
        """Convert a single security scheme."""
        if isinstance(scheme, BasicAuthenticationSecurity):
            result: dict[str, Any] = {
                'type': 'http',
                'scheme': 'basic',
            }
            if scheme.description:
                result['description'] = scheme.description
            result.update(self._extract_vendor_extensions(scheme))
            return result

        elif isinstance(scheme, ApiKeySecurity):
            result = {
                'type': 'apiKey',
                'name': scheme.name,
                'in': scheme.in_.value,
            }
            if scheme.description:
                result['description'] = scheme.description
            result.update(self._extract_vendor_extensions(scheme))
            return result

        elif isinstance(scheme, OAuth2ImplicitSecurity):
            flows: dict[str, Any] = {
                'implicit': {
                    'authorizationUrl': str(scheme.authorization_url),
                    'scopes': scheme.scopes,
                }
            }
            result = {
                'type': 'oauth2',
                'flows': flows,
            }
            if scheme.description:
                result['description'] = scheme.description
            result.update(self._extract_vendor_extensions(scheme))
            warnings.append('OAuth2 implicit flow restructured for OpenAPI 3.0')
            return result

        elif isinstance(scheme, OAuth2PasswordSecurity):
            flows = {
                'password': {
                    'tokenUrl': str(scheme.token_url),
                    'scopes': scheme.scopes,
                }
            }
            result = {
                'type': 'oauth2',
                'flows': flows,
            }
            if scheme.description:
                result['description'] = scheme.description
            result.update(self._extract_vendor_extensions(scheme))
            warnings.append('OAuth2 password flow restructured for OpenAPI 3.0')
            return result

        elif isinstance(scheme, OAuth2ApplicationSecurity):
            flows = {
                'clientCredentials': {
                    'tokenUrl': str(scheme.token_url),
                    'scopes': scheme.scopes,
                }
            }
            result = {
                'type': 'oauth2',
                'flows': flows,
            }
            if scheme.description:
                result['description'] = scheme.description
            result.update(self._extract_vendor_extensions(scheme))
            warnings.append(
                'OAuth2 application flow converted to clientCredentials for OpenAPI 3.0'
            )
            return result

        elif isinstance(scheme, OAuth2AccessCodeSecurity):
            flows = {
                'authorizationCode': {
                    'authorizationUrl': str(scheme.authorization_url),
                    'tokenUrl': str(scheme.token_url),
                    'scopes': scheme.scopes,
                }
            }
            result = {
                'type': 'oauth2',
                'flows': flows,
            }
            if scheme.description:
                result['description'] = scheme.description
            result.update(self._extract_vendor_extensions(scheme))
            warnings.append(
                'OAuth2 accessCode flow converted to authorizationCode for OpenAPI 3.0'
            )
            return result

        return {}

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
        self, param: Parameter, warnings: list[str]
    ) -> dict[str, Any]:
        """Convert a single parameter (handles both body and non-body)."""
        if isinstance(param, JsonReference):
            return {'$ref': self._update_ref(param.ref)}
        elif isinstance(param, BodyParameter):
            # This shouldn't happen at path level in Swagger 2.0, but handle it
            warnings.append(
                'Body parameter found at path level, converting to inline requestBody'
            )
            return self._convert_body_parameter_to_dict(
                param, ['application/json'], warnings
            )
        elif isinstance(param, NonBodyParameter):
            return self._convert_non_body_parameter_to_dict(param, warnings)
        return {}


# Update forward references for recursive models
Schema.model_rebuild()
PrimitivesItems.model_rebuild()
