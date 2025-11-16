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
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator, model_validator


# ============================================================================
# Enums
# ============================================================================


class SchemeType(str, Enum):
    """Transfer protocol schemes."""

    HTTP = "http"
    HTTPS = "https"
    WS = "ws"
    WSS = "wss"


class ParameterLocation(str, Enum):
    """Parameter location types."""

    QUERY = "query"
    HEADER = "header"
    PATH = "path"
    FORM_DATA = "formData"
    BODY = "body"


class PrimitiveType(str, Enum):
    """Primitive types for non-body parameters."""

    STRING = "string"
    NUMBER = "number"
    INTEGER = "integer"
    BOOLEAN = "boolean"
    ARRAY = "array"
    FILE = "file"


class CollectionFormat(str, Enum):
    """Collection format for array parameters."""

    CSV = "csv"
    SSV = "ssv"
    TSV = "tsv"
    PIPES = "pipes"


class CollectionFormatWithMulti(str, Enum):
    """Collection format including multi for query/formData parameters."""

    CSV = "csv"
    SSV = "ssv"
    TSV = "tsv"
    PIPES = "pipes"
    MULTI = "multi"


class SecuritySchemeType(str, Enum):
    """Security scheme types."""

    BASIC = "basic"
    API_KEY = "apiKey"
    OAUTH2 = "oauth2"


class OAuth2Flow(str, Enum):
    """OAuth2 flow types."""

    IMPLICIT = "implicit"
    PASSWORD = "password"
    APPLICATION = "application"
    ACCESS_CODE = "accessCode"


class ApiKeyLocation(str, Enum):
    """API key location."""

    HEADER = "header"
    QUERY = "query"


# ============================================================================
# Base Models
# ============================================================================


class BaseModelWithVendorExtensions(BaseModel):
    """Base model that allows vendor extensions (x- fields)."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)


class JsonReference(BaseModel):
    """JSON Reference object."""

    ref: str = Field(..., alias="$ref")

    model_config = ConfigDict(populate_by_name=True)


# ============================================================================
# Info Models
# ============================================================================


class Contact(BaseModelWithVendorExtensions):
    """Contact information for the API."""

    name: Optional[str] = None
    url: Optional[HttpUrl] = None
    email: Optional[str] = None


class License(BaseModelWithVendorExtensions):
    """License information for the API."""

    name: str
    url: Optional[HttpUrl] = None


class Info(BaseModelWithVendorExtensions):
    """General information about the API."""

    title: str
    version: str
    description: Optional[str] = None
    terms_of_service: Optional[str] = Field(None, alias="termsOfService")
    contact: Optional[Contact] = None
    license: Optional[License] = None


class ExternalDocs(BaseModelWithVendorExtensions):
    """External documentation reference."""

    url: HttpUrl
    description: Optional[str] = None


class Tag(BaseModelWithVendorExtensions):
    """API tag for grouping operations."""

    name: str
    description: Optional[str] = None
    external_docs: Optional[ExternalDocs] = Field(None, alias="externalDocs")


# ============================================================================
# XML Model
# ============================================================================


class XML(BaseModelWithVendorExtensions):
    """XML representation metadata."""

    name: Optional[str] = None
    namespace: Optional[str] = None
    prefix: Optional[str] = None
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

    ref: Optional[str] = Field(None, alias="$ref")
    format: Optional[str] = None
    title: Optional[str] = None
    description: Optional[str] = None
    default: Optional[Any] = None
    multiple_of: Optional[float] = Field(None, alias="multipleOf", gt=0)
    maximum: Optional[float] = None
    exclusive_maximum: Optional[bool] = Field(None, alias="exclusiveMaximum")
    minimum: Optional[float] = None
    exclusive_minimum: Optional[bool] = Field(None, alias="exclusiveMinimum")
    max_length: Optional[int] = Field(None, alias="maxLength", ge=0)
    min_length: Optional[int] = Field(None, alias="minLength", ge=0)
    pattern: Optional[str] = None
    max_items: Optional[int] = Field(None, alias="maxItems", ge=0)
    min_items: Optional[int] = Field(None, alias="minItems", ge=0)
    unique_items: Optional[bool] = Field(None, alias="uniqueItems")
    max_properties: Optional[int] = Field(None, alias="maxProperties", ge=0)
    min_properties: Optional[int] = Field(None, alias="minProperties", ge=0)
    required: Optional[List[str]] = None
    enum: Optional[List[Any]] = None
    type: Optional[Union[str, List[str]]] = None
    items: Optional[Union["Schema", List["Schema"]]] = None
    all_of: Optional[List["Schema"]] = Field(None, alias="allOf")
    properties: Optional[Dict[str, "Schema"]] = None
    additional_properties: Optional[Union["Schema", bool]] = Field(
        None, alias="additionalProperties"
    )
    discriminator: Optional[str] = None
    read_only: bool = Field(False, alias="readOnly")
    xml: Optional[XML] = None
    external_docs: Optional[ExternalDocs] = Field(None, alias="externalDocs")
    example: Optional[Any] = None


class FileSchema(BaseModelWithVendorExtensions):
    """Schema for file uploads."""

    type: Literal["file"]
    format: Optional[str] = None
    title: Optional[str] = None
    description: Optional[str] = None
    default: Optional[Any] = None
    required: Optional[List[str]] = None
    read_only: bool = Field(False, alias="readOnly")
    external_docs: Optional[ExternalDocs] = Field(None, alias="externalDocs")
    example: Optional[Any] = None


# ============================================================================
# Parameter Models
# ============================================================================


class PrimitivesItems(BaseModelWithVendorExtensions):
    """Items object for primitive array parameters."""

    type: Optional[PrimitiveType] = None
    format: Optional[str] = None
    items: Optional["PrimitivesItems"] = None
    collection_format: Optional[CollectionFormat] = Field(
        CollectionFormat.CSV, alias="collectionFormat"
    )
    default: Optional[Any] = None
    maximum: Optional[float] = None
    exclusive_maximum: Optional[bool] = Field(None, alias="exclusiveMaximum")
    minimum: Optional[float] = None
    exclusive_minimum: Optional[bool] = Field(None, alias="exclusiveMinimum")
    max_length: Optional[int] = Field(None, alias="maxLength", ge=0)
    min_length: Optional[int] = Field(None, alias="minLength", ge=0)
    pattern: Optional[str] = None
    max_items: Optional[int] = Field(None, alias="maxItems", ge=0)
    min_items: Optional[int] = Field(None, alias="minItems", ge=0)
    unique_items: Optional[bool] = Field(None, alias="uniqueItems")
    enum: Optional[List[Any]] = None
    multiple_of: Optional[float] = Field(None, alias="multipleOf", gt=0)


class BaseParameterFields(BaseModelWithVendorExtensions):
    """Common fields for all parameter types."""

    name: str
    in_: ParameterLocation = Field(..., alias="in")
    description: Optional[str] = None
    required: bool = False


class BodyParameter(BaseParameterFields):
    """Body parameter definition."""

    in_: Literal[ParameterLocation.BODY] = Field(
        ParameterLocation.BODY, alias="in"
    )
    schema_: Schema = Field(..., alias="schema")
    required: bool = False

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class NonBodyParameter(BaseParameterFields):
    """Non-body parameter (query, header, path, formData)."""

    in_: Union[
        Literal[ParameterLocation.QUERY],
        Literal[ParameterLocation.HEADER],
        Literal[ParameterLocation.PATH],
        Literal[ParameterLocation.FORM_DATA],
    ] = Field(..., alias="in")
    type: PrimitiveType
    format: Optional[str] = None
    allow_empty_value: Optional[bool] = Field(None, alias="allowEmptyValue")
    items: Optional[PrimitivesItems] = None
    collection_format: Optional[
        Union[CollectionFormat, CollectionFormatWithMulti]
    ] = Field(CollectionFormat.CSV, alias="collectionFormat")
    default: Optional[Any] = None
    maximum: Optional[float] = None
    exclusive_maximum: Optional[bool] = Field(None, alias="exclusiveMaximum")
    minimum: Optional[float] = None
    exclusive_minimum: Optional[bool] = Field(None, alias="exclusiveMinimum")
    max_length: Optional[int] = Field(None, alias="maxLength", ge=0)
    min_length: Optional[int] = Field(None, alias="minLength", ge=0)
    pattern: Optional[str] = None
    max_items: Optional[int] = Field(None, alias="maxItems", ge=0)
    min_items: Optional[int] = Field(None, alias="minItems", ge=0)
    unique_items: Optional[bool] = Field(None, alias="uniqueItems")
    enum: Optional[List[Any]] = None
    multiple_of: Optional[float] = Field(None, alias="multipleOf", gt=0)

    @model_validator(mode="after")
    def validate_path_required(self) -> "NonBodyParameter":
        """Path parameters must be required."""
        if self.in_ == ParameterLocation.PATH and not self.required:
            raise ValueError("Path parameters must have required=True")
        return self

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


Parameter = Union[BodyParameter, NonBodyParameter, JsonReference]


# ============================================================================
# Response Models
# ============================================================================


class Header(BaseModelWithVendorExtensions):
    """Response header definition."""

    type: PrimitiveType
    format: Optional[str] = None
    items: Optional[PrimitivesItems] = None
    collection_format: Optional[CollectionFormat] = Field(
        CollectionFormat.CSV, alias="collectionFormat"
    )
    default: Optional[Any] = None
    maximum: Optional[float] = None
    exclusive_maximum: Optional[bool] = Field(None, alias="exclusiveMaximum")
    minimum: Optional[float] = None
    exclusive_minimum: Optional[bool] = Field(None, alias="exclusiveMinimum")
    max_length: Optional[int] = Field(None, alias="maxLength", ge=0)
    min_length: Optional[int] = Field(None, alias="minLength", ge=0)
    pattern: Optional[str] = None
    max_items: Optional[int] = Field(None, alias="maxItems", ge=0)
    min_items: Optional[int] = Field(None, alias="minItems", ge=0)
    unique_items: Optional[bool] = Field(None, alias="uniqueItems")
    enum: Optional[List[Any]] = None
    multiple_of: Optional[float] = Field(None, alias="multipleOf", gt=0)
    description: Optional[str] = None


class Response(BaseModelWithVendorExtensions):
    """Response object."""

    description: str
    schema_: Optional[Union[Schema, FileSchema]] = Field(None, alias="schema")
    headers: Optional[Dict[str, Header]] = None
    examples: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


ResponseValue = Union[Response, JsonReference]


class Responses(BaseModelWithVendorExtensions):
    """
    Response definitions for an operation.
    
    Keys can be HTTP status codes (as strings) or "default".
    """

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    def __getitem__(self, key: str) -> ResponseValue:
        """Allow dict-like access to response codes."""
        return self.__pydantic_extra__.get(key) or getattr(self, key, None)

    @model_validator(mode="before")
    @classmethod
    def validate_and_convert_responses(cls, data: Any) -> Any:
        """Validate response keys and convert to Response objects."""
        if not isinstance(data, dict):
            return data
        
        result = {}
        for key, value in data.items():
            # Validate response keys (except vendor extensions)
            if not key.startswith("x-"):
                if key != "default" and not (key.isdigit() and len(key) == 3):
                    raise ValueError(
                        f"Response key must be a 3-digit status code or 'default', got: {key}"
                    )
            
            # Convert dict values to Response objects (or keep JsonReference)
            if isinstance(value, dict):
                if "$ref" in value:
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

    tags: Optional[List[str]] = None
    summary: Optional[str] = None
    description: Optional[str] = None
    external_docs: Optional[ExternalDocs] = Field(None, alias="externalDocs")
    operation_id: Optional[str] = Field(None, alias="operationId")
    consumes: Optional[List[str]] = None
    produces: Optional[List[str]] = None
    parameters: Optional[List[Parameter]] = None
    responses: Responses
    schemes: Optional[List[SchemeType]] = None
    deprecated: bool = False
    security: Optional[List[Dict[str, List[str]]]] = None

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class PathItem(BaseModelWithVendorExtensions):
    """Path item with operations."""

    ref: Optional[str] = Field(None, alias="$ref")
    get: Optional[Operation] = None
    put: Optional[Operation] = None
    post: Optional[Operation] = None
    delete: Optional[Operation] = None
    options: Optional[Operation] = None
    head: Optional[Operation] = None
    patch: Optional[Operation] = None
    parameters: Optional[List[Parameter]] = None

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class Paths(BaseModelWithVendorExtensions):
    """
    Paths object containing all API paths.
    
    Keys must start with "/" (except vendor extensions starting with "x-").
    """

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    @model_validator(mode="before")
    @classmethod
    def validate_and_convert_paths(cls, data: Any) -> Any:
        """Validate path keys and convert path items to PathItem objects."""
        if not isinstance(data, dict):
            return data
        
        result = {}
        for key, value in data.items():
            # Validate path keys
            if not key.startswith("x-") and not key.startswith("/"):
                raise ValueError(f"Path must start with '/', got: {key}")
            
            # Convert dict values to PathItem objects for paths
            if key.startswith("/") and isinstance(value, dict):
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
    description: Optional[str] = None

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class ApiKeySecurity(BaseModelWithVendorExtensions):
    """API key security scheme."""

    type: Literal[SecuritySchemeType.API_KEY]
    name: str
    in_: ApiKeyLocation = Field(..., alias="in")
    description: Optional[str] = None

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class OAuth2ImplicitSecurity(BaseModelWithVendorExtensions):
    """OAuth2 implicit flow security scheme."""

    type: Literal[SecuritySchemeType.OAUTH2]
    flow: Literal[OAuth2Flow.IMPLICIT]
    authorization_url: HttpUrl = Field(..., alias="authorizationUrl")
    scopes: Dict[str, str] = {}
    description: Optional[str] = None

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class OAuth2PasswordSecurity(BaseModelWithVendorExtensions):
    """OAuth2 password flow security scheme."""

    type: Literal[SecuritySchemeType.OAUTH2]
    flow: Literal[OAuth2Flow.PASSWORD]
    token_url: HttpUrl = Field(..., alias="tokenUrl")
    scopes: Dict[str, str] = {}
    description: Optional[str] = None

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class OAuth2ApplicationSecurity(BaseModelWithVendorExtensions):
    """OAuth2 application flow security scheme."""

    type: Literal[SecuritySchemeType.OAUTH2]
    flow: Literal[OAuth2Flow.APPLICATION]
    token_url: HttpUrl = Field(..., alias="tokenUrl")
    scopes: Dict[str, str] = {}
    description: Optional[str] = None

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class OAuth2AccessCodeSecurity(BaseModelWithVendorExtensions):
    """OAuth2 access code flow security scheme."""

    type: Literal[SecuritySchemeType.OAUTH2]
    flow: Literal[OAuth2Flow.ACCESS_CODE]
    authorization_url: HttpUrl = Field(..., alias="authorizationUrl")
    token_url: HttpUrl = Field(..., alias="tokenUrl")
    scopes: Dict[str, str] = {}
    description: Optional[str] = None

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


SecurityScheme = Union[
    BasicAuthenticationSecurity,
    ApiKeySecurity,
    OAuth2ImplicitSecurity,
    OAuth2PasswordSecurity,
    OAuth2ApplicationSecurity,
    OAuth2AccessCodeSecurity,
]


# ============================================================================
# Main Swagger Model
# ============================================================================


class Swagger(BaseModelWithVendorExtensions):
    """
    Root Swagger 2.0 specification object.
    
    This is the main model representing a complete Swagger/OpenAPI 2.0 document.
    """

    swagger: Literal["2.0"]
    info: Info
    host: Optional[str] = Field(None, pattern=r"^[^{}/ :\\]+(?::\d+)?$")
    base_path: Optional[str] = Field(None, alias="basePath", pattern=r"^/")
    schemes: Optional[List[SchemeType]] = None
    consumes: Optional[List[str]] = None
    produces: Optional[List[str]] = None
    paths: Paths
    definitions: Optional[Dict[str, Schema]] = None
    parameters: Optional[Dict[str, Parameter]] = None
    responses: Optional[Dict[str, Response]] = None
    security_definitions: Optional[Dict[str, SecurityScheme]] = Field(
        None, alias="securityDefinitions"
    )
    security: Optional[List[Dict[str, List[str]]]] = None
    tags: Optional[List[Tag]] = None
    external_docs: Optional[ExternalDocs] = Field(None, alias="externalDocs")

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


# Update forward references for recursive models
Schema.model_rebuild()
PrimitivesItems.model_rebuild()

