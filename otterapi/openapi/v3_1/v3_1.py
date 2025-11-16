"""OpenAPI 3.1 specification models.

This module provides Pydantic models for the OpenAPI 3.1 specification,
which is aligned with JSON Schema 2020-12.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import TYPE_CHECKING, Annotated, Any, Dict, List, Optional, Union

from pydantic import (
    AnyUrl,
    BaseModel,
    ConfigDict,
    Field,
    RootModel,
)

if TYPE_CHECKING:
    from ..v3_2 import v3_2


class Reference(BaseModel):
    """Reference object for OpenAPI 3.1."""
    ref: str = Field(..., alias='$ref')
    summary: Optional[str] = None
    description: Optional[str] = None


class Contact(BaseModel):
    """Contact information for the API."""
    model_config = ConfigDict(extra='forbid')

    name: Optional[str] = None
    url: Optional[str] = None
    email: Optional[str] = None  # Using str instead of EmailStr to avoid optional dependency


class License(BaseModel):
    """License information for the API."""
    model_config = ConfigDict(extra='forbid')

    name: str
    identifier: Optional[str] = None
    url: Optional[str] = None


class ServerVariable(BaseModel):
    """Server variable for URL templating."""
    model_config = ConfigDict(extra='forbid')

    enum: Optional[List[str]] = None
    default: str
    description: Optional[str] = None


class Type(Enum):
    """JSON Schema types."""
    array = 'array'
    boolean = 'boolean'
    integer = 'integer'
    number = 'number'
    object = 'object'
    string = 'string'
    null = 'null'


class Discriminator(BaseModel):
    """Discriminator for polymorphism."""
    propertyName: str
    mapping: Optional[Dict[str, str]] = None


class XML(BaseModel):
    """XML representation metadata."""
    model_config = ConfigDict(extra='forbid')

    name: Optional[str] = None
    namespace: Optional[AnyUrl] = None
    prefix: Optional[str] = None
    attribute: Optional[bool] = False
    wrapped: Optional[bool] = False


class Example(BaseModel):
    """Example object."""
    model_config = ConfigDict(extra='forbid')

    summary: Optional[str] = None
    description: Optional[str] = None
    value: Optional[Any] = None
    externalValue: Optional[str] = None


class Style(Enum):
    """Parameter style for simple parameters."""
    simple = 'simple'


class SecurityRequirement(RootModel[Dict[str, List[str]]]):
    """Security requirement object."""
    root: Dict[str, List[str]]


class ExternalDocumentation(BaseModel):
    """External documentation reference."""
    model_config = ConfigDict(extra='forbid')

    description: Optional[str] = None
    url: str


class ExampleXORExamples(RootModel[Any]):
    """Ensures example and examples are mutually exclusive."""
    root: Any = Field(
        ..., description='Example and examples are mutually exclusive'
    )


class SchemaXORContent1(BaseModel):
    """Helper for schema/content mutual exclusion."""
    pass


class SchemaXORContent(RootModel[Union[Any, SchemaXORContent1]]):
    """Ensures schema and content are mutually exclusive."""
    root: Union[Any, SchemaXORContent1] = Field(
        ...,
        description='Schema and content are mutually exclusive, at least one is required',
    )


class In(Enum):
    """Path parameter location."""
    path = 'path'


class Style1(Enum):
    """Path parameter styles."""
    matrix = 'matrix'
    label = 'label'
    simple = 'simple'


class Required(Enum):
    """Required enum for path parameters."""
    bool_True = True


class PathParameter(BaseModel):
    """Path parameter definition."""
    in_: Optional[In] = Field(None, alias='in')
    style: Optional[Style1] = 'simple'
    required: Required


class In1(Enum):
    """Query parameter location."""
    query = 'query'


class Style2(Enum):
    """Query parameter styles."""
    form = 'form'
    spaceDelimited = 'spaceDelimited'
    pipeDelimited = 'pipeDelimited'
    deepObject = 'deepObject'


class QueryParameter(BaseModel):
    """Query parameter definition."""
    in_: Optional[In1] = Field(None, alias='in')
    style: Optional[Style2] = 'form'


class In2(Enum):
    """Header parameter location."""
    header = 'header'


class Style3(Enum):
    """Header parameter style."""
    simple = 'simple'


class HeaderParameter(BaseModel):
    """Header parameter definition."""
    in_: Optional[In2] = Field(None, alias='in')
    style: Optional[Style3] = 'simple'


class In3(Enum):
    """Cookie parameter location."""
    cookie = 'cookie'


class Style4(Enum):
    """Cookie parameter style."""
    form = 'form'


class CookieParameter(BaseModel):
    """Cookie parameter definition."""
    in_: Optional[In3] = Field(None, alias='in')
    style: Optional[Style4] = 'form'


class Type1(Enum):
    """API Key security type."""
    apiKey = 'apiKey'


class In4(Enum):
    """API Key location."""
    header = 'header'
    query = 'query'
    cookie = 'cookie'


class APIKeySecurityScheme(BaseModel):
    """API Key security scheme."""
    model_config = ConfigDict(extra='forbid')

    type: Type1
    name: str
    in_: In4 = Field(..., alias='in')
    description: Optional[str] = None


class Type2(Enum):
    """HTTP security type."""
    http = 'http'


class HTTPSecurityScheme1(BaseModel):
    """HTTP Bearer security scheme."""
    model_config = ConfigDict(extra='forbid')

    scheme: Annotated[str, Field(pattern=r'^[Bb][Ee][Aa][Rr][Ee][Rr]$')]
    bearerFormat: Optional[str] = None
    description: Optional[str] = None
    type: Type2


class HTTPSecurityScheme2(BaseModel):
    """HTTP non-Bearer security scheme."""
    model_config = ConfigDict(extra='forbid')

    scheme: str
    bearerFormat: Optional[str] = None
    description: Optional[str] = None
    type: Type2


class HTTPSecurityScheme(RootModel[Union[HTTPSecurityScheme1, HTTPSecurityScheme2]]):
    """HTTP security scheme union."""
    root: Union[HTTPSecurityScheme1, HTTPSecurityScheme2]


class Type4(Enum):
    """OAuth2 security type."""
    oauth2 = 'oauth2'


class Type5(Enum):
    """OpenID Connect security type."""
    openIdConnect = 'openIdConnect'


class OpenIdConnectSecurityScheme(BaseModel):
    """OpenID Connect security scheme."""
    model_config = ConfigDict(extra='forbid')

    type: Type5
    openIdConnectUrl: str
    description: Optional[str] = None


class ImplicitOAuthFlow(BaseModel):
    """OAuth2 implicit flow."""
    model_config = ConfigDict(extra='forbid')

    authorizationUrl: str
    refreshUrl: Optional[str] = None
    scopes: Dict[str, str]


class PasswordOAuthFlow(BaseModel):
    """OAuth2 password flow."""
    model_config = ConfigDict(extra='forbid')

    tokenUrl: str
    refreshUrl: Optional[str] = None
    scopes: Dict[str, str]


class ClientCredentialsFlow(BaseModel):
    """OAuth2 client credentials flow."""
    model_config = ConfigDict(extra='forbid')

    tokenUrl: str
    refreshUrl: Optional[str] = None
    scopes: Dict[str, str]


class AuthorizationCodeOAuthFlow(BaseModel):
    """OAuth2 authorization code flow."""
    model_config = ConfigDict(extra='forbid')

    authorizationUrl: str
    tokenUrl: str
    refreshUrl: Optional[str] = None
    scopes: Dict[str, str]


class Callback(RootModel[Dict[Annotated[str, Field(pattern=r'^x-')], Any]]):
    """Callback object."""
    root: Dict[Annotated[str, Field(pattern=r'^x-')], Any]


class Style5(Enum):
    """Encoding styles."""
    form = 'form'
    spaceDelimited = 'spaceDelimited'
    pipeDelimited = 'pipeDelimited'
    deepObject = 'deepObject'


class Info(BaseModel):
    """API metadata."""
    model_config = ConfigDict(extra='forbid')

    title: str
    summary: Optional[str] = None
    description: Optional[str] = None
    termsOfService: Optional[str] = None
    contact: Optional[Contact] = None
    license: Optional[License] = None
    version: str


class Server(BaseModel):
    """Server object."""
    model_config = ConfigDict(extra='forbid')

    url: str
    description: Optional[str] = None
    variables: Optional[Dict[str, ServerVariable]] = None


class Schema(BaseModel):
    """JSON Schema object for OpenAPI 3.1.
    
    OpenAPI 3.1 uses JSON Schema 2020-12 with some modifications.
    """
    model_config = ConfigDict(extra='forbid')

    # Core JSON Schema keywords
    title: Optional[str] = None
    multipleOf: Optional[Annotated[float, Field(gt=0)]] = None
    maximum: Optional[float] = None
    exclusiveMaximum: Optional[float] = None  # Changed from boolean in 3.0
    minimum: Optional[float] = None
    exclusiveMinimum: Optional[float] = None  # Changed from boolean in 3.0
    maxLength: Optional[Annotated[int, Field(ge=0)]] = None
    minLength: Optional[Annotated[int, Field(ge=0)]] = 0
    pattern: Optional[str] = None
    maxItems: Optional[Annotated[int, Field(ge=0)]] = None
    minItems: Optional[Annotated[int, Field(ge=0)]] = 0
    uniqueItems: Optional[bool] = False
    maxProperties: Optional[Annotated[int, Field(ge=0)]] = None
    minProperties: Optional[Annotated[int, Field(ge=0)]] = 0
    required: Optional[List[str]] = None
    enum: Optional[List[Any]] = None
    
    # Type can now be an array for nullable types
    type: Optional[Union[Type, List[Type]]] = None
    
    # Composition keywords
    not_: Optional[Union[Schema, Reference]] = Field(None, alias='not')
    allOf: Optional[List[Union[Schema, Reference]]] = None
    oneOf: Optional[List[Union[Schema, Reference]]] = None
    anyOf: Optional[List[Union[Schema, Reference]]] = None
    
    # Array keywords
    items: Optional[Union[Schema, Reference]] = None
    prefixItems: Optional[List[Union[Schema, Reference]]] = None  # New in 3.1
    
    # Object keywords
    properties: Optional[Dict[str, Union[Schema, Reference]]] = None
    additionalProperties: Optional[Union[Schema, Reference, bool]] = True
    patternProperties: Optional[Dict[str, Union[Schema, Reference]]] = None  # New in 3.1
    
    # String keywords
    format: Optional[str] = None
    
    # Metadata
    description: Optional[str] = None
    default: Optional[Any] = None
    
    # OpenAPI-specific keywords
    discriminator: Optional[Discriminator] = None
    readOnly: Optional[bool] = False
    writeOnly: Optional[bool] = False
    example: Optional[Any] = None
    examples: Optional[List[Any]] = None  # New in 3.1
    externalDocs: Optional[ExternalDocumentation] = None
    deprecated: Optional[bool] = False
    xml: Optional[XML] = None
    
    # Note: nullable is removed in 3.1, use type arrays instead


class Tag(BaseModel):
    """Tag for API operations."""
    model_config = ConfigDict(extra='forbid')

    name: str
    description: Optional[str] = None
    externalDocs: Optional[ExternalDocumentation] = None


class OAuthFlows(BaseModel):
    """OAuth2 flows configuration."""
    model_config = ConfigDict(extra='forbid')

    implicit: Optional[ImplicitOAuthFlow] = None
    password: Optional[PasswordOAuthFlow] = None
    clientCredentials: Optional[ClientCredentialsFlow] = None
    authorizationCode: Optional[AuthorizationCodeOAuthFlow] = None


class Link(BaseModel):
    """Link object for response links."""
    model_config = ConfigDict(extra='forbid')

    operationId: Optional[str] = None
    operationRef: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    requestBody: Optional[Any] = None
    description: Optional[str] = None
    server: Optional[Server] = None


class OAuth2SecurityScheme(BaseModel):
    """OAuth2 security scheme."""
    model_config = ConfigDict(extra='forbid')

    type: Type4
    flows: OAuthFlows
    description: Optional[str] = None


class SecurityScheme(RootModel[Union[
    APIKeySecurityScheme,
    HTTPSecurityScheme,
    OAuth2SecurityScheme,
    OpenIdConnectSecurityScheme,
]]):
    """Security scheme union."""
    root: Union[
        APIKeySecurityScheme,
        HTTPSecurityScheme,
        OAuth2SecurityScheme,
        OpenIdConnectSecurityScheme,
    ]


class Components(BaseModel):
    """Components object for reusable definitions."""
    model_config = ConfigDict(extra='forbid')

    schemas: Optional[
        Dict[Annotated[str, Field(pattern=r'^[a-zA-Z0-9\.\-_]+$')], Union[Schema, Reference]]
    ] = None
    responses: Optional[
        Dict[Annotated[str, Field(pattern=r'^[a-zA-Z0-9\.\-_]+$')], Union[Reference, Response]]
    ] = None
    parameters: Optional[
        Dict[Annotated[str, Field(pattern=r'^[a-zA-Z0-9\.\-_]+$')], Union[Reference, Parameter]]
    ] = None
    examples: Optional[
        Dict[Annotated[str, Field(pattern=r'^[a-zA-Z0-9\.\-_]+$')], Union[Reference, Example]]
    ] = None
    requestBodies: Optional[
        Dict[Annotated[str, Field(pattern=r'^[a-zA-Z0-9\.\-_]+$')], Union[Reference, RequestBody]]
    ] = None
    headers: Optional[
        Dict[Annotated[str, Field(pattern=r'^[a-zA-Z0-9\.\-_]+$')], Union[Reference, Header]]
    ] = None
    securitySchemes: Optional[
        Dict[Annotated[str, Field(pattern=r'^[a-zA-Z0-9\.\-_]+$')], Union[Reference, SecurityScheme]]
    ] = None
    links: Optional[
        Dict[Annotated[str, Field(pattern=r'^[a-zA-Z0-9\.\-_]+$')], Union[Reference, Link]]
    ] = None
    callbacks: Optional[
        Dict[Annotated[str, Field(pattern=r'^[a-zA-Z0-9\.\-_]+$')], Union[Reference, Callback]]
    ] = None
    pathItems: Optional[
        Dict[Annotated[str, Field(pattern=r'^[a-zA-Z0-9\.\-_]+$')], Union[Reference, PathItem]]
    ] = None  # New in 3.1


class Response(BaseModel):
    """Response object."""
    model_config = ConfigDict(extra='forbid')

    description: str
    headers: Optional[Dict[str, Union[Header, Reference]]] = None
    content: Optional[Dict[str, MediaType]] = None
    links: Optional[Dict[str, Union[Link, Reference]]] = None


class MediaType(BaseModel):
    """Media type object."""
    model_config = ConfigDict(extra='forbid')

    schema_: Optional[Union[Schema, Reference]] = Field(None, alias='schema')
    example: Optional[Any] = None
    examples: Optional[Dict[str, Union[Example, Reference]]] = None
    encoding: Optional[Dict[str, Encoding]] = None


class Header(BaseModel):
    """Header object."""
    model_config = ConfigDict(extra='forbid')

    description: Optional[str] = None
    required: Optional[bool] = False
    deprecated: Optional[bool] = False
    allowEmptyValue: Optional[bool] = False
    style: Optional[Style] = 'simple'
    explode: Optional[bool] = None
    allowReserved: Optional[bool] = False
    schema_: Optional[Union[Schema, Reference]] = Field(None, alias='schema')
    content: Optional[Dict[str, MediaType]] = None
    example: Optional[Any] = None
    examples: Optional[Dict[str, Union[Example, Reference]]] = None


class Paths(RootModel[Dict[str, Union['PathItem', Any]]]):
    """Paths object."""
    root: Dict[str, Union['PathItem', Any]]


class PathItem(BaseModel):
    """Path item object."""
    model_config = ConfigDict(extra='forbid')

    field_ref: Optional[str] = Field(None, alias='$ref')
    summary: Optional[str] = None
    description: Optional[str] = None
    get: Optional[Operation] = None
    put: Optional[Operation] = None
    post: Optional[Operation] = None
    delete: Optional[Operation] = None
    options: Optional[Operation] = None
    head: Optional[Operation] = None
    patch: Optional[Operation] = None
    trace: Optional[Operation] = None
    servers: Optional[List[Server]] = None
    parameters: Optional[List[Union[Parameter, Reference]]] = None


class Operation(BaseModel):
    """Operation object."""
    model_config = ConfigDict(extra='forbid')

    tags: Optional[List[str]] = None
    summary: Optional[str] = None
    description: Optional[str] = None
    externalDocs: Optional[ExternalDocumentation] = None
    operationId: Optional[str] = None
    parameters: Optional[List[Union[Parameter, Reference]]] = None
    requestBody: Optional[Union[RequestBody, Reference]] = None
    responses: Optional[Responses] = None  # Made optional in 3.1
    callbacks: Optional[Dict[str, Union[Callback, Reference]]] = None
    deprecated: Optional[bool] = False
    security: Optional[List[SecurityRequirement]] = None
    servers: Optional[List[Server]] = None


class Responses(BaseModel):
    """Responses object."""
    model_config = ConfigDict(extra='forbid')

    default: Optional[Union[Response, Reference]] = None


class Parameter(BaseModel):
    """Parameter object."""
    model_config = ConfigDict(extra='forbid')

    name: str
    in_: str = Field(..., alias='in')
    description: Optional[str] = None
    required: Optional[bool] = False
    deprecated: Optional[bool] = False
    allowEmptyValue: Optional[bool] = False
    style: Optional[str] = None
    explode: Optional[bool] = None
    allowReserved: Optional[bool] = False
    schema_: Optional[Union[Schema, Reference]] = Field(None, alias='schema')
    content: Optional[Dict[str, MediaType]] = None
    example: Optional[Any] = None
    examples: Optional[Dict[str, Union[Example, Reference]]] = None


class RequestBody(BaseModel):
    """Request body object."""
    model_config = ConfigDict(extra='forbid')

    description: Optional[str] = None
    content: Dict[str, MediaType]
    required: Optional[bool] = False


class Encoding(BaseModel):
    """Encoding object."""
    model_config = ConfigDict(extra='forbid')

    contentType: Optional[str] = None
    headers: Optional[Dict[str, Union[Header, Reference]]] = None
    style: Optional[Style5] = None
    explode: Optional[bool] = None
    allowReserved: Optional[bool] = False


class Webhook(RootModel[Dict[str, Union['PathItem', Reference]]]):
    """Webhook object (new in OpenAPI 3.1)."""
    root: Dict[str, Union['PathItem', Reference]]


class OpenAPI(BaseModel):
    """OpenAPI 3.1 root document."""
    model_config = ConfigDict(extra='forbid')

    openapi: Annotated[str, Field(pattern=r'^3\.1\.\d+(-.+)?$')]  # Updated for 3.1.x
    info: Info
    jsonSchemaDialect: Optional[str] = 'https://spec.openapis.org/oas/3.1/dialect/base'  # New in 3.1
    servers: Optional[List[Server]] = None
    paths: Optional[Paths] = None  # Made optional in 3.1
    webhooks: Optional[Dict[str, Union[PathItem, Reference]]] = None  # New in 3.1
    components: Optional[Components] = None
    security: Optional[List[SecurityRequirement]] = None
    tags: Optional[List[Tag]] = None
    externalDocs: Optional[ExternalDocumentation] = None

    def upgrade_to_v3_2(self) -> 'v3_2.OpenAPI':
        """Upgrade this OpenAPI 3.1 document to OpenAPI 3.2.
        
        Converts the current OpenAPI 3.1 specification to OpenAPI 3.2 format.
        The conversion includes:
        - Updating the openapi version string from 3.1.x to 3.2.0
        - Updating the jsonSchemaDialect to the 3.2 dialect
        - All other fields are compatible and transferred as-is
        
        Returns:
            v3_2.OpenAPI: The upgraded OpenAPI 3.2 document
        """
        from ..v3_2 import v3_2
        
        # Get the model as a dictionary with aliases (e.g., 'schema' instead of 'schema_')
        data = self.model_dump(mode='json', by_alias=True, exclude_none=False)
        
        # Update the openapi version from 3.1.x to 3.2.0
        version_match = re.match(r'^3\.1\.(\d+)(-.+)?$', data['openapi'])
        if version_match:
            # Keep the patch version but change major.minor to 3.2
            patch = version_match.group(1)
            suffix = version_match.group(2) or ''
            data['openapi'] = f'3.2.{patch}{suffix}'
        else:
            # Fallback to 3.2.0 if pattern doesn't match
            data['openapi'] = '3.2.0'
        
        # Update jsonSchemaDialect if it's the 3.1 default
        if data.get('jsonSchemaDialect') == 'https://spec.openapis.org/oas/3.1/dialect/base':
            data['jsonSchemaDialect'] = 'https://spec.openapis.org/oas/3.2/dialect/base'
        
        # Parse and validate with v3_2.OpenAPI model
        return v3_2.OpenAPI.model_validate(data)


# Rebuild models to resolve forward references
Schema.model_rebuild()
OpenAPI.model_rebuild()
Components.model_rebuild()
Response.model_rebuild()
MediaType.model_rebuild()
Paths.model_rebuild()
PathItem.model_rebuild()
Operation.model_rebuild()
Parameter.model_rebuild()
Header.model_rebuild()

