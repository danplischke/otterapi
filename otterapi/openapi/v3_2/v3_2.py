"""OpenAPI 3.2 specification models.

This module provides Pydantic models for the OpenAPI 3.2 specification,
building on top of OpenAPI 3.1 with additional features.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Literal, Union

from pydantic import (
    AnyUrl,
    BaseModel,
    ConfigDict,
    Field,
    PositiveFloat,
    RootModel,
    StringConstraints,
    field_validator,
)


class Reference(BaseModel):
    """Reference object for OpenAPI 3.2."""

    ref: str = Field(..., alias='$ref')
    summary: str | None = None
    description: str | None = None


class Contact(BaseModel):
    """Contact information for the API."""

    model_config = ConfigDict(extra='forbid')

    name: str | None = None
    url: str | None = None
    email: str | None = None


class License(BaseModel):
    """License information for the API."""

    model_config = ConfigDict(extra='forbid')

    name: str
    identifier: str | None = None
    url: str | None = None


class ServerVariable(BaseModel):
    """Server variable for URL templating."""

    model_config = ConfigDict(extra='forbid')

    enum: list[str] | None = None
    default: str
    description: str | None = None


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
    mapping: dict[str, str] | None = None


class XML(BaseModel):
    """XML representation metadata."""

    model_config = ConfigDict(extra='forbid')

    name: str | None = None
    namespace: AnyUrl | None = None
    prefix: str | None = None
    attribute: bool | None = False
    wrapped: bool | None = False


class Example(BaseModel):
    """Example object."""

    model_config = ConfigDict(extra='forbid')

    summary: str | None = None
    description: str | None = None
    value: Any | None = None
    externalValue: str | None = None


class ParameterLocation(Enum):
    path = 'path'
    query = 'query'
    header = 'header'
    cookie = 'cookie'


class ParameterStyle(Enum):
    matrix = 'matrix'
    label = 'label'
    simple = 'simple'
    form = 'form'
    spaceDelimited = 'spaceDelimited'
    pipeDelimited = 'pipeDelimited'
    deepObject = 'deepObject'


class SecurityRequirement(RootModel[dict[str, list[str]]]):
    """Security requirement object."""

    root: dict[str, list[str]]


class ExternalDocumentation(BaseModel):
    """External documentation reference."""

    model_config = ConfigDict(extra='forbid')

    description: str | None = None
    url: str


class PathParameter(BaseModel):
    """Path parameter definition."""

    in_: Literal['path'] | None = Field(None, alias='in')
    style: ParameterStyle | None = ParameterStyle.simple
    required: Literal[True]


class QueryParameter(BaseModel):
    """Query parameter definition."""

    in_: Literal['query'] | None = Field(None, alias='in')
    style: ParameterStyle | None = ParameterStyle.form


class HeaderParameter(BaseModel):
    """Header parameter definition."""

    in_: Literal['header'] | None = Field(None, alias='in')
    style: ParameterStyle | None = ParameterStyle.simple


class CookieParameter(BaseModel):
    """Cookie parameter definition."""

    in_: Literal['cookie'] | None = Field(None, alias='in')
    style: ParameterStyle | None = ParameterStyle.form


class SecuritySchemeType(Enum):
    apiKey = 'apiKey'
    http = 'http'
    oauth2 = 'oauth2'
    openIdConnect = 'openIdConnect'


class APIKeySecurityScheme(BaseModel):
    """API Key security scheme."""

    model_config = ConfigDict(extra='forbid')

    type: SecuritySchemeType
    name: str
    in_: ParameterLocation = Field(..., alias='in')
    description: str | None = None


class HTTPSecurityScheme(BaseModel):
    """HTTP security scheme."""

    model_config = ConfigDict(extra='forbid')

    type: SecuritySchemeType
    scheme: str
    bearerFormat: str | None = None
    description: str | None = None


class OpenIdConnectSecurityScheme(BaseModel):
    """OpenID Connect security scheme."""

    model_config = ConfigDict(extra='forbid')

    type: SecuritySchemeType
    openIdConnectUrl: str
    description: str | None = None


class ImplicitOAuthFlow(BaseModel):
    """OAuth2 implicit flow."""

    model_config = ConfigDict(extra='forbid')

    authorizationUrl: str
    refreshUrl: str | None = None
    scopes: dict[str, str]


class PasswordOAuthFlow(BaseModel):
    """OAuth2 password flow."""

    model_config = ConfigDict(extra='forbid')

    tokenUrl: str
    refreshUrl: str | None = None
    scopes: dict[str, str]


class ClientCredentialsFlow(BaseModel):
    """OAuth2 client credentials flow."""

    model_config = ConfigDict(extra='forbid')

    tokenUrl: str
    refreshUrl: str | None = None
    scopes: dict[str, str]


class AuthorizationCodeOAuthFlow(BaseModel):
    """OAuth2 authorization code flow."""

    model_config = ConfigDict(extra='forbid')

    authorizationUrl: str
    tokenUrl: str
    refreshUrl: str | None = None
    scopes: dict[str, str]


class Callback(RootModel[dict[Annotated[str, StringConstraints(pattern=r'^x-')], Any]]):
    """Callback object."""

    root: dict[Annotated[str, StringConstraints(pattern=r'^x-')], Any]


class Info(BaseModel):
    """API metadata."""

    model_config = ConfigDict(extra='forbid')

    title: str
    summary: str | None = None
    description: str | None = None
    termsOfService: str | None = None
    contact: Contact | None = None
    license: License | None = None
    version: str


class Server(BaseModel):
    """Server object."""

    model_config = ConfigDict(extra='forbid')

    url: str
    description: str | None = None
    variables: dict[str, ServerVariable] | None = None


class Schema(BaseModel):
    """JSON Schema object for OpenAPI 3.2.

    OpenAPI 3.2 continues to use JSON Schema 2020-12 with OpenAPI vocabulary.
    """

    model_config = ConfigDict(extra='forbid')

    # Core JSON Schema keywords
    title: str | None = None
    multipleOf: PositiveFloat | None = None
    maximum: float | None = None
    exclusiveMaximum: float | None = None
    minimum: float | None = None
    exclusiveMinimum: float | None = None
    maxLength: Annotated[int, Field(ge=0)] | None = None
    minLength: Annotated[int, Field(ge=0)] | None = 0
    pattern: str | None = None
    maxItems: Annotated[int, Field(ge=0)] | None = None
    minItems: Annotated[int, Field(ge=0)] | None = 0
    uniqueItems: bool | None = False
    maxProperties: Annotated[int, Field(ge=0)] | None = None
    minProperties: Annotated[int, Field(ge=0)] | None = 0
    required: list[str] | None = None
    enum: list[Any] | None = None

    # Type can be an array for nullable types
    type: Type | list[Type] | None = None

    # Composition keywords
    not_: Schema | Reference | None = Field(None, alias='not')
    allOf: list[Schema | Reference] | None = None
    oneOf: list[Schema | Reference] | None = None
    anyOf: list[Schema | Reference] | None = None

    # Array keywords
    items: Schema | Reference | None = None
    prefixItems: list[Schema | Reference] | None = None
    contains: Schema | Reference | None = None  # New in 3.2

    # Object keywords
    properties: dict[str, Schema | Reference] | None = None
    additionalProperties: Schema | Reference | bool | None = True
    patternProperties: dict[str, Schema | Reference] | None = None
    propertyNames: Schema | Reference | None = None  # New in 3.2

    # String keywords
    format: str | None = None
    contentMediaType: str | None = None  # New in 3.2
    contentEncoding: str | None = None  # New in 3.2

    # Metadata
    description: str | None = None
    default: Any | None = None

    # Conditional keywords
    if_: Schema | Reference | None = Field(None, alias='if')  # New in 3.2
    then: Schema | Reference | None = None  # New in 3.2
    else_: Schema | Reference | None = Field(None, alias='else')  # New in 3.2
    dependentSchemas: dict[str, Schema | Reference] | None = None  # New in 3.2

    # OpenAPI-specific keywords
    discriminator: Discriminator | None = None
    readOnly: bool | None = False
    writeOnly: bool | None = False
    example: Any | None = None
    examples: list[Any] | None = None
    externalDocs: ExternalDocumentation | None = None
    deprecated: bool | None = False
    xml: XML | None = None


class Tag(BaseModel):
    """Tag for API operations."""

    model_config = ConfigDict(extra='forbid')

    name: str
    description: str | None = None
    externalDocs: ExternalDocumentation | None = None


class OAuthFlows(BaseModel):
    """OAuth2 flows configuration."""

    model_config = ConfigDict(extra='forbid')

    implicit: ImplicitOAuthFlow | None = None
    password: PasswordOAuthFlow | None = None
    clientCredentials: ClientCredentialsFlow | None = None
    authorizationCode: AuthorizationCodeOAuthFlow | None = None


class Link(BaseModel):
    """Link object for response links."""

    model_config = ConfigDict(extra='forbid')

    operationId: str | None = None
    operationRef: str | None = None
    parameters: dict[str, Any] | None = None
    requestBody: Any | None = None
    description: str | None = None
    server: Server | None = None


class OAuth2SecurityScheme(BaseModel):
    """OAuth2 security scheme."""

    model_config = ConfigDict(extra='forbid')

    type: SecuritySchemeType
    flows: OAuthFlows
    description: str | None = None


class SecurityScheme(
    RootModel[
        APIKeySecurityScheme
        | HTTPSecurityScheme
        | OAuth2SecurityScheme
        | OpenIdConnectSecurityScheme
    ]
):
    """Security scheme union."""

    root: (
        APIKeySecurityScheme
        | HTTPSecurityScheme
        | OAuth2SecurityScheme
        | OpenIdConnectSecurityScheme
    )


class Components(BaseModel):
    """Components object for reusable definitions."""

    model_config = ConfigDict(extra='forbid')

    schemas: (
        dict[
            Annotated[str, StringConstraints(pattern=r'^[a-zA-Z0-9\.\-_]+$')],
            Schema | Reference,
        ]
        | None
    ) = None
    responses: (
        dict[
            Annotated[str, StringConstraints(pattern=r'^[a-zA-Z0-9\.\-_]+$')],
            Reference | Response,
        ]
        | None
    ) = None
    parameters: (
        dict[
            Annotated[str, StringConstraints(pattern=r'^[a-zA-Z0-9\.\-_]+$')],
            Reference | Parameter,
        ]
        | None
    ) = None
    examples: (
        dict[
            Annotated[str, StringConstraints(pattern=r'^[a-zA-Z0-9\.\-_]+$')],
            Reference | Example,
        ]
        | None
    ) = None
    requestBodies: (
        dict[
            Annotated[str, StringConstraints(pattern=r'^[a-zA-Z0-9\.\-_]+$')],
            Reference | RequestBody,
        ]
        | None
    ) = None
    headers: (
        dict[
            Annotated[str, StringConstraints(pattern=r'^[a-zA-Z0-9\.\-_]+$')],
            Reference | Header,
        ]
        | None
    ) = None
    securitySchemes: (
        dict[
            Annotated[str, StringConstraints(pattern=r'^[a-zA-Z0-9\.\-_]+$')],
            Reference | SecurityScheme,
        ]
        | None
    ) = None
    links: (
        dict[
            Annotated[str, StringConstraints(pattern=r'^[a-zA-Z0-9\.\-_]+$')],
            Reference | Link,
        ]
        | None
    ) = None
    callbacks: (
        dict[
            Annotated[str, StringConstraints(pattern=r'^[a-zA-Z0-9\.\-_]+$')],
            Reference | Callback,
        ]
        | None
    ) = None
    pathItems: (
        dict[
            Annotated[str, StringConstraints(pattern=r'^[a-zA-Z0-9\.\-_]+$')],
            Reference | PathItem,
        ]
        | None
    ) = None


class Response(BaseModel):
    """Response object."""

    model_config = ConfigDict(extra='forbid')

    description: str
    headers: dict[str, Header | Reference] | None = None
    content: dict[str, MediaType] | None = None
    links: dict[str, Link | Reference] | None = None


class MediaType(BaseModel):
    """Media type object."""

    model_config = ConfigDict(extra='forbid')

    schema_: Schema | Reference | None = Field(None, alias='schema')
    example: Any | None = None
    examples: dict[str, Example | Reference] | None = None
    encoding: dict[str, Encoding] | None = None


class Header(BaseModel):
    """Header object."""

    model_config = ConfigDict(extra='forbid')

    description: str | None = None
    required: bool | None = False
    deprecated: bool | None = False
    allowEmptyValue: bool | None = False
    style: ParameterStyle | None = ParameterStyle.simple
    explode: bool | None = None
    allowReserved: bool | None = False
    schema_: Schema | Reference | None = Field(None, alias='schema')
    content: dict[str, MediaType] | None = None
    example: Any | None = None
    examples: dict[str, Example | Reference] | None = None

    @field_validator('style', mode='before')
    @classmethod
    def _coerce_style(cls, v: Any) -> Any:
        if v is None or isinstance(v, ParameterStyle):
            return v
        try:
            return ParameterStyle(v)
        except (ValueError, KeyError):
            return None


class Paths(RootModel[dict[str, 'PathItem']]):
    """Paths object.

    Keys should be path templates (starting with /) or extensions (starting with x-).
    """

    root: dict[str, PathItem]


class PathItem(BaseModel):
    """Path item object."""

    model_config = ConfigDict(extra='forbid')

    field_ref: str | None = Field(None, alias='$ref')
    summary: str | None = None
    description: str | None = None
    get: Operation | None = None
    put: Operation | None = None
    post: Operation | None = None
    delete: Operation | None = None
    options: Operation | None = None
    head: Operation | None = None
    patch: Operation | None = None
    trace: Operation | None = None
    servers: list[Server] | None = None
    parameters: list[Parameter | Reference] | None = None


class Operation(BaseModel):
    """Operation object."""

    model_config = ConfigDict(extra='forbid')

    tags: list[str] | None = None
    summary: str | None = None
    description: str | None = None
    externalDocs: ExternalDocumentation | None = None
    operationId: str | None = None
    parameters: list[Parameter | Reference] | None = None
    requestBody: RequestBody | Reference | None = None
    responses: Responses | None = None
    callbacks: dict[str, Callback | Reference] | None = None
    deprecated: bool | None = False
    security: list[SecurityRequirement] | None = None
    servers: list[Server] | None = None


class Responses(RootModel[dict[str, Response | Reference]]):
    pass


class Parameter(BaseModel):
    """Parameter object."""

    model_config = ConfigDict(extra='forbid')

    name: str
    in_: str = Field(..., alias='in')
    description: str | None = None
    required: bool | None = False
    deprecated: bool | None = False
    allowEmptyValue: bool | None = False
    style: str | None = None
    explode: bool | None = None
    allowReserved: bool | None = False
    schema_: Schema | Reference | None = Field(None, alias='schema')
    content: dict[str, MediaType] | None = None
    example: Any | None = None
    examples: dict[str, Example | Reference] | None = None


class RequestBody(BaseModel):
    """Request body object."""

    model_config = ConfigDict(extra='forbid')

    description: str | None = None
    content: dict[str, MediaType]
    required: bool | None = False


class Encoding(BaseModel):
    """Encoding object."""

    model_config = ConfigDict(extra='forbid')

    contentType: str | None = None
    headers: dict[str, Header | Reference] | None = None
    style: ParameterStyle | None = None
    explode: bool | None = None
    allowReserved: bool | None = False


class Webhook(RootModel[dict[str, Union['PathItem', Reference]]]):
    """Webhook object."""

    root: dict[str, PathItem | Reference]


class OpenAPI(BaseModel):
    """OpenAPI 3.2 root document."""

    model_config = ConfigDict(extra='forbid')

    openapi: Annotated[
        str, StringConstraints(pattern=r'^3\.2\.\d+(-.+)?$')
    ]  # Updated for 3.2.x
    info: Info
    jsonSchemaDialect: str | None = (
        'https://spec.openapis.org/oas/3.2/dialect/base'  # Updated for 3.2
    )
    servers: list[Server] | None = None
    paths: Paths | None = None
    webhooks: dict[str, PathItem | Reference] | None = None
    components: Components | None = None
    security: list[SecurityRequirement] | None = None
    tags: list[Tag] | None = None
    externalDocs: ExternalDocumentation | None = None


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
Webhook.model_rebuild()
