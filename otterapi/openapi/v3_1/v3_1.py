"""OpenAPI 3.1 specification models.

This module provides Pydantic models for the OpenAPI 3.1 specification,
which is aligned with JSON Schema 2020-12.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import TYPE_CHECKING, Annotated, Any, Union

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
    summary: str | None = None
    description: str | None = None


class Contact(BaseModel):
    """Contact information for the API."""

    model_config = ConfigDict(extra='forbid')

    name: str | None = None
    url: str | None = None
    email: str | None = (
        None  # Using str instead of EmailStr to avoid optional dependency
    )


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


class Style(Enum):
    """Parameter style for simple parameters."""

    simple = 'simple'


class SecurityRequirement(RootModel[dict[str, list[str]]]):
    """Security requirement object."""

    root: dict[str, list[str]]


class ExternalDocumentation(BaseModel):
    """External documentation reference."""

    model_config = ConfigDict(extra='forbid')

    description: str | None = None
    url: str


class ExampleXORExamples(RootModel[Any]):
    """Ensures example and examples are mutually exclusive."""

    root: Any = Field(..., description='Example and examples are mutually exclusive')


class SchemaXORContent1(BaseModel):
    """Helper for schema/content mutual exclusion."""

    pass


class SchemaXORContent(RootModel[Any | SchemaXORContent1]):
    """Ensures schema and content are mutually exclusive."""

    root: Any | SchemaXORContent1 = Field(
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

    in_: In | None = Field(None, alias='in')
    style: Style1 | None = 'simple'
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

    in_: In1 | None = Field(None, alias='in')
    style: Style2 | None = 'form'


class In2(Enum):
    """Header parameter location."""

    header = 'header'


class Style3(Enum):
    """Header parameter style."""

    simple = 'simple'


class HeaderParameter(BaseModel):
    """Header parameter definition."""

    in_: In2 | None = Field(None, alias='in')
    style: Style3 | None = 'simple'


class In3(Enum):
    """Cookie parameter location."""

    cookie = 'cookie'


class Style4(Enum):
    """Cookie parameter style."""

    form = 'form'


class CookieParameter(BaseModel):
    """Cookie parameter definition."""

    in_: In3 | None = Field(None, alias='in')
    style: Style4 | None = 'form'


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
    description: str | None = None


class Type2(Enum):
    """HTTP security type."""

    http = 'http'


class HTTPSecurityScheme1(BaseModel):
    """HTTP Bearer security scheme."""

    model_config = ConfigDict(extra='forbid')

    scheme: Annotated[str, Field(pattern=r'^[Bb][Ee][Aa][Rr][Ee][Rr]$')]
    bearerFormat: str | None = None
    description: str | None = None
    type: Type2


class HTTPSecurityScheme2(BaseModel):
    """HTTP non-Bearer security scheme."""

    model_config = ConfigDict(extra='forbid')

    scheme: str
    bearerFormat: str | None = None
    description: str | None = None
    type: Type2


class HTTPSecurityScheme(RootModel[HTTPSecurityScheme1 | HTTPSecurityScheme2]):
    """HTTP security scheme union."""

    root: HTTPSecurityScheme1 | HTTPSecurityScheme2


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


class Callback(RootModel[dict[Annotated[str, Field(pattern=r'^x-')], Any]]):
    """Callback object."""

    root: dict[Annotated[str, Field(pattern=r'^x-')], Any]


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
    """JSON Schema object for OpenAPI 3.1.

    OpenAPI 3.1 uses JSON Schema 2020-12 with some modifications.
    """

    model_config = ConfigDict(extra='forbid')

    # Core JSON Schema keywords
    title: str | None = None
    multipleOf: Annotated[float, Field(gt=0)] | None = None
    maximum: float | None = None
    exclusiveMaximum: float | None = None  # Changed from boolean in 3.0
    minimum: float | None = None
    exclusiveMinimum: float | None = None  # Changed from boolean in 3.0
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

    # Type can now be an array for nullable types
    type: Type | list[Type] | None = None

    # Composition keywords
    not_: Schema | Reference | None = Field(None, alias='not')
    allOf: list[Schema | Reference] | None = None
    oneOf: list[Schema | Reference] | None = None
    anyOf: list[Schema | Reference] | None = None

    # Array keywords
    items: Schema | Reference | None = None
    prefixItems: list[Schema | Reference] | None = None  # New in 3.1

    # Object keywords
    properties: dict[str, Schema | Reference] | None = None
    additionalProperties: Schema | Reference | bool | None = True
    patternProperties: dict[str, Schema | Reference] | None = None  # New in 3.1

    # String keywords
    format: str | None = None

    # Metadata
    description: str | None = None
    default: Any | None = None

    # OpenAPI-specific keywords
    discriminator: Discriminator | None = None
    readOnly: bool | None = False
    writeOnly: bool | None = False
    example: Any | None = None
    examples: list[Any] | None = None  # New in 3.1
    externalDocs: ExternalDocumentation | None = None
    deprecated: bool | None = False
    xml: XML | None = None

    # Note: nullable is removed in 3.1, use type arrays instead


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

    type: Type4
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
        dict[Annotated[str, Field(pattern=r'^[a-zA-Z0-9\.\-_]+$')], Schema | Reference]
        | None
    ) = None
    responses: (
        dict[
            Annotated[str, Field(pattern=r'^[a-zA-Z0-9\.\-_]+$')], Reference | Response
        ]
        | None
    ) = None
    parameters: (
        dict[
            Annotated[str, Field(pattern=r'^[a-zA-Z0-9\.\-_]+$')], Reference | Parameter
        ]
        | None
    ) = None
    examples: (
        dict[Annotated[str, Field(pattern=r'^[a-zA-Z0-9\.\-_]+$')], Reference | Example]
        | None
    ) = None
    requestBodies: (
        dict[
            Annotated[str, Field(pattern=r'^[a-zA-Z0-9\.\-_]+$')],
            Reference | RequestBody,
        ]
        | None
    ) = None
    headers: (
        dict[Annotated[str, Field(pattern=r'^[a-zA-Z0-9\.\-_]+$')], Reference | Header]
        | None
    ) = None
    securitySchemes: (
        dict[
            Annotated[str, Field(pattern=r'^[a-zA-Z0-9\.\-_]+$')],
            Reference | SecurityScheme,
        ]
        | None
    ) = None
    links: (
        dict[Annotated[str, Field(pattern=r'^[a-zA-Z0-9\.\-_]+$')], Reference | Link]
        | None
    ) = None
    callbacks: (
        dict[
            Annotated[str, Field(pattern=r'^[a-zA-Z0-9\.\-_]+$')], Reference | Callback
        ]
        | None
    ) = None
    pathItems: (
        dict[
            Annotated[str, Field(pattern=r'^[a-zA-Z0-9\.\-_]+$')], Reference | PathItem
        ]
        | None
    ) = None  # New in 3.1


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
    style: Style | None = 'simple'
    explode: bool | None = None
    allowReserved: bool | None = False
    schema_: Schema | Reference | None = Field(None, alias='schema')
    content: dict[str, MediaType] | None = None
    example: Any | None = None
    examples: dict[str, Example | Reference] | None = None


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
    responses: Responses | None = None  # Made optional in 3.1
    callbacks: dict[str, Callback | Reference] | None = None
    deprecated: bool | None = False
    security: list[SecurityRequirement] | None = None
    servers: list[Server] | None = None


class Responses(RootModel[dict[str, Response | Reference]]):
    """Responses object containing response definitions by HTTP status code.

    Keys are HTTP status codes (200, 400, etc.) or 'default'.
    """

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
    style: Style5 | None = None
    explode: bool | None = None
    allowReserved: bool | None = False


class Webhook(RootModel[dict[str, Union['PathItem', Reference]]]):
    """Webhook object (new in OpenAPI 3.1)."""

    root: dict[str, PathItem | Reference]


class OpenAPI(BaseModel):
    """OpenAPI 3.1 root document."""

    model_config = ConfigDict(extra='forbid')

    openapi: Annotated[str, Field(pattern=r'^3\.1\.\d+(-.+)?$')]  # Updated for 3.1.x
    info: Info
    jsonSchemaDialect: str | None = (
        'https://spec.openapis.org/oas/3.1/dialect/base'  # New in 3.1
    )
    servers: list[Server] | None = None
    paths: Paths | None = None  # Made optional in 3.1
    webhooks: dict[str, PathItem | Reference] | None = None  # New in 3.1
    components: Components | None = None
    security: list[SecurityRequirement] | None = None
    tags: list[Tag] | None = None
    externalDocs: ExternalDocumentation | None = None

    def upgrade(self) -> tuple[v3_2.OpenAPI, list[str]]:
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
        if (
            data.get('jsonSchemaDialect')
            == 'https://spec.openapis.org/oas/3.1/dialect/base'
        ):
            data['jsonSchemaDialect'] = 'https://spec.openapis.org/oas/3.2/dialect/base'

        # Parse and validate with v3_2.OpenAPI model
        return v3_2.OpenAPI.model_validate(data), []


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
