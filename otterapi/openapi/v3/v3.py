from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import (
    AnyUrl,
    BaseModel,
    EmailStr,
    Extra,
    Field,
    PositiveFloat,
    conint,
    constr,
)


class Reference(BaseModel):
    __root__: Dict[constr(regex=r'^\$ref$'), str]


class Contact(BaseModel):
    class Config:
        extra = Extra.forbid

    name: Optional[str] = None
    url: Optional[str] = None
    email: Optional[EmailStr] = None


class License(BaseModel):
    class Config:
        extra = Extra.forbid

    name: str
    url: Optional[str] = None


class ServerVariable(BaseModel):
    class Config:
        extra = Extra.forbid

    enum: Optional[List[str]] = None
    default: str
    description: Optional[str] = None


class Type(Enum):
    array = 'array'
    boolean = 'boolean'
    integer = 'integer'
    number = 'number'
    object = 'object'
    string = 'string'


class Discriminator(BaseModel):
    propertyName: str
    mapping: Optional[Dict[str, str]] = None


class XML(BaseModel):
    class Config:
        extra = Extra.forbid

    name: Optional[str] = None
    namespace: Optional[AnyUrl] = None
    prefix: Optional[str] = None
    attribute: Optional[bool] = False
    wrapped: Optional[bool] = False


class Example(BaseModel):
    class Config:
        extra = Extra.forbid

    summary: Optional[str] = None
    description: Optional[str] = None
    value: Optional[Any] = None
    externalValue: Optional[str] = None


class Style(Enum):
    simple = 'simple'


class SecurityRequirement(BaseModel):
    __root__: Dict[str, List[str]]


class ExternalDocumentation(BaseModel):
    class Config:
        extra = Extra.forbid

    description: Optional[str] = None
    url: str


class ExampleXORExamples(BaseModel):
    __root__: Any = Field(
        ..., description='Example and examples are mutually exclusive'
    )


class SchemaXORContent1(BaseModel):
    pass


class SchemaXORContent(BaseModel):
    __root__: Union[Any, SchemaXORContent1] = Field(
        ...,
        description='Schema and content are mutually exclusive, at least one is required',
    )


class In(Enum):
    path = 'path'


class Style1(Enum):
    matrix = 'matrix'
    label = 'label'
    simple = 'simple'


class Required(Enum):
    bool_True = True


class PathParameter(BaseModel):
    in_: Optional[In] = Field(None, alias='in')
    style: Optional[Style1] = 'simple'
    required: Required


class In1(Enum):
    query = 'query'


class Style2(Enum):
    form = 'form'
    spaceDelimited = 'spaceDelimited'
    pipeDelimited = 'pipeDelimited'
    deepObject = 'deepObject'


class QueryParameter(BaseModel):
    in_: Optional[In1] = Field(None, alias='in')
    style: Optional[Style2] = 'form'


class In2(Enum):
    header = 'header'


class Style3(Enum):
    simple = 'simple'


class HeaderParameter(BaseModel):
    in_: Optional[In2] = Field(None, alias='in')
    style: Optional[Style3] = 'simple'


class In3(Enum):
    cookie = 'cookie'


class Style4(Enum):
    form = 'form'


class CookieParameter(BaseModel):
    in_: Optional[In3] = Field(None, alias='in')
    style: Optional[Style4] = 'form'


class Type1(Enum):
    apiKey = 'apiKey'


class In4(Enum):
    header = 'header'
    query = 'query'
    cookie = 'cookie'


class APIKeySecurityScheme(BaseModel):
    class Config:
        extra = Extra.forbid

    type: Type1
    name: str
    in_: In4 = Field(..., alias='in')
    description: Optional[str] = None


class Type2(Enum):
    http = 'http'


class HTTPSecurityScheme1(BaseModel):
    class Config:
        extra = Extra.forbid

    scheme: constr(regex=r'^[Bb][Ee][Aa][Rr][Ee][Rr]$')
    bearerFormat: Optional[str] = None
    description: Optional[str] = None
    type: Type2


class HTTPSecurityScheme2(BaseModel):
    class Config:
        extra = Extra.forbid

    scheme: str
    bearerFormat: Optional[str] = None
    description: Optional[str] = None
    type: Type2


class HTTPSecurityScheme(BaseModel):
    class Config:
        extra = Extra.forbid

    __root__: Union[HTTPSecurityScheme1, HTTPSecurityScheme2]


class Type4(Enum):
    oauth2 = 'oauth2'


class Type5(Enum):
    openIdConnect = 'openIdConnect'


class OpenIdConnectSecurityScheme(BaseModel):
    class Config:
        extra = Extra.forbid

    type: Type5
    openIdConnectUrl: str
    description: Optional[str] = None


class ImplicitOAuthFlow(BaseModel):
    class Config:
        extra = Extra.forbid

    authorizationUrl: str
    refreshUrl: Optional[str] = None
    scopes: Dict[str, str]


class PasswordOAuthFlow(BaseModel):
    class Config:
        extra = Extra.forbid

    tokenUrl: str
    refreshUrl: Optional[str] = None
    scopes: Dict[str, str]


class ClientCredentialsFlow(BaseModel):
    class Config:
        extra = Extra.forbid

    tokenUrl: str
    refreshUrl: Optional[str] = None
    scopes: Dict[str, str]


class AuthorizationCodeOAuthFlow(BaseModel):
    class Config:
        extra = Extra.forbid

    authorizationUrl: str
    tokenUrl: str
    refreshUrl: Optional[str] = None
    scopes: Dict[str, str]


class Callback(BaseModel):
    __root__: Dict[constr(regex=r'^x-'), Any]


class Style5(Enum):
    form = 'form'
    spaceDelimited = 'spaceDelimited'
    pipeDelimited = 'pipeDelimited'
    deepObject = 'deepObject'


class Info(BaseModel):
    class Config:
        extra = Extra.forbid

    title: str
    description: Optional[str] = None
    termsOfService: Optional[str] = None
    contact: Optional[Contact] = None
    license: Optional[License] = None
    version: str


class Server(BaseModel):
    class Config:
        extra = Extra.forbid

    url: str
    description: Optional[str] = None
    variables: Optional[Dict[str, ServerVariable]] = None


class Schema(BaseModel):
    class Config:
        extra = Extra.forbid

    title: Optional[str] = None
    multipleOf: Optional[PositiveFloat] = None
    maximum: Optional[float] = None
    exclusiveMaximum: Optional[bool] = False
    minimum: Optional[float] = None
    exclusiveMinimum: Optional[bool] = False
    maxLength: Optional[conint(ge=0)] = None
    minLength: Optional[conint(ge=0)] = 0
    pattern: Optional[str] = None
    maxItems: Optional[conint(ge=0)] = None
    minItems: Optional[conint(ge=0)] = 0
    uniqueItems: Optional[bool] = False
    maxProperties: Optional[conint(ge=0)] = None
    minProperties: Optional[conint(ge=0)] = 0
    required: Optional[List[str]] = Field(None, min_items=1, unique_items=True)
    enum: Optional[List] = Field(None, min_items=1, unique_items=False)
    type: Optional[Type] = None
    not_: Optional[Union[Schema, Reference]] = Field(None, alias='not')
    allOf: Optional[List[Union[Schema, Reference]]] = None
    oneOf: Optional[List[Union[Schema, Reference]]] = None
    anyOf: Optional[List[Union[Schema, Reference]]] = None
    items: Optional[Union[Schema, Reference]] = None
    properties: Optional[Dict[str, Union[Schema, Reference]]] = None
    additionalProperties: Optional[Union[Schema, Reference, bool]] = True
    description: Optional[str] = None
    format: Optional[str] = None
    default: Optional[Any] = None
    nullable: Optional[bool] = False
    discriminator: Optional[Discriminator] = None
    readOnly: Optional[bool] = False
    writeOnly: Optional[bool] = False
    example: Optional[Any] = None
    externalDocs: Optional[ExternalDocumentation] = None
    deprecated: Optional[bool] = False
    xml: Optional[XML] = None


class Tag(BaseModel):
    class Config:
        extra = Extra.forbid

    name: str
    description: Optional[str] = None
    externalDocs: Optional[ExternalDocumentation] = None


class OAuthFlows(BaseModel):
    class Config:
        extra = Extra.forbid

    implicit: Optional[ImplicitOAuthFlow] = None
    password: Optional[PasswordOAuthFlow] = None
    clientCredentials: Optional[ClientCredentialsFlow] = None
    authorizationCode: Optional[AuthorizationCodeOAuthFlow] = None


class Link(BaseModel):
    class Config:
        extra = Extra.forbid

    operationId: Optional[str] = None
    operationRef: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    requestBody: Optional[Any] = None
    description: Optional[str] = None
    server: Optional[Server] = None


class OAuth2SecurityScheme(BaseModel):
    class Config:
        extra = Extra.forbid

    type: Type4
    flows: OAuthFlows
    description: Optional[str] = None


class SecurityScheme(BaseModel):
    __root__: Union[
        APIKeySecurityScheme,
        HTTPSecurityScheme,
        OAuth2SecurityScheme,
        OpenIdConnectSecurityScheme,
    ]


class Model(BaseModel):
    class Config:
        extra = Extra.forbid

    openapi: constr(regex=r'^3\.0\.\d(-.+)?$')
    info: Info
    externalDocs: Optional[ExternalDocumentation] = None
    servers: Optional[List[Server]] = None
    security: Optional[List[SecurityRequirement]] = None
    tags: Optional[List[Tag]] = Field(None, unique_items=True)
    paths: Paths
    components: Optional[Components] = None


class Components(BaseModel):
    class Config:
        extra = Extra.forbid

    schemas: Optional[
        Dict[constr(regex=r'^[a-zA-Z0-9\.\-_]+$'), Union[Schema, Reference]]
    ] = None
    responses: Optional[
        Dict[constr(regex=r'^[a-zA-Z0-9\.\-_]+$'), Union[Reference, Response]]
    ] = None
    parameters: Optional[
        Dict[constr(regex=r'^[a-zA-Z0-9\.\-_]+$'), Union[Reference, Parameter]]
    ] = None
    examples: Optional[
        Dict[constr(regex=r'^[a-zA-Z0-9\.\-_]+$'), Union[Reference, Example]]
    ] = None
    requestBodies: Optional[
        Dict[constr(regex=r'^[a-zA-Z0-9\.\-_]+$'), Union[Reference, RequestBody]]
    ] = None
    headers: Optional[
        Dict[constr(regex=r'^[a-zA-Z0-9\.\-_]+$'), Union[Reference, Header]]
    ] = None
    securitySchemes: Optional[
        Dict[constr(regex=r'^[a-zA-Z0-9\.\-_]+$'), Union[Reference, SecurityScheme]]
    ] = None
    links: Optional[
        Dict[constr(regex=r'^[a-zA-Z0-9\.\-_]+$'), Union[Reference, Link]]
    ] = None
    callbacks: Optional[
        Dict[constr(regex=r'^[a-zA-Z0-9\.\-_]+$'), Union[Reference, Callback]]
    ] = None


class Response(BaseModel):
    class Config:
        extra = Extra.forbid

    description: str
    headers: Optional[Dict[str, Union[Header, Reference]]] = None
    content: Optional[Dict[str, MediaType]] = None
    links: Optional[Dict[str, Union[Link, Reference]]] = None


class MediaType(BaseModel):
    class Config:
        extra = Extra.forbid

    schema_: Optional[Union[Schema, Reference]] = Field(None, alias='schema')
    example: Optional[Any] = None
    examples: Optional[Dict[str, Union[Example, Reference]]] = None
    encoding: Optional[Dict[str, Encoding]] = None


class Header(BaseModel):
    class Config:
        extra = Extra.forbid

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


class Paths(BaseModel):
    class Config:
        extra = Extra.forbid

    __root__: Union[
        Dict[constr(regex=r'^\/'), PathItem], Dict[constr(regex=r'^x-'), Any]
    ]


class PathItem(BaseModel):
    class Config:
        extra = Extra.forbid

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
    parameters: Optional[List[Union[Parameter, Reference]]] = Field(
        None, unique_items=True
    )


class Operation(BaseModel):
    class Config:
        extra = Extra.forbid

    tags: Optional[List[str]] = None
    summary: Optional[str] = None
    description: Optional[str] = None
    externalDocs: Optional[ExternalDocumentation] = None
    operationId: Optional[str] = None
    parameters: Optional[List[Union[Parameter, Reference]]] = Field(
        None, unique_items=True
    )
    requestBody: Optional[Union[RequestBody, Reference]] = None
    responses: Responses
    callbacks: Optional[Dict[str, Union[Callback, Reference]]] = None
    deprecated: Optional[bool] = False
    security: Optional[List[SecurityRequirement]] = None
    servers: Optional[List[Server]] = None


class Responses(BaseModel):
    class Config:
        extra = Extra.forbid

    default: Optional[Union[Response, Reference]] = None


class Parameter(BaseModel):
    class Config:
        extra = Extra.forbid

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
    class Config:
        extra = Extra.forbid

    description: Optional[str] = None
    content: Dict[str, MediaType]
    required: Optional[bool] = False


class Encoding(BaseModel):
    class Config:
        extra = Extra.forbid

    contentType: Optional[str] = None
    headers: Optional[Dict[str, Union[Header, Reference]]] = None
    style: Optional[Style5] = None
    explode: Optional[bool] = None
    allowReserved: Optional[bool] = False


Schema.model_rebuild()
Model.model_rebuild()
Components.model_rebuild()
Response.model_rebuild()
MediaType.model_rebuild()
Paths.model_rebuild()
PathItem.model_rebuild()
Operation.model_rebuild()
