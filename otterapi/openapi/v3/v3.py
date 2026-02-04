from __future__ import annotations

from enum import Enum
from typing import Annotated, Any

from pydantic import (
    AnyUrl,
    BaseModel,
    ConfigDict,
    EmailStr,
    Field,
    PositiveFloat,
    RootModel,
    StringConstraints,
)

# Import OpenAPI 3.1 models for upgrade functionality
from otterapi.openapi.v3_1 import v3_1 as openapi_v3_1


class Reference(
    RootModel[dict[Annotated[str, StringConstraints(pattern=r'^\$ref$')], str]]
):
    pass


class Contact(BaseModel):
    model_config = ConfigDict(extra='forbid')

    name: str | None = None
    url: str | None = None
    email: EmailStr | None = None


class License(BaseModel):
    model_config = ConfigDict(extra='forbid')

    name: str
    url: str | None = None


class ServerVariable(BaseModel):
    model_config = ConfigDict(extra='forbid')

    enum: list[str] | None = None
    default: str
    description: str | None = None


class Type(Enum):
    array = 'array'
    boolean = 'boolean'
    integer = 'integer'
    number = 'number'
    object = 'object'
    string = 'string'


class Discriminator(BaseModel):
    propertyName: str
    mapping: dict[str, str] | None = None


class XML(BaseModel):
    model_config = ConfigDict(extra='forbid')

    name: str | None = None
    namespace: AnyUrl | None = None
    prefix: str | None = None
    attribute: bool | None = False
    wrapped: bool | None = False


class Example(BaseModel):
    model_config = ConfigDict(extra='forbid')

    summary: str | None = None
    description: str | None = None
    value: Any | None = None
    externalValue: str | None = None


class Style(Enum):
    simple = 'simple'


class SecurityRequirement(RootModel[dict[str, list[str]]]):
    pass


class ExternalDocumentation(BaseModel):
    model_config = ConfigDict(extra='forbid')

    description: str | None = None
    url: str


class ExampleXORExamples(RootModel[Any]):
    root: Any = Field(..., description='Example and examples are mutually exclusive')


class SchemaXORContent1(BaseModel):
    pass


class SchemaXORContent(RootModel[Any | SchemaXORContent1]):
    root: Any | SchemaXORContent1 = Field(
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
    in_: In | None = Field(None, alias='in')
    style: Style1 | None = 'simple'
    required: Required


class In1(Enum):
    query = 'query'


class Style2(Enum):
    form = 'form'
    spaceDelimited = 'spaceDelimited'
    pipeDelimited = 'pipeDelimited'
    deepObject = 'deepObject'


class QueryParameter(BaseModel):
    in_: In1 | None = Field(None, alias='in')
    style: Style2 | None = 'form'


class In2(Enum):
    header = 'header'


class Style3(Enum):
    simple = 'simple'


class HeaderParameter(BaseModel):
    in_: In2 | None = Field(None, alias='in')
    style: Style3 | None = 'simple'


class In3(Enum):
    cookie = 'cookie'


class Style4(Enum):
    form = 'form'


class CookieParameter(BaseModel):
    in_: In3 | None = Field(None, alias='in')
    style: Style4 | None = 'form'


class Type1(Enum):
    apiKey = 'apiKey'


class In4(Enum):
    header = 'header'
    query = 'query'
    cookie = 'cookie'


class APIKeySecurityScheme(BaseModel):
    model_config = ConfigDict(extra='forbid')

    type: Type1
    name: str
    in_: In4 = Field(..., alias='in')
    description: str | None = None


class Type2(Enum):
    http = 'http'


class HTTPSecurityScheme1(BaseModel):
    model_config = ConfigDict(extra='forbid')

    scheme: Annotated[str, StringConstraints(pattern=r'^[Bb][Ee][Aa][Rr][Ee][Rr]$')]
    bearerFormat: str | None = None
    description: str | None = None
    type: Type2


class HTTPSecurityScheme2(BaseModel):
    model_config = ConfigDict(extra='forbid')

    scheme: str
    bearerFormat: str | None = None
    description: str | None = None
    type: Type2


class HTTPSecurityScheme(RootModel[HTTPSecurityScheme1 | HTTPSecurityScheme2]):
    pass


class Type4(Enum):
    oauth2 = 'oauth2'


class Type5(Enum):
    openIdConnect = 'openIdConnect'


class OpenIdConnectSecurityScheme(BaseModel):
    model_config = ConfigDict(extra='forbid')

    type: Type5
    openIdConnectUrl: str
    description: str | None = None


class ImplicitOAuthFlow(BaseModel):
    model_config = ConfigDict(extra='forbid')

    authorizationUrl: str
    refreshUrl: str | None = None
    scopes: dict[str, str]


class PasswordOAuthFlow(BaseModel):
    model_config = ConfigDict(extra='forbid')

    tokenUrl: str
    refreshUrl: str | None = None
    scopes: dict[str, str]


class ClientCredentialsFlow(BaseModel):
    model_config = ConfigDict(extra='forbid')

    tokenUrl: str
    refreshUrl: str | None = None
    scopes: dict[str, str]


class AuthorizationCodeOAuthFlow(BaseModel):
    model_config = ConfigDict(extra='forbid')

    authorizationUrl: str
    tokenUrl: str
    refreshUrl: str | None = None
    scopes: dict[str, str]


class Callback(RootModel[dict[Annotated[str, StringConstraints(pattern=r'^x-')], Any]]):
    pass


class Style5(Enum):
    form = 'form'
    spaceDelimited = 'spaceDelimited'
    pipeDelimited = 'pipeDelimited'
    deepObject = 'deepObject'


class Info(BaseModel):
    model_config = ConfigDict(extra='forbid')

    title: str
    description: str | None = None
    termsOfService: str | None = None
    contact: Contact | None = None
    license: License | None = None
    version: str


class Server(BaseModel):
    model_config = ConfigDict(extra='forbid')

    url: str
    description: str | None = None
    variables: dict[str, ServerVariable] | None = None


class Schema(BaseModel):
    model_config = ConfigDict(extra='forbid')

    title: str | None = None
    multipleOf: PositiveFloat | None = None
    maximum: float | None = None
    exclusiveMaximum: bool | None = False
    minimum: float | None = None
    exclusiveMinimum: bool | None = False
    maxLength: Annotated[int, Field(ge=0)] | None = None
    minLength: Annotated[int, Field(ge=0)] | None = 0
    pattern: str | None = None
    maxItems: Annotated[int, Field(ge=0)] | None = None
    minItems: Annotated[int, Field(ge=0)] | None = 0
    uniqueItems: bool | None = False
    maxProperties: Annotated[int, Field(ge=0)] | None = None
    minProperties: Annotated[int, Field(ge=0)] | None = 0
    required: list[str] | None = Field(None, min_length=1)
    enum: list | None = Field(None, min_length=1)
    type: Type | None = None
    not_: Schema | Reference | None = Field(None, alias='not')
    allOf: list[Schema | Reference] | None = None
    oneOf: list[Schema | Reference] | None = None
    anyOf: list[Schema | Reference] | None = None
    items: Schema | Reference | None = None
    properties: dict[str, Schema | Reference] | None = None
    additionalProperties: Schema | Reference | bool | None = True
    description: str | None = None
    format: str | None = None
    default: Any | None = None
    nullable: bool | None = False
    discriminator: Discriminator | None = None
    readOnly: bool | None = False
    writeOnly: bool | None = False
    example: Any | None = None
    externalDocs: ExternalDocumentation | None = None
    deprecated: bool | None = False
    xml: XML | None = None


class Tag(BaseModel):
    model_config = ConfigDict(extra='forbid')

    name: str
    description: str | None = None
    externalDocs: ExternalDocumentation | None = None


class OAuthFlows(BaseModel):
    model_config = ConfigDict(extra='forbid')

    implicit: ImplicitOAuthFlow | None = None
    password: PasswordOAuthFlow | None = None
    clientCredentials: ClientCredentialsFlow | None = None
    authorizationCode: AuthorizationCodeOAuthFlow | None = None


class Link(BaseModel):
    model_config = ConfigDict(extra='forbid')

    operationId: str | None = None
    operationRef: str | None = None
    parameters: dict[str, Any] | None = None
    requestBody: Any | None = None
    description: str | None = None
    server: Server | None = None


class OAuth2SecurityScheme(BaseModel):
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
    pass


class OpenAPI(BaseModel):
    model_config = ConfigDict(extra='forbid')

    openapi: Annotated[str, StringConstraints(pattern=r'^3\.0\.\d(-.+)?$')]
    info: Info
    externalDocs: ExternalDocumentation | None = None
    servers: list[Server] | None = None
    security: list[SecurityRequirement] | None = None
    tags: list[Tag] | None = None
    paths: Paths
    components: Components | None = None

    def upgrade(self) -> tuple[openapi_v3_1.OpenAPI, list[str]]:
        """
        Upgrade this OpenAPI 3.0 specification to OpenAPI 3.1.

        Returns:
            A tuple of (OpenAPI 3.1 model, list of warnings)

        Key changes in 3.1:
        - nullable property removed, use type arrays instead
        - exclusiveMaximum/exclusiveMinimum changed from boolean to numeric
        - New jsonSchemaDialect field
        - Version updated to 3.1.x
        """
        warnings: list[str] = []

        # Convert basic metadata
        info = self._convert_info_to_3_1()

        # Convert components
        components = (
            self._convert_components_to_3_1(warnings) if self.components else None
        )

        # Convert paths
        paths = self._convert_paths_to_3_1(warnings) if self.paths else None

        # Convert servers
        servers = (
            [self._convert_server_to_3_1(s) for s in self.servers]
            if self.servers
            else None
        )

        # Convert security requirements
        security = None
        if self.security:
            security = [
                openapi_v3_1.SecurityRequirement(
                    root={k: v for k, v in req.root.items()}
                )
                for req in self.security
            ]

        # Convert tags
        tags = None
        if self.tags:
            tags = [
                openapi_v3_1.Tag(
                    name=tag.name,
                    description=tag.description,
                    externalDocs=self._convert_external_docs_to_3_1(tag.externalDocs)
                    if tag.externalDocs
                    else None,
                )
                for tag in self.tags
            ]

        # Convert external docs
        external_docs = (
            self._convert_external_docs_to_3_1(self.externalDocs)
            if self.externalDocs
            else None
        )

        # Build OpenAPI 3.1 object
        openapi_3_1 = openapi_v3_1.OpenAPI(
            openapi='3.1.0',
            info=info,
            jsonSchemaDialect='https://spec.openapis.org/oas/3.1/dialect/base',
            servers=servers,
            paths=paths,
            components=components,
            security=security,
            tags=tags,
            externalDocs=external_docs,
        )

        return openapi_3_1, warnings

    def _convert_info_to_3_1(self) -> openapi_v3_1.Info:
        """Convert Info object from OpenAPI 3.0 to 3.1."""
        contact = None
        if self.info.contact:
            contact = openapi_v3_1.Contact(
                name=self.info.contact.name,
                url=self.info.contact.url,
                email=self.info.contact.email,
            )

        license_obj = None
        if self.info.license:
            license_obj = openapi_v3_1.License(
                name=self.info.license.name, url=self.info.license.url
            )

        return openapi_v3_1.Info(
            title=self.info.title,
            version=self.info.version,
            summary=None,  # New in 3.1, not present in 3.0
            description=self.info.description,
            termsOfService=self.info.termsOfService,
            contact=contact,
            license=license_obj,
        )

    def _convert_server_to_3_1(self, server: Server) -> openapi_v3_1.Server:
        """Convert Server object from OpenAPI 3.0 to 3.1."""
        variables = None
        if server.variables:
            variables = {
                name: openapi_v3_1.ServerVariable(
                    enum=var.enum, default=var.default, description=var.description
                )
                for name, var in server.variables.items()
            }

        return openapi_v3_1.Server(
            url=server.url, description=server.description, variables=variables
        )

    def _convert_external_docs_to_3_1(
        self, docs: ExternalDocumentation
    ) -> openapi_v3_1.ExternalDocumentation:
        """Convert ExternalDocumentation from OpenAPI 3.0 to 3.1."""
        return openapi_v3_1.ExternalDocumentation(
            url=docs.url, description=docs.description
        )

    def _convert_components_to_3_1(
        self, warnings: list[str]
    ) -> openapi_v3_1.Components | None:
        """Convert Components object from OpenAPI 3.0 to 3.1."""
        if not self.components:
            return None

        # Convert schemas
        schemas = None
        if self.components.schemas:
            schemas = {}
            for name, schema in self.components.schemas.items():
                schemas[name] = self._convert_schema_or_ref_to_3_1(schema, warnings)

        # Convert responses
        responses = None
        if self.components.responses:
            responses = {}
            for name, response in self.components.responses.items():
                responses[name] = self._convert_response_or_ref_to_3_1(
                    response, warnings
                )

        # Convert parameters
        parameters = None
        if self.components.parameters:
            parameters = {}
            for name, param in self.components.parameters.items():
                parameters[name] = self._convert_parameter_or_ref_to_3_1(
                    param, warnings
                )

        # Convert examples
        examples = None
        if self.components.examples:
            examples = {}
            for name, example in self.components.examples.items():
                examples[name] = self._convert_example_or_ref_to_3_1(example)

        # Convert request bodies
        request_bodies = None
        if self.components.requestBodies:
            request_bodies = {}
            for name, body in self.components.requestBodies.items():
                request_bodies[name] = self._convert_request_body_or_ref_to_3_1(
                    body, warnings
                )

        # Convert headers
        headers = None
        if self.components.headers:
            headers = {}
            for name, header in self.components.headers.items():
                headers[name] = self._convert_header_or_ref_to_3_1(header, warnings)

        # Convert security schemes
        security_schemes = None
        if self.components.securitySchemes:
            security_schemes = {}
            for name, scheme in self.components.securitySchemes.items():
                security_schemes[name] = self._convert_security_scheme_or_ref_to_3_1(
                    scheme
                )

        # Convert links
        links = None
        if self.components.links:
            links = {}
            for name, link in self.components.links.items():
                links[name] = self._convert_link_or_ref_to_3_1(link)

        # Convert callbacks
        callbacks = None
        if self.components.callbacks:
            callbacks = {}
            for name, callback in self.components.callbacks.items():
                callbacks[name] = self._convert_callback_or_ref_to_3_1(
                    callback, warnings
                )

        return openapi_v3_1.Components(
            schemas=schemas,
            responses=responses,
            parameters=parameters,
            examples=examples,
            requestBodies=request_bodies,
            headers=headers,
            securitySchemes=security_schemes,
            links=links,
            callbacks=callbacks,
            pathItems=None,  # New in 3.1, not in 3.0
        )

    def _convert_schema_or_ref_to_3_1(
        self, schema: Schema | Reference, warnings: list[str]
    ) -> openapi_v3_1.Schema | openapi_v3_1.Reference:
        """Convert a Schema or Reference from OpenAPI 3.0 to 3.1."""
        if isinstance(schema, Reference):
            return self._convert_reference_to_3_1(schema)
        return self._convert_schema_to_3_1(schema, warnings)

    def _convert_reference_to_3_1(self, ref: Reference) -> openapi_v3_1.Reference:
        """Convert a Reference from OpenAPI 3.0 to 3.1."""
        # Extract the $ref value from the RootModel
        ref_value = ref.root.get('$ref', '')
        return openapi_v3_1.Reference(
            **{'$ref': ref_value, 'summary': None, 'description': None}
        )

    def _convert_schema_to_3_1(
        self, schema: Schema, warnings: list[str]
    ) -> openapi_v3_1.Schema:
        """Convert a Schema from OpenAPI 3.0 to 3.1."""
        # Handle nullable conversion
        type_value = None
        if schema.type:
            # Convert v3.Type enum to v3_1.Type enum
            type_3_1 = openapi_v3_1.Type(schema.type.value)
            if schema.nullable:
                # Convert nullable: true to type array
                warnings.append('Converting nullable field to type array for schema')
                type_value = [type_3_1, openapi_v3_1.Type.null]
            else:
                type_value = type_3_1

        # Handle exclusiveMaximum/exclusiveMinimum conversion
        exclusive_maximum = None
        if schema.exclusiveMaximum and schema.maximum is not None:
            # In 3.0, exclusiveMaximum is boolean, in 3.1 it's numeric
            warnings.append('Converting exclusiveMaximum from boolean to numeric')
            exclusive_maximum = schema.maximum
        elif not schema.exclusiveMaximum and schema.maximum is not None:
            exclusive_maximum = None

        exclusive_minimum = None
        if schema.exclusiveMinimum and schema.minimum is not None:
            warnings.append('Converting exclusiveMinimum from boolean to numeric')
            exclusive_minimum = schema.minimum
        elif not schema.exclusiveMinimum and schema.minimum is not None:
            exclusive_minimum = None

        # Use maximum/minimum only if not exclusive
        maximum = schema.maximum if not schema.exclusiveMaximum else None
        minimum = schema.minimum if not schema.exclusiveMinimum else None

        # Convert nested schemas
        not_ = (
            self._convert_schema_or_ref_to_3_1(schema.not_, warnings)
            if schema.not_
            else None
        )
        allOf = (
            [self._convert_schema_or_ref_to_3_1(s, warnings) for s in schema.allOf]
            if schema.allOf
            else None
        )
        oneOf = (
            [self._convert_schema_or_ref_to_3_1(s, warnings) for s in schema.oneOf]
            if schema.oneOf
            else None
        )
        anyOf = (
            [self._convert_schema_or_ref_to_3_1(s, warnings) for s in schema.anyOf]
            if schema.anyOf
            else None
        )
        items = (
            self._convert_schema_or_ref_to_3_1(schema.items, warnings)
            if schema.items
            else None
        )

        properties = None
        if schema.properties:
            properties = {
                name: self._convert_schema_or_ref_to_3_1(prop, warnings)
                for name, prop in schema.properties.items()
            }

        additional_properties = None
        if schema.additionalProperties is not None:
            if isinstance(schema.additionalProperties, bool):
                additional_properties = schema.additionalProperties
            else:
                additional_properties = self._convert_schema_or_ref_to_3_1(
                    schema.additionalProperties, warnings
                )

        # Convert discriminator
        discriminator = None
        if schema.discriminator:
            discriminator = openapi_v3_1.Discriminator(
                propertyName=schema.discriminator.propertyName,
                mapping=schema.discriminator.mapping,
            )

        # Convert XML
        xml = None
        if schema.xml:
            xml = openapi_v3_1.XML(
                name=schema.xml.name,
                namespace=schema.xml.namespace,
                prefix=schema.xml.prefix,
                attribute=schema.xml.attribute,
                wrapped=schema.xml.wrapped,
            )

        # Convert external docs
        external_docs = (
            self._convert_external_docs_to_3_1(schema.externalDocs)
            if schema.externalDocs
            else None
        )

        # Build schema dict excluding None values for optional fields
        schema_dict = {
            'title': schema.title,
            'multipleOf': schema.multipleOf,
            'maximum': maximum,
            'exclusiveMaximum': exclusive_maximum,
            'minimum': minimum,
            'exclusiveMinimum': exclusive_minimum,
            'maxLength': schema.maxLength,
            'minLength': schema.minLength,
            'pattern': schema.pattern,
            'maxItems': schema.maxItems,
            'minItems': schema.minItems,
            'uniqueItems': schema.uniqueItems,
            'maxProperties': schema.maxProperties,
            'minProperties': schema.minProperties,
            'required': schema.required,
            'enum': schema.enum,
            'type': type_value,
            'not': not_,
            'allOf': allOf,
            'oneOf': oneOf,
            'anyOf': anyOf,
            'items': items,
            'prefixItems': None,
            'properties': properties,
            'additionalProperties': additional_properties,
            'patternProperties': None,
            'format': schema.format,
            'description': schema.description,
            'default': schema.default,
            'discriminator': discriminator,
            'readOnly': schema.readOnly,
            'writeOnly': schema.writeOnly,
            'example': schema.example,
            'examples': None,
            'externalDocs': external_docs,
            'deprecated': schema.deprecated,
            'xml': xml,
        }

        # Remove None values to avoid extra_forbid issues
        schema_dict = {k: v for k, v in schema_dict.items() if v is not None}

        return openapi_v3_1.Schema.model_validate(schema_dict)

    def _convert_response_or_ref_to_3_1(
        self, response: Response | Reference, warnings: list[str]
    ) -> openapi_v3_1.Response | openapi_v3_1.Reference:
        """Convert a Response or Reference from OpenAPI 3.0 to 3.1."""
        if isinstance(response, Reference):
            return self._convert_reference_to_3_1(response)
        return self._convert_response_to_3_1(response, warnings)

    def _convert_response_to_3_1(
        self, response: Response, warnings: list[str]
    ) -> openapi_v3_1.Response:
        """Convert a Response from OpenAPI 3.0 to 3.1."""
        headers = None
        if response.headers:
            headers = {}
            for name, header in response.headers.items():
                headers[name] = self._convert_header_or_ref_to_3_1(header, warnings)

        content = None
        if response.content:
            content = {}
            for media_type, media_type_obj in response.content.items():
                content[media_type] = self._convert_media_type_to_3_1(
                    media_type_obj, warnings
                )

        links = None
        if response.links:
            links = {}
            for name, link in response.links.items():
                links[name] = self._convert_link_or_ref_to_3_1(link)

        return openapi_v3_1.Response(
            description=response.description,
            headers=headers,
            content=content,
            links=links,
        )

    def _convert_media_type_to_3_1(
        self, media_type: MediaType, warnings: list[str]
    ) -> openapi_v3_1.MediaType:
        """Convert a MediaType from OpenAPI 3.0 to 3.1."""
        schema_ = None
        if media_type.schema_:
            schema_ = self._convert_schema_or_ref_to_3_1(media_type.schema_, warnings)

        examples = None
        if media_type.examples:
            examples = {}
            for name, example in media_type.examples.items():
                examples[name] = self._convert_example_or_ref_to_3_1(example)

        encoding = None
        if media_type.encoding:
            encoding = {}
            for name, enc in media_type.encoding.items():
                encoding[name] = self._convert_encoding_to_3_1(enc, warnings)

        media_type_dict = {
            'schema': schema_,
            'example': media_type.example,
            'examples': examples,
            'encoding': encoding,
        }

        # Remove None values
        media_type_dict = {k: v for k, v in media_type_dict.items() if v is not None}

        return openapi_v3_1.MediaType.model_validate(media_type_dict)

    def _convert_encoding_to_3_1(
        self, encoding: Encoding, warnings: list[str]
    ) -> openapi_v3_1.Encoding:
        """Convert an Encoding from OpenAPI 3.0 to 3.1."""
        headers = None
        if encoding.headers:
            headers = {}
            for name, header in encoding.headers.items():
                headers[name] = self._convert_header_or_ref_to_3_1(header, warnings)

        return openapi_v3_1.Encoding(
            contentType=encoding.contentType,
            headers=headers,
            style=encoding.style,
            explode=encoding.explode,
            allowReserved=encoding.allowReserved,
        )

    def _convert_parameter_or_ref_to_3_1(
        self, param: Parameter | Reference, warnings: list[str]
    ) -> openapi_v3_1.Parameter | openapi_v3_1.Reference:
        """Convert a Parameter or Reference from OpenAPI 3.0 to 3.1."""
        if isinstance(param, Reference):
            return self._convert_reference_to_3_1(param)
        return self._convert_parameter_to_3_1(param, warnings)

    def _convert_parameter_to_3_1(
        self, param: Parameter, warnings: list[str]
    ) -> openapi_v3_1.Parameter:
        """Convert a Parameter from OpenAPI 3.0 to 3.1."""
        schema_ = None
        if param.schema_:
            schema_ = self._convert_schema_or_ref_to_3_1(param.schema_, warnings)

        content = None
        if param.content:
            content = {}
            for media_type, media_type_obj in param.content.items():
                content[media_type] = self._convert_media_type_to_3_1(
                    media_type_obj, warnings
                )

        examples = None
        if param.examples:
            examples = {}
            for name, example in param.examples.items():
                examples[name] = self._convert_example_or_ref_to_3_1(example)

        param_dict = {
            'name': param.name,
            'in': param.in_,
            'description': param.description,
            'required': param.required,
            'deprecated': param.deprecated,
            'allowEmptyValue': param.allowEmptyValue,
            'style': param.style,
            'explode': param.explode,
            'allowReserved': param.allowReserved,
            'schema': schema_,
            'content': content,
            'example': param.example,
            'examples': examples,
        }

        # Remove None values
        param_dict = {k: v for k, v in param_dict.items() if v is not None}

        return openapi_v3_1.Parameter.model_validate(param_dict)

    def _convert_header_or_ref_to_3_1(
        self, header: Header | Reference, warnings: list[str]
    ) -> openapi_v3_1.Header | openapi_v3_1.Reference:
        """Convert a Header or Reference from OpenAPI 3.0 to 3.1."""
        if isinstance(header, Reference):
            return self._convert_reference_to_3_1(header)
        return self._convert_header_to_3_1(header, warnings)

    def _convert_header_to_3_1(
        self, header: Header, warnings: list[str]
    ) -> openapi_v3_1.Header:
        """Convert a Header from OpenAPI 3.0 to 3.1."""
        schema_ = None
        if header.schema_:
            schema_ = self._convert_schema_or_ref_to_3_1(header.schema_, warnings)

        content = None
        if header.content:
            content = {}
            for media_type, media_type_obj in header.content.items():
                content[media_type] = self._convert_media_type_to_3_1(
                    media_type_obj, warnings
                )

        examples = None
        if header.examples:
            examples = {}
            for name, example in header.examples.items():
                examples[name] = self._convert_example_or_ref_to_3_1(example)

        header_dict = {
            'description': header.description,
            'required': header.required,
            'deprecated': header.deprecated,
            'allowEmptyValue': header.allowEmptyValue,
            'style': header.style,
            'explode': header.explode,
            'allowReserved': header.allowReserved,
            'schema': schema_,
            'content': content,
            'example': header.example,
            'examples': examples,
        }

        # Remove None values
        header_dict = {k: v for k, v in header_dict.items() if v is not None}

        return openapi_v3_1.Header.model_validate(header_dict)

    def _convert_example_or_ref_to_3_1(
        self, example: Example | Reference
    ) -> openapi_v3_1.Example | openapi_v3_1.Reference:
        """Convert an Example or Reference from OpenAPI 3.0 to 3.1."""
        if isinstance(example, Reference):
            return self._convert_reference_to_3_1(example)
        return openapi_v3_1.Example(
            summary=example.summary,
            description=example.description,
            value=example.value,
            externalValue=example.externalValue,
        )

    def _convert_request_body_or_ref_to_3_1(
        self, body: RequestBody | Reference, warnings: list[str]
    ) -> openapi_v3_1.RequestBody | openapi_v3_1.Reference:
        """Convert a RequestBody or Reference from OpenAPI 3.0 to 3.1."""
        if isinstance(body, Reference):
            return self._convert_reference_to_3_1(body)

        content = {}
        for media_type, media_type_obj in body.content.items():
            content[media_type] = self._convert_media_type_to_3_1(
                media_type_obj, warnings
            )

        return openapi_v3_1.RequestBody(
            description=body.description, content=content, required=body.required
        )

    def _convert_link_or_ref_to_3_1(
        self, link: Link | Reference
    ) -> openapi_v3_1.Link | openapi_v3_1.Reference:
        """Convert a Link or Reference from OpenAPI 3.0 to 3.1."""
        if isinstance(link, Reference):
            return self._convert_reference_to_3_1(link)

        server = self._convert_server_to_3_1(link.server) if link.server else None

        return openapi_v3_1.Link(
            operationId=link.operationId,
            operationRef=link.operationRef,
            parameters=link.parameters,
            requestBody=link.requestBody,
            description=link.description,
            server=server,
        )

    def _convert_security_scheme_or_ref_to_3_1(
        self, scheme: SecurityScheme | Reference
    ) -> openapi_v3_1.SecurityScheme | openapi_v3_1.Reference:
        """Convert a SecurityScheme or Reference from OpenAPI 3.0 to 3.1."""
        if isinstance(scheme, Reference):
            return self._convert_reference_to_3_1(scheme)

        # SecurityScheme is a RootModel with Union, need to access root
        sec_scheme = scheme.root

        if isinstance(sec_scheme, APIKeySecurityScheme):
            inner = openapi_v3_1.APIKeySecurityScheme.model_validate(
                {
                    'type': openapi_v3_1.Type1.apiKey,
                    'name': sec_scheme.name,
                    'in': sec_scheme.in_.value,
                    'description': sec_scheme.description,
                }
            )
            return openapi_v3_1.SecurityScheme(root=inner)
        elif isinstance(sec_scheme, HTTPSecurityScheme):
            http_scheme = sec_scheme.root
            # Determine if it's Bearer or non-Bearer based on scheme pattern
            scheme_value = http_scheme.scheme
            if scheme_value.lower() == 'bearer':
                inner_http = openapi_v3_1.HTTPSecurityScheme1(
                    type=openapi_v3_1.Type2.http,
                    scheme=scheme_value,
                    bearerFormat=http_scheme.bearerFormat,
                    description=http_scheme.description,
                )
            else:
                inner_http = openapi_v3_1.HTTPSecurityScheme2(
                    type=openapi_v3_1.Type2.http,
                    scheme=scheme_value,
                    description=http_scheme.description,
                )
            inner = openapi_v3_1.HTTPSecurityScheme(root=inner_http)
            return openapi_v3_1.SecurityScheme(root=inner)
        elif isinstance(sec_scheme, OAuth2SecurityScheme):
            flows = sec_scheme.flows
            oauth_flows = openapi_v3_1.OAuthFlows(
                implicit=openapi_v3_1.ImplicitOAuthFlow(
                    authorizationUrl=flows.implicit.authorizationUrl,
                    refreshUrl=flows.implicit.refreshUrl,
                    scopes=flows.implicit.scopes,
                )
                if flows.implicit
                else None,
                password=openapi_v3_1.PasswordOAuthFlow(
                    tokenUrl=flows.password.tokenUrl,
                    refreshUrl=flows.password.refreshUrl,
                    scopes=flows.password.scopes,
                )
                if flows.password
                else None,
                clientCredentials=openapi_v3_1.ClientCredentialsFlow(
                    tokenUrl=flows.clientCredentials.tokenUrl,
                    refreshUrl=flows.clientCredentials.refreshUrl,
                    scopes=flows.clientCredentials.scopes,
                )
                if flows.clientCredentials
                else None,
                authorizationCode=openapi_v3_1.AuthorizationCodeOAuthFlow(
                    authorizationUrl=flows.authorizationCode.authorizationUrl,
                    tokenUrl=flows.authorizationCode.tokenUrl,
                    refreshUrl=flows.authorizationCode.refreshUrl,
                    scopes=flows.authorizationCode.scopes,
                )
                if flows.authorizationCode
                else None,
            )
            inner = openapi_v3_1.OAuth2SecurityScheme(
                type=openapi_v3_1.Type4.oauth2,
                flows=oauth_flows,
                description=sec_scheme.description,
            )
            return openapi_v3_1.SecurityScheme(root=inner)
        elif isinstance(sec_scheme, OpenIdConnectSecurityScheme):
            inner = openapi_v3_1.OpenIdConnectSecurityScheme(
                type=openapi_v3_1.Type5.openIdConnect,
                openIdConnectUrl=sec_scheme.openIdConnectUrl,
                description=sec_scheme.description,
            )
            return openapi_v3_1.SecurityScheme(root=inner)

        raise ValueError(f'Unknown security scheme type: {type(sec_scheme)}')

    def _convert_callback_or_ref_to_3_1(
        self, callback: Callback | Reference, warnings: list[str]
    ) -> openapi_v3_1.Callback | openapi_v3_1.Reference:
        """Convert a Callback or Reference from OpenAPI 3.0 to 3.1."""
        if isinstance(callback, Reference):
            return self._convert_reference_to_3_1(callback)

        # Callback is a RootModel, just pass through the root
        return openapi_v3_1.Callback(root=callback.root)

    def _convert_paths_to_3_1(self, warnings: list[str]) -> openapi_v3_1.Paths | None:
        """Convert Paths from OpenAPI 3.0 to 3.1."""
        if not self.paths:
            return None

        # Paths is a RootModel containing a dict
        paths_dict = {}

        if hasattr(self.paths, 'root') and isinstance(self.paths.root, dict):
            for path, path_item in self.paths.root.items():
                if path.startswith('x-'):
                    # Vendor extension, pass through
                    paths_dict[path] = path_item
                else:
                    # Convert PathItem
                    paths_dict[path] = self._convert_path_item_to_3_1(
                        path_item, warnings
                    )

        return openapi_v3_1.Paths(root=paths_dict)

    def _convert_path_item_to_3_1(
        self, path_item: PathItem, warnings: list[str]
    ) -> openapi_v3_1.PathItem:
        """Convert a PathItem from OpenAPI 3.0 to 3.1."""
        parameters = None
        if path_item.parameters:
            parameters = [
                self._convert_parameter_or_ref_to_3_1(p, warnings)
                for p in path_item.parameters
            ]

        servers = None
        if path_item.servers:
            servers = [self._convert_server_to_3_1(s) for s in path_item.servers]

        # Convert operations
        get = (
            self._convert_operation_to_3_1(path_item.get, warnings)
            if path_item.get
            else None
        )
        put = (
            self._convert_operation_to_3_1(path_item.put, warnings)
            if path_item.put
            else None
        )
        post = (
            self._convert_operation_to_3_1(path_item.post, warnings)
            if path_item.post
            else None
        )
        delete = (
            self._convert_operation_to_3_1(path_item.delete, warnings)
            if path_item.delete
            else None
        )
        options = (
            self._convert_operation_to_3_1(path_item.options, warnings)
            if path_item.options
            else None
        )
        head = (
            self._convert_operation_to_3_1(path_item.head, warnings)
            if path_item.head
            else None
        )
        patch = (
            self._convert_operation_to_3_1(path_item.patch, warnings)
            if path_item.patch
            else None
        )
        trace = (
            self._convert_operation_to_3_1(path_item.trace, warnings)
            if path_item.trace
            else None
        )

        path_item_dict = {
            '$ref': path_item.field_ref,
            'summary': path_item.summary,
            'description': path_item.description,
            'get': get,
            'put': put,
            'post': post,
            'delete': delete,
            'options': options,
            'head': head,
            'patch': patch,
            'trace': trace,
            'servers': servers,
            'parameters': parameters,
        }

        # Remove None values
        path_item_dict = {k: v for k, v in path_item_dict.items() if v is not None}

        return openapi_v3_1.PathItem.model_validate(path_item_dict)

    def _convert_operation_to_3_1(
        self, operation: Operation, warnings: list[str]
    ) -> openapi_v3_1.Operation:
        """Convert an Operation from OpenAPI 3.0 to 3.1."""
        external_docs = (
            self._convert_external_docs_to_3_1(operation.externalDocs)
            if operation.externalDocs
            else None
        )

        parameters = None
        if operation.parameters:
            parameters = [
                self._convert_parameter_or_ref_to_3_1(p, warnings)
                for p in operation.parameters
            ]

        request_body = None
        if operation.requestBody:
            request_body = self._convert_request_body_or_ref_to_3_1(
                operation.requestBody, warnings
            )

        responses = (
            self._convert_responses_to_3_1(operation.responses, warnings)
            if operation.responses
            else None
        )

        callbacks = None
        if operation.callbacks:
            callbacks = {}
            for name, callback in operation.callbacks.items():
                callbacks[name] = self._convert_callback_or_ref_to_3_1(
                    callback, warnings
                )

        security = None
        if operation.security:
            security = [
                openapi_v3_1.SecurityRequirement(
                    root={k: v for k, v in req.root.items()}
                )
                for req in operation.security
            ]

        servers = None
        if operation.servers:
            servers = [self._convert_server_to_3_1(s) for s in operation.servers]

        return openapi_v3_1.Operation(
            tags=operation.tags,
            summary=operation.summary,
            description=operation.description,
            externalDocs=external_docs,
            operationId=operation.operationId,
            parameters=parameters,
            requestBody=request_body,
            responses=responses,
            callbacks=callbacks,
            deprecated=operation.deprecated,
            security=security,
            servers=servers,
        )

    def _convert_responses_to_3_1(
        self, responses: Responses, warnings: list[str]
    ) -> openapi_v3_1.Responses:
        """Convert Responses from OpenAPI 3.0 to 3.1."""
        converted_responses = {}
        for status_code, response in responses.root.items():
            converted_responses[status_code] = self._convert_response_or_ref_to_3_1(
                response, warnings
            )
        return openapi_v3_1.Responses(root=converted_responses)


class Components(BaseModel):
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


class Response(BaseModel):
    model_config = ConfigDict(extra='forbid')

    description: str
    headers: dict[str, Header | Reference] | None = None
    content: dict[str, MediaType] | None = None
    links: dict[str, Link | Reference] | None = None


class MediaType(BaseModel):
    model_config = ConfigDict(extra='forbid')

    schema_: Schema | Reference | None = Field(None, alias='schema')
    example: Any | None = None
    examples: dict[str, Example | Reference] | None = None
    encoding: dict[str, Encoding] | None = None


class Header(BaseModel):
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
    """Paths object containing path items and optional extensions.

    Keys should be path templates (starting with /) or extensions (starting with x-).
    """

    pass


class PathItem(BaseModel):
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
    model_config = ConfigDict(extra='forbid')

    tags: list[str] | None = None
    summary: str | None = None
    description: str | None = None
    externalDocs: ExternalDocumentation | None = None
    operationId: str | None = None
    parameters: list[Parameter | Reference] | None = None
    requestBody: RequestBody | Reference | None = None
    responses: Responses
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
    model_config = ConfigDict(extra='forbid')

    description: str | None = None
    content: dict[str, MediaType]
    required: bool | None = False


class Encoding(BaseModel):
    model_config = ConfigDict(extra='forbid')

    contentType: str | None = None
    headers: dict[str, Header | Reference] | None = None
    style: Style5 | None = None
    explode: bool | None = None
    allowReserved: bool | None = False


Schema.model_rebuild()
OpenAPI.model_rebuild()
Components.model_rebuild()
Response.model_rebuild()
MediaType.model_rebuild()
Paths.model_rebuild()
PathItem.model_rebuild()
Operation.model_rebuild()
