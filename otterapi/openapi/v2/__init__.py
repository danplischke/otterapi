"""OpenAPI/Swagger 2.0 specification models."""

from otterapi.openapi.v2.v2 import (
    ApiKeySecurity,
    BasicAuthenticationSecurity,
    BodyParameter,
    CollectionFormat,
    CollectionFormatWithMulti,
    Contact,
    ExternalDocs,
    FileSchema,
    Header,
    Info,
    JsonReference,
    License,
    NonBodyParameter,
    OAuth2AccessCodeSecurity,
    OAuth2ApplicationSecurity,
    OAuth2Flow,
    OAuth2ImplicitSecurity,
    OAuth2PasswordSecurity,
    Operation,
    Parameter,
    ParameterLocation,
    PathItem,
    Paths,
    PrimitiveType,
    PrimitivesItems,
    Response,
    Responses,
    ResponseValue,
    Schema,
    SchemeType,
    SecurityScheme,
    SecuritySchemeType,
    Swagger,
    Tag,
    XML,
)

__all__ = [
    # Main model
    "Swagger",
    # Info models
    "Info",
    "Contact",
    "License",
    "Tag",
    "ExternalDocs",
    # Schema models
    "Schema",
    "FileSchema",
    "XML",
    # Parameter models
    "Parameter",
    "BodyParameter",
    "NonBodyParameter",
    "PrimitivesItems",
    # Response models
    "Response",
    "Responses",
    "ResponseValue",
    "Header",
    # Operation models
    "Operation",
    "PathItem",
    "Paths",
    # Security models
    "SecurityScheme",
    "BasicAuthenticationSecurity",
    "ApiKeySecurity",
    "OAuth2ImplicitSecurity",
    "OAuth2PasswordSecurity",
    "OAuth2ApplicationSecurity",
    "OAuth2AccessCodeSecurity",
    # Enums
    "SchemeType",
    "ParameterLocation",
    "PrimitiveType",
    "CollectionFormat",
    "CollectionFormatWithMulti",
    "SecuritySchemeType",
    "OAuth2Flow",
    # Utilities
    "JsonReference",
]

