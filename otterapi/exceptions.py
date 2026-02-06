"""Custom exceptions for OtterAPI.

This module defines a hierarchy of exceptions used throughout the OtterAPI library
to provide clear, actionable error messages for different failure scenarios.
"""


class OtterAPIError(Exception):
    """Base exception for all OtterAPI errors.

    All exceptions raised by OtterAPI inherit from this class, making it easy
    to catch all OtterAPI-related errors with a single except clause.

    Example:
        try:
            codegen.generate()
        except OtterAPIError as e:
            print(f"OtterAPI error: {e}")
    """

    def __init__(self, message: str, *args, **kwargs):
        self.message = message
        super().__init__(message, *args, **kwargs)


class SchemaError(OtterAPIError):
    """Base exception for schema-related errors."""

    pass


class SchemaLoadError(SchemaError):
    """Failed to load an OpenAPI schema from a source.

    This exception is raised when the schema cannot be loaded from
    the specified URL or file path.

    Attributes:
        source: The source path or URL that failed to load.
        cause: The underlying exception that caused the failure.
    """

    def __init__(self, source: str, cause: Exception | None = None):
        self.source = source
        self.cause = cause
        message = f"Failed to load schema from '{source}'"
        if cause:
            message += f': {cause}'
        super().__init__(message)


class SchemaValidationError(SchemaError):
    """Schema failed OpenAPI specification validation.

    This exception is raised when the loaded schema does not conform
    to the OpenAPI specification.

    Attributes:
        source: The source path or URL of the invalid schema.
        errors: List of validation error messages.
    """

    def __init__(self, source: str, errors: list[str] | None = None):
        self.source = source
        self.errors = errors or []
        message = f"Schema validation failed for '{source}'"
        if errors:
            message += f': {"; ".join(errors)}'
        super().__init__(message)


class SchemaReferenceError(SchemaError):
    """Failed to resolve a $ref reference in the schema.

    This exception is raised when a JSON Reference ($ref) cannot be resolved,
    either because the referenced component doesn't exist or the reference
    format is not supported.

    Attributes:
        reference: The $ref string that could not be resolved.
        reason: Explanation of why the reference couldn't be resolved.
    """

    def __init__(self, reference: str, reason: str | None = None):
        self.reference = reference
        self.reason = reason
        message = f"Failed to resolve reference '{reference}'"
        if reason:
            message += f': {reason}'
        super().__init__(message)


class CodeGenerationError(OtterAPIError):
    """Error during code generation.

    This exception is raised when the code generation process fails
    after successfully loading and validating the schema.

    Attributes:
        context: Additional context about what was being generated.
        cause: The underlying exception that caused the failure.
    """

    def __init__(
        self, message: str, context: str | None = None, cause: Exception | None = None
    ):
        self.context = context
        self.cause = cause
        full_message = message
        if context:
            full_message = f'{message} (while generating {context})'
        if cause:
            full_message += f': {cause}'
        super().__init__(full_message)


class TypeGenerationError(CodeGenerationError):
    """Error generating a type from a schema.

    This exception is raised when a specific type cannot be generated
    from an OpenAPI schema definition.

    Attributes:
        type_name: The name of the type being generated.
        schema_path: The path to the schema in the OpenAPI document.
    """

    def __init__(
        self,
        type_name: str,
        schema_path: str | None = None,
        cause: Exception | None = None,
    ):
        self.type_name = type_name
        self.schema_path = schema_path
        message = f"Failed to generate type '{type_name}'"
        if schema_path:
            message += f" at '{schema_path}'"
        super().__init__(message, context=type_name, cause=cause)


class EndpointGenerationError(CodeGenerationError):
    """Error generating an endpoint function.

    This exception is raised when an endpoint function cannot be generated
    from an OpenAPI operation definition.

    Attributes:
        operation_id: The operationId of the endpoint.
        method: The HTTP method of the endpoint.
        path: The URL path of the endpoint.
    """

    def __init__(
        self,
        operation_id: str,
        method: str | None = None,
        path: str | None = None,
        cause: Exception | None = None,
    ):
        self.operation_id = operation_id
        self.method = method
        self.path = path
        message = f"Failed to generate endpoint '{operation_id}'"
        if method and path:
            message += f' ({method.upper()} {path})'
        super().__init__(message, context=operation_id, cause=cause)


class ConfigurationError(OtterAPIError):
    """Error in configuration.

    This exception is raised when the configuration is invalid or
    cannot be loaded.

    Attributes:
        config_path: The path to the configuration file, if applicable.
        field: The specific configuration field that is invalid.
    """

    def __init__(
        self, message: str, config_path: str | None = None, field: str | None = None
    ):
        self.config_path = config_path
        self.field = field
        full_message = message
        if config_path:
            full_message = f"{message} in '{config_path}'"
        if field:
            full_message += f' (field: {field})'
        super().__init__(full_message)


class OutputError(OtterAPIError):
    """Error writing generated output.

    This exception is raised when the generated code cannot be written
    to the output location.

    Attributes:
        output_path: The path where output was being written.
        cause: The underlying exception that caused the failure.
    """

    def __init__(self, output_path: str, cause: Exception | None = None):
        self.output_path = output_path
        self.cause = cause
        message = f"Failed to write output to '{output_path}'"
        if cause:
            message += f': {cause}'
        super().__init__(message)


class UnsupportedFeatureError(OtterAPIError):
    """Attempted to use an unsupported feature.

    This exception is raised when the schema uses a feature that
    is not yet supported by OtterAPI.

    Attributes:
        feature: Description of the unsupported feature.
        suggestion: Optional suggestion for a workaround.
    """

    def __init__(self, feature: str, suggestion: str | None = None):
        self.feature = feature
        self.suggestion = suggestion
        message = f'Unsupported feature: {feature}'
        if suggestion:
            message += f'. {suggestion}'
        super().__init__(message)
