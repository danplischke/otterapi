from openapi_pydantic import Reference, Schema
from openapi_pydantic.v3.parser import OpenAPIv3

from otterapi.codegen.utils import sanitize_identifier


class OpenAPIProcessor:
    def __init__(self, openapi: OpenAPIv3 | None):
        self.openapi: OpenAPIv3 | None = openapi

    def _resolve_reference(self, reference: Reference | Schema) -> tuple[Schema, str]:
        if isinstance(reference, Reference):
            if not reference.ref.startswith('#/components/schemas/'):
                raise ValueError(f'Unsupported reference format: {reference.ref}')

            schema_name = reference.ref.split('/')[-1]
            schemas = self.openapi.components.schemas

            if schema_name not in schemas:
                raise ValueError(
                    f"Referenced schema '{schema_name}' not found in components.schemas"
                )

            return schemas[schema_name], sanitize_identifier(schema_name)
        return reference, sanitize_identifier(reference.title) if hasattr(
            reference, 'title'
        ) and reference.title else None
