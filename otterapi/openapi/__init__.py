from typing import Annotated, Any, Literal

from pydantic import Discriminator, RootModel, Tag

from otterapi.openapi.v2 import Swagger
from otterapi.openapi.v3 import OpenAPI as OpenAPIv3_0
from otterapi.openapi.v3_1 import OpenAPI as OpenAPIv3_1
from otterapi.openapi.v3_2 import OpenAPI as OpenAPIv3_2

__all__ = [
    'UniversalOpenAPI',
]


def _get_openapi_version(data: Any) -> Literal['2.0', '3.0', '3.1', '3.2']:
    """Discriminator function to determine the OpenAPI version from raw data."""
    if isinstance(data, dict):
        # Check for Swagger 2.0 (uses 'swagger' field)
        if 'swagger' in data:
            return '2.0'
        # Check for OpenAPI 3.x (uses 'openapi' field)
        openapi_version = data.get('openapi', '')
        if openapi_version.startswith('3.0'):
            return '3.0'
        elif openapi_version.startswith('3.1'):
            return '3.1'
        elif openapi_version.startswith('3.2'):
            return '3.2'
    # Default fallback - will let Pydantic try to match
    return '3.0'


class UniversalOpenAPI(
    RootModel[
        Annotated[
            Annotated[Swagger, Tag('2.0')]
            | Annotated[OpenAPIv3_0, Tag('3.0')]
            | Annotated[OpenAPIv3_1, Tag('3.1')]
            | Annotated[OpenAPIv3_2, Tag('3.2')],
            Discriminator(_get_openapi_version),
        ]
    ]
):
    """Universal OpenAPI model that can parse Swagger 2.0, OpenAPI 3.0, 3.1, or 3.2 documents."""

    pass
