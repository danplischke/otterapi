from typing import Annotated, Any, Union
from pydantic.fields import Field
from .client import Client
from .models import Cat, Dog

__all__ = ('async_list_animals', 'list_animals')


def list_animals(
    *, client: Client | None = None, **kwargs: Any
) -> list[Annotated[Dog | Cat, Field(discriminator='kind')]]:
    c = client or Client()
    response = c._request(method='get', path='/animals', **kwargs)
    return c._parse_response(
        response, list[Annotated[Dog | Cat, Field(discriminator='kind')]]
    )


async def async_list_animals(
    *, client: Client | None = None, **kwargs: Any
) -> list[Annotated[Dog | Cat, Field(discriminator='kind')]]:
    c = client or Client()
    response = await c._request_async(method='get', path='/animals', **kwargs)
    return await c._parse_response_async(
        response, list[Annotated[Dog | Cat, Field(discriminator='kind')]]
    )
