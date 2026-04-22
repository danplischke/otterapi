from typing import Any, Union
from .client import Client
from .models import User

__all__ = ('async_list_users', 'list_users')


def list_users(*, client: Client | None = None, **kwargs: Any) -> list[User]:
    c = client or Client()
    response = c._request(method='get', path='/users', **kwargs)
    return c._parse_response(response, list[User])


async def async_list_users(
    *, client: Client | None = None, **kwargs: Any
) -> list[User]:
    c = client or Client()
    response = await c._request_async(method='get', path='/users', **kwargs)
    return await c._parse_response_async(response, list[User])
