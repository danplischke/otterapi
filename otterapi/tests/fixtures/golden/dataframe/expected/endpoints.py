from typing import Any, TYPE_CHECKING, Union
from ._dataframe import to_pandas, to_polars
from .client import Client
from .models import Row

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl

__all__ = (
    'async_list_rows',
    'async_list_rows_df',
    'async_list_rows_pl',
    'list_rows',
    'list_rows_df',
    'list_rows_pl',
)


def list_rows(*, client: Client | None = None, **kwargs: Any) -> list[Row]:
    c = client or Client()
    response = c._request(method='get', path='/rows', **kwargs)
    return c._parse_response(response, list[Row])


async def async_list_rows(*, client: Client | None = None, **kwargs: Any) -> list[Row]:
    c = client or Client()
    response = await c._request_async(method='get', path='/rows', **kwargs)
    return await c._parse_response_async(response, list[Row])


def list_rows_df(
    *, path: str | None = None, client: Client | None = None, **kwargs: Any
) -> 'pd.DataFrame':
    """
    Returns:
        pd.DataFrame
    """
    c = client or Client()
    data = c._request_json(method='get', path='/rows', **kwargs)
    return to_pandas(data, path=path)


async def async_list_rows_df(
    *, path: str | None = None, client: Client | None = None, **kwargs: Any
) -> 'pd.DataFrame':
    """
    Returns:
        pd.DataFrame
    """
    c = client or Client()
    data = await c._request_json_async(method='get', path='/rows', **kwargs)
    return to_pandas(data, path=path)


def list_rows_pl(
    *, path: str | None = None, client: Client | None = None, **kwargs: Any
) -> 'pl.DataFrame':
    """
    Returns:
        pl.DataFrame
    """
    c = client or Client()
    data = c._request_json(method='get', path='/rows', **kwargs)
    return to_polars(data, path=path)


async def async_list_rows_pl(
    *, path: str | None = None, client: Client | None = None, **kwargs: Any
) -> 'pl.DataFrame':
    """
    Returns:
        pl.DataFrame
    """
    c = client or Client()
    data = await c._request_json_async(method='get', path='/rows', **kwargs)
    return to_polars(data, path=path)
