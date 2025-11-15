from builtins import list
from datetime import datetime
from httpx import AsyncClient, request
from pydantic import Json, RootModel, TypeAdapter
from typing import Any, Type, TypeVar, Union
from .models import ApiResponse, Category, Order, Pet, Tag, User

__all__ = ('aaddPet', 'acreateUser', 'acreateUsersWithListInput', 'addPet', 'adeleteOrder', 'adeletePet', 'adeleteUser',
           'afindPetsByStatus', 'afindPetsByTags', 'agetInventory', 'agetOrderById', 'agetPetById', 'agetUserByName',
           'aloginUser', 'alogoutUser', 'aplaceOrder', 'aupdatePet', 'aupdatePetWithForm', 'aupdateUser', 'auploadFile',
           'createUser', 'createUsersWithListInput', 'deleteOrder', 'deletePet', 'deleteUser', 'findPetsByStatus',
           'findPetsByTags', 'getInventory', 'getOrderById', 'getPetById', 'getUserByName', 'loginUser', 'logoutUser',
           'placeOrder', 'updatePet', 'updatePetWithForm', 'updateUser', 'uploadFile')
BASE_URL = '/api/v3'
T = TypeVar('T')


def request_sync(method: str, path: str, *, response_model: Union[Type[T], None] = Json,
                 supported_status_codes: list[int] = None, **kwargs: dict) -> T:
    response = request(method,
                       f'{BASE_URL}{path}', **kwargs)
    if not supported_status_codes or response.status_code not in supported_status_codes:
        response.raise_for_status()
    data = response.json()
    if not response_model:
        return data
    validated_data = TypeAdapter(response_model).validate_python(data)
    if isinstance(validated_data, RootModel):
        return validated_data.root
    return validated_data


async def request_async(method: str, path: str, *, response_model: Union[Type[T], None] = Json,
                        supported_status_codes: list[int] = None, **kwargs: dict) -> T:
    async with AsyncClient() as client:
        response = await client.request(method,
                                        f'{BASE_URL}{path}', **kwargs)
        if not supported_status_codes or response.status_code not in supported_status_codes:
            response.raise_for_status()
        data = response.json()
        if not response_model:
            return data
        validated_data = TypeAdapter(response_model).validate_python(data)
        if isinstance(validated_data, RootModel):
            return validated_data.root
        return validated_data


def addPet(body: Pet, **kwargs: dict) -> Pet:
    """Add a new pet to the store."""
    return request_sync(method='post', path='/pet', response_model=Pet, supported_status_codes=[200],
                        json=body.model_dump(), **kwargs)


async def aaddPet(body: Pet, **kwargs: dict) -> Pet:
    """Add a new pet to the store."""
    return await request_async(method='post', path='/pet', response_model=Pet, supported_status_codes=[200],
                               json=body.model_dump(), **kwargs)


def updatePet(body: Pet, **kwargs: dict) -> Pet:
    """Update an existing pet by Id."""
    return request_sync(method='put', path='/pet', response_model=Pet, supported_status_codes=[200],
                        json=body.model_dump(), **kwargs)


async def aupdatePet(body: Pet, **kwargs: dict) -> Pet:
    """Update an existing pet by Id."""
    return await request_async(method='put', path='/pet', response_model=Pet, supported_status_codes=[200],
                               json=body.model_dump(), **kwargs)


def findPetsByStatus(status: str, **kwargs: dict) -> list[Pet]:
    """Multiple status values can be provided with comma separated strings."""
    return request_sync(method='get', path='/pet/findByStatus', response_model=list[Pet], supported_status_codes=[200],
                        params={'status': status}, **kwargs)


async def afindPetsByStatus(status: str, **kwargs: dict) -> list[Pet]:
    """Multiple status values can be provided with comma separated strings."""
    return await request_async(method='get', path='/pet/findByStatus', response_model=list[Pet],
                               supported_status_codes=[200], params={'status': status}, **kwargs)


def findPetsByTags(tags: list[str], **kwargs: dict) -> list[Pet]:
    """Multiple tags can be provided with comma separated strings. Use tag1, tag2, tag3 for testing."""
    return request_sync(method='get', path='/pet/findByTags', response_model=list[Pet], supported_status_codes=[200],
                        params={'tags': tags}, **kwargs)


async def afindPetsByTags(tags: list[str], **kwargs: dict) -> list[Pet]:
    """Multiple tags can be provided with comma separated strings. Use tag1, tag2, tag3 for testing."""
    return await request_async(method='get', path='/pet/findByTags', response_model=list[Pet],
                               supported_status_codes=[200], params={'tags': tags}, **kwargs)


def deletePet(petId: int, *, api_key: str = None, **kwargs: dict) -> Any:
    """Delete a pet."""
    return request_sync(method='delete', path=f'/pet/{petId}', response_model=None, supported_status_codes=None,
                        headers={'api_key': api_key}, **kwargs)


async def adeletePet(petId: int, *, api_key: str = None, **kwargs: dict) -> Any:
    """Delete a pet."""
    return await request_async(method='delete', path=f'/pet/{petId}', response_model=None, supported_status_codes=None,
                               headers={'api_key': api_key}, **kwargs)


def getPetById(petId: int, **kwargs: dict) -> Pet:
    """Returns a single pet."""
    return request_sync(method='get', path=f'/pet/{petId}', response_model=Pet, supported_status_codes=[200], **kwargs)


async def agetPetById(petId: int, **kwargs: dict) -> Pet:
    """Returns a single pet."""
    return await request_async(method='get', path=f'/pet/{petId}', response_model=Pet, supported_status_codes=[200],
                               **kwargs)


def updatePetWithForm(petId: int, *, name: str = None, status: str = None, **kwargs: dict) -> Pet:
    """Updates a pet resource based on the form data."""
    return request_sync(method='post', path=f'/pet/{petId}', response_model=Pet, supported_status_codes=[200],
                        params={'name': name, 'status': status}, **kwargs)


async def aupdatePetWithForm(petId: int, *, name: str = None, status: str = None, **kwargs: dict) -> Pet:
    """Updates a pet resource based on the form data."""
    return await request_async(method='post', path=f'/pet/{petId}', response_model=Pet, supported_status_codes=[200],
                               params={'name': name, 'status': status}, **kwargs)


def uploadFile(petId: int, *, additionalMetadata: str = None, **kwargs: dict) -> ApiResponse:
    """Upload image of the pet."""
    return request_sync(method='post', path=f'/pet/{petId}/uploadImage', response_model=ApiResponse,
                        supported_status_codes=[200], params={'additionalMetadata': additionalMetadata}, **kwargs)


async def auploadFile(petId: int, *, additionalMetadata: str = None, **kwargs: dict) -> ApiResponse:
    """Upload image of the pet."""
    return await request_async(method='post', path=f'/pet/{petId}/uploadImage', response_model=ApiResponse,
                               supported_status_codes=[200], params={'additionalMetadata': additionalMetadata},
                               **kwargs)


def getInventory(**kwargs: dict) -> dict[str, Any]:
    """Returns a map of status codes to quantities."""
    return request_sync(method='get', path='/store/inventory', response_model=dict[str, Any],
                        supported_status_codes=[200], **kwargs)


async def agetInventory(**kwargs: dict) -> dict[str, Any]:
    """Returns a map of status codes to quantities."""
    return await request_async(method='get', path='/store/inventory', response_model=dict[str, Any],
                               supported_status_codes=[200], **kwargs)


def placeOrder(*, body: Order = None, **kwargs: dict) -> Order:
    """Place a new order in the store."""
    return request_sync(method='post', path='/store/order', response_model=Order, supported_status_codes=[200],
                        json=body.model_dump(), **kwargs)


async def aplaceOrder(*, body: Order = None, **kwargs: dict) -> Order:
    """Place a new order in the store."""
    return await request_async(method='post', path='/store/order', response_model=Order, supported_status_codes=[200],
                               json=body.model_dump(), **kwargs)


def deleteOrder(orderId: int, **kwargs: dict) -> Any:
    """For valid response try integer IDs with value < 1000. Anything above 1000 or non-integers will generate API errors."""
    return request_sync(method='delete', path=f'/store/order/{orderId}', response_model=None,
                        supported_status_codes=None, **kwargs)


async def adeleteOrder(orderId: int, **kwargs: dict) -> Any:
    """For valid response try integer IDs with value < 1000. Anything above 1000 or non-integers will generate API errors."""
    return await request_async(method='delete', path=f'/store/order/{orderId}', response_model=None,
                               supported_status_codes=None, **kwargs)


def getOrderById(orderId: int, **kwargs: dict) -> Order:
    """For valid response try integer IDs with value <= 5 or > 10. Other values will generate exceptions."""
    return request_sync(method='get', path=f'/store/order/{orderId}', response_model=Order,
                        supported_status_codes=[200], **kwargs)


async def agetOrderById(orderId: int, **kwargs: dict) -> Order:
    """For valid response try integer IDs with value <= 5 or > 10. Other values will generate exceptions."""
    return await request_async(method='get', path=f'/store/order/{orderId}', response_model=Order,
                               supported_status_codes=[200], **kwargs)


def createUser(*, body: User = None, **kwargs: dict) -> User:
    """This can only be done by the logged in user."""
    return request_sync(method='post', path='/user', response_model=User, supported_status_codes=[200],
                        json=body.model_dump(), **kwargs)


async def acreateUser(*, body: User = None, **kwargs: dict) -> User:
    """This can only be done by the logged in user."""
    return await request_async(method='post', path='/user', response_model=User, supported_status_codes=[200],
                               json=body.model_dump(), **kwargs)


def createUsersWithListInput(*, body: list[User] = None, **kwargs: dict) -> User:
    """Creates list of users with given input array."""
    return request_sync(method='post', path='/user/createWithList', response_model=User, supported_status_codes=[200],
                        json=body, **kwargs)


async def acreateUsersWithListInput(*, body: list[User] = None, **kwargs: dict) -> User:
    """Creates list of users with given input array."""
    return await request_async(method='post', path='/user/createWithList', response_model=User,
                               supported_status_codes=[200], json=body, **kwargs)


def loginUser(*, username: str = None, password: str = None, **kwargs: dict) -> str:
    """Log into the system."""
    return request_sync(method='get', path='/user/login', response_model=str, supported_status_codes=[200],
                        params={'username': username, 'password': password}, **kwargs)


async def aloginUser(*, username: str = None, password: str = None, **kwargs: dict) -> str:
    """Log into the system."""
    return await request_async(method='get', path='/user/login', response_model=str, supported_status_codes=[200],
                               params={'username': username, 'password': password}, **kwargs)


def logoutUser(**kwargs: dict) -> Any:
    """Log user out of the system."""
    return request_sync(method='get', path='/user/logout', response_model=None, supported_status_codes=None, **kwargs)


async def alogoutUser(**kwargs: dict) -> Any:
    """Log user out of the system."""
    return await request_async(method='get', path='/user/logout', response_model=None, supported_status_codes=None,
                               **kwargs)


def deleteUser(username: str, **kwargs: dict) -> Any:
    """This can only be done by the logged in user."""
    return request_sync(method='delete', path=f'/user/{username}', response_model=None, supported_status_codes=None,
                        **kwargs)


async def adeleteUser(username: str, **kwargs: dict) -> Any:
    """This can only be done by the logged in user."""
    return await request_async(method='delete', path=f'/user/{username}', response_model=None,
                               supported_status_codes=None, **kwargs)


def getUserByName(username: str, **kwargs: dict) -> User:
    """Get user detail based on username."""
    return request_sync(method='get', path=f'/user/{username}', response_model=User, supported_status_codes=[200],
                        **kwargs)


async def agetUserByName(username: str, **kwargs: dict) -> User:
    """Get user detail based on username."""
    return await request_async(method='get', path=f'/user/{username}', response_model=User,
                               supported_status_codes=[200], **kwargs)


def updateUser(username: str, *, body: User = None, **kwargs: dict) -> Any:
    """This can only be done by the logged in user."""
    return request_sync(method='put', path=f'/user/{username}', response_model=None, supported_status_codes=None,
                        json=body.model_dump(), **kwargs)


async def aupdateUser(username: str, *, body: User = None, **kwargs: dict) -> Any:
    """This can only be done by the logged in user."""
    return await request_async(method='put', path=f'/user/{username}', response_model=None, supported_status_codes=None,
                               json=body.model_dump(), **kwargs)
