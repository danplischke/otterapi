from builtins import list
from datetime import datetime
from pydantic.fields import Field
from pydantic.main import BaseModel
__all__ = ('ApiResponse', 'Category', 'Order', 'Pet', 'Tag', 'User')

class Category(BaseModel):
    id: int = Field(default=None)
    name: str = Field(default=None)

class Tag(BaseModel):
    id: int = Field(default=None)
    name: str = Field(default=None)

class Pet(BaseModel):
    id: int = Field(default=None)
    name: str = Field(default=None)
    category: Category = Field(default=None)
    photoUrls: list[str] = Field(default=None)
    tags: list[Tag] = Field(default=None)
    status: str = Field(default=None)

class ApiResponse(BaseModel):
    code: int = Field(default=None)
    type: str = Field(default=None)
    message: str = Field(default=None)

class Order(BaseModel):
    id: int = Field(default=None)
    petId: int = Field(default=None)
    quantity: int = Field(default=None)
    shipDate: datetime = Field(default=None)
    status: str = Field(default=None)
    complete: bool = Field(default=None)

class User(BaseModel):
    id: int = Field(default=None)
    username: str = Field(default=None)
    firstName: str = Field(default=None)
    lastName: str = Field(default=None)
    email: str = Field(default=None)
    password: str = Field(default=None)
    phone: str = Field(default=None)
    userStatus: int = Field(default=None)