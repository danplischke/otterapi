from pydantic.main import BaseModel
from pydantic.fields import Field

__all__ = ('Cat', 'Dog')


class Dog(BaseModel):
    kind: str
    bark: str


class Cat(BaseModel):
    kind: str
    meow: str
