from pydantic.main import BaseModel
from pydantic.fields import Field

__all__ = ('Item',)


class Item(BaseModel):
    id: int
    name: str
