from pydantic.main import BaseModel
from pydantic.fields import Field

__all__ = ('Row',)


class Row(BaseModel):
    id: int
    value: float
