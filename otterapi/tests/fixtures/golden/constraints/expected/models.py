from pydantic.main import BaseModel
from pydantic.fields import Field

__all__ = ('User',)


class User(BaseModel):
    model_config = {'extra': 'forbid'}
    id: int = Field(ge=1.0)
    username: str = Field(min_length=3, max_length=20, pattern='^[a-z][a-z0-9_]*$')
    age: int = Field(ge=0.0, le=150.0)
    tags: list[str] = Field(default=None, max_length=10)
