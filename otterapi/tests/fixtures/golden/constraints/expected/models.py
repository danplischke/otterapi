from pydantic.main import BaseModel
from pydantic.fields import Field

__all__ = ('User',)


def _html_val(v):
    if v is None:
        return '<em style="color:#aaa">None</em>'
    if isinstance(v, list):
        n = len(v)
        preview = ', '.join((str(x) for x in v[:3]))
        suffix = f', … ({n})' if n > 3 else ''
        return f'[{preview}{suffix}]'
    if hasattr(v, '_repr_html_'):
        return v._repr_html_()
    return str(v)


class _HtmlReprMixin:
    """Mixin that renders a Pydantic model as an HTML table in Jupyter notebooks."""

    def _repr_html_(self) -> str:
        fields = getattr(self.__class__, 'model_fields', {})
        rows = []
        for i, name in enumerate(fields):
            val = getattr(self, name, None)
            bg = ' style="background:#f8f8f8"' if i % 2 == 0 else ''
            rows.append(
                f'<tr{bg}><td style="font-weight:bold;padding:2px 8px;white-space:nowrap;color:#444">{name}</td><td style="padding:2px 8px">{_html_val(val)}</td></tr>'
            )
        class_name = type(self).__name__
        inner = ''.join(rows)
        return f'<details open><summary style="font-weight:bold;cursor:pointer;font-family:monospace">{class_name}</summary><table style="border-collapse:collapse;font-size:13px;font-family:monospace">{inner}</table></details>'


class User(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    id: int = Field(ge=1.0)
    username: str = Field(min_length=3, max_length=20, pattern='^[a-z][a-z0-9_]*$')
    age: int = Field(ge=0.0, le=150.0)
    tags: list[str] = Field(default=None, max_length=10)
