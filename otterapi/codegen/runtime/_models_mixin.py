# ``str.translate`` rather than ``html.escape``: this source is spliced into the
# generated models module, which has no ``html`` import to lean on, and a schema
# named ``html`` or ``escape`` would shadow one.  Quotes are escaped too, so a
# value is safe in an attribute as well as in element content.
_HTML_ESCAPES = str.maketrans(
    {'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#x27;'}
)


def _html_escape(value):
    return str(value).translate(_HTML_ESCAPES)


def _html_val(v):
    if v is None:
        return '<em style="color:#aaa">None</em>'
    if isinstance(v, list):
        n = len(v)
        # Bracketed on purpose: this module is re-unparsed into the
        # generated package, and ``ast.unparse`` parenthesizes a lone
        # generator argument (UP034 in the user's tree).
        preview = ', '.join([str(x) for x in v[:3]])
        suffix = f', … ({n})' if n > 3 else ''
        return _html_escape(f'[{preview}{suffix}]')
    if hasattr(v, '_repr_html_'):
        # A nested model escapes its own values; its markup is ours to embed.
        return v._repr_html_()
    return _html_escape(v)


class _HtmlReprMixin:
    """Mixin that renders a Pydantic model as an HTML table in Jupyter notebooks.

    Every value is HTML-escaped: field values come straight from API
    responses, and a notebook renders ``_repr_html_`` output as live markup.
    """

    def _repr_html_(self) -> str:
        fields = getattr(self.__class__, 'model_fields', {})
        rows = []
        for i, name in enumerate(fields):
            val = getattr(self, name, None)
            bg = ' style="background:#f8f8f8"' if i % 2 == 0 else ''
            rows.append(
                f'<tr{bg}><td style="font-weight:bold;padding:2px 8px;'
                f'white-space:nowrap;color:#444">{_html_escape(name)}</td>'
                f'<td style="padding:2px 8px">{_html_val(val)}</td></tr>'
            )
        class_name = _html_escape(type(self).__name__)
        inner = ''.join(rows)
        return (
            f'<details open><summary style="font-weight:bold;cursor:pointer;'
            f'font-family:monospace">{class_name}</summary>'
            f'<table style="border-collapse:collapse;font-size:13px;'
            f'font-family:monospace">{inner}</table></details>'
        )
