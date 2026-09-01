HTTP_METHODS: tuple[str, ...] = (
    'get',
    'post',
    'put',
    'delete',
    'patch',
    'head',
    'options',
    'trace',
)


class MediaType:
    JSON = 'application/json'
    TEXT_JSON = 'text/json'
    OCTET_STREAM = 'application/octet-stream'
    FORM_URLENCODED = 'application/x-www-form-urlencoded'
    MULTIPART = 'multipart/form-data'


# Media types a spec may spell with parameters (``application/json;charset=utf-8``)
# and in any case (RFC 9110 8.3.1: type, subtype and parameter names are all
# case-insensitive). Everything that classifies a spec's content-type key goes
# through these helpers rather than comparing the raw string.
JSON_MEDIA_TYPES: frozenset[str] = frozenset({MediaType.JSON, MediaType.TEXT_JSON})


def base_media_type(content_type: str) -> str:
    """Strip media-type parameters and case from a content-type key.

    ``'Application/JSON; charset=utf-8'`` -> ``'application/json'``.
    """
    return content_type.split(';', 1)[0].strip().lower()


def is_json_media_type(content_type: str) -> bool:
    """Whether *content_type* names a JSON media type, parameters and all."""
    base = base_media_type(content_type)
    return base in JSON_MEDIA_TYPES or base.endswith('+json')
