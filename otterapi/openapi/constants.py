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
