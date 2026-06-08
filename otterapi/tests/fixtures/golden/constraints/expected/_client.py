from pydantic import RootModel, TypeAdapter
from httpx import AsyncClient, Client, Response
from typing import Any, TypeVar
from types import UnionType

T = TypeVar('T')
__all__ = (
    'BaseConstraintsAPIClient',
    'BaseAPIError',
    'ClientError',
    'ServerError',
    'BadRequestError',
    'UnauthorizedError',
    'ForbiddenError',
    'NotFoundError',
    'ConflictError',
    'UnprocessableEntityError',
    'RateLimitError',
    'InternalServerError',
    'BadGatewayError',
    'ServiceUnavailableError',
    'GatewayTimeoutError',
)


class BaseAPIError(Exception):
    """Exception raised when an API request fails with an error response.

    This exception provides detailed error information from the API response,
    including the HTTP status code, error message, and full response body.

    Attributes:
        status_code: The HTTP status code of the response.
        response: The httpx Response object.
        detail: Parsed error detail from the response body (if available).
        body: Raw response body text.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int,
        response: Response,
        detail: Any | None = None,
        body: str = '',
    ) -> None:
        self.status_code = status_code
        self.response = response
        self.detail = detail
        self.body = body
        super().__init__(message)

    @classmethod
    def from_response(cls, response: Response) -> 'BaseAPIError':
        status_code = response.status_code
        body = response.text
        detail = None
        try:
            json_body = response.json()
            if isinstance(json_body, dict):
                detail = json_body.get('detail', json_body)
            else:
                detail = json_body
        except Exception:
            detail = body if body else None
        message = f'HTTP {status_code} Error'
        if detail:
            if isinstance(detail, list):
                error_msgs = []
                for err in detail:
                    if isinstance(err, dict):
                        loc = err.get('loc', [])
                        msg = err.get('msg', str(err))
                        loc_str = (
                            ' -> '.join((str(x) for x in loc)) if loc else 'unknown'
                        )
                        error_msgs.append(f'  - {loc_str}: {msg}')
                    else:
                        error_msgs.append(f'  - {err}')
                if error_msgs:
                    message = f'HTTP {status_code} Validation Error:\n' + '\n'.join(
                        error_msgs
                    )
            elif isinstance(detail, str):
                message = f'HTTP {status_code} Error: {detail}'
            else:
                message = f'HTTP {status_code} Error: {detail}'
        cls = _resolve_error_class(status_code, cls)
        return cls(
            message,
            status_code=status_code,
            response=response,
            detail=detail,
            body=body,
        )

    def __str__(self) -> str:
        return self.args[0]

    def __repr__(self) -> str:
        return f'BaseAPIError(status_code={self.status_code}, detail={self.detail!r})'


class ClientError(BaseAPIError):
    """Base class for 4xx HTTP errors."""

    pass


class ServerError(BaseAPIError):
    """Base class for 5xx HTTP errors."""

    pass


class BadRequestError(ClientError):
    """Raised on HTTP 400."""

    pass


class UnauthorizedError(ClientError):
    """Raised on HTTP 401."""

    pass


class ForbiddenError(ClientError):
    """Raised on HTTP 403."""

    pass


class NotFoundError(ClientError):
    """Raised on HTTP 404."""

    pass


class ConflictError(ClientError):
    """Raised on HTTP 409."""

    pass


class UnprocessableEntityError(ClientError):
    """Raised on HTTP 422."""

    pass


class RateLimitError(ClientError):
    """Raised on HTTP 429."""

    pass


class InternalServerError(ServerError):
    """Raised on HTTP 500."""

    pass


class BadGatewayError(ServerError):
    """Raised on HTTP 502."""

    pass


class ServiceUnavailableError(ServerError):
    """Raised on HTTP 503."""

    pass


class GatewayTimeoutError(ServerError):
    """Raised on HTTP 504."""

    pass


_STATUS_ERROR_MAP: dict[int, type[BaseAPIError]] = {
    400: BadRequestError,
    401: UnauthorizedError,
    403: ForbiddenError,
    404: NotFoundError,
    409: ConflictError,
    422: UnprocessableEntityError,
    429: RateLimitError,
    500: InternalServerError,
    502: BadGatewayError,
    503: ServiceUnavailableError,
    504: GatewayTimeoutError,
}


def _resolve_error_class(
    status_code: int, default: type[BaseAPIError]
) -> type[BaseAPIError]:
    """Pick the most specific ``BaseAPIError`` subclass for a status code.

    Falls through ``_STATUS_ERROR_MAP`` -> ``ClientError`` (4xx) ->
    ``ServerError`` (5xx) -> ``default`` (typically ``BaseAPIError``).
    """
    explicit = _STATUS_ERROR_MAP.get(status_code)
    if explicit is not None:
        return explicit
    if 400 <= status_code < 500:
        return ClientError
    if 500 <= status_code < 600:
        return ServerError
    return default


APIError = BaseAPIError


class BaseConstraintsAPIClient:
    """Base HTTP client with request infrastructure.

    This class is regenerated on each code generation run.
    To customize, subclass this in client.py.

    Endpoint implementations are in the module files (e.g., pet.py, store.py).

    Args:
        base_url: Base URL for API requests. Default: https://example.test
        timeout: Request timeout in seconds. Default: 30.0
        headers: Default headers to include in all requests.
        http_client: Custom httpx.Client for sync requests.
        async_http_client: Custom httpx.AsyncClient for async requests.
    """

    def __init__(
        self,
        base_url: str = 'https://example.test',
        timeout: float = 30.0,
        headers: dict[str, str] | None = None,
        http_client: Client | None = None,
        async_http_client: AsyncClient | None = None,
    ) -> None:
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.headers = headers or {}
        self._client = http_client
        self._async_client = async_http_client

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict | None = None,
        headers: dict | None = None,
        json: Any | None = None,
        data: Any | None = None,
        files: Any | None = None,
        content: Any | None = None,
        timeout: float | None = None,
    ) -> Response:
        filtered_params = (
            {k: v for k, v in params.items() if v is not None} if params else None
        )
        if self._client:
            response = self._client.request(
                method,
                f'{self.base_url}{path}',
                params=filtered_params,
                headers={**self.headers, **(headers or {})},
                json=json,
                data=data,
                files=files,
                content=content,
                timeout=timeout if timeout is not None else self.timeout,
            )
        else:
            with Client() as client:
                response = client.request(
                    method,
                    f'{self.base_url}{path}',
                    params=filtered_params,
                    headers={**self.headers, **(headers or {})},
                    json=json,
                    data=data,
                    files=files,
                    content=content,
                    timeout=timeout if timeout is not None else self.timeout,
                )
        if response.is_error:
            raise APIError.from_response(response)
        return response

    async def _request_async(
        self,
        method: str,
        path: str,
        *,
        params: dict | None = None,
        headers: dict | None = None,
        json: Any | None = None,
        data: Any | None = None,
        files: Any | None = None,
        content: Any | None = None,
        timeout: float | None = None,
    ) -> Response:
        filtered_params = (
            {k: v for k, v in params.items() if v is not None} if params else None
        )
        if self._async_client:
            response = await self._async_client.request(
                method,
                f'{self.base_url}{path}',
                params=filtered_params,
                headers={**self.headers, **(headers or {})},
                json=json,
                data=data,
                files=files,
                content=content,
                timeout=timeout if timeout is not None else self.timeout,
            )
        else:
            async with AsyncClient() as client:
                response = await client.request(
                    method,
                    f'{self.base_url}{path}',
                    params=filtered_params,
                    headers={**self.headers, **(headers or {})},
                    json=json,
                    data=data,
                    files=files,
                    content=content,
                    timeout=timeout if timeout is not None else self.timeout,
                )
        if response.is_error:
            raise APIError.from_response(response)
        return response

    def _request_json(
        self,
        method: str,
        path: str,
        *,
        params: dict | None = None,
        headers: dict | None = None,
        json: Any | None = None,
        data: Any | None = None,
        files: Any | None = None,
        content: Any | None = None,
        timeout: float | None = None,
    ) -> Any:
        response = self._request(
            method,
            path,
            params=params,
            headers=headers,
            json=json,
            data=data,
            files=files,
            content=content,
            timeout=timeout,
        )
        return response.json()

    async def _request_json_async(
        self,
        method: str,
        path: str,
        *,
        params: dict | None = None,
        headers: dict | None = None,
        json: Any | None = None,
        data: Any | None = None,
        files: Any | None = None,
        content: Any | None = None,
        timeout: float | None = None,
    ) -> Any:
        response = await self._request_async(
            method,
            path,
            params=params,
            headers=headers,
            json=json,
            data=data,
            files=files,
            content=content,
            timeout=timeout,
        )
        return response.json()

    def _validate_response(self, response: Response, validated: Any) -> None:
        """Validate the parsed response before returning.

        Override this method to add custom validation logic, such as
        checking for API-level errors in wrapper objects.

        This hook is called after Pydantic validation but before unwrapping
        RootModel responses.

        Args:
            response: The raw httpx Response object.
            validated: The parsed and validated Pydantic model.

        Raises:
            APIError: If validation fails (or any other exception).

        Example:
            def _validate_response(self, response, validated):
                # Check for API-level errors in wrapper objects
                if hasattr(validated, 'error') and validated.error is not None:
                    raise APIError(
                        status_code=response.status_code,
                        response=response,
                        detail=validated.error,
                    )
        """
        pass

    def _parse_response(self, response: Response, response_type: type | UnionType) -> T:
        data = response.json()
        validated = TypeAdapter(response_type).validate_python(data)
        self._validate_response(response, validated)
        if isinstance(validated, RootModel):
            return validated.root
        return validated

    async def _parse_response_async(
        self, response: Response, response_type: type | UnionType
    ) -> T:
        data = response.json()
        validated = TypeAdapter(response_type).validate_python(data)
        self._validate_response(response, validated)
        if isinstance(validated, RootModel):
            return validated.root
        return validated
