"""API Client.

This file is generated once and will NOT be overwritten on regeneration.
You can safely customize this file to add authentication, logging,
error handling, or other client-specific functionality.
"""

from ._client import BasePaginatedAPIClient, BaseAPIError


class PaginatedAPIClient(BasePaginatedAPIClient):
    """API client with customizable configuration.

    This class inherits from the generated BasePaginatedAPIClient and can be
    customized without being overwritten on code regeneration.

    Example:
        >>> client = PaginatedAPIClient()
        >>> # Use default base URL from OpenAPI spec

        >>> client = PaginatedAPIClient(base_url="https://staging.api.example.com")
        >>> # Override base URL

        >>> client = PaginatedAPIClient(timeout=60.0, headers={"Authorization": "Bearer token"})
        >>> # Custom timeout and headers

        >>> import httpx
        >>> with httpx.Client() as http_client:
        ...     client = PaginatedAPIClient(http_client=http_client)
        ...     # Use custom HTTP client (useful for testing/mocking)
    """

    pass

    # Add custom methods or override base class methods below.
    #
    # Example - adding authentication:
    #
    # def __init__(self, api_key: str | None = None, **kwargs):
    #     super().__init__(**kwargs)
    #     if api_key:
    #         self.headers["Authorization"] = f"Bearer {api_key}"
    #
    # Example - validating API responses (e.g., checking for errors in wrapper objects):
    #
    # def _validate_response(self, response, validated):
    #     """Check for API-level errors in response wrapper."""
    #     if hasattr(validated, 'error') and validated.error is not None:
    #         raise APIError(
    #             status_code=response.status_code,
    #             response=response,
    #             detail=validated.error,
    #         )


class APIError(BaseAPIError):
    """Customizable API error class.

    Override the from_response() classmethod to customize how error details
    are extracted from API responses. The default implementation looks for
    a 'detail' key in the JSON response body.

    Example - customizing error detail extraction:

        class APIError(BaseAPIError):
            @classmethod
            def from_response(cls, response):
                status_code = response.status_code
                body = response.text
                detail = None
                try:
                    json_body = response.json()
                    # Custom keys for this API
                    detail = (
                        json_body.get('error')
                        or json_body.get('message')
                        or json_body.get('detail')
                        or json_body
                    )
                except Exception:
                    detail = body if body else None
                return cls(
                    f'HTTP {status_code} Error: {detail}',
                    status_code=status_code,
                    response=response,
                    detail=detail,
                    body=body,
                )
    """

    pass


# Convenience aliases for shorter imports
Client = PaginatedAPIClient
Error = APIError

__all__ = ["PaginatedAPIClient", "Client", "APIError", "Error"]
