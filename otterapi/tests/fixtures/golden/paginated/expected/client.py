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

    The client holds a persistent connection pool and retries transient errors
    (429/5xx) automatically.  Use it as a context manager so the pool is
    released cleanly::

        with PaginatedAPIClient() as client:
            data = list_genes(client=client)

    Async context manager::

        async with PaginatedAPIClient() as client:
            data = await async_list_genes(client=client)

    Calling async endpoints from a Jupyter notebook or plain script::

        from ._client import run_sync
        gene = run_sync(async_get_gene(symbol="BRCA1"))

    Fan-out over many targets in parallel::

        from ._client import run_concurrently
        genes = run_concurrently(
            [async_get_gene(symbol=g) for g in gene_list],
            concurrency=10,
        )

    Other examples::

        >>> client = PaginatedAPIClient(base_url="https://staging.api.example.com")
        >>> client = PaginatedAPIClient(max_retries=0)  # disable retry
        >>> client = PaginatedAPIClient(timeout=60.0, headers={"X-Request-ID": "abc"})
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

__all__ = ["APIError", "Client", "Error", "PaginatedAPIClient"]
