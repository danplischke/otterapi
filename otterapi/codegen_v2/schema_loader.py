"""Schema loading utilities for OpenAPI documents.

This module provides utilities for loading OpenAPI schemas from various sources
including URLs and local file paths.
"""

import json
from pathlib import Path

import httpx
from pydantic import TypeAdapter

from otterapi.openapi import UniversalOpenAPI
from otterapi.openapi.v3_2.v3_2 import OpenAPI as OpenAPIv3


class SchemaLoader:
    """Loads OpenAPI schemas from URLs or file paths.
    
    This class provides a unified interface for loading OpenAPI schemas
    from different sources (HTTP URLs or local files) and validates them
    against the OpenAPI v3 specification.
    
    Example:
        >>> loader = SchemaLoader()
        >>> schema = loader.load('https://api.example.com/openapi.json')
        >>> # or
        >>> schema = loader.load('/path/to/openapi.yaml')
    """
    
    def __init__(self, http_client: httpx.Client | None = None):
        """Initialize the schema loader.
        
        Args:
            http_client: Optional HTTP client to use for URL requests.
                        If not provided, a default client will be created.
        """
        self._http_client = http_client
    
    def load(self, source: str) -> OpenAPIv3:
        """Load and validate an OpenAPI schema from a URL or file path.
        
        Args:
            source: URL or file path to the OpenAPI schema.
        
        Returns:
            Validated OpenAPIv3 object.
        
        Raises:
            httpx.HTTPError: If loading from URL fails.
            FileNotFoundError: If the file path doesn't exist.
            json.JSONDecodeError: If the content is not valid JSON.
            pydantic.ValidationError: If the schema is not valid OpenAPI v3.
        """
        if self._is_url(source):
            content = self._load_from_url(source)
        else:
            content = self._load_from_file(source)
        
        return self._validate_schema(content)
    
    def _is_url(self, text: str) -> bool:
        """Check if a string is a URL.
        
        Args:
            text: String to check.
        
        Returns:
            True if the string appears to be a URL, False otherwise.
        """
        from urllib.parse import urlparse
        
        try:
            result = urlparse(text)
            return all([result.scheme, result.netloc])
        except ValueError:
            return False
    
    def _load_from_url(self, url: str) -> dict:
        """Load schema content from a URL.
        
        Args:
            url: URL to load from.
        
        Returns:
            Parsed JSON content as a dictionary.
        
        Raises:
            httpx.HTTPError: If the request fails.
        """
        if self._http_client:
            response = self._http_client.get(url)
        else:
            response = httpx.get(url)
        
        response.raise_for_status()
        return response.json()
    
    def _load_from_file(self, file_path: str) -> dict:
        """Load schema content from a file.
        
        Args:
            file_path: Path to the file.
        
        Returns:
            Parsed JSON content as a dictionary.
        
        Raises:
            FileNotFoundError: If the file doesn't exist.
            json.JSONDecodeError: If the file is not valid JSON.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Schema file not found: {file_path}")
        
        with open(path, 'rb') as f:
            return json.loads(f.read())
    
    def _validate_schema(self, content: dict) -> OpenAPIv3:
        """Validate and parse schema content as OpenAPI v3.
        
        Args:
            content: Dictionary containing the schema data.
        
        Returns:
            Validated OpenAPIv3 object.
        
        Raises:
            pydantic.ValidationError: If the schema is not valid OpenAPI v3.
        """
        schema: UniversalOpenAPI = TypeAdapter(UniversalOpenAPI).validate_python(content)
        if isinstance(schema.root, OpenAPIv3):
            return schema.root
        else:
            schema = schema.root
            # Upgrade to V3_2 if schema is an older version
            while not isinstance(schema, OpenAPIv3):
                schema, warnings = schema.upgrade()
            return schema

