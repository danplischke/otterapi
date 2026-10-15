"""Schema loading and resolution utilities for OpenAPI documents.

This module provides utilities for:
- Loading OpenAPI schemas from various sources (URLs, local files, YAML/JSON)
- Automatic version detection and upgrading of older OpenAPI/Swagger specifications
- Resolving $ref references and managing schema lookups in OpenAPI documents
- Bounding what an untrusted specification can cost to load: download size and
  time, redirects, and YAML alias expansion
"""

import ipaddress
import json
import logging
import socket
import time
from pathlib import Path
from typing import Any, cast
from urllib.parse import urljoin, urlparse

import httpx
import yaml
from pydantic import TypeAdapter, ValidationError

from otterapi.exceptions import (
    SchemaLoadError,
    SchemaReferenceError,
    SchemaValidationError,
)
from otterapi.openapi import UniversalOpenAPI
from otterapi.openapi.v3_2 import Reference, Schema
from otterapi.openapi.v3_2.v3_2 import OpenAPI as OpenAPIv3

logger = logging.getLogger(__name__)

__all__ = [
    'SchemaLoader',
    'SchemaResolver',
]


# =============================================================================
# Limits for untrusted specifications
# =============================================================================

# Largest response body accepted from a URL, measured after decompression.
MAX_SPEC_BYTES = 50 * 1024 * 1024

# Wall-clock budget, in seconds, for downloading one spec, redirects included.
MAX_FETCH_SECONDS = 60.0

# Redirects followed for one download before giving up.
MAX_REDIRECTS = 5

# A YAML document is refused when its aliases expand it to more than
# max(MAX_ALIAS_AMPLIFICATION * distinct nodes, MIN_AMPLIFICATION_NODES) nodes.
# The floor keeps ordinary anchor and merge-key use far away from the limit.
MAX_ALIAS_AMPLIFICATION = 100
MIN_AMPLIFICATION_NODES = 100_000

# Per-read network timeout of the client the loader creates for itself. The
# overall budget above is enforced separately, as the body arrives.
_READ_TIMEOUT_SECONDS = 30.0

# The only content codings requested or accepted. One gzip or deflate stream
# inflates at most about 1000:1 per network read, which the size check absorbs;
# stacked codings, brotli and zstd can inflate a single read without bound.
_CONTENT_CODINGS = ('gzip', 'deflate')

# YAML node counts saturate here, so the alias walk stays cheap and its figures
# printable however deeply aliases nest.
_NODE_COUNT_CEILING = 1 << 62

_YAML_CONTAINERS = (dict, list, tuple, set, frozenset)

_NAT64_PREFIX = ipaddress.IPv6Network('64:ff9b::/96')


def _describe_bytes(count: int) -> str:
    """Render a byte count for error messages."""
    if count >= 1024 * 1024:
        return f'{count / (1024 * 1024):g} MiB'
    return f'{count} bytes'


def _resolve_addresses(host: str) -> list[str]:
    """Resolve ``host`` to every IP address it maps to.

    A function of its own so tests can swap DNS for a lookup table.
    """
    infos = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    return list(dict.fromkeys(str(info[4][0]) for info in infos))


def _is_non_public(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """Whether ``ip`` is anything other than an ordinary public address.

    Covers loopback, private (RFC 1918 and unique-local), link-local (which
    includes the cloud metadata address 169.254.169.254), site-local,
    multicast, reserved, unspecified and shared (carrier-grade NAT) space. An
    IPv4 address wrapped in IPv6 (mapped, 6to4 or NAT64 form) is judged by the
    address inside.
    """
    if isinstance(ip, ipaddress.IPv6Address):
        embedded = ip.ipv4_mapped
        if embedded is None:
            embedded = ip.sixtofour
        if embedded is None and ip in _NAT64_PREFIX:
            embedded = ipaddress.IPv4Address(int(ip) & 0xFFFFFFFF)
        if embedded is not None:
            return _is_non_public(embedded)
        if ip.is_site_local:
            return True
    return (
        ip.is_loopback
        or ip.is_private
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
        or not ip.is_global
    )


def _host_addresses(host: str) -> list[ipaddress.IPv4Address | ipaddress.IPv6Address]:
    """Return the addresses ``host`` stands for.

    An IP literal stands for itself; any other name is resolved, which also
    covers the numeric spellings (``2130706433``, ``127.1``) that the system
    resolver, and therefore the connection, reads as IPv4 addresses.

    Raises:
        OSError: If the name does not resolve.
    """
    try:
        return [ipaddress.ip_address(host)]
    except ValueError:
        return [ipaddress.ip_address(address) for address in _resolve_addresses(host)]


def _is_internal_host(host: str) -> bool:
    """Whether every address ``host`` stands for is non-public."""
    try:
        addresses = _host_addresses(host)
    except (OSError, ValueError):
        return False
    return bool(addresses) and all(_is_non_public(a) for a in addresses)


def _vet_redirect(
    source: str, current: httpx.URL, location: str, origin_host: str
) -> httpx.URL:
    """Resolve a ``Location`` header against ``current`` and check it may be followed.

    Only http(s) URLs with a host are followed. A target on a non-public
    address (loopback, private, link-local, ...) is refused unless the spec was
    itself served from one (``origin_host``), so a dev server on localhost can
    still redirect within localhost. The target is parsed by httpx, the same
    parser that will make the request, so the host checked is the host fetched.
    Names are resolved here, which makes the check best effort: DNS may answer
    differently when the connection is actually made. A name that cannot be
    resolved from here is followed, not refused: behind a proxy it is the
    proxy that resolves names, and without one a name nobody can resolve cannot
    be connected to either, so refusing would only break ordinary redirects on
    networks where DNS happens at the proxy.

    Args:
        source: The URL the user asked for, used in error messages.
        current: The URL that answered with the redirect.
        location: The ``Location`` header value.
        origin_host: Host of ``source``.

    Returns:
        The absolute URL to fetch next.

    Raises:
        SchemaLoadError: If the redirect must not be followed.
        httpx.InvalidURL: If ``location`` cannot be parsed as a URL.
    """
    target = current.join(location)
    host = target.raw_host.decode('ascii')
    if target.scheme not in ('http', 'https') or not host:
        raise SchemaLoadError(
            source,
            cause=ValueError(
                f"redirect to '{target}' refused: only http(s) URLs are followed"
            ),
        )

    try:
        addresses = _host_addresses(host)
    except OSError as e:
        logger.warning(
            "Could not resolve '%s' to check the redirect to '%s' (%s); following it.",
            host,
            target,
            e,
        )
        return target
    except ValueError as e:
        raise SchemaLoadError(
            source,
            cause=ValueError(
                f"redirect to '{target}' refused: {host} resolved to an "
                f'unusable address ({e})'
            ),
        )

    blocked = next((str(a) for a in addresses if _is_non_public(a)), None)
    if blocked is not None and not _is_internal_host(origin_host):
        points_to = host if blocked == host else f'{host} resolves to {blocked}'
        raise SchemaLoadError(
            source,
            cause=ValueError(
                f"redirect to '{target}' refused: {points_to}, "
                'which is not a public address'
            ),
        )
    return target


def _check_content_coding(source: str, response: httpx.Response) -> None:
    """Refuse content codings other than a single gzip or deflate stream."""
    header = response.headers.get('content-encoding', '')
    codings = [c.strip().lower() for c in header.split(',')]
    codings = [c for c in codings if c and c != 'identity']
    if len(codings) > 1 or (codings and codings[0] not in _CONTENT_CODINGS):
        raise SchemaLoadError(
            source,
            cause=ValueError(
                f'unsupported Content-Encoding {header!r} '
                f'(only {" or ".join(_CONTENT_CODINGS)} are accepted)'
            ),
        )


def _measure_aliasing(document: Any) -> tuple[int, int]:
    """Measure how far YAML aliases inflate a parsed document.

    ``yaml.safe_load`` shares aliased objects instead of copying them, so a few
    hundred bytes can describe a tree with billions of nodes that validation
    later walks reference by reference. This walk is iterative, so nesting
    depth is no concern, and it visits every distinct container once, so its
    cost follows the size of the text however heavily the document is aliased.

    Returns:
        ``(expanded, unique)``: the node count a consumer that follows every
        reference separately would see, and the count of nodes that exist
        (each container once, plus its scalar children). ``expanded`` saturates
        at ``_NODE_COUNT_CEILING``.

    Raises:
        ValueError: If a container contains itself through an alias.
    """
    if not isinstance(document, _YAML_CONTAINERS):
        return 1, 1

    expanded: dict[int, int] = {}
    in_progress: set[int] = set()
    unique = 0
    # (container, False) enters a container, (container, True) leaves it once
    # everything below it has been measured.
    stack: list[tuple[Any, bool]] = [(document, False)]
    while stack:
        node, leaving = stack.pop()
        node_id = id(node)
        children = node.values() if isinstance(node, dict) else node
        if leaving:
            total = 1
            for child in children:
                if isinstance(child, _YAML_CONTAINERS):
                    total += expanded[id(child)]
                else:
                    total += 1
            expanded[node_id] = min(total, _NODE_COUNT_CEILING)
            in_progress.discard(node_id)
            continue
        if node_id in expanded:
            continue
        in_progress.add(node_id)
        unique += 1
        stack.append((node, True))
        for child in children:
            if not isinstance(child, _YAML_CONTAINERS):
                unique += 1
            elif id(child) in in_progress:
                raise ValueError(
                    'an alias refers to the node that contains it (recursive YAML)'
                )
            elif id(child) not in expanded:
                stack.append((child, False))
    return expanded[id(document)], unique


def _check_yaml_aliases(document: Any, source: str) -> None:
    """Reject YAML documents whose aliases are recursive or expand explosively.

    Ordinary anchors and merge keys share a handful of nodes and stay far below
    the limit. Documents without any sharing (all JSON, YAML without aliases)
    measure ``expanded == unique`` and can never trip it.

    Raises:
        SchemaLoadError: If the document is recursive or its aliases expand it
            past ``max(MAX_ALIAS_AMPLIFICATION * unique, MIN_AMPLIFICATION_NODES)``
            nodes.
    """
    try:
        expanded, unique = _measure_aliasing(document)
    except ValueError as e:
        raise SchemaLoadError(source, cause=e)

    limit = max(MAX_ALIAS_AMPLIFICATION * unique, MIN_AMPLIFICATION_NODES)
    if expanded > limit:
        reached = f'{expanded:,}'
        if expanded >= _NODE_COUNT_CEILING:
            reached = f'over {reached}'
        raise SchemaLoadError(
            source,
            cause=ValueError(
                f'YAML aliases expand {unique:,} distinct nodes to {reached}, '
                f'past the limit of {limit:,}; refusing what looks like an '
                'alias bomb'
            ),
        )


def _parse_document(text: str, source: str, *, is_yaml: bool) -> Any:
    """Parse JSON or YAML text into plain Python data.

    Raises:
        SchemaLoadError: If the text is not valid, nests too deeply to parse,
            or (YAML only) uses aliases recursively or explosively.
    """
    try:
        document = yaml.safe_load(text) if is_yaml else json.loads(text)
    except (json.JSONDecodeError, yaml.YAMLError) as e:
        raise SchemaLoadError(source, cause=e)
    except RecursionError:
        raise SchemaLoadError(
            source, cause=ValueError('the document is nested too deeply to parse')
        )
    if is_yaml:
        _check_yaml_aliases(document, source)
    return document


# =============================================================================
# Schema Loader
# =============================================================================


class SchemaLoader:
    """Loads OpenAPI schemas from URLs or file paths.

    This class provides a unified interface for loading OpenAPI schemas
    from different sources (HTTP URLs or local files), supporting both
    JSON and YAML formats. It also handles automatic version detection
    and upgrading of Swagger 2.0 and older OpenAPI 3.x specifications.

    Features:
        - Load from URLs (http/https) or local file paths
        - Support for both JSON and YAML formats
        - Automatic version detection (Swagger 2.0, OpenAPI 3.0, 3.1, 3.2)
        - Automatic upgrade to OpenAPI 3.2 for code generation
        - External $ref resolution for URLs and relative files
        - Caching of loaded external schemas
        - Bounded downloads: capped decoded size, an overall time limit, and
          redirects that are limited, http(s)-only and kept off internal
          addresses (unless the spec itself came from one)
        - YAML alias bombs and recursive aliases are rejected

    Example:
        >>> loader = SchemaLoader()
        >>> schema = loader.load('https://api.example.com/openapi.json')
        >>> # or
        >>> schema = loader.load('/path/to/openapi.yaml')
        >>> # or with external ref resolution
        >>> loader = SchemaLoader(resolve_external_refs=True)
        >>> schema = loader.load('./api.yaml')
    """

    def __init__(
        self,
        http_client: httpx.Client | None = None,
        resolve_external_refs: bool = False,
        base_path: str | Path | None = None,
        lenient: bool = False,
        max_bytes: int | None = None,
        fetch_timeout: float | None = None,
    ):
        """Initialize the schema loader.

        Args:
            http_client: Optional HTTP client to use for URL requests.
                        If not provided, a default client will be created.
                        Redirects are followed by the loader itself (at most
                        ``MAX_REDIRECTS``, whatever the client's own setting),
                        so the client's default headers and auth are sent on
                        every hop: do not attach credentials to a client used
                        for specs from hosts you do not control.
            resolve_external_refs: Whether to resolve external $ref references.
                                  If True, external URLs and relative file paths
                                  will be loaded and inlined.
            base_path: Base path for resolving relative file references.
                      Defaults to current working directory.
            lenient: When True, drop unrecognized, structurally-invalid fields
                    from the specification (e.g. stray keys emitted by some
                    generators) instead of failing validation. Dropped fields
                    are reported as warnings. Vendor extensions (``x-*``) are
                    always preserved, in both strict and lenient mode. The one
                    exception is the type-discriminated security-scheme objects,
                    which still reject unknown keys (including ``x-*``) so their
                    variant union stays unambiguous.
            max_bytes: Largest response body, counted after decompression,
                      accepted from a URL. Defaults to ``MAX_SPEC_BYTES``.
            fetch_timeout: Seconds a URL download may take overall, redirects
                          included. Defaults to ``MAX_FETCH_SECONDS``.
        """
        self._http_client = http_client
        self._resolve_external_refs = resolve_external_refs
        self._base_path = Path(base_path) if base_path else Path.cwd()
        self._external_cache: dict[str, dict] = {}
        self._upgrade_warnings: list[str] = []
        self._lenient = lenient
        self._max_bytes = MAX_SPEC_BYTES if max_bytes is None else max_bytes
        self._fetch_timeout = (
            MAX_FETCH_SECONDS if fetch_timeout is None else fetch_timeout
        )

    def load(self, source: str) -> OpenAPIv3:
        """Load and validate an OpenAPI schema from a URL or file path.

        This method:
        1. Loads the content from the source (URL or file)
        2. Parses JSON or YAML content
        3. Detects the OpenAPI version
        4. Optionally resolves external $ref references
        5. Upgrades older versions to OpenAPI 3.2
        6. Validates the final schema

        Args:
            source: URL or file path to the OpenAPI schema.

        Returns:
            Validated OpenAPIv3 (3.2) object.

        Raises:
            SchemaLoadError: If the schema cannot be loaded from the source.
            SchemaValidationError: If the schema is not valid OpenAPI.
        """
        try:
            if self._is_url(source):
                content = self._load_from_url(source)
            else:
                content = self._load_from_file(source)
                source_path = Path(source)
                if source_path.is_absolute():
                    self._base_path = source_path.parent
                else:
                    self._base_path = (Path.cwd() / source_path).parent

            if self._resolve_external_refs:
                content = self._resolve_refs(content, source)

            return self._validate_and_upgrade(content, source)

        except SchemaLoadError:
            raise
        except SchemaValidationError:
            raise
        except Exception as e:
            raise SchemaLoadError(source, cause=e)

    def get_upgrade_warnings(self) -> list[str]:
        """Get any warnings generated during schema upgrade."""
        return self._upgrade_warnings.copy()

    def get_detected_version(self, content: dict) -> str:
        """Detect the OpenAPI/Swagger version from schema content."""
        if 'swagger' in content:
            return content.get('swagger', '2.0')
        openapi_version = content.get('openapi', '3.0.0')
        if openapi_version.startswith('3.0'):
            return '3.0'
        elif openapi_version.startswith('3.1'):
            return '3.1'
        elif openapi_version.startswith('3.2'):
            return '3.2'
        return openapi_version

    def _is_url(self, text: str) -> bool:
        """Check if a string is a URL."""
        try:
            result = urlparse(text)
            return result.scheme in ('http', 'https')
        except ValueError:
            return False

    def _load_from_url(self, url: str) -> dict:
        """Load schema content from a URL.

        The download is bounded in size and time and its redirects are vetted
        hop by hop, see ``_download``.
        """
        try:
            text, content_type = self._download(url)
        except (httpx.HTTPError, httpx.InvalidURL) as e:
            raise SchemaLoadError(url, cause=e)

        is_yaml = 'yaml' in content_type or url.endswith(('.yaml', '.yml'))
        return _parse_document(text, url, is_yaml=is_yaml)

    def _download(self, url: str) -> tuple[str, str]:
        """Download ``url`` within the size, time and redirect limits.

        Redirects are followed here, never by httpx, so each hop can be vetted
        by ``_vet_redirect``; the first URL is the user's own choice and is
        fetched as given. The client is asked not to follow redirects on every
        request, which also holds for an injected client configured to follow
        them.

        Returns:
            The decoded response body and its ``Content-Type`` header.

        Raises:
            SchemaLoadError: If a limit is exceeded or a redirect is refused.
            httpx.HTTPError: If the request or the response status fails.
        """
        client = self._http_client
        if client is None:
            client = httpx.Client(
                timeout=min(_READ_TIMEOUT_SECONDS, self._fetch_timeout),
                follow_redirects=False,
            )

        try:
            deadline = time.monotonic() + self._fetch_timeout
            current = httpx.URL(url)
            origin_host = current.raw_host.decode('ascii')
            headers = {'Accept-Encoding': ', '.join(_CONTENT_CODINGS)}
            redirects = 0
            while True:
                if time.monotonic() > deadline:
                    raise self._too_slow(url)
                with client.stream(
                    'GET', current, headers=headers, follow_redirects=False
                ) as response:
                    if not response.has_redirect_location:
                        response.raise_for_status()
                        text = self._read_body(url, response, deadline)
                        return text, response.headers.get('content-type', '')
                    location = response.headers['location']

                redirects += 1
                if redirects > MAX_REDIRECTS:
                    raise SchemaLoadError(
                        url,
                        cause=ValueError(
                            f'too many redirects (more than {MAX_REDIRECTS})'
                        ),
                    )
                current = _vet_redirect(url, current, location, origin_host)
        finally:
            if client is not self._http_client:
                client.close()

    def _read_body(self, url: str, response: httpx.Response, deadline: float) -> str:
        """Read a response body, enforcing the size limit and the deadline.

        Bytes are counted after decompression, as each chunk arrives, so a
        small compressed body that inflates enormously is stopped as soon as it
        passes the limit instead of being held in memory first.
        """
        limit = self._max_bytes
        _check_content_coding(url, response)

        try:
            announced = int(response.headers.get('content-length', ''))
        except ValueError:
            announced = 0
        if announced > limit:
            raise SchemaLoadError(
                url,
                cause=ValueError(
                    f'the server announced a {announced} byte response, over '
                    f'the {_describe_bytes(limit)} limit'
                ),
            )

        chunks: list[bytes] = []
        size = 0
        for chunk in response.iter_bytes():
            size += len(chunk)
            if size > limit:
                raise SchemaLoadError(
                    url,
                    cause=ValueError(
                        f'response is larger than the {_describe_bytes(limit)} '
                        'limit (counted after decompression)'
                    ),
                )
            if time.monotonic() > deadline:
                raise self._too_slow(url)
            chunks.append(chunk)
        return b''.join(chunks).decode(response.encoding or 'utf-8', errors='replace')

    def _too_slow(self, url: str) -> SchemaLoadError:
        """Build the error for a download that overran its overall budget."""
        return SchemaLoadError(
            url,
            cause=ValueError(
                f'download did not finish within {self._fetch_timeout:g} seconds'
            ),
        )

    def _load_from_file(self, file_path: str) -> dict:
        """Load schema content from a file."""
        path = Path(file_path)
        if not path.is_absolute():
            path = self._base_path / path

        if not path.exists():
            raise SchemaLoadError(
                str(file_path), cause=FileNotFoundError(f'File not found: {path}')
            )

        try:
            content = path.read_text(encoding='utf-8')
        except OSError as e:
            raise SchemaLoadError(str(file_path), cause=e)

        is_yaml = path.suffix.lower() in ('.yaml', '.yml')
        return _parse_document(content, str(file_path), is_yaml=is_yaml)

    def _resolve_refs(self, content: dict, source: str) -> dict:
        """Resolve all external $ref references in the schema."""
        return self._resolve_refs_recursive(content, source, set())

    def _resolve_refs_recursive(self, obj: Any, base: str, visited: set[str]) -> Any:
        """Recursively resolve external references."""
        if isinstance(obj, dict):
            if '$ref' in obj:
                ref = obj['$ref']
                if not ref.startswith('#'):
                    return self._resolve_external_ref(ref, base, visited)
            return {
                k: self._resolve_refs_recursive(v, base, visited)
                for k, v in obj.items()
            }
        elif isinstance(obj, list):
            return [self._resolve_refs_recursive(item, base, visited) for item in obj]
        return obj

    def _resolve_external_ref(self, ref: str, base: str, visited: set[str]) -> dict:
        """Resolve an external $ref reference."""
        if '#' in ref:
            file_part, pointer = ref.split('#', 1)
        else:
            file_part, pointer = ref, ''

        if self._is_url(file_part):
            location = file_part
        elif self._is_url(base):
            location = urljoin(base, file_part)
        else:
            base_path = Path(base).parent if not Path(base).is_dir() else Path(base)
            location = str(base_path / file_part)

        cache_key = f'{location}#{pointer}'
        if cache_key in visited:
            logger.warning(f'Circular reference detected: {cache_key}')
            return {'$ref': ref}

        visited = visited | {cache_key}

        if location not in self._external_cache:
            try:
                if self._is_url(location):
                    self._external_cache[location] = self._load_from_url(location)
                else:
                    self._external_cache[location] = self._load_from_file(location)
            except SchemaLoadError:
                logger.warning(f'Failed to resolve external reference: {ref}')
                return {'$ref': ref}

        content = self._external_cache[location]

        if pointer:
            content = self._resolve_json_pointer(content, pointer)

        return self._resolve_refs_recursive(content, location, visited)

    def _resolve_json_pointer(self, obj: Any, pointer: str) -> Any:
        """Resolve a JSON pointer within an object."""
        if not pointer or pointer == '/':
            return obj

        parts = pointer.strip('/').split('/')
        current = obj

        for part in parts:
            part = part.replace('~1', '/').replace('~0', '~')
            if isinstance(current, dict):
                if part not in current:
                    raise ValueError(f'JSON pointer path not found: {pointer}')
                current = current[part]
            elif isinstance(current, list):
                try:
                    index = int(part)
                    current = current[index]
                except (ValueError, IndexError):
                    raise ValueError(f'JSON pointer path not found: {pointer}')
            else:
                raise ValueError(f'JSON pointer path not found: {pointer}')

        return current

    def _validate_and_upgrade(self, content: dict, source: str) -> OpenAPIv3:
        """Validate and upgrade schema content to OpenAPI 3.2."""
        self._upgrade_warnings = []
        dropped_keys: set[str] = set()
        context = {'lenient': self._lenient, 'dropped_keys': dropped_keys}

        try:
            universal: UniversalOpenAPI = TypeAdapter(UniversalOpenAPI).validate_python(
                content, context=context
            )
        except ValidationError as e:
            raise SchemaValidationError(
                source, errors=self._format_validation_errors(e)
            )

        if dropped_keys:
            summary = (
                f'Lenient mode dropped {len(dropped_keys)} unrecognized field '
                f'name(s) from the specification: '
                f'{", ".join(sorted(dropped_keys))}'
            )
            logger.warning(summary)
            self._upgrade_warnings.append(summary)

        try:
            schema = universal.root

            if isinstance(schema, OpenAPIv3):
                return schema

            while not isinstance(schema, OpenAPIv3):
                schema, warnings = schema.upgrade()
                self._upgrade_warnings.extend(warnings)

            return schema

        except Exception as e:
            raise SchemaValidationError(source, errors=[str(e)])

    def _format_validation_errors(
        self, error: ValidationError, limit: int = 15
    ) -> list[str]:
        """Render a pydantic ``ValidationError`` into concise messages.

        Identical problems that repeat across the document are grouped, so a
        spec with one mistake in fifty places reports it once.
        """
        grouped: dict[str, str] = {}
        for err in error.errors():
            msg = err.get('msg', '')
            loc = '.'.join(str(part) for part in err.get('loc', ()) if str(part))
            grouped.setdefault(msg, loc)

        messages = [f'{loc}: {msg}' if loc else msg for msg, loc in grouped.items()]
        if len(messages) > limit:
            hidden = len(messages) - limit
            messages = messages[:limit]
            messages.append(f'... and {hidden} more distinct validation error(s)')

        if not self._lenient:
            messages.append(
                'Hint: re-run with --lenient (or set lenient: true in your '
                'config) to drop unrecognized fields instead of failing.'
            )
        return messages


# =============================================================================
# Schema Resolver
# =============================================================================


class SchemaResolver:
    """Resolves $ref references and manages schema lookups in OpenAPI documents.

    This class provides a centralized way to resolve JSON References ($ref) in
    OpenAPI documents, supporting both local references (within the same document)
    and caching resolved references for performance.

    Example:
        >>> resolver = SchemaResolver(openapi_doc)
        >>> schema, name = resolver.resolve_reference(ref)
        >>> all_schemas = resolver.get_all_schemas()
    """

    def __init__(self, openapi: OpenAPIv3):
        """Initialize the schema resolver.

        Args:
            openapi: The OpenAPI document to resolve references from.
        """
        self.openapi = openapi
        self._cache: dict[str, tuple[Schema, str | None]] = {}

    def resolve_reference(
        self, reference: Reference | Schema
    ) -> tuple[Schema, str | None]:
        """Resolve a $ref reference to its schema and name.

        If the input is already a Schema (not a Reference), it is returned as-is
        with its title as the name (if available).

        Args:
            reference: A Reference or Schema object to resolve.

        Returns:
            Tuple of (resolved Schema, schema name).

        Raises:
            SchemaReferenceError: If the reference cannot be resolved.
        """
        if isinstance(reference, Schema):
            name = (
                reference.title
                if hasattr(reference, 'title') and reference.title
                else None
            )
            return reference, self._sanitize_name(name) if name else None

        ref_str = reference.ref

        # Check cache first
        if ref_str in self._cache:
            return self._cache[ref_str]

        # Resolve the reference
        schema, name = self._resolve_ref_string(ref_str)

        # Cache the result
        self._cache[ref_str] = (schema, name)

        return schema, name

    def _resolve_ref_string(self, ref: str) -> tuple[Schema, str | None]:
        """Resolve a $ref string to its schema.

        Args:
            ref: The $ref string (e.g., '#/components/schemas/Pet').

        Returns:
            Tuple of (resolved Schema, schema name).

        Raises:
            SchemaReferenceError: If the reference format is unsupported or
                                  the referenced schema doesn't exist.
        """
        # Handle local references
        if ref.startswith('#/'):
            return self._resolve_local_reference(ref)

        # Handle external references (not yet supported)
        if ref.startswith(('http://', 'https://')):
            raise SchemaReferenceError(
                ref,
                'External URL references are not yet supported. '
                'Consider inlining the referenced schema.',
            )

        if ref.startswith(('./', '../')):
            raise SchemaReferenceError(
                ref,
                'Relative file references are not yet supported. '
                'Consider using a tool to bundle your OpenAPI spec.',
            )

        raise SchemaReferenceError(ref, 'Unknown reference format')

    def _resolve_local_reference(self, ref: str) -> tuple[Schema, str | None]:
        """Resolve a local JSON Pointer reference.

        Args:
            ref: The local reference (e.g., '#/components/schemas/Pet').

        Returns:
            Tuple of (resolved Schema, schema name).

        Raises:
            SchemaReferenceError: If the reference path is invalid or
                                  the schema doesn't exist.
        """
        # Parse the JSON Pointer
        if not ref.startswith('#/'):
            raise SchemaReferenceError(ref, 'Local reference must start with #/')

        path_parts = ref[2:].split('/')

        # Currently only support components/schemas references
        if (
            len(path_parts) >= 3
            and path_parts[0] == 'components'
            and path_parts[1] == 'schemas'
        ):
            schema_name = path_parts[2]
            return self._get_component_schema(schema_name, ref)

        # Support components/parameters references
        if (
            len(path_parts) >= 3
            and path_parts[0] == 'components'
            and path_parts[1] == 'parameters'
        ):
            raise SchemaReferenceError(
                ref,
                'Parameter references should be resolved through the parameter resolver',
            )

        # Support components/responses references
        if (
            len(path_parts) >= 3
            and path_parts[0] == 'components'
            and path_parts[1] == 'responses'
        ):
            raise SchemaReferenceError(
                ref,
                'Response references should be resolved through the response resolver',
            )

        raise SchemaReferenceError(
            ref,
            'Unsupported reference path. Only #/components/schemas/... is currently supported.',
        )

    def _get_component_schema(
        self, schema_name: str, ref: str
    ) -> tuple[Schema, str | None]:
        """Get a schema from the components/schemas section.

        Args:
            schema_name: The name of the schema in components/schemas.
            ref: The original reference string (for error messages).

        Returns:
            Tuple of (Schema, sanitized schema name).

        Raises:
            SchemaReferenceError: If the schema doesn't exist.
        """
        if not self.openapi.components:
            raise SchemaReferenceError(
                ref,
                'Document has no components section',
            )

        if not self.openapi.components.schemas:
            raise SchemaReferenceError(
                ref,
                'Document has no schemas in components',
            )

        schemas = self.openapi.components.schemas

        if schema_name not in schemas:
            available = ', '.join(sorted(schemas.keys())[:10])
            if len(schemas) > 10:
                available += f', ... ({len(schemas)} total)'
            raise SchemaReferenceError(
                ref,
                f"Schema '{schema_name}' not found. Available schemas: {available}",
            )

        return cast(Schema, schemas[schema_name]), self._sanitize_name(schema_name)

    def get_all_schemas(self) -> dict[str, Schema | Reference]:
        """Get all schemas defined in the components/schemas section.

        Returns:
            Dictionary mapping schema names to Schema (or Reference) objects.
            Returns an empty dict if no schemas are defined.
        """
        if not self.openapi.components or not self.openapi.components.schemas:
            return {}
        return dict(self.openapi.components.schemas)

    @staticmethod
    def _sanitize_name(name: str | None) -> str | None:
        """Sanitize a name to be a valid Python identifier.

        Args:
            name: The name to sanitize.

        Returns:
            Sanitized name, or None if input is None.
        """
        if name is None:
            return None

        # Import here to avoid circular imports
        from otterapi.codegen.utils import sanitize_identifier

        return sanitize_identifier(name)
