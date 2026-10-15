"""Tests for the limits SchemaLoader puts on untrusted specifications.

A spec fetched from a URL is attacker-influenced input, and YAML adds a second
way to turn a tiny document into a huge amount of work. The loader therefore:

* caps the decoded size of a download and the time it may take,
* follows at most five redirects itself, over http(s) only, and never into
  loopback / private / link-local / cloud-metadata addresses -- unless the spec
  itself was served from such an address (a dev server on localhost),
* rejects YAML whose aliases are recursive or expand it explosively, for URL
  and file sources alike, and reports over-deep documents as load errors.

Every test runs offline: responses come from ``httpx.MockTransport`` and name
resolution goes through a lookup table, so nothing touches the network or DNS.
"""

from __future__ import annotations

import gzip
import ipaddress
import json
import socket
import time
import zlib
from collections.abc import Callable, Iterator

import httpx
import pytest
import yaml

from otterapi.codegen import schema as schema_module
from otterapi.codegen.schema import SchemaLoader
from otterapi.exceptions import SchemaLoadError

SPEC = {
    'openapi': '3.0.3',
    'info': {'title': 'Limits API', 'version': '1.0.0'},
    'paths': {},
}
SPEC_JSON = json.dumps(SPEC).encode()
PUBLIC_URL = 'https://public.example/openapi.json'
LOCAL_URL = 'http://localhost:8000/openapi.json'
LIMIT = 1_000

# The only names the fake resolver knows; anything else fails to resolve.
HOSTS = {
    'localhost': ['127.0.0.1', '::1'],
    'public.example': ['93.184.216.34'],
    'cdn.example.org': ['151.101.1.69', '2606:4700:4700::1111'],
    'internal.example': ['10.1.2.3'],
    'split.example': ['93.184.216.34', '10.0.0.5'],
    # Numeric spellings the system resolver (and so the connection) reads as IPv4.
    '2130706433': ['127.0.0.1'],
    '127.1': ['127.0.0.1'],
}

_SYSTEM_RESOLVER = schema_module._resolve_addresses


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


class Chunks(httpx.SyncByteStream):
    """A response body that arrives in pieces, the way it would from a socket."""

    def __init__(
        self,
        data: bytes,
        size: int = 64,
        on_chunk: Callable[[], None] | None = None,
    ):
        self.data = data
        self.size = size
        self.on_chunk = on_chunk
        self.sent = 0

    def __iter__(self) -> Iterator[bytes]:
        for start in range(0, len(self.data), self.size):
            if self.on_chunk is not None:
                self.on_chunk()
            self.sent += 1
            yield self.data[start : start + self.size]


class Untouchable(httpx.SyncByteStream):
    """A body the loader must refuse before reading a single byte of it."""

    def __iter__(self) -> Iterator[bytes]:
        raise AssertionError('the response body should not have been read')


class FakeClock:
    """Stands in for ``time.monotonic`` so a deadline can be crossed instantly."""

    def __init__(self) -> None:
        self.now = 1_000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def spec_response() -> httpx.Response:
    return httpx.Response(
        200, content=SPEC_JSON, headers={'Content-Type': 'application/json'}
    )


def redirect(location: str, status: int = 302) -> httpx.Response:
    return httpx.Response(status, headers={'Location': location})


def padded_spec(size: int) -> bytes:
    """The spec as JSON, padded with trailing whitespace to exactly ``size`` bytes."""
    body = SPEC_JSON + b' ' * (size - len(SPEC_JSON))
    assert len(body) == size
    return body


def redirect_once(location: str, status: int = 302):
    """A server whose ``/openapi.json`` redirects and everything else is the spec."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == '/openapi.json':
            return redirect(location, status)
        return spec_response()

    return handler


def redirect_chain(length: int):
    """A server that redirects ``length`` times before serving the spec."""

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        hop = 0 if path == '/openapi.json' else int(path[len('/hop') :])
        if hop < length:
            return redirect(f'/hop{hop + 1}')
        return spec_response()

    return handler


def alias_bomb(levels: int = 6, fanout: int = 10) -> str:
    """YAML where every level is a list of ``fanout`` aliases of the level below."""
    lines = ['a0: &a0 [' + ', '.join(['x'] * fanout) + ']']
    for level in range(1, levels):
        refs = ', '.join([f'*a{level - 1}'] * fanout)
        lines.append(f'a{level}: &a{level} [{refs}]')
    return '\n'.join(lines)


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def fake_dns(monkeypatch):
    """Replace DNS with ``HOSTS`` and record every lookup."""
    lookups: list[str] = []

    def resolve(host: str) -> list[str]:
        lookups.append(host)
        if host not in HOSTS:
            raise socket.gaierror(socket.EAI_NONAME, f'no such host: {host}')
        return list(HOSTS[host])

    monkeypatch.setattr(schema_module, '_resolve_addresses', resolve)
    return lookups


@pytest.fixture
def clock(monkeypatch):
    fake = FakeClock()
    monkeypatch.setattr(time, 'monotonic', fake)
    return fake


@pytest.fixture
def make_loader():
    """Build a loader on a scripted transport: ``(loader, requested URLs)``."""
    clients: list[httpx.Client] = []

    def build(handler, *, follow_redirects: bool = False, **kwargs):
        requested: list[str] = []

        def recording(request: httpx.Request) -> httpx.Response:
            requested.append(str(request.url))
            return handler(request)

        client = httpx.Client(
            transport=httpx.MockTransport(recording), follow_redirects=follow_redirects
        )
        clients.append(client)
        return SchemaLoader(http_client=client, **kwargs), requested

    yield build
    for client in clients:
        client.close()


@pytest.fixture(params=['file', 'url'])
def load_text(request, tmp_path, make_loader):
    """Load JSON or YAML text from a local file or from a URL, in turn."""

    def load(text: str, kind: str):
        if request.param == 'file':
            path = tmp_path / f'openapi.{kind}'
            path.write_text(text, encoding='utf-8')
            return SchemaLoader().load(str(path))

        content_type = 'application/json' if kind == 'json' else 'application/yaml'

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200, content=text.encode(), headers={'Content-Type': content_type}
            )

        loader, _ = make_loader(handler)
        return loader.load(f'https://public.example/openapi.{kind}')

    return load


# -----------------------------------------------------------------------------
# Download size
# -----------------------------------------------------------------------------


class TestDownloadSize:
    def test_default_limit_is_fifty_mebibytes(self):
        assert schema_module.MAX_SPEC_BYTES == 50 * 1024 * 1024
        assert schema_module._describe_bytes(schema_module.MAX_SPEC_BYTES) == '50 MiB'
        assert schema_module._describe_bytes(LIMIT) == '1000 bytes'

    def test_body_over_the_limit_is_rejected(self, make_loader):
        body = padded_spec(LIMIT + 1)
        loader, _ = make_loader(
            lambda request: httpx.Response(200, stream=Chunks(body)), max_bytes=LIMIT
        )
        with pytest.raises(SchemaLoadError, match=f'{LIMIT} bytes'):
            loader.load(PUBLIC_URL)

    def test_body_at_the_limit_is_accepted(self, make_loader):
        body = padded_spec(LIMIT)
        loader, _ = make_loader(
            lambda request: httpx.Response(200, stream=Chunks(body)), max_bytes=LIMIT
        )
        assert loader.load(PUBLIC_URL).info.title == 'Limits API'

    def test_limit_defaults_to_the_module_constant(self, make_loader, monkeypatch):
        monkeypatch.setattr(schema_module, 'MAX_SPEC_BYTES', LIMIT)
        body = padded_spec(LIMIT + 1)

        def handler(request):
            return httpx.Response(200, stream=Chunks(body))

        loader, _ = make_loader(handler)
        with pytest.raises(SchemaLoadError, match=f'{LIMIT} bytes'):
            loader.load(PUBLIC_URL)

        loader, _ = make_loader(handler, max_bytes=LIMIT + 1)
        assert loader.load(PUBLIC_URL).info.title == 'Limits API'

    @pytest.mark.parametrize(
        ('coding', 'compress'), [('gzip', gzip.compress), ('deflate', zlib.compress)]
    )
    def test_compressed_body_is_counted_after_decoding(
        self, make_loader, coding, compress
    ):
        inflated = b' ' * 200_000
        deflated = compress(inflated)
        assert len(deflated) < LIMIT < len(inflated)

        def handler(request):
            return httpx.Response(
                200,
                headers={
                    'Content-Encoding': coding,
                    'Content-Length': str(len(deflated)),
                },
                stream=Chunks(deflated, size=20),
            )

        loader, _ = make_loader(handler, max_bytes=LIMIT)
        with pytest.raises(SchemaLoadError, match=f'{LIMIT} bytes'):
            loader.load(PUBLIC_URL)

    @pytest.mark.parametrize(
        ('coding', 'compress'), [('gzip', gzip.compress), ('deflate', zlib.compress)]
    )
    def test_compressed_body_within_the_limit_loads(
        self, make_loader, coding, compress
    ):
        deflated = compress(SPEC_JSON)

        def handler(request):
            return httpx.Response(
                200,
                headers={'Content-Encoding': coding},
                stream=Chunks(deflated, size=16),
            )

        loader, _ = make_loader(handler, max_bytes=LIMIT)
        assert loader.load(PUBLIC_URL).info.title == 'Limits API'

    def test_reading_stops_once_the_limit_is_passed(self, make_loader):
        class Bottomless(httpx.SyncByteStream):
            """Gzip data that inflates forever, 20 kB of zeros per piece."""

            def __init__(self) -> None:
                self.compressor = zlib.compressobj(wbits=31)  # gzip framing
                self.sent = 0

            def __iter__(self) -> Iterator[bytes]:
                while True:
                    self.sent += 1
                    if self.sent > 500:
                        raise AssertionError('the loader kept reading')
                    yield self.compressor.compress(
                        b'\0' * 20_000
                    ) + self.compressor.flush(zlib.Z_SYNC_FLUSH)

        bottomless = Bottomless()

        def handler(request):
            return httpx.Response(
                200, headers={'Content-Encoding': 'gzip'}, stream=bottomless
            )

        loader, _ = make_loader(handler, max_bytes=100_000)
        with pytest.raises(SchemaLoadError, match='100000 bytes'):
            loader.load(PUBLIC_URL)
        assert bottomless.sent <= 10

    def test_declared_length_over_the_limit_is_rejected_unread(self, make_loader):
        def handler(request):
            return httpx.Response(
                200, headers={'Content-Length': str(LIMIT + 1)}, stream=Untouchable()
            )

        loader, _ = make_loader(handler, max_bytes=LIMIT)
        with pytest.raises(SchemaLoadError, match=f'announced a {LIMIT + 1} byte'):
            loader.load(PUBLIC_URL)

    def test_unparseable_content_length_is_ignored(self, make_loader):
        def handler(request):
            return httpx.Response(
                200, headers={'Content-Length': 'lots'}, stream=Chunks(SPEC_JSON)
            )

        loader, _ = make_loader(handler, max_bytes=LIMIT)
        assert loader.load(PUBLIC_URL).info.title == 'Limits API'

    def test_error_response_body_is_never_downloaded(self, make_loader):
        loader, _ = make_loader(
            lambda request: httpx.Response(500, stream=Untouchable())
        )
        with pytest.raises(SchemaLoadError) as raised:
            loader.load(PUBLIC_URL)
        assert isinstance(raised.value.cause, httpx.HTTPStatusError)

    @pytest.mark.parametrize(
        'coding', ['gzip, gzip', 'gzip,deflate', 'br', 'zstd', 'compress']
    )
    def test_unsupported_content_encoding_is_refused_unread(self, make_loader, coding):
        def handler(request):
            return httpx.Response(
                200, headers={'Content-Encoding': coding}, stream=Untouchable()
            )

        loader, _ = make_loader(handler)
        with pytest.raises(SchemaLoadError, match='Content-Encoding'):
            loader.load(PUBLIC_URL)

    @pytest.mark.parametrize('coding', ['identity', 'GZIP', ' gzip , identity '])
    def test_harmless_content_encoding_spellings_are_accepted(
        self, make_loader, coding
    ):
        deflated = gzip.compress(SPEC_JSON) if 'gzip' in coding.lower() else SPEC_JSON

        def handler(request):
            return httpx.Response(
                200, headers={'Content-Encoding': coding}, stream=Chunks(deflated)
            )

        loader, _ = make_loader(handler)
        assert loader.load(PUBLIC_URL).info.title == 'Limits API'

    def test_only_gzip_and_deflate_are_requested(self, make_loader):
        accepted: list[str] = []

        def handler(request):
            accepted.append(request.headers['accept-encoding'])
            return spec_response()

        loader, _ = make_loader(handler)
        loader.load(PUBLIC_URL)
        assert accepted == ['gzip, deflate']


# -----------------------------------------------------------------------------
# Download time
# -----------------------------------------------------------------------------


class TestDownloadTime:
    def test_default_budget_is_a_minute(self):
        assert schema_module.MAX_FETCH_SECONDS == 60.0

    def test_slow_drip_runs_into_the_deadline(self, make_loader, clock):
        # Six chunks, each arriving 15 seconds after the one before.
        stream = Chunks(padded_spec(600), size=100, on_chunk=lambda: clock.advance(15))
        loader, _ = make_loader(
            lambda request: httpx.Response(200, stream=stream), fetch_timeout=50
        )
        with pytest.raises(SchemaLoadError, match='within 50 seconds'):
            loader.load(PUBLIC_URL)
        assert stream.sent < 6

    def test_download_inside_the_deadline_succeeds(self, make_loader, clock):
        stream = Chunks(padded_spec(600), size=100, on_chunk=lambda: clock.advance(5))
        loader, _ = make_loader(
            lambda request: httpx.Response(200, stream=stream), fetch_timeout=50
        )
        assert loader.load(PUBLIC_URL).info.title == 'Limits API'

    def test_budget_defaults_to_the_module_constant(
        self, make_loader, clock, monkeypatch
    ):
        monkeypatch.setattr(schema_module, 'MAX_FETCH_SECONDS', 50)
        stream = Chunks(padded_spec(600), size=100, on_chunk=lambda: clock.advance(15))
        loader, _ = make_loader(lambda request: httpx.Response(200, stream=stream))
        with pytest.raises(SchemaLoadError, match='within 50 seconds'):
            loader.load(PUBLIC_URL)

    def test_budget_covers_the_whole_redirect_chain(self, make_loader, clock):
        def handler(request):
            clock.advance(25)
            return redirect('/openapi.json?again')

        loader, requested = make_loader(handler)
        with pytest.raises(SchemaLoadError, match='within 60 seconds'):
            loader.load(PUBLIC_URL)
        assert len(requested) == 3


# -----------------------------------------------------------------------------
# The client the loader creates for itself
# -----------------------------------------------------------------------------


class TestOwnClient:
    @pytest.fixture
    def created(self, monkeypatch):
        """Make ``httpx.Client`` build mock-transport clients; record their kwargs."""
        made: list[tuple[httpx.Client, dict]] = []
        real_client = httpx.Client

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == '/broken':
                return httpx.Response(500)
            return spec_response()

        def factory(**kwargs):
            client = real_client(transport=httpx.MockTransport(handler), **kwargs)
            made.append((client, kwargs))
            return client

        def no_shortcut(*args, **kwargs):
            raise AssertionError('httpx.get follows redirects on its own')

        monkeypatch.setattr(httpx, 'Client', factory)
        monkeypatch.setattr(httpx, 'get', no_shortcut)
        return made

    def test_client_is_short_lived_and_does_not_follow_redirects(self, created):
        assert SchemaLoader().load(PUBLIC_URL).info.title == 'Limits API'

        [(client, kwargs)] = created
        assert kwargs == {'timeout': 30.0, 'follow_redirects': False}
        assert client.is_closed

    def test_read_timeout_never_exceeds_the_budget(self, created):
        SchemaLoader(fetch_timeout=5).load(PUBLIC_URL)

        [(_, kwargs)] = created
        assert kwargs['timeout'] == 5

    def test_client_is_closed_when_the_download_fails(self, created):
        with pytest.raises(SchemaLoadError):
            SchemaLoader().load('https://public.example/broken')

        [(client, _)] = created
        assert client.is_closed

    def test_injected_client_is_left_open(self):
        client = httpx.Client(
            transport=httpx.MockTransport(lambda request: spec_response())
        )
        SchemaLoader(http_client=client).load(PUBLIC_URL)
        assert not client.is_closed
        client.close()


# -----------------------------------------------------------------------------
# Redirects
# -----------------------------------------------------------------------------

INTERNAL_TARGETS = [
    pytest.param('http://127.0.0.1:8000/spec.json', id='loopback-v4'),
    pytest.param('http://localhost/spec.json', id='localhost-name'),
    pytest.param('http://[::1]/spec.json', id='loopback-v6'),
    pytest.param('http://169.254.169.254/latest/meta-data/', id='cloud-metadata'),
    pytest.param('http://10.0.0.5/spec.json', id='rfc1918-10'),
    pytest.param('http://172.16.0.1/spec.json', id='rfc1918-172'),
    pytest.param('http://192.168.1.1/spec.json', id='rfc1918-192'),
    pytest.param('http://[fd00::1]/spec.json', id='unique-local-v6'),
    pytest.param('http://[fe80::1]/spec.json', id='link-local-v6'),
    pytest.param('http://[::ffff:127.0.0.1]/spec.json', id='mapped-loopback'),
    pytest.param('http://[::ffff:169.254.169.254]/spec.json', id='mapped-metadata'),
    pytest.param('http://[64:ff9b::a9fe:a9fe]/spec.json', id='nat64-metadata'),
    pytest.param('http://[2002:a9fe:a9fe::1]/spec.json', id='6to4-metadata'),
    pytest.param('http://0.0.0.0/spec.json', id='unspecified-v4'),
    pytest.param('http://[::]/spec.json', id='unspecified-v6'),
    pytest.param('http://224.0.0.1/spec.json', id='multicast-v4'),
    pytest.param('http://[ff02::1]/spec.json', id='multicast-v6'),
    pytest.param('http://240.0.0.1/spec.json', id='reserved'),
    pytest.param('http://100.64.0.1/spec.json', id='shared-address-space'),
    pytest.param('http://internal.example/spec.json', id='name-to-private'),
    pytest.param('http://split.example/spec.json', id='name-to-public-and-private'),
    pytest.param('http://2130706433/spec.json', id='decimal-ip'),
    pytest.param('http://127.1/spec.json', id='short-ip'),
    pytest.param('//localhost/spec.json', id='scheme-relative-localhost'),
]


class TestRedirects:
    @pytest.mark.parametrize('target', INTERNAL_TARGETS)
    def test_internal_target_is_refused_from_a_public_origin(self, make_loader, target):
        loader, requested = make_loader(redirect_once(target))
        with pytest.raises(SchemaLoadError, match='not a public address'):
            loader.load(PUBLIC_URL)
        assert requested == [PUBLIC_URL]

    @pytest.mark.parametrize('target', INTERNAL_TARGETS)
    def test_the_same_redirect_is_followed_from_a_local_origin(
        self, make_loader, target
    ):
        loader, requested = make_loader(redirect_once(target))
        assert loader.load(LOCAL_URL).info.title == 'Limits API'
        assert len(requested) == 2

    @pytest.mark.parametrize(
        'origin',
        [
            'http://localhost:8000/openapi.json',
            'http://127.0.0.1:8000/openapi.json',
            'http://[::1]:8000/openapi.json',
            'http://10.0.0.5/openapi.json',
            'http://internal.example/openapi.json',
        ],
    )
    def test_any_internal_origin_may_redirect_inside_the_internal_network(
        self, make_loader, origin
    ):
        loader, requested = make_loader(
            redirect_once('http://127.0.0.1:9000/spec.json')
        )
        assert loader.load(origin).info.title == 'Limits API'
        assert len(requested) == 2

    def test_origin_that_is_only_partly_internal_counts_as_public(self, make_loader):
        loader, requested = make_loader(redirect_once('http://127.0.0.1/spec.json'))
        with pytest.raises(SchemaLoadError, match='not a public address'):
            loader.load('http://split.example/openapi.json')
        assert len(requested) == 1

    @pytest.mark.parametrize('origin', [PUBLIC_URL, LOCAL_URL])
    @pytest.mark.parametrize(
        'target',
        [
            'file:///etc/passwd',
            'ftp://public.example/spec.json',
            'gopher://public.example/',
            'ws://public.example/spec.json',
        ],
    )
    def test_non_http_target_is_refused(self, make_loader, origin, target):
        loader, requested = make_loader(redirect_once(target))
        with pytest.raises(SchemaLoadError, match='only http'):
            loader.load(origin)
        assert len(requested) == 1

    @pytest.mark.parametrize('origin', [PUBLIC_URL, LOCAL_URL])
    @pytest.mark.parametrize(
        'target', ['data:application/json,{}', 'javascript:alert(1)']
    )
    def test_hostless_target_is_refused(self, make_loader, origin, target):
        # httpx itself rejects these before the loader sees them; either way
        # they are refused without a second request.
        loader, requested = make_loader(redirect_once(target))
        with pytest.raises(SchemaLoadError):
            loader.load(origin)
        assert len(requested) == 1

    @pytest.mark.parametrize('status', [301, 302, 303, 307, 308])
    def test_redirect_to_another_public_host_is_followed(
        self, make_loader, fake_dns, status
    ):
        target = 'https://cdn.example.org/spec.json'
        loader, requested = make_loader(redirect_once(target, status))
        assert loader.load(PUBLIC_URL).info.title == 'Limits API'
        assert requested == [PUBLIC_URL, target]
        # Only the redirect target was looked up; the first URL never is.
        assert fake_dns == ['cdn.example.org']

    @pytest.mark.parametrize(
        ('location', 'expected'),
        [
            ('/v2/spec.json', 'https://public.example/v2/spec.json'),
            ('spec.json', 'https://public.example/spec.json'),
            ('//cdn.example.org/spec.json', 'https://cdn.example.org/spec.json'),
            (
                'http://cdn.example.org:8080/spec.json',
                'http://cdn.example.org:8080/spec.json',
            ),
        ],
    )
    def test_location_is_resolved_against_the_current_url(
        self, make_loader, location, expected
    ):
        loader, requested = make_loader(redirect_once(location))
        loader.load(PUBLIC_URL)
        assert requested == [PUBLIC_URL, expected]

    def test_five_redirects_are_followed(self, make_loader):
        loader, requested = make_loader(redirect_chain(5))
        assert loader.load(PUBLIC_URL).info.title == 'Limits API'
        assert len(requested) == 6

    def test_a_sixth_redirect_is_refused(self, make_loader):
        loader, requested = make_loader(redirect_chain(6))
        with pytest.raises(SchemaLoadError, match='too many redirects'):
            loader.load(PUBLIC_URL)
        assert len(requested) == 6

    def test_redirect_loop_is_bounded(self, make_loader):
        def handler(request):
            other = 'b' if request.url.path.endswith('/a') else 'a'
            return redirect(f'https://public.example/{other}')

        loader, requested = make_loader(handler)
        with pytest.raises(SchemaLoadError, match='too many redirects'):
            loader.load('https://public.example/a')
        assert len(requested) == 6

    def test_the_first_url_is_never_address_checked(self, make_loader, fake_dns):
        for origin in (
            LOCAL_URL,
            'http://127.0.0.1:8000/openapi.json',
            'http://[::1]/openapi.json',
            'http://10.0.0.5/openapi.json',
            'http://169.254.169.254/openapi.json',
        ):
            loader, requested = make_loader(lambda request: spec_response())
            assert loader.load(origin).info.title == 'Limits API'
            assert len(requested) == 1
        assert fake_dns == []

    def test_target_that_does_not_resolve_is_followed(self, make_loader, caplog):
        """Behind a proxy, DNS happens at the proxy and this machine cannot tell.

        Refusing would break ordinary redirects there without stopping DNS
        rebinding, which answers the check honestly and the connection not.
        """
        loader, requested = make_loader(redirect_once('https://nowhere.example/x'))
        with caplog.at_level('WARNING', logger='otterapi.codegen.schema'):
            assert loader.load(PUBLIC_URL).info.title == 'Limits API'
        assert requested == [PUBLIC_URL, 'https://nowhere.example/x']
        assert any('nowhere.example' in r.getMessage() for r in caplog.records)

    def test_target_that_resolves_to_garbage_is_refused(self, make_loader, monkeypatch):
        monkeypatch.setattr(
            schema_module, '_resolve_addresses', lambda host: ['not-an-address']
        )
        loader, requested = make_loader(redirect_once('https://odd.example/x'))
        with pytest.raises(SchemaLoadError, match='unusable address'):
            loader.load(PUBLIC_URL)
        assert requested == [PUBLIC_URL]

    @pytest.mark.parametrize(
        'location', ['http://[::1/spec.json', 'http://[invalid]/', 'http://a.b:port/']
    )
    def test_malformed_location_is_refused(self, make_loader, location):
        loader, requested = make_loader(redirect_once(location))
        with pytest.raises(SchemaLoadError):
            loader.load(PUBLIC_URL)
        assert requested == [PUBLIC_URL]

    @pytest.mark.parametrize(
        ('status', 'headers'),
        [
            (302, {}),
            (300, {'Location': '/elsewhere'}),
            (304, {'Location': '/elsewhere'}),
        ],
        ids=['no-location', 'multiple-choices', 'not-modified'],
    )
    def test_3xx_that_is_not_a_redirect_is_an_http_error(
        self, make_loader, status, headers
    ):
        loader, requested = make_loader(
            lambda request: httpx.Response(status, headers=headers)
        )
        with pytest.raises(SchemaLoadError) as raised:
            loader.load(PUBLIC_URL)
        assert isinstance(raised.value.cause, httpx.HTTPStatusError)
        assert requested == [PUBLIC_URL]

    def test_injected_client_cannot_be_told_to_follow_redirects(self, make_loader):
        loader, requested = make_loader(
            redirect_once('http://169.254.169.254/latest/meta-data/'),
            follow_redirects=True,
        )
        with pytest.raises(SchemaLoadError, match='not a public address'):
            loader.load(PUBLIC_URL)
        assert requested == [PUBLIC_URL]

    @pytest.mark.parametrize(
        'address',
        [
            '8.8.8.8',
            '93.184.216.34',
            '2606:4700:4700::1111',
            '::ffff:8.8.8.8',
            '64:ff9b::808:808',
            '2002:808:808::1',
        ],
    )
    def test_public_addresses_are_not_flagged(self, address):
        assert not schema_module._is_non_public(ipaddress.ip_address(address))

    @pytest.mark.parametrize(
        'address',
        ['fec0::1', '::7f00:1', '255.255.255.255', '0.1.2.3', 'ff00::1', '2001:db8::1'],
    )
    def test_unusual_non_public_addresses_are_flagged(self, address):
        assert schema_module._is_non_public(ipaddress.ip_address(address))

    def test_system_resolver_returns_address_strings(self):
        # An IP literal needs no DNS, so this is safe offline.
        assert _SYSTEM_RESOLVER('127.0.0.1') == ['127.0.0.1']


# -----------------------------------------------------------------------------
# YAML aliases
# -----------------------------------------------------------------------------

ANCHORED_HEAD = """\
openapi: 3.0.3
info:
  title: Anchored API
  version: 1.0.0
x-defaults:
  error-responses: &error-responses
    '400':
      description: Malformed request
      content:
        application/json:
          schema:
            $ref: '#/components/schemas/Error'
    '404':
      description: No such resource
      content:
        application/json:
          schema:
            $ref: '#/components/schemas/Error'
  pagination: &pagination
    - name: limit
      in: query
      schema:
        type: integer
        minimum: 1
    - name: offset
      in: query
      schema:
        type: integer
        minimum: 0
  audit-fields: &audit-fields
    created_at:
      type: string
      format: date-time
    updated_at:
      type: string
      format: date-time
paths:
"""

ANCHORED_PATH = """\
  /things{n}:
    get:
      operationId: listThings{n}
      parameters: *pagination
      responses:
        <<: *error-responses
        '200':
          description: A page of things
          content:
            application/json:
              schema:
                type: array
                items:
                  $ref: '#/components/schemas/Thing{n}'
"""

ANCHORED_SCHEMA = """\
    Thing{n}:
      type: object
      required: [id]
      properties:
        <<: *audit-fields
        id:
          type: integer
"""


def anchored_spec(operations: int = 25) -> str:
    """A spec that shares parameters, responses and fields through anchors."""
    text = ANCHORED_HEAD
    text += ''.join(ANCHORED_PATH.format(n=n) for n in range(operations))
    text += (
        'components:\n'
        '  schemas:\n'
        '    Error:\n'
        '      type: object\n'
        '      properties:\n'
        '        message:\n'
        '          type: string\n'
    )
    text += ''.join(ANCHORED_SCHEMA.format(n=n) for n in range(operations))
    return text


class TestYamlAliases:
    def test_alias_bomb_is_rejected_quickly(self, load_text):
        text = alias_bomb(levels=6, fanout=10)
        assert len(text) < 1_000

        started = time.perf_counter()
        with pytest.raises(
            SchemaLoadError, match='YAML aliases expand 17 distinct nodes to 1,234,567'
        ):
            load_text(text, 'yaml')
        assert time.perf_counter() - started < 1.0

    def test_astronomical_alias_bomb_is_rejected_quickly(self, load_text):
        # Every level doubles the one below it: 2**5000 nodes from ~5000.
        lines = ['a0: &a0 [x]']
        lines += [f'a{i}: &a{i} [*a{i - 1}, *a{i - 1}]' for i in range(1, 5_000)]

        started = time.perf_counter()
        with pytest.raises(SchemaLoadError, match='to over 4,611,686,018,427,387,904'):
            load_text('\n'.join(lines), 'yaml')
        assert time.perf_counter() - started < 5.0

    @pytest.mark.parametrize(
        'text',
        [
            '&a [*a]',
            '&a {x: *a}',
            'a: &a {b: &b {c: *a}}',
            "openapi: 3.0.3\ninfo: &info\n  title: Loop\n  version: '1'\n  x-self: *info\n",
        ],
        ids=['list', 'mapping', 'mutual', 'inside-a-spec'],
    )
    def test_recursive_alias_is_rejected(self, load_text, text):
        with pytest.raises(SchemaLoadError, match='recursive'):
            load_text(text, 'yaml')

    def test_realistic_anchors_and_merge_keys_load(self, load_text):
        spec = load_text(anchored_spec(), 'yaml')

        assert spec.info.title == 'Anchored API'
        assert len(spec.paths.root) == 25
        operation = spec.paths.root['/things3'].get
        assert sorted(operation.responses.root) == ['200', '400', '404']
        assert [p.name for p in operation.parameters] == ['limit', 'offset']
        assert sorted(spec.components.schemas['Thing3'].properties) == [
            'created_at',
            'id',
            'updated_at',
        ]

    def test_realistic_spec_really_shares_nodes_below_the_limit(self):
        document = yaml.safe_load(anchored_spec())
        expanded, unique = schema_module._measure_aliasing(document)

        assert expanded > unique
        assert expanded < schema_module.MIN_AMPLIFICATION_NODES

    def test_large_json_document_is_not_alias_checked(self, load_text):
        document = dict(SPEC, **{'x-padding': list(range(200_000))})
        spec = load_text(json.dumps(document), 'json')
        assert spec.info.title == 'Limits API'


class TestMeasureAliasing:
    @staticmethod
    def measure(document):
        return schema_module._measure_aliasing(document)

    @staticmethod
    def check(document):
        schema_module._check_yaml_aliases(document, 'test.yaml')

    def test_scalar_document(self):
        assert self.measure(42) == (1, 1)
        assert self.measure(None) == (1, 1)

    def test_unshared_document_measures_the_same_both_ways(self):
        assert self.measure({'a': [1, 2], 'b': {'c': 3}}) == (6, 6)

    def test_shared_node_counts_once_unique_and_every_time_expanded(self):
        shared = [1, 2, 3]
        assert self.measure({'a': shared, 'b': shared}) == (9, 5)

    def test_tuples_and_sets_are_walked_too(self):
        omap = yaml.safe_load('--- !!omap\n- a: 1\n- b: [1, 2]\n')
        assert self.measure(omap) == (9, 9)
        assert self.measure({'x': {1, 2, 3}}) == (5, 5)

    def test_walk_is_not_limited_by_recursion_depth(self):
        root: list = []
        node = root
        for _ in range(100_000):
            child: list = []
            node.append(child)
            node = child
        assert self.measure(root) == (100_001, 100_001)

    def test_doubling_chain_is_exact_and_cheap(self):
        node: list = [0]
        for _ in range(40):
            node = [node, node]
        assert self.measure(node) == (3 * 2**40 - 1, 42)

    def test_counts_saturate_instead_of_growing_without_bound(self):
        node: list = [0]
        for _ in range(500):
            node = [node, node]
        assert self.measure(node) == (schema_module._NODE_COUNT_CEILING, 502)

    def test_cycles_are_detected(self):
        looped: list = []
        looped.append(looped)
        with pytest.raises(ValueError, match='recursive'):
            self.measure(looped)

        first: dict = {}
        first['next'] = {'back': first}
        with pytest.raises(ValueError, match='recursive'):
            self.measure(first)

    def test_a_node_shared_without_a_cycle_is_not_a_cycle(self):
        shared: list = [1]
        assert self.measure([shared, [shared], {'k': shared}]) == (9, 5)

    def test_limit_is_the_larger_of_the_ratio_and_the_floor(self):
        # 2,000 distinct nodes: the 100x ratio (200,000) is the limit.
        leaf = [0] * 999
        at_limit = [0] * 999 + [leaf] * 199
        assert self.measure(at_limit) == (200_000, 2_000)
        self.check(at_limit)

        over = [0] * 999 + [leaf] * 200
        assert self.measure(over) == (201_000, 2_000)
        with pytest.raises(SchemaLoadError, match='past the limit of 200,000'):
            self.check(over)

        # 100 distinct nodes: the 100,000 floor is the limit.
        small_leaf = [0] * 98
        under_floor = [small_leaf] * 1_010
        assert self.measure(under_floor) == (99_991, 100)
        self.check(under_floor)

        over_floor = [small_leaf] * 1_011
        with pytest.raises(SchemaLoadError, match='past the limit of 100,000'):
            self.check(over_floor)

    def test_large_document_without_sharing_is_never_rejected(self):
        document = [0] * 150_000
        expanded, unique = self.measure(document)
        assert expanded == unique > schema_module.MIN_AMPLIFICATION_NODES
        self.check(document)


# -----------------------------------------------------------------------------
# Documents that are too deep to parse
# -----------------------------------------------------------------------------


class TestDeepNesting:
    @pytest.mark.parametrize(
        'text',
        ['[' * 100_000 + ']' * 100_000, '{"a":' * 100_000 + '1' + '}' * 100_000],
        ids=['arrays', 'objects'],
    )
    def test_deep_json_is_a_load_error(self, load_text, text):
        with pytest.raises(SchemaLoadError, match='nested too deeply'):
            load_text(text, 'json')

    def test_deep_yaml_is_a_load_error(self, load_text):
        with pytest.raises(SchemaLoadError, match='nested too deeply'):
            load_text('[' * 5_000 + ']' * 5_000, 'yaml')
