"""Smoke test for scripts/regen_openapi_models.py.

We don't actually run the regenerator (it hits the network and rewrites
4 large modules), but we *do* sanity-check that the script imports, the
SOURCES list is complete, and the URLs are well-formed.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from urllib.parse import urlparse

SCRIPT = Path(__file__).resolve().parents[2] / 'scripts' / 'regen_openapi_models.py'


def _load():
    spec = importlib.util.spec_from_file_location('regen_openapi_models', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_script_exists():
    assert SCRIPT.is_file()


def test_module_imports_cleanly():
    _load()


def test_sources_cover_every_supported_version():
    module = _load()
    packages = {s.package for s in module.SOURCES}
    assert packages == {'v2', 'v3', 'v3_1', 'v3_2'}


def test_schema_urls_are_https():
    module = _load()
    for source in module.SOURCES:
        parsed = urlparse(source.schema_url)
        assert parsed.scheme == 'https', source.schema_url
        assert parsed.netloc, source.schema_url


def test_module_filename_matches_package():
    module = _load()
    for source in module.SOURCES:
        assert source.module_filename == f'{source.package}.py'
