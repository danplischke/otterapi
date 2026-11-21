"""End-to-end smoke test against large, messy, real-world OpenAPI specs.

The curated fixtures prove *type-correctness* on small hand-picked shapes. This
proves *robustness* on the real thing: OtterAPI must read Stripe (587 ops, 1400+
schemas) and the GitHub REST API (1200+ ops) and emit a client that is valid,
importable Python whose every pydantic model actually builds.

These specs are what surfaced the parser/upgrade crash, the infinite type
recursion, the ``None | None`` annotation, and the leading-underscore /
BaseModel-shadow field bugs -- none of which the small fixtures could reach. The
specs are vendored gzipped under ``fixtures/real_specs`` (see its README).

Marked ``slow``: it generates and imports ~3 MB of code, so it runs in a
dedicated CI job rather than the fast unit matrix.
"""

from __future__ import annotations

import ast
import gzip
import importlib
import json
import sys
import warnings
from pathlib import Path

import pytest

from otterapi.codegen.codegen import Codegen
from otterapi.config import DocumentConfig

pytestmark = pytest.mark.slow

REAL_SPECS = Path(__file__).parent / 'fixtures' / 'real_specs'

# name -> gzipped spec fixture
SPECS = {
    'stripe': 'stripe.openapi.json.gz',
    'github': 'github.openapi.json.gz',
}


def _load_spec(fixture: str, dest: Path) -> Path:
    """Decompress a vendored ``*.json.gz`` spec into *dest* and return the path."""
    with gzip.open(REAL_SPECS / fixture) as fh:
        dest.write_bytes(fh.read())
    return dest


def _import_package(pkg_dir: Path) -> None:
    """Import a generated client package (and its modules) under strict warnings.

    Importing -- not just compiling -- forces every pydantic model class to be
    built, so a shadowed BaseModel attribute, an un-evaluatable annotation, or an
    illegal field name fails here instead of silently shipping.
    """
    parent = str(pkg_dir.parent)
    pkg = pkg_dir.name
    added = parent not in sys.path
    if added:
        sys.path.insert(0, parent)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            for module in (pkg, f'{pkg}.models', f'{pkg}.client', f'{pkg}.endpoints'):
                importlib.import_module(module)
    finally:
        for mod_name in list(sys.modules):
            if mod_name == pkg or mod_name.startswith(f'{pkg}.'):
                del sys.modules[mod_name]
        if added and parent in sys.path:
            sys.path.remove(parent)


@pytest.mark.parametrize('name', sorted(SPECS))
def test_real_spec_generates_compiles_and_imports(name, tmp_path):
    spec_path = _load_spec(SPECS[name], tmp_path / f'{name}.json')

    # Sanity-check the vendored fixture round-trips before blaming the generator.
    assert json.loads(spec_path.read_text()).get('openapi', '').startswith('3.')

    pkg_dir = tmp_path / f'otterclient_{name}'
    config = DocumentConfig(
        source=str(spec_path),
        output=str(pkg_dir),
        base_url='https://example.test',
    )
    # Formatting is pure cosmetics and slow on megabytes of output; skip it.
    Codegen(config, format_output=False).generate()

    for module in ('__init__', 'models', 'client', 'endpoints'):
        path = pkg_dir / f'{module}.py'
        assert path.exists(), f'{name}: missing {module}.py'
        # Valid Python (syntax) ...
        compile(path.read_text(), str(path), 'exec')

    # ... and actually importable (every pydantic model builds).
    _import_package(pkg_dir)


# (client_style, resource_naming, result_objects): every class-based layout,
# since each derives its own identifiers (method names, resource properties,
# resource classes) from the spec.
CLASS_LAYOUTS = {
    'client': ('client', 'tag', True),
    'resource-tag': ('resource', 'tag', False),
    'resource-path': ('resource', 'path', True),
    'resource-operation_id': ('resource', 'operation_id', False),
}


@pytest.mark.parametrize('layout', sorted(CLASS_LAYOUTS))
@pytest.mark.parametrize('name', sorted(SPECS))
def test_real_spec_class_layouts_compile_and_import(name, layout, tmp_path):
    spec_path = _load_spec(SPECS[name], tmp_path / f'{name}.json')
    client_style, resource_naming, result_objects = CLASS_LAYOUTS[layout]

    pkg_dir = tmp_path / f'otterclient_{name}_{layout.replace("-", "_")}'
    config = DocumentConfig(
        source=str(spec_path),
        output=str(pkg_dir),
        base_url='https://example.test',
        client_style=client_style,
        resource_naming=resource_naming,
        result_objects=result_objects,
    )
    Codegen(config, format_output=False).generate()

    clients = pkg_dir / '_clients.py'
    compile(clients.read_text(), str(clients), 'exec')
    tree = ast.parse(clients.read_text())
    class_names = [n.name for n in tree.body if isinstance(n, ast.ClassDef)]
    assert len(class_names) == len(set(class_names)), f'{name}: duplicate classes'

    # __init__ imports _clients, so this builds every resource class too.
    _import_package(pkg_dir)
