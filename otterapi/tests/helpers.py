"""Shared helpers for tests that generate and import a client package."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import yaml

from otterapi.codegen.codegen import Codegen
from otterapi.config import DocumentConfig


def generate(spec: dict | Path, output: Path, **config: Any) -> Path:
    """Generate a client for ``spec`` (a dict or a spec file) into ``output``.

    ``format_output=False`` skips formatting, for speed; every other keyword is
    a ``DocumentConfig`` field.
    """
    format_output = config.pop('format_output', True)
    if isinstance(spec, dict):
        source = output.parent / f'{output.name}.yaml'
        source.write_text(yaml.safe_dump(spec), encoding='utf-8')
    else:
        source = spec
    document = DocumentConfig.model_validate(
        {'source': str(source), 'output': str(output), **config}
    )
    Codegen(document, format_output=format_output).generate()
    return output


def import_fresh(parent: Path, package: str) -> ModuleType:
    """Import the generated ``package`` under ``parent``, dropping stale copies.

    Submodules import through the package afterwards
    (``importlib.import_module(f'{package}.models')``); ``sys.path`` is only
    touched for the top-level import.
    """
    top = package.split('.')[0]
    for name in [m for m in sys.modules if m.split('.')[0] == top]:
        del sys.modules[name]
    sys.path.insert(0, str(parent))
    try:
        return importlib.import_module(package)
    finally:
        sys.path.remove(str(parent))


def generate_package(
    tmp_path: Path, package: str, spec: dict | Path, **config: Any
) -> ModuleType:
    """Generate ``spec`` into ``tmp_path/package`` and import it afresh."""
    generate(spec, tmp_path / package, **config)
    return import_fresh(tmp_path, package)


#: The golden fixtures: one directory per spec, with an optional config.yaml.
GOLDEN = Path(__file__).parent / 'fixtures' / 'golden'


def generate_golden(fixture: str, output: Path, **overrides: Any) -> Path:
    """Generate a golden fixture with its own ``config.yaml`` plus ``overrides``."""
    spec_dir = GOLDEN / fixture
    config_path = spec_dir / 'config.yaml'
    base = (
        yaml.safe_load(config_path.read_text(encoding='utf-8')) or {}
        if config_path.is_file()
        else {}
    )
    return generate(
        spec_dir / 'spec.yaml',
        output,
        base_url='https://example.test',
        **{**base, **overrides},
    )
