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


def generate_package(
    tmp_path: Path, package: str, spec: dict, **config: Any
) -> ModuleType:
    """Generate ``spec`` into ``tmp_path/package`` and import it afresh.

    Submodules import through the package afterwards
    (``importlib.import_module(f'{package}.models')``); ``sys.path`` is only
    touched for the top-level import.
    """
    spec_file = tmp_path / f'{package}.yaml'
    spec_file.write_text(yaml.safe_dump(spec), encoding='utf-8')
    Codegen(
        DocumentConfig(source=str(spec_file), output=str(tmp_path / package), **config)
    ).generate()
    for name in [m for m in sys.modules if m.split('.')[0] == package]:
        del sys.modules[name]
    sys.path.insert(0, str(tmp_path))
    try:
        return importlib.import_module(package)
    finally:
        sys.path.remove(str(tmp_path))
