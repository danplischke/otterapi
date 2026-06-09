"""Wave 1.6: Codegen runtime guards must be real exceptions, not bare ``assert``.

``python -O`` strips asserts; if a guard against an uninitialized state
relies on ``assert``, the failure mode under ``-O`` is an obscure
``AttributeError`` deep inside generation. Real ``RuntimeError``s with
actionable messages are required.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from otterapi.codegen.codegen import Codegen
from otterapi.config import DocumentConfig


def test_generate_models_file_without_load_raises_runtime_error(tmp_path: Path):
    config = DocumentConfig(
        source='/dev/null',
        output=str(tmp_path),
        base_url='https://example.test',
    )
    codegen = Codegen(config)
    # _generate_models_file is the path that used to ``assert self.typegen``.
    with pytest.raises(RuntimeError, match='TypeGenerator is not initialized'):
        codegen._generate_models_file(tmp_path / 'models.py')
