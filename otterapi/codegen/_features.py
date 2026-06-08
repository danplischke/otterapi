"""Shared base for code-generation features that emit a runtime helper module.

Pagination, DataFrame, and Export each ship a static Python module into the
user's generated package (``_pagination.py``, ``_dataframe.py``,
``_export.py``). They all follow the same pattern:

1. A static module-content string.
2. A ``generate_*_module(output_dir)`` writer.
3. A "should this feature run for this endpoint" predicate driven by a
   ``DocumentConfig`` section.

Without an abstraction, every new feature reimplements the writer (and forgets
things like ``encoding='utf-8'`` -- which bit us on Windows for the export
module). ``FeatureModule`` factors out that common shape; per-feature AST
emitters stay separate because their shapes diverge significantly.

Subclasses declare the file name + content + an ``is_enabled`` predicate.
The base class handles directory creation, encoding, and the iteration
contract used by ``Codegen``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

from upath import UPath

if TYPE_CHECKING:
    from otterapi.config import DocumentConfig


class FeatureModule(ABC):
    """A code-generation feature that emits a single runtime helper module.

    Concrete subclasses provide:

    - :pyattr:`module_filename` -- the file name written into the output dir
      (e.g. ``"_pagination.py"``).
    - :pyattr:`module_content` -- the static Python source to write.
    - :pymeth:`is_enabled` -- whether the feature is turned on for a given
      :class:`~otterapi.config.DocumentConfig`.
    """

    @property
    @abstractmethod
    def module_filename(self) -> str: ...

    @property
    @abstractmethod
    def module_content(self) -> str: ...

    @abstractmethod
    def is_enabled(self, config: DocumentConfig) -> bool: ...

    def write(self, output_dir: Path | UPath) -> Path | UPath:
        """Write :pyattr:`module_content` into ``output_dir``.

        Always opens with UTF-8 encoding; relying on the platform locale
        broke the export module on Windows in PR #2.
        """
        output_dir = UPath(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        target = output_dir / self.module_filename
        target.write_text(self.module_content, encoding='utf-8')
        return target


def write_enabled_features(
    config: DocumentConfig,
    output_dir: Path | UPath,
    features: list[FeatureModule],
) -> list[Path | UPath]:
    """Write every enabled feature's runtime module. Returns the written paths."""
    return [
        feature.write(output_dir) for feature in features if feature.is_enabled(config)
    ]


# =============================================================================
# Concrete features
# =============================================================================


class PaginationFeature(FeatureModule):
    """Emits ``_pagination.py`` (offset / cursor / page iterators)."""

    module_filename = '_pagination.py'

    @property
    def module_content(self) -> str:
        from otterapi.codegen.pagination import PAGINATION_MODULE_CONTENT

        return PAGINATION_MODULE_CONTENT

    def is_enabled(self, config: DocumentConfig) -> bool:
        return bool(config.pagination.enabled)


class DataFrameFeature(FeatureModule):
    """Emits ``_dataframe.py`` (pandas / polars converters)."""

    module_filename = '_dataframe.py'

    @property
    def module_content(self) -> str:
        from otterapi.codegen.dataframes import DATAFRAME_MODULE_CONTENT

        return DATAFRAME_MODULE_CONTENT

    def is_enabled(self, config: DocumentConfig) -> bool:
        return bool(config.dataframe.enabled) and (
            bool(config.dataframe.pandas) or bool(config.dataframe.polars)
        )


class ExportFeature(FeatureModule):
    """Emits ``_export.py`` (streaming CSV / TSV / JSONL / Parquet writers)."""

    module_filename = '_export.py'

    @property
    def module_content(self) -> str:
        from otterapi.codegen.export import EXPORT_MODULE_CONTENT

        return EXPORT_MODULE_CONTENT

    def is_enabled(self, config: DocumentConfig) -> bool:
        return bool(config.export.enabled)


def all_features() -> list[FeatureModule]:
    """Canonical ordered list of every runtime-helper feature."""
    return [PaginationFeature(), DataFrameFeature(), ExportFeature()]
