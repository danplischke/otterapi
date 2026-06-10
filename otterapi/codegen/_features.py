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
from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING

from upath import UPath

if TYPE_CHECKING:
    from otterapi.config import DocumentConfig


class FeatureModule(ABC):
    """A code-generation feature that emits a single runtime helper module.

    Concrete subclasses provide:

    - :pyattr:`module_filename` -- the file name written into the output dir
      (e.g. ``"_pagination.py"``). Must match the corresponding file in
      ``otterapi.codegen.runtime``.
    - :pymeth:`is_enabled` -- whether the feature is turned on for a given
      :class:`~otterapi.config.DocumentConfig`.
    """

    @property
    @abstractmethod
    def module_filename(self) -> str: ...

    @abstractmethod
    def is_enabled(self, config: DocumentConfig) -> bool: ...

    @property
    def module_content(self) -> str:
        """Read the runtime source from the ``otterapi.codegen.runtime`` package.

        Subclasses may override this with a plain class attribute to supply
        content directly (useful in tests or custom extensions).
        """
        return (
            files('otterapi.codegen.runtime')
            .joinpath(self.module_filename)
            .read_text('utf-8')
        )

    def transform_content(self, content: str, config: DocumentConfig) -> str:
        """Optionally transform the module source before writing.

        The default implementation returns *content* unchanged.  Subclasses
        override this to perform pydantic-version-specific (or other
        config-driven) rewrites of the static template source.

        Args:
            content: Raw source text read from ``otterapi.codegen.runtime``.
            config: The active :class:`~otterapi.config.DocumentConfig`.

        Returns:
            The (possibly transformed) source text to write to disk.
        """
        return content

    def write(self, output_dir: Path | UPath, config: DocumentConfig | None = None) -> Path | UPath:
        """Write :pyattr:`module_content` into ``output_dir``.

        Always opens with UTF-8 encoding; relying on the platform locale
        broke the export module on Windows in PR #2.

        Args:
            output_dir: Directory where the module file is written.
            config: Optional :class:`~otterapi.config.DocumentConfig` passed to
                :pymeth:`transform_content`.  When *None*, no transformation is
                applied (backwards-compatible for callers that do not yet pass
                config, e.g. ``generate_dataframe_module``).
        """
        output_dir = UPath(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        content = self.module_content
        if config is not None:
            content = self.transform_content(content, config)
        target = output_dir / self.module_filename
        target.write_text(content, encoding='utf-8')
        return target


def write_enabled_features(
    config: DocumentConfig,
    output_dir: Path | UPath,
    features: list[FeatureModule],
) -> list[Path | UPath]:
    """Write every enabled feature's runtime module. Returns the written paths."""
    return [
        feature.write(output_dir, config)
        for feature in features
        if feature.is_enabled(config)
    ]


def all_features() -> list[FeatureModule]:
    """Ordered list of every runtime-helper feature."""
    return [
        PaginationFeature(),
        DataFrameFeature(),
        ExportFeature(),
        ConcurrencyFeature(),
        RetryFeature(),
    ]


# =============================================================================
# Concrete features
# =============================================================================


class PaginationFeature(FeatureModule):
    """Emits ``_pagination.py`` (offset / cursor / page iterators)."""

    module_filename = '_pagination.py'

    def is_enabled(self, config: DocumentConfig) -> bool:
        return bool(config.pagination.enabled)


class DataFrameFeature(FeatureModule):
    """Emits ``_dataframe.py`` (pandas / polars converters).

    When ``config.pydantic_version == 1`` the emitted helper uses ``obj.dict()``
    (the Pydantic v1 serialisation method).  For Pydantic v2 (the default) it
    uses ``obj.model_dump()`` and removes the dead v1 fallback branch.
    """

    module_filename = '_dataframe.py'

    def is_enabled(self, config: DocumentConfig) -> bool:
        return bool(config.dataframe.enabled) and (
            bool(config.dataframe.pandas) or bool(config.dataframe.polars)
        )

    def transform_content(self, content: str, config: DocumentConfig) -> str:
        """Rewrite ``_to_dict`` and ``_normalize_data`` for the target Pydantic version."""
        if config.pydantic_version == 1:
            return content.replace(
                # Remove model_dump branch and make dict() the only call path
                'if hasattr(obj, \'model_dump\'):\n        return obj.model_dump()\n'
                '    elif hasattr(obj, \'dict\'):\n        # Pydantic v1 compatibility\n'
                '        return obj.dict()',
                '# Pydantic v1\n    if hasattr(obj, \'dict\'):\n        return obj.dict()',
            ).replace(
                # _normalize_data: replace the model_dump / dict check with dict only
                'if hasattr(first, \'model_dump\') or hasattr(first, \'dict\'):',
                'if hasattr(first, \'dict\'):',
            ).replace(
                # single-item path
                'if hasattr(data, \'model_dump\') or hasattr(data, \'dict\'):',
                'if hasattr(data, \'dict\'):',
            )
        # pydantic_version == 2 (default): strip the dead v1 .dict() fallback branch
        return content.replace(
            '    elif hasattr(obj, \'dict\'):\n        # Pydantic v1 compatibility\n'
            '        return obj.dict()\n',
            '',
        ).replace(
            'if hasattr(first, \'model_dump\') or hasattr(first, \'dict\'):',
            'if hasattr(first, \'model_dump\'):',
        ).replace(
            'if hasattr(data, \'model_dump\') or hasattr(data, \'dict\'):',
            'if hasattr(data, \'model_dump\'):',
        )


class ExportFeature(FeatureModule):
    """Emits ``_export.py`` (streaming CSV / TSV / JSONL / Parquet writers)."""

    module_filename = '_export.py'

    def is_enabled(self, config: DocumentConfig) -> bool:
        return bool(config.export.enabled)


class ConcurrencyFeature(FeatureModule):
    """Emits ``_concurrency.py`` (run_sync / run_concurrently helpers)."""

    module_filename = '_concurrency.py'

    def is_enabled(self, _config: DocumentConfig) -> bool:
        return True


class RetryFeature(FeatureModule):
    """Emits ``_retry.py`` (backoff sleep helpers used by the generated _client.py)."""

    module_filename = '_retry.py'

    def is_enabled(self, _config: DocumentConfig) -> bool:
        return True
