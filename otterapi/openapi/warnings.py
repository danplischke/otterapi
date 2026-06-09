from __future__ import annotations


class WarningCollector:
    """Collects and deduplicates warnings, emitting summaries instead of per-occurrence noise.

    Pass a mapping of warning_key → message template at construction time; keys not in
    the map fall back to the raw key as the message.
    """

    def __init__(self, warning_templates: dict[str, str] | None = None) -> None:
        self._counts: dict[str, int] = {}
        self._unique_warnings: list[str] = []
        self._warning_templates: dict[str, str] = warning_templates or {}

    def add(self, warning_key: str, count: int = 1) -> None:
        """Add a counted warning (deduplicated into a summary)."""
        self._counts[warning_key] = self._counts.get(warning_key, 0) + count

    def add_unique(self, warning: str) -> None:
        """Add a warning that appears as-is (not deduplicated)."""
        self._unique_warnings.append(warning)

    def get_warnings(self) -> list[str]:
        """Return unique warnings followed by summarized counted warnings."""
        result = list(self._unique_warnings)
        for key, count in self._counts.items():
            msg = self._warning_templates.get(key, key)
            result.append(msg if count == 1 else f'{msg} ({count} occurrences)')
        return result
