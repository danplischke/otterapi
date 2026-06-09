"""Facade between the parser layer and the code generator.

The codegen modules (``codegen.py``, ``types.py``, ``schema.py``,
``utils.py``) import directly from ``otterapi.openapi.v3_2.v3_2`` today.
That means every refactor of the parser layer ripples into codegen, and
swapping out the OpenAPI representation (e.g. for a vendor-extended fork)
requires touching every codegen file.

``OpenAPIAdapter`` is a thin wrapper around an ``OpenAPI`` v3.2 document
that exposes the *intent* of what codegen wants from the parser
("paths()", "components_schema(name)", "info_title()") rather than raw
attribute access. New call sites in codegen should reach for the adapter
first; existing direct uses can migrate incrementally as touched (audited
and tracked in issue #3, item 10).

Keeping this minimal on purpose: the goal is to *establish the seam*, not
to wholesale-translate every operation. Each method lands as a real
caller adopts it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from otterapi.openapi.constants import HTTP_METHODS

if TYPE_CHECKING:
    from otterapi.openapi.v3_2.v3_2 import (
        OpenAPI as OpenAPIv3_2,
        PathItem,
        Reference,
        Schema,
        Server,
    )


class OpenAPIAdapter:
    """Read-only facade over an OpenAPI 3.2 document.

    Wrapping the document this way lets the rest of codegen stay agnostic
    of the underlying parser's class names and attribute paths.
    """

    __slots__ = ('_doc',)

    def __init__(self, document: OpenAPIv3_2) -> None:
        self._doc = document

    # -- Document-wide accessors --------------------------------------

    @property
    def document(self) -> OpenAPIv3_2:
        """Escape hatch for callers that still need the raw v3.2 model."""
        return self._doc

    def title(self) -> str:
        return self._doc.info.title

    def version(self) -> str:
        return self._doc.info.version

    def servers(self) -> list[Server]:
        return list(self._doc.servers or [])

    def has_paths(self) -> bool:
        return bool(self._doc.paths)

    # -- Paths / operations -------------------------------------------

    def paths(self) -> dict[str, PathItem]:
        """Return the path -> PathItem mapping (always a real dict)."""
        if not self._doc.paths:
            return {}
        # ``paths.root`` is keyed by URL string in the v3.2 RootModel.
        return dict(self._doc.paths.root)

    def operations(self):
        """Yield ``(path, method, operation)`` triples across the document.

        ``method`` is lower-cased to match HTTP method conventions used
        elsewhere in codegen.
        """
        for path, item in self.paths().items():
            for method in HTTP_METHODS:
                op = getattr(item, method, None)
                if op is not None:
                    yield path, method, op

    # -- Components ---------------------------------------------------

    def components_schemas(self) -> dict[str, Schema | Reference]:
        components = self._doc.components
        if components is None or components.schemas is None:
            return {}
        return dict(components.schemas)

    def components_parameter(self, name: str):
        components = self._doc.components
        if components is None or components.parameters is None:
            return None
        return components.parameters.get(name)

    def components_response(self, name: str):
        components = self._doc.components
        if components is None or components.responses is None:
            return None
        return components.responses.get(name)

    def components_request_body(self, name: str):
        components = self._doc.components
        if components is None or components.requestBodies is None:
            return None
        return components.requestBodies.get(name)
