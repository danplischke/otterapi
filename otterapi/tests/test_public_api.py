"""Tests for the stable public API surface (Wave 3.15)."""

from __future__ import annotations

import otterapi.codegen as codegen

# Names users should be able to rely on across minor releases. Adding to
# this list is fine; removing/renaming is a breaking change.
STABLE_PUBLIC = {
    'Codegen',
    'TypeGenerator',
    'Type',
    'Endpoint',
    'Parameter',
    'RequestBodyInfo',
    'ResponseInfo',
    'SchemaLoader',
    'SchemaResolver',
}


class TestStableSurface:
    def test_all_lists_only_stable_names(self):
        assert set(codegen.__all__) == STABLE_PUBLIC

    def test_every_stable_name_is_actually_exported(self):
        for name in STABLE_PUBLIC:
            assert hasattr(codegen, name), f'missing public symbol: {name}'

    def test_all_is_a_list_or_tuple(self):
        # ``import *`` requires ``__all__`` to be iterable of strings.
        assert isinstance(codegen.__all__, (list, tuple))
        assert all(isinstance(x, str) for x in codegen.__all__)
