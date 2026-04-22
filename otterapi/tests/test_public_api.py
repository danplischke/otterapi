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


class TestBackwardsCompatibility:
    """Internal names stay reachable via direct attribute access for now."""

    def test_internal_names_still_importable(self):
        # These are not in ``__all__`` but are still attribute-accessible to
        # avoid breaking existing third-party code overnight (audit issue
        # #3, item 15: shrink, don't shatter).
        for name in (
            'EndpointFunctionFactory',
            'EndpointFunctionConfig',
            'FunctionSignatureBuilder',
            'ParameterASTBuilder',
            'ImportCollector',
            'CodeEmitter',
            'FileEmitter',
            'StringEmitter',
            'TypeRegistry',
            'TypeInfo',
            'ModelNameCollector',
            'DataFrameMethodConfig',
            'ModuleTree',
            'ModuleTreeBuilder',
            'ModuleMapResolver',
            'SplitModuleEmitter',
        ):
            assert hasattr(codegen, name), f'lost internal symbol: {name}'

    def test_internal_names_excluded_from_all(self):
        # Internal names should NOT appear in ``__all__`` -- ``import *``
        # must give users only the stable surface.
        for name in (
            'EndpointFunctionFactory',
            'FunctionSignatureBuilder',
            'ImportCollector',
        ):
            assert name not in codegen.__all__
