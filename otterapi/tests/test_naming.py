"""Endpoint naming: rename rules, overrides, path-version routing (issue #30)."""

from __future__ import annotations

import pytest

from otterapi.codegen.naming import AssignedName, NamedOperation, assign_names
from otterapi.config import NamingConfig, VersioningConfig
from otterapi.exceptions import ConfigurationError

STRIP_V = NamingConfig(rules=[{'pattern': r'_v\d+(?=_|$)', 'replace': ''}])


def _ops(*specs: tuple[str, str]) -> list[NamedOperation]:
    return [NamedOperation(raw_name=n, method='get', path=p) for n, p in specs]


def test_no_config_keeps_historical_numbering() -> None:
    ops = _ops(('list', '/a'), ('list', '/b'), ('list_2', '/c'), ('List', '/d'))
    names = [a.fn_name for a in assign_names(ops, NamingConfig(), None)]
    assert names == ['list', 'list_2', 'list_2_2', 'list_3']


def test_rules_apply_in_order_and_overrides_win() -> None:
    naming = NamingConfig(
        rules=[
            {'pattern': '^api_', 'replace': ''},
            {'pattern': r'_v(\d+)$', 'replace': r'_version\1'},
        ],
        overrides={'api_special_v9': 'special'},
    )
    ops = _ops(('api_users_v2', '/u'), ('api_special_v9', '/s'))
    names = [a.fn_name for a in assign_names(ops, naming, None)]
    assert names == ['users_version2', 'special']


def test_collision_created_by_a_rename_is_an_error() -> None:
    ops = _ops(('get_x', '/x'), ('get_x_v2', '/x2'))
    with pytest.raises(ConfigurationError, match="function name 'get_x'"):
        assign_names(ops, STRIP_V, None)


def test_collision_without_distinct_versions_is_an_error() -> None:
    ops = _ops(('get_x', '/v1/x'), ('get_x_v2', '/v1/y'))
    with pytest.raises(ConfigurationError, match='naming.overrides'):
        assign_names(ops, STRIP_V, VersioningConfig())


def test_newest_version_keeps_the_name_older_go_to_submodule() -> None:
    ops = _ops(
        ('get_x', '/api/x/v1/by-id'),
        ('get_x_v3', '/api/x/v3/by-id'),
        ('get_x_v2', '/api/x/v2/by-id'),
        ('only_v1', '/api/y/v1/z'),
    )
    assert assign_names(ops, STRIP_V, VersioningConfig()) == [
        AssignedName('get_x', legacy_version=1, config_key='get_x_v1'),
        AssignedName('get_x'),
        AssignedName('get_x', legacy_version=2, config_key='get_x_v2'),
        AssignedName('only'),  # exists in one version only: stays put
    ]


def test_suffix_mode_keeps_older_versions_in_the_module() -> None:
    ops = _ops(('get_x', '/v1/x'), ('get_x_v2', '/v2/x'))
    names = assign_names(ops, STRIP_V, VersioningConfig(legacy='suffix'))
    assert [a.fn_name for a in names] == ['get_x_v1', 'get_x']
    assert all(a.legacy_version is None for a in names)


def test_versioning_without_rules_routes_reused_operation_ids() -> None:
    ops = _ops(('get_x', '/v1/x'), ('get_x', '/v2/x'))
    names = assign_names(ops, NamingConfig(), VersioningConfig())
    assert names[1] == AssignedName('get_x')
    assert names[0].legacy_version == 1


def test_legacy_config_key_clashing_with_a_real_name_is_an_error() -> None:
    ops = _ops(('get_x', '/v1/x'), ('get_x_v2', '/v2/x'), ('get_x_v1_', '/z'))
    naming = NamingConfig(
        rules=[{'pattern': r'_v2$', 'replace': ''}],
        overrides={'get_x_v1_': 'get_x_v1'},
    )
    with pytest.raises(ConfigurationError, match='config key'):
        assign_names(ops, naming, VersioningConfig())


@pytest.mark.parametrize(
    ('pattern', 'message'),
    [('(?P<v>\\d+)', 'P<version>'), ('(', 'invalid regular expression')],
)
def test_path_pattern_is_validated(pattern: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        VersioningConfig(path_pattern=pattern)


def test_override_must_be_an_identifier() -> None:
    with pytest.raises(ValueError, match='not a valid identifier'):
        NamingConfig(overrides={'op': 'not-valid'})
