"""Endpoint function naming: rename rules, path-version routing, collisions.

Every operation starts from a raw name (its operationId, or the path-derived
name). :func:`assign_names` turns the raw names of all operations into final
function names in one pass, so it can see every collision at once:

1. ``naming`` rules and overrides rewrite each raw name.
2. Operations that end up sharing a name but carry different path versions
   are routed by ``versioning``: the newest keeps the name, older ones become
   legacy (a ``v<N>`` submodule or a ``_v<N>`` suffix).
3. A collision a rename *created* is an error -- silently numbering one
   operation ``_2`` would hide which one the user meant.
4. Any remaining collision (the spec itself reuses an operationId, or two raw
   names snake-case alike) keeps the historical ``_2`` / ``_3`` suffixing.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

from otterapi.codegen.utils import to_snake_case
from otterapi.exceptions import ConfigurationError

if TYPE_CHECKING:
    from otterapi.config import NamingConfig, VersioningConfig

__all__ = ['AssignedName', 'NamedOperation', 'assign_names']


@dataclass(frozen=True)
class NamedOperation:
    """What naming needs to know about one operation."""

    raw_name: str
    method: str
    path: str


@dataclass(frozen=True)
class AssignedName:
    """The final name of one operation.

    Attributes:
        fn_name: The sync function name.
        legacy_version: Set when the operation is an older path version
            routed into a ``v<N>`` submodule under the same ``fn_name``.
        config_key: The key per-endpoint config is looked up by, when it
            differs from ``fn_name`` (``<name>_v<N>`` for submodule legacy).
    """

    fn_name: str
    legacy_version: int | None = None
    config_key: str | None = None


def _label(op: NamedOperation) -> str:
    return f'{op.method.upper()} {op.path} ({op.raw_name})'


def _route_versions(
    members: list[int],
    ops: list[NamedOperation],
    versioning: VersioningConfig,
) -> dict[int, int] | None:
    """``{op index: version}`` when the group differs only in path version."""
    versions = {i: versioning.version_of(ops[i].path) for i in members}
    known = [v for v in versions.values() if v is not None]
    if len(known) != len(members) or len(set(known)) != len(known):
        return None
    return {i: v for i, v in versions.items() if v is not None}


def assign_names(
    ops: list[NamedOperation],
    naming: NamingConfig,
    versioning: VersioningConfig | None,
) -> list[AssignedName]:
    """Final function names for ``ops``, in the same order."""
    renamed = [naming.apply(op.raw_name) for op in ops]
    bases = [to_snake_case(name) for name in renamed]
    was_renamed = [new != op.raw_name for new, op in zip(renamed, ops, strict=True)]

    groups: dict[str, list[int]] = defaultdict(list)
    for i, base in enumerate(bases):
        groups[base].append(i)

    fixed: dict[int, AssignedName] = {}
    reserved: set[str] = set()
    for base, members in groups.items():
        if len(members) < 2:
            continue
        routed = (
            _route_versions(members, ops, versioning)
            if versioning is not None and versioning.enabled
            else None
        )
        if routed is not None:
            newest = max(routed.values())
            for i, version in routed.items():
                if version == newest:
                    fixed[i] = AssignedName(base)
                    reserved.add(base)
                elif versioning is not None and versioning.legacy == 'suffix':
                    name = f'{base}_v{version}'
                    fixed[i] = AssignedName(name)
                    reserved.add(name)
                else:
                    fixed[i] = AssignedName(
                        base, legacy_version=version, config_key=f'{base}_v{version}'
                    )
            continue
        if any(was_renamed[i] for i in members):
            listed = ', '.join(_label(ops[i]) for i in members)
            hint = (
                ''
                if versioning is not None
                else ' If they are versions of one endpoint, configure '
                'versioning; otherwise'
            )
            raise ConfigurationError(
                f'naming rules give {len(members)} operations the function name '
                f'{base!r}: {listed}.{hint} add a naming.overrides entry for one '
                'of them.',
                field='naming',
            )

    used: set[str] = set(reserved)
    result: list[AssignedName] = []
    for i, base in enumerate(bases):
        if i in fixed:
            result.append(fixed[i])
            continue
        name = base
        if name in used:
            suffix = 2
            while f'{name}_{suffix}' in used:
                suffix += 1
            name = f'{name}_{suffix}'
        used.add(name)
        result.append(AssignedName(name))

    # A legacy name living in a submodule must not also be some other
    # operation's name in the main module.
    for assigned in result:
        if assigned.legacy_version is not None and assigned.config_key in used:
            raise ConfigurationError(
                f'{assigned.config_key!r} is both an operation name and the '
                f'config key of the v{assigned.legacy_version} '
                f'{assigned.fn_name!r}; add a naming.overrides entry.',
                field='naming',
            )
    return result
