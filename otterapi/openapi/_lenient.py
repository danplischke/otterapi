"""Shared lenient-validation base for the strict (``extra='forbid'``) v3.x models.

The OpenAPI 3.1/3.2 object models use ``extra='forbid'`` so that ``X | Reference``
unions discriminate correctly: a ``{"$ref": ...}`` payload is *rejected* by the
object branch (which forbids the unknown ``$ref`` key) and therefore falls through
to the ``Reference`` branch.

Lenient mode must not break that discrimination. The before-validator below strips
unrecognised keys *only* when a ``lenient`` validation context is supplied, and it
never strips ``$ref``/``ref``. Keeping ``$ref`` in place means ``extra='forbid'``
still rejects it, so the ``Reference`` branch continues to win.

In strict mode (no context, or ``lenient`` falsey) the validator is a no-op and the
models behave exactly as before -- including the v2->v3 upgrade path, which validates
without a context.
"""

from __future__ import annotations

from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, ValidationInfo, model_validator

# Keys that identify a Reference object. These are never stripped, so that
# ``extra='forbid'`` still rejects them on object branches and the ``Reference``
# branch of an ``X | Reference`` union wins.
_REFERENCE_KEYS = frozenset({'$ref', 'ref'})


class LenientRefModel(BaseModel):
    """Base for strict v3.x object models that supports opt-in lenient stripping.

    Subclasses keep ``extra='forbid'`` (inherited here and typically re-declared
    for clarity). When validated with ``context={'lenient': True}`` any field name
    that is neither declared, nor an alias, nor a reference key is dropped before
    validation and recorded in ``context['dropped_keys']`` (a set) when present.
    """

    model_config = ConfigDict(extra='forbid')

    # Union variants discriminated by an unknown field should set this to False so
    # that they *raise* (via forbid) instead of stripping, avoiding cross-branch
    # pollution of ``dropped_keys``. Ref-union object models keep it True.
    LENIENT_STRIPPABLE: ClassVar[bool] = True

    @model_validator(mode='before')
    @classmethod
    def _lenient_drop_unknown_fields(cls, data: Any, info: ValidationInfo) -> Any:
        if not isinstance(data, dict):
            return data
        context = info.context or {}
        if not (context.get('lenient') and cls.LENIENT_STRIPPABLE):
            return data

        allowed = set(cls.model_fields)
        allowed.update(
            field.alias for field in cls.model_fields.values() if field.alias
        )
        unknown = [
            key for key in data if key not in allowed and key not in _REFERENCE_KEYS
        ]
        if not unknown:
            return data

        sink = context.get('dropped_keys')
        if sink is not None:
            sink.update(str(key) for key in unknown)
        return {key: value for key, value in data.items() if key not in unknown}
