"""Shared validation base for the OpenAPI 3.0/3.1/3.2 object models.

These models must satisfy three requirements at once:

* **Preserve specification extensions.** The OpenAPI spec permits ``x-*`` fields
  on almost every object, so they are accepted and retained (in ``model_extra``)
  rather than rejected.
* **Keep ``$ref`` discrimination.** An *undeclared* ``$ref``/``ref`` key means the
  payload is really a Reference, so this object branch must fail and let the
  ``Reference`` branch of an ``X | Reference`` union win. (Models that *declare*
  ``$ref`` -- e.g. Path Item -- keep it as a normal field.)
* **Reject unknown junk.** Any other unrecognised key raises in strict mode, or --
  when validated with ``context={'lenient': True}`` -- is dropped and recorded in
  ``context['dropped_keys']``.

Pydantic's ``extra`` setting is all-or-nothing, so we use ``extra='allow'`` (to
retain ``x-*``) and enforce the rules above in a ``mode='before'`` validator. In
strict mode (no context, or ``lenient`` falsey) the validator still preserves
``x-*`` and still rejects junk, and the v2->v3 upgrade path (which validates
without a context) is unaffected.
"""

from __future__ import annotations

from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, ValidationInfo, model_validator

# Keys that identify a Reference object. When present but *not* declared by the
# model, the object branch fails so the ``Reference`` branch of an
# ``X | Reference`` union wins.
_REFERENCE_KEYS = frozenset({'$ref', 'ref'})


def _is_vendor_extension(key: Any) -> bool:
    """Return True for OpenAPI specification extensions (``x-*`` keys)."""
    return isinstance(key, str) and key.startswith('x-')


class LenientRefModel(BaseModel):
    """Base for v3.x object models: preserves ``x-*`` and keeps ``$ref`` discrimination.

    Uses ``extra='allow'`` so vendor extensions survive validation. Subclasses
    re-declare ``extra='allow'`` (plus e.g. ``populate_by_name``) for clarity;
    the shared ``mode='before'`` validator enforces the handling of every other
    unknown key.
    """

    model_config = ConfigDict(extra='allow')

    # Union variants discriminated by an unknown field set this to False so that
    # they *raise* on unexpected keys instead of stripping them, avoiding
    # cross-branch pollution of ``dropped_keys``.
    LENIENT_STRIPPABLE: ClassVar[bool] = True

    @model_validator(mode='before')
    @classmethod
    def _enforce_known_fields(cls, data: Any, info: ValidationInfo) -> Any:
        if not isinstance(data, dict):
            return data

        allowed = set(cls.model_fields)
        allowed.update(
            field.alias for field in cls.model_fields.values() if field.alias
        )

        # An undeclared $ref/ref means the payload is really a Reference: fail
        # this branch so the `X | Reference` union falls through to Reference.
        if any(key in data for key in _REFERENCE_KEYS if key not in allowed):
            raise ValueError(
                f'{cls.__name__}: unexpected reference key; treating payload as '
                'a Reference'
            )

        # Vendor extensions (x-*) are always kept; declared fields validate as
        # usual; everything else is unexpected.
        unknown = [
            key for key in data if key not in allowed and not _is_vendor_extension(key)
        ]
        if not unknown:
            return data

        context = info.context or {}
        if context.get('lenient') and cls.LENIENT_STRIPPABLE:
            sink = context.get('dropped_keys')
            if sink is not None:
                sink.update(str(key) for key in unknown)
            return {key: value for key, value in data.items() if key not in unknown}

        raise ValueError(
            f'unexpected field(s) on {cls.__name__}: '
            f'{", ".join(sorted(str(key) for key in unknown))}'
        )
