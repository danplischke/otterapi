"""Model-generation regressions found by the extended branch review.

Each case compiled and imported but generated the wrong thing, silently:

1. Two schemas sharing a ``title`` were merged into one model, so the second
   one's fields were dropped (``$ref``s were cached by title).
2. A renamed field (``_id`` -> ``field_id``) replaced a real ``field_id``.
3. A schema named like a class the package itself defines (``Client``,
   ``Query``, ``NotFoundError``) shadowed it.
4. Which field names count as reserved depended on the installed pydantic.
5. String enums are ``StrEnum`` now that 3.11 is the oldest target; values
   YAML read as booleans must not break them.
"""

import importlib
import json
import sys
from enum import StrEnum
from pathlib import Path

import pytest
from pydantic import BaseModel

from otterapi.codegen.codegen import Codegen
from otterapi.codegen.types import _PYDANTIC_RESERVED_NAMES
from otterapi.config import DocumentConfig


def _spec(schemas: dict, paths: dict | None = None) -> dict:
    return {
        'openapi': '3.0.3',
        'info': {'title': 'T', 'version': '1'},
        'paths': paths
        or {
            '/a': {
                'get': {
                    'operationId': 'get_a',
                    'responses': {
                        '200': {
                            'description': 'ok',
                            'content': {
                                'application/json': {
                                    'schema': {'$ref': '#/components/schemas/A'}
                                }
                            },
                        }
                    },
                }
            }
        },
        'components': {'schemas': schemas},
    }


def _generate(tmp_path: Path, name: str, spec: dict, **overrides):
    """Generate a package from ``spec`` and import it fresh."""
    spec_file = tmp_path / f'{name}.json'
    spec_file.write_text(json.dumps(spec))
    config = DocumentConfig.model_validate(
        {
            'source': str(spec_file),
            'output': str(tmp_path / name),
            'base_url': 'https://example.test',
            **overrides,
        }
    )
    Codegen(config, format_output=False).generate()
    sys.path.insert(0, str(tmp_path))
    try:
        for mod in [m for m in sys.modules if m == name or m.startswith(name + '.')]:
            sys.modules.pop(mod)
        return importlib.import_module(name)
    finally:
        sys.path.remove(str(tmp_path))


def _object(properties: dict, **extra) -> dict:
    return {'type': 'object', 'properties': properties, **extra}


class TestSharedTitles:
    def test_schemas_sharing_a_title_stay_distinct(self, tmp_path):
        spec = _spec(
            {
                'A': _object(
                    {
                        'home': {'$ref': '#/components/schemas/UserAddress'},
                        'office': {'$ref': '#/components/schemas/CompanyAddress'},
                    }
                ),
                'UserAddress': _object({'street': {'type': 'string'}}, title='Address'),
                'CompanyAddress': _object(
                    {'company': {'type': 'string'}, 'vat': {'type': 'string'}},
                    title='Address',
                ),
            }
        )
        pkg = _generate(tmp_path, 'titles', spec)
        a = pkg.models.A.model_validate(
            {'home': {'street': 's'}, 'office': {'company': 'c', 'vat': 'v'}}
        )
        assert a.home.street == 's'
        assert type(a.office) is not type(a.home)
        assert (a.office.company, a.office.vat) == ('c', 'v')

    def test_recursive_titled_schema_still_terminates(self, tmp_path):
        # Stripe's shape: component key ``charge`` titled ``Charge``, referring
        # back to itself. Caching by $ref must still break the cycle.
        spec = _spec(
            {
                'A': _object({'charge': {'$ref': '#/components/schemas/charge'}}),
                'charge': _object(
                    {
                        'id': {'type': 'string'},
                        'refunded_by': {'$ref': '#/components/schemas/charge'},
                    },
                    title='Charge',
                ),
            }
        )
        pkg = _generate(tmp_path, 'recursive', spec)
        a = pkg.models.A.model_validate(
            {'charge': {'id': '1', 'refunded_by': {'id': '2'}}}
        )
        assert a.charge.refunded_by.id == '2'


class TestFieldNamesStayUnique:
    @pytest.mark.parametrize('order', ['renamed_first', 'natural_first'])
    def test_renamed_field_never_replaces_a_real_one(self, tmp_path, order):
        props = {'_id': {'type': 'string'}, 'field_id': {'type': 'integer'}}
        if order == 'natural_first':
            props = dict(reversed(list(props.items())))
        pkg = _generate(tmp_path, f'uniq_{order}', _spec({'A': _object(props)}))
        a = pkg.models.A.model_validate({'_id': 'x', 'field_id': 7})
        # The real ``field_id`` keeps its name; ``_id`` steps around it.
        assert a.field_id == 7
        assert a.field_id_ == 'x'
        assert a.model_dump(by_alias=True) == {'_id': 'x', 'field_id': 7}


class TestReservedClassNames:
    @pytest.mark.parametrize('client_style', ['functions', 'client'])
    def test_schemas_named_like_package_classes_are_renamed(
        self, tmp_path, client_style
    ):
        names = ['Client', 'AsyncClient', 'Query', 'AsyncQuery', 'NotFoundError']
        schemas = {n: _object({'v': {'type': 'string'}}) for n in names}
        paths = {
            f'/{n.lower()}': {
                'get': {
                    'operationId': f'get_{n.lower()}',
                    'responses': {
                        '200': {
                            'description': 'ok',
                            'content': {
                                'application/json': {
                                    'schema': {
                                        'type': 'array',
                                        'items': {'$ref': f'#/components/schemas/{n}'},
                                    }
                                }
                            },
                        }
                    },
                }
            }
            for n in names
        }
        overrides = {'client_style': client_style}
        if client_style != 'functions':
            overrides['result_objects'] = True
        pkg = _generate(
            tmp_path, f'reserved_{client_style}', _spec(schemas, paths), **overrides
        )
        # The package's own classes keep their names ...
        assert issubclass(pkg.NotFoundError, Exception)
        assert not issubclass(pkg.Client, BaseModel)
        # ... and the schemas moved aside.
        for name in names:
            assert issubclass(getattr(pkg.models, f'{name}_'), BaseModel)


def test_reserved_names_cover_installed_pydantic():
    # The list is pinned so output does not depend on the installed pydantic;
    # this flags a pydantic release that adds a public BaseModel name.
    public = {n for n in dir(BaseModel) if not n.startswith('_')}
    assert public <= _PYDANTIC_RESERVED_NAMES, sorted(public - _PYDANTIC_RESERVED_NAMES)


class TestStringEnums:
    def test_string_enums_are_strenums_and_coerce_yaml_booleans(self, tmp_path):
        # YAML 1.1 reads unquoted ``ON`` / ``OFF`` as booleans; a ``type:
        # string`` enum must still import (StrEnum rejects non-str values).
        spec = _spec(
            {
                'A': _object(
                    {
                        'mode': {
                            'type': 'string',
                            'enum': [True, False, 'AUTO'],
                            'default': 'AUTO',
                        },
                        'level': {'type': 'integer', 'enum': [1, 2]},
                    }
                )
            }
        )
        pkg = _generate(tmp_path, 'strenums', spec)
        mode_enum = type(pkg.models.A().mode)
        assert issubclass(mode_enum, StrEnum)
        assert [m.value for m in mode_enum] == ['True', 'False', 'AUTO']
        assert str(pkg.models.A().mode) == 'AUTO'
        level = pkg.models.A.model_validate({'level': 2}).level
        assert not isinstance(level, str) and level.value == 2
