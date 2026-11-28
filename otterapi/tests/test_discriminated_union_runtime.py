"""End-to-end: generated discriminated unions parse, and their lists export.

``test_discriminated_unions.py`` checks the emitted ``Annotated[...,
Field(discriminator=...)]`` source and runs pydantic against hand-written
``Literal`` models. These tests generate the whole client instead, which is
what exposed two bugs:

1. The variants typed their tag as a one-member enum (``kind: DogKind``), so
   pydantic rejected the union ("needs field 'kind' to be of type Literal")
   and every call of a union-returning endpoint raised.
2. The export wrapper passed the union annotation itself as ``model=``, which
   the writers (they want a model class) could not handle.
"""

import asyncio
import csv
import json
from pathlib import Path

import httpx2
import pytest

from otterapi.tests.helpers import generate_package

SPEC = Path(__file__).parent / 'fixtures' / 'golden' / 'discriminator' / 'spec.yaml'
ROWS = [{'kind': 'dog', 'bark': 'woof'}, {'kind': 'cat', 'meow': 'mew'}]
EXPORT = {'enabled': True, 'formats': ['csv', 'jsonl', 'parquet']}


def _generate(tmp_path: Path, name: str, source: Path = SPEC, **overrides):
    return generate_package(
        tmp_path,
        name,
        source,
        base_url='https://example.test',
        format_output=False,
        **overrides,
    )


def _http(rows=ROWS) -> httpx2.Client:
    return httpx2.Client(
        transport=httpx2.MockTransport(lambda _r: httpx2.Response(200, json=rows))
    )


class TestUnionParses:
    def test_variants_type_their_tag_as_a_literal(self, tmp_path):
        pkg = _generate(tmp_path, 'du_parse')
        with _http() as http:
            animals = pkg.endpoints.list_animals(
                client=pkg.client.Client(http_client=http)
            )
        assert [type(a).__name__ for a in animals] == ['Dog', 'Cat']
        assert animals[0].bark == 'woof' and animals[1].meow == 'mew'
        source = (tmp_path / 'du_parse' / 'models.py').read_text()
        assert "kind: Literal['dog']" in source
        assert 'DogKind' not in source  # no one-member enum for the tag

    def test_unknown_tag_is_rejected(self, tmp_path):
        pkg = _generate(tmp_path, 'du_reject')
        with _http([{'kind': 'fish', 'bark': 'glub'}]) as http:
            with pytest.raises(Exception, match='fish'):
                pkg.endpoints.list_animals(client=pkg.client.Client(http_client=http))


class TestUnionExports:
    @pytest.fixture
    def pkg(self, tmp_path):
        return _generate(tmp_path, 'du_export', export=EXPORT)

    def test_export_wrapper_passes_the_member_models(self, pkg, tmp_path):
        source = (tmp_path / 'du_export' / 'endpoints.py').read_text()
        assert 'model=(Dog, Cat)' in source

    def test_csv_has_every_variants_columns(self, pkg, tmp_path):
        out = tmp_path / 'animals.csv'
        with _http() as http:
            client = pkg.client.Client(http_client=http)
            assert pkg.endpoints.list_animals_export(str(out), client=client) == 2
        with out.open(newline='') as fh:
            rows = list(csv.DictReader(fh))
        assert rows == [
            {'kind': 'dog', 'bark': 'woof', 'meow': ''},
            {'kind': 'cat', 'bark': '', 'meow': 'mew'},
        ]

    def test_jsonl_keeps_each_variants_shape(self, pkg, tmp_path):
        out = tmp_path / 'animals.jsonl'
        with _http() as http:
            client = pkg.client.Client(http_client=http)
            pkg.endpoints.list_animals_export(str(out), format='jsonl', client=client)
        lines = [json.loads(line) for line in out.read_text().splitlines()]
        assert lines == ROWS

    def test_parquet_merges_the_variant_schemas(self, pkg, tmp_path):
        pq = pytest.importorskip('pyarrow.parquet')
        out = tmp_path / 'animals.parquet'
        with _http() as http:
            client = pkg.client.Client(http_client=http)
            pkg.endpoints.list_animals_export(str(out), format='parquet', client=client)
        table = pq.read_table(out)
        assert table.column_names == ['kind', 'bark', 'meow']
        assert table.to_pylist() == [
            {'kind': 'dog', 'bark': 'woof', 'meow': None},
            {'kind': 'cat', 'bark': None, 'meow': 'mew'},
        ]
        # A field only some variants have must be nullable.
        assert table.schema.field('bark').nullable

    def test_async_export(self, pkg, tmp_path):
        out = tmp_path / 'animals_async.csv'

        async def run() -> int:
            transport = httpx2.MockTransport(lambda _r: httpx2.Response(200, json=ROWS))
            async with httpx2.AsyncClient(transport=transport) as http:
                client = pkg.client.Client(async_http_client=http)
                return await pkg.endpoints.async_list_animals_export(
                    str(out), client=client
                )

        assert asyncio.run(run()) == 2
        assert out.read_text().splitlines()[0] == 'kind,bark,meow'

    def test_query_export_on_the_class_client(self, tmp_path):
        pkg = _generate(
            tmp_path,
            'du_query',
            export=EXPORT,
            client_style='client',
            result_objects=True,
        )
        out = tmp_path / 'animals_query.csv'
        with _http() as http:
            assert pkg.Client(http_client=http).list_animals().export(str(out)) == 2

    def test_dict_rows_validate_against_the_union(self, pkg, tmp_path):
        # Calling the writer directly with dicts picks each row's variant.
        out = tmp_path / 'direct.jsonl'
        models = (pkg.models.Dog, pkg.models.Cat)
        assert pkg.export(ROWS, out, model=models, format='jsonl') == 2
        assert [json.loads(line) for line in out.read_text().splitlines()] == ROWS


def _write_spec(tmp_path: Path, name: str, spec: dict) -> Path:
    path = tmp_path / f'{name}.json'
    path.write_text(json.dumps(spec))
    return path


def _union_spec(*, tag_required: bool) -> dict:
    def variant(tag: str, field: str) -> dict:
        return {
            'type': 'object',
            'required': (['kind'] if tag_required else []) + [field],
            'properties': {
                'kind': {'type': 'string', 'enum': [tag]},
                field: {'type': 'string'},
                'status': {'type': 'string', 'enum': ['active']},
            },
        }

    return {
        'openapi': '3.0.3',
        'info': {'title': 'T', 'version': '1'},
        'paths': {
            '/animals': {
                'get': {
                    'operationId': 'list_animals',
                    'responses': {
                        '200': {
                            'description': 'ok',
                            'content': {
                                'application/json': {
                                    'schema': {
                                        'type': 'array',
                                        'items': {
                                            '$ref': '#/components/schemas/Animal'
                                        },
                                    }
                                }
                            },
                        }
                    },
                }
            },
            '/names': {
                'get': {
                    'operationId': 'list_names',
                    'responses': {
                        '200': {
                            'description': 'ok',
                            'content': {
                                'application/json': {
                                    'schema': {
                                        'type': 'array',
                                        'items': {'type': 'string'},
                                    }
                                }
                            },
                        }
                    },
                }
            },
        },
        'components': {
            'schemas': {
                'Animal': {
                    'oneOf': [
                        {'$ref': '#/components/schemas/Dog'},
                        {'$ref': '#/components/schemas/Cat'},
                    ],
                    'discriminator': {'propertyName': 'kind'},
                },
                'Dog': variant('dog', 'bark'),
                'Cat': variant('cat', 'meow'),
            }
        },
    }


class TestTagShapes:
    def test_optional_tag_defaults_to_its_value(self, tmp_path):
        spec = _write_spec(tmp_path, 'opt', _union_spec(tag_required=False))
        pkg = _generate(tmp_path, 'du_optional', spec)
        assert pkg.models.Dog(bark='woof').kind == 'dog'
        with _http(ROWS) as http:
            animals = pkg.endpoints.list_animals(
                client=pkg.client.Client(http_client=http)
            )
        assert [type(a).__name__ for a in animals] == ['Dog', 'Cat']

    def test_other_single_value_enums_stay_enums(self, tmp_path):
        # Only the discriminator becomes a Literal; ``status`` is a plain field.
        spec = _write_spec(tmp_path, 'req', _union_spec(tag_required=True))
        pkg = _generate(tmp_path, 'du_status', spec)
        status = pkg.models.Dog(kind='dog', bark='woof', status='active').status
        assert type(status).__name__.endswith('Status') and status == 'active'

    def test_primitive_list_gets_no_export_variant(self, tmp_path):
        # ``list[str]`` has no columns to write; no broken wrapper is emitted.
        spec = _write_spec(tmp_path, 'prim', _union_spec(tag_required=True))
        pkg = _generate(tmp_path, 'du_prim', spec, export=EXPORT)
        assert hasattr(pkg.endpoints, 'list_animals_export')
        assert not hasattr(pkg.endpoints, 'list_names_export')
