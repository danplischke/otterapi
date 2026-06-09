"""Tests for export configuration and runtime writers."""

from __future__ import annotations

import ast
import asyncio
import csv
import importlib.util
import json
import sys
from datetime import date, datetime, time
from decimal import Decimal
from enum import Enum
from pathlib import Path
from uuid import UUID

import pytest
from pydantic import BaseModel

from otterapi.codegen.export import generate_export_module
from otterapi.config import (
    DocumentConfig,
    EndpointExportConfig,
    ExportConfig,
)

# -----------------------------------------------------------------------------
# Fixtures: pydantic models used across the runtime writer tests
# -----------------------------------------------------------------------------


class Status(str, Enum):
    ACTIVE = 'active'
    INACTIVE = 'inactive'


class User(BaseModel):
    id: int
    name: str
    email: str | None = None
    status: Status = Status.ACTIVE
    created_at: datetime | None = None
    tags: list[str] = []


def _sample_users(n: int = 3) -> list[User]:
    return [
        User(
            id=i,
            name=f'user-{i}',
            email=f'user-{i}@example.com' if i % 2 == 0 else None,
            status=Status.ACTIVE if i % 2 == 0 else Status.INACTIVE,
            created_at=datetime(2024, 1, i + 1, 12, 0, 0),
            tags=['a', 'b'] if i % 2 else [],
        )
        for i in range(n)
    ]


@pytest.fixture(scope='module')
def exported_runtime(tmp_path_factory):
    """Generate and import the runtime ``_export.py`` module once per session."""
    directory = tmp_path_factory.mktemp('exported_runtime')
    generate_export_module(directory)
    module_path = Path(directory) / '_export.py'

    spec = importlib.util.spec_from_file_location('_otterapi_export_test', module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules['_otterapi_export_test'] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop('_otterapi_export_test', None)


# -----------------------------------------------------------------------------
# Config-model tests
# -----------------------------------------------------------------------------


class TestEndpointExportConfig:
    def test_default_values(self):
        config = EndpointExportConfig()
        assert config.enabled is None
        assert config.path is None
        assert config.formats is None

    def test_explicit_values(self):
        config = EndpointExportConfig(
            enabled=True, path='data.items', formats=['parquet']
        )
        assert config.enabled is True
        assert config.path == 'data.items'
        assert config.formats == ['parquet']

    def test_rejects_extra_fields(self):
        with pytest.raises(Exception):
            EndpointExportConfig(unknown='value')


class TestExportConfig:
    def test_defaults(self):
        config = ExportConfig()
        assert config.enabled is False
        assert config.formats == ['csv', 'jsonl']
        assert config.default_path is None
        assert config.include_all is True
        assert config.batch_size == 1000
        assert config.endpoints == {}

    def test_disabled_skips_generation(self):
        config = ExportConfig(enabled=False)
        should_gen, formats, path = config.should_generate_for_endpoint(
            'list_users', returns_list=True
        )
        assert should_gen is False
        assert formats == []
        assert path is None

    def test_enabled_default_applies_to_all_list_endpoints(self):
        config = ExportConfig(enabled=True, formats=['csv', 'parquet'])
        should_gen, formats, path = config.should_generate_for_endpoint(
            'list_users', returns_list=True
        )
        assert should_gen is True
        assert formats == ['csv', 'parquet']
        assert path is None

    def test_non_list_endpoint_skipped_by_default(self):
        config = ExportConfig(enabled=True)
        should_gen, _, _ = config.should_generate_for_endpoint(
            'get_user', returns_list=False
        )
        assert should_gen is False

    def test_endpoint_override_enables_non_list(self):
        config = ExportConfig(
            enabled=True,
            endpoints={'get_user': EndpointExportConfig(enabled=True, path='data')},
        )
        should_gen, formats, path = config.should_generate_for_endpoint(
            'get_user', returns_list=False
        )
        assert should_gen is True
        assert formats == ['csv', 'jsonl']
        assert path == 'data'

    def test_endpoint_override_disables_list(self):
        config = ExportConfig(
            enabled=True,
            endpoints={'list_users': EndpointExportConfig(enabled=False)},
        )
        should_gen, _, _ = config.should_generate_for_endpoint(
            'list_users', returns_list=True
        )
        assert should_gen is False

    def test_endpoint_override_formats(self):
        config = ExportConfig(
            enabled=True,
            formats=['csv'],
            endpoints={
                'list_users': EndpointExportConfig(formats=['parquet', 'jsonl'])
            },
        )
        should_gen, formats, _ = config.should_generate_for_endpoint(
            'list_users', returns_list=True
        )
        assert should_gen is True
        assert formats == ['parquet', 'jsonl']

    def test_rejects_extra_fields(self):
        with pytest.raises(Exception):
            ExportConfig(unknown='value')


class TestDocumentConfigExport:
    def test_document_has_export_field(self):
        doc = DocumentConfig(source='spec.json', output='./out')
        assert isinstance(doc.export, ExportConfig)
        assert doc.export.enabled is False

    def test_document_accepts_export_overrides(self):
        doc = DocumentConfig(
            source='spec.json',
            output='./out',
            export={'enabled': True, 'formats': ['parquet']},
        )
        assert doc.export.enabled is True
        assert doc.export.formats == ['parquet']


# -----------------------------------------------------------------------------
# Module generation smoke test
# -----------------------------------------------------------------------------


class TestGenerateExportModule:
    def test_writes_file(self, tmp_path):
        result = generate_export_module(tmp_path)
        assert result.exists()
        content = Path(result).read_text(encoding='utf-8')
        assert 'def to_csv(' in content
        assert 'def to_parquet(' in content
        assert 'pydantic_to_arrow_schema' in content

    def test_generated_module_is_syntactically_valid(self, tmp_path):
        import ast as _ast

        result = generate_export_module(tmp_path)
        _ast.parse(Path(result).read_text(encoding='utf-8'))


# -----------------------------------------------------------------------------
# Runtime writer tests (exercise the generated _export.py in-place)
# -----------------------------------------------------------------------------


class TestPydanticToArrowSchema:
    def test_maps_scalar_fields(self, exported_runtime):
        schema = exported_runtime.pydantic_to_arrow_schema(User)
        names = schema.names
        assert names == ['id', 'name', 'email', 'status', 'created_at', 'tags']

        types_by_name = {field.name: field for field in schema}
        assert str(types_by_name['id'].type) == 'int64'
        assert str(types_by_name['name'].type) == 'string'
        assert types_by_name['email'].nullable is True
        assert types_by_name['created_at'].nullable is True
        assert 'timestamp' in str(types_by_name['created_at'].type)
        # list[str] → list<string>
        assert 'list' in str(types_by_name['tags'].type)

    def test_naive_datetime_maps_to_no_tz_timestamp(self, exported_runtime):
        """Naive datetime must not produce a UTC-timezone column (breaks at write time)."""
        schema = exported_runtime.pydantic_to_arrow_schema(User)
        dt_type = str(schema.field('created_at').type)
        assert 'timestamp' in dt_type
        assert 'UTC' not in dt_type

    def test_pydantic_datetime_sentinels_map_correctly(self, exported_runtime):
        """AwareDatetime (a Pydantic sentinel class) must resolve to timestamp, not string."""
        try:
            from pydantic import AwareDatetime
        except ImportError:
            pytest.skip('pydantic.AwareDatetime not available')

        class EventModel(BaseModel):
            ts: AwareDatetime

        schema = exported_runtime.pydantic_to_arrow_schema(EventModel)
        assert 'timestamp' in str(schema.field('ts').type)

    def test_tuple_homogeneous_maps_to_list(self, exported_runtime):
        """tuple[T, ...] must map to list<T>, not crash."""

        class TupleModel(BaseModel):
            coords: tuple[float, ...]

        schema = exported_runtime.pydantic_to_arrow_schema(TupleModel)
        assert 'list' in str(schema.field('coords').type)

    def test_tuple_heterogeneous_maps_to_string(self, exported_runtime):
        """tuple[A, B] must fall back to string rather than crashing."""

        class PairModel(BaseModel):
            pair: tuple[int, str]

        schema = exported_runtime.pydantic_to_arrow_schema(PairModel)
        assert str(schema.field('pair').type) == 'string'


class TestParquetTypeHandlingRegression:
    """Regression tests for the four type-handling bugs."""

    def test_naive_datetime_writes_without_error(self, exported_runtime, tmp_path):
        """Naive datetime values must write to Parquet without ArrowInvalid."""
        pytest.importorskip('pyarrow')
        pq = pytest.importorskip('pyarrow.parquet')

        path = tmp_path / 'naive_dt.parquet'
        rows = [User(id=1, name='a', created_at=datetime(2024, 6, 1, 10, 0, 0))]
        exported_runtime.to_parquet(rows, path, model=User)
        table = pq.read_table(path)
        assert table.num_rows == 1

    def test_homogeneous_tuple_round_trips(self, exported_runtime, tmp_path):
        """tuple[T, ...] fields must write and read back correctly."""
        pytest.importorskip('pyarrow')
        pq = pytest.importorskip('pyarrow.parquet')

        class CoordModel(BaseModel):
            pts: tuple[float, ...]

        path = tmp_path / 'tuples.parquet'
        exported_runtime.to_parquet(
            [CoordModel(pts=(1.0, 2.0, 3.0))], path, model=CoordModel
        )
        table = pq.read_table(path)
        assert table.num_rows == 1

    def test_multi_union_field_coerced_to_string(self, exported_runtime, tmp_path):
        """Union[int, str] values must be coerced to string to match the schema."""
        pytest.importorskip('pyarrow')
        pq = pytest.importorskip('pyarrow.parquet')

        class FlexModel(BaseModel):
            value: int | str

        path = tmp_path / 'union.parquet'
        exported_runtime.to_parquet(
            [FlexModel(value=42), FlexModel(value='hello')], path, model=FlexModel
        )
        table = pq.read_table(path)
        assert table.num_rows == 2
        assert str(table.schema.field('value').type) == 'string'


class TestCsvWriter:
    def test_round_trip(self, exported_runtime, tmp_path):
        path = tmp_path / 'users.csv'
        rows = _sample_users(3)
        written = exported_runtime.to_csv(iter(rows), path, model=User)
        assert written == 3

        with open(path, newline='', encoding='utf-8') as fh:
            reader = csv.DictReader(fh)
            records = list(reader)
        assert [r['name'] for r in records] == ['user-0', 'user-1', 'user-2']
        # Missing-optional rendered as empty
        assert records[1]['email'] == ''
        # list[str] encoded as JSON
        assert records[1]['tags'] == '["a","b"]'

    def test_empty_iterable(self, exported_runtime, tmp_path):
        path = tmp_path / 'empty.csv'
        written = exported_runtime.to_csv(iter([]), path, model=User)
        assert written == 0
        # Header is still written
        assert (
            path.read_text(encoding='utf-8').splitlines()[0].startswith('id,name,email')
        )

    def test_accepts_dicts(self, exported_runtime, tmp_path):
        path = tmp_path / 'from_dicts.csv'
        written = exported_runtime.to_csv(
            [{'id': 1, 'name': 'a'}, {'id': 2, 'name': 'b'}],
            path,
            model=User,
        )
        assert written == 2


class TestTsvWriter:
    def test_uses_tab_delimiter(self, exported_runtime, tmp_path):
        path = tmp_path / 'users.tsv'
        exported_runtime.to_tsv(_sample_users(2), path, model=User)
        line = path.read_text(encoding='utf-8').splitlines()[0]
        assert '\t' in line and ',' not in line


class TestJsonlWriter:
    def test_round_trip(self, exported_runtime, tmp_path):
        path = tmp_path / 'users.jsonl'
        rows = _sample_users(3)
        written = exported_runtime.to_jsonl(iter(rows), path, model=User)
        assert written == 3

        lines = [
            json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()
        ]
        assert [r['name'] for r in lines] == ['user-0', 'user-1', 'user-2']
        # Datetime serialized as ISO-8601 string
        assert isinstance(lines[0]['created_at'], str)

    def test_empty_iterable(self, exported_runtime, tmp_path):
        path = tmp_path / 'empty.jsonl'
        assert exported_runtime.to_jsonl([], path, model=User) == 0
        assert path.read_text(encoding='utf-8') == ''


class TestParquetWriter:
    def test_round_trip(self, exported_runtime, tmp_path):
        pytest.importorskip('pyarrow')
        pq = pytest.importorskip('pyarrow.parquet')

        path = tmp_path / 'users.parquet'
        rows = _sample_users(5)
        written = exported_runtime.to_parquet(iter(rows), path, model=User)
        assert written == 5

        table = pq.read_table(path)
        assert table.num_rows == 5
        assert table.column_names == [
            'id',
            'name',
            'email',
            'status',
            'created_at',
            'tags',
        ]
        # Schema comes from the model, not the data.
        assert str(table.schema.field('id').type) == 'int64'
        assert table.schema.field('email').nullable is True

    def test_empty_iterable_writes_valid_empty_file(self, exported_runtime, tmp_path):
        pytest.importorskip('pyarrow')
        pq = pytest.importorskip('pyarrow.parquet')

        path = tmp_path / 'empty.parquet'
        assert exported_runtime.to_parquet([], path, model=User) == 0
        table = pq.read_table(path)
        assert table.num_rows == 0
        assert table.column_names == list(User.model_fields.keys())

    def test_batches_produce_same_schema(self, exported_runtime, tmp_path):
        pytest.importorskip('pyarrow')
        pq = pytest.importorskip('pyarrow.parquet')

        path = tmp_path / 'batched.parquet'
        # First batch has null email, second has strings; schema must stay stable.
        rows = [
            User(id=1, name='a', email=None),
            User(id=2, name='b', email=None),
            User(id=3, name='c', email='c@x.io'),
        ]
        exported_runtime.to_parquet(iter(rows), path, model=User, batch_size=2)
        table = pq.read_table(path)
        assert table.num_rows == 3
        assert table.schema.field('email').nullable is True


class TestExportDispatch:
    @pytest.mark.parametrize('fmt', ['csv', 'tsv', 'jsonl'])
    def test_dispatch_writes(self, exported_runtime, tmp_path, fmt):
        path = tmp_path / f'users.{fmt}'
        written = exported_runtime.export(
            _sample_users(2), path, model=User, format=fmt
        )
        assert written == 2
        assert path.exists()

    def test_dispatch_rejects_unknown_format(self, exported_runtime, tmp_path):
        with pytest.raises(ValueError, match='Unsupported export format'):
            exported_runtime.export(
                _sample_users(1), tmp_path / 'x', model=User, format='xml'
            )


class TestAsyncWriters:
    async def _aiter(self, items):
        for it in items:
            yield it

    def test_csv_async(self, exported_runtime, tmp_path):
        path = tmp_path / 'async.csv'

        async def run():
            return await exported_runtime.to_csv_async(
                self._aiter(_sample_users(3)), path, model=User
            )

        written = asyncio.run(run())
        assert written == 3
        assert 'user-0' in path.read_text(encoding='utf-8')

    def test_jsonl_async(self, exported_runtime, tmp_path):
        path = tmp_path / 'async.jsonl'

        async def run():
            return await exported_runtime.to_jsonl_async(
                self._aiter(_sample_users(2)), path, model=User
            )

        written = asyncio.run(run())
        assert written == 2
        assert len(path.read_text(encoding='utf-8').splitlines()) == 2

    def test_parquet_async(self, exported_runtime, tmp_path):
        pytest.importorskip('pyarrow')
        pq = pytest.importorskip('pyarrow.parquet')

        path = tmp_path / 'async.parquet'

        async def run():
            return await exported_runtime.to_parquet_async(
                self._aiter(_sample_users(4)), path, model=User
            )

        written = asyncio.run(run())
        assert written == 4
        assert pq.read_table(path).num_rows == 4


# -----------------------------------------------------------------------------
# Extended schema-mapping tests
# -----------------------------------------------------------------------------


class IntStatus(int, Enum):
    ACTIVE = 1
    INACTIVE = 0


class Address(BaseModel):
    street: str
    city: str


class RichModel(BaseModel):
    uid: UUID
    flag: bool
    score: float
    amount: Decimal
    raw: bytes
    born: date
    wakeup: time
    meta: dict[str, str]
    addr: Address
    rank: IntStatus
    labels: set[str]
    coords: frozenset[float]
    counts: list[int]


class TestPydanticToArrowSchemaExtended:
    def test_uuid_maps_to_string(self, exported_runtime):
        pytest.importorskip('pyarrow')
        schema = exported_runtime.pydantic_to_arrow_schema(RichModel)
        assert str(schema.field('uid').type) == 'string'

    def test_bool_maps_to_bool(self, exported_runtime):
        pytest.importorskip('pyarrow')
        schema = exported_runtime.pydantic_to_arrow_schema(RichModel)
        assert str(schema.field('flag').type) == 'bool'

    def test_decimal_maps_to_decimal128(self, exported_runtime):
        pytest.importorskip('pyarrow')
        schema = exported_runtime.pydantic_to_arrow_schema(RichModel)
        assert 'decimal' in str(schema.field('amount').type)

    def test_bytes_maps_to_binary(self, exported_runtime):
        pytest.importorskip('pyarrow')
        schema = exported_runtime.pydantic_to_arrow_schema(RichModel)
        assert str(schema.field('raw').type) == 'binary'

    def test_date_maps_to_date32(self, exported_runtime):
        pytest.importorskip('pyarrow')
        schema = exported_runtime.pydantic_to_arrow_schema(RichModel)
        assert str(schema.field('born').type) == 'date32[day]'

    def test_time_maps_to_time64(self, exported_runtime):
        pytest.importorskip('pyarrow')
        schema = exported_runtime.pydantic_to_arrow_schema(RichModel)
        assert 'time64' in str(schema.field('wakeup').type)

    def test_dict_falls_back_to_string(self, exported_runtime):
        pytest.importorskip('pyarrow')
        schema = exported_runtime.pydantic_to_arrow_schema(RichModel)
        assert str(schema.field('meta').type) == 'string'

    def test_nested_model_maps_to_struct(self, exported_runtime):
        pytest.importorskip('pyarrow')
        schema = exported_runtime.pydantic_to_arrow_schema(RichModel)
        assert 'struct' in str(schema.field('addr').type)

    def test_int_enum_maps_to_int64(self, exported_runtime):
        pytest.importorskip('pyarrow')
        schema = exported_runtime.pydantic_to_arrow_schema(RichModel)
        assert str(schema.field('rank').type) == 'int64'

    def test_set_maps_to_list(self, exported_runtime):
        pytest.importorskip('pyarrow')
        schema = exported_runtime.pydantic_to_arrow_schema(RichModel)
        assert 'list' in str(schema.field('labels').type)

    def test_frozenset_maps_to_list(self, exported_runtime):
        pytest.importorskip('pyarrow')
        schema = exported_runtime.pydantic_to_arrow_schema(RichModel)
        assert 'list' in str(schema.field('coords').type)

    def test_list_int_maps_to_list_int64(self, exported_runtime):
        pytest.importorskip('pyarrow')
        schema = exported_runtime.pydantic_to_arrow_schema(RichModel)
        field_str = str(schema.field('counts').type)
        assert 'list' in field_str and 'int64' in field_str

    def test_annotated_constrained_type_unwrapped(self, exported_runtime):
        pytest.importorskip('pyarrow')
        from pydantic import PositiveInt

        class ConstrainedModel(BaseModel):
            count: PositiveInt

        schema = exported_runtime.pydantic_to_arrow_schema(ConstrainedModel)
        assert str(schema.field('count').type) == 'int64'

    def test_optional_field_marked_nullable(self, exported_runtime):
        pytest.importorskip('pyarrow')

        class NullableModel(BaseModel):
            x: bool | None = None

        schema = exported_runtime.pydantic_to_arrow_schema(NullableModel)
        assert schema.field('x').nullable is True
        assert str(schema.field('x').type) == 'bool'

    def test_bare_unparameterized_list_falls_back_to_string(self, exported_runtime):
        """Bare ``list`` without type args has no origin, so it falls back to string."""
        pytest.importorskip('pyarrow')

        class BareModel(BaseModel):
            items: list

        schema = exported_runtime.pydantic_to_arrow_schema(BareModel)
        assert str(schema.field('items').type) == 'string'


# -----------------------------------------------------------------------------
# Extended CSV / TSV writer tests
# -----------------------------------------------------------------------------


class TestCsvWriterExtended:
    def test_no_header(self, exported_runtime, tmp_path):
        path = tmp_path / 'no_header.csv'
        exported_runtime.to_csv(_sample_users(2), path, model=User, header=False)
        lines = path.read_text(encoding='utf-8').splitlines()
        # First line should be data, not a header
        assert lines[0].startswith('0,')

    def test_batch_size_one_produces_correct_output(self, exported_runtime, tmp_path):
        path = tmp_path / 'batched.csv'
        written = exported_runtime.to_csv(_sample_users(3), path, model=User, batch_size=1)
        assert written == 3
        with open(path, newline='', encoding='utf-8') as fh:
            records = list(csv.DictReader(fh))
        assert len(records) == 3

    def test_list_field_json_encoded(self, exported_runtime, tmp_path):
        path = tmp_path / 'lists.csv'
        exported_runtime.to_csv([User(id=1, name='a', tags=['x', 'y'])], path, model=User)
        with open(path, newline='', encoding='utf-8') as fh:
            rows = list(csv.DictReader(fh))
        assert rows[0]['tags'] == '["x","y"]'

    def test_none_optional_renders_as_empty(self, exported_runtime, tmp_path):
        path = tmp_path / 'nulls.csv'
        exported_runtime.to_csv([User(id=1, name='a', email=None)], path, model=User)
        with open(path, newline='', encoding='utf-8') as fh:
            rows = list(csv.DictReader(fh))
        assert rows[0]['email'] == ''

    def test_enum_field_rendered_as_value(self, exported_runtime, tmp_path):
        path = tmp_path / 'enum.csv'
        exported_runtime.to_csv(
            [User(id=1, name='a', status=Status.INACTIVE)], path, model=User
        )
        with open(path, newline='', encoding='utf-8') as fh:
            rows = list(csv.DictReader(fh))
        assert rows[0]['status'] == 'inactive'


# -----------------------------------------------------------------------------
# Extended JSONL writer tests
# -----------------------------------------------------------------------------


class TestJsonlWriterExtended:
    def test_dict_rows_accepted(self, exported_runtime, tmp_path):
        path = tmp_path / 'dicts.jsonl'
        written = exported_runtime.to_jsonl(
            [{'id': 1, 'name': 'alice'}, {'id': 2, 'name': 'bob'}],
            path,
            model=User,
        )
        assert written == 2
        records = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
        assert records[0]['id'] == 1
        assert records[1]['name'] == 'bob'

    def test_none_optional_serialized_as_null(self, exported_runtime, tmp_path):
        path = tmp_path / 'nulls.jsonl'
        exported_runtime.to_jsonl([User(id=1, name='a', email=None)], path, model=User)
        record = json.loads(path.read_text(encoding='utf-8').strip())
        assert record['email'] is None

    def test_enum_serialized_as_value(self, exported_runtime, tmp_path):
        path = tmp_path / 'enum.jsonl'
        exported_runtime.to_jsonl(
            [User(id=1, name='a', status=Status.INACTIVE)], path, model=User
        )
        record = json.loads(path.read_text(encoding='utf-8').strip())
        assert record['status'] == 'inactive'

    def test_batch_size_one(self, exported_runtime, tmp_path):
        path = tmp_path / 'batched.jsonl'
        written = exported_runtime.to_jsonl(_sample_users(4), path, model=User, batch_size=1)
        assert written == 4
        assert len(path.read_text(encoding='utf-8').splitlines()) == 4


# -----------------------------------------------------------------------------
# Extended Parquet writer tests
# -----------------------------------------------------------------------------


class TestParquetWriterExtended:
    def test_uuid_field_written_as_string(self, exported_runtime, tmp_path):
        pytest.importorskip('pyarrow')
        pq = pytest.importorskip('pyarrow.parquet')

        class UUIDModel(BaseModel):
            id: UUID
            name: str

        path = tmp_path / 'uuids.parquet'
        u = UUID('12345678-1234-5678-1234-567812345678')
        exported_runtime.to_parquet([UUIDModel(id=u, name='test')], path, model=UUIDModel)
        table = pq.read_table(path)
        assert str(table.schema.field('id').type) == 'string'
        assert table.column('id')[0].as_py() == str(u)

    def test_decimal_field_preserved(self, exported_runtime, tmp_path):
        pytest.importorskip('pyarrow')
        pq = pytest.importorskip('pyarrow.parquet')

        class DecimalModel(BaseModel):
            price: Decimal

        path = tmp_path / 'decimals.parquet'
        exported_runtime.to_parquet(
            [DecimalModel(price=Decimal('19.99'))], path, model=DecimalModel
        )
        table = pq.read_table(path)
        assert 'decimal' in str(table.schema.field('price').type)

    def test_int_enum_written_as_integer(self, exported_runtime, tmp_path):
        pytest.importorskip('pyarrow')
        pq = pytest.importorskip('pyarrow.parquet')

        class EnumModel(BaseModel):
            rank: IntStatus

        path = tmp_path / 'int_enum.parquet'
        exported_runtime.to_parquet(
            [EnumModel(rank=IntStatus.ACTIVE)], path, model=EnumModel
        )
        table = pq.read_table(path)
        assert str(table.schema.field('rank').type) == 'int64'
        assert table.column('rank')[0].as_py() == 1

    def test_str_enum_written_as_string(self, exported_runtime, tmp_path):
        pytest.importorskip('pyarrow')
        pq = pytest.importorskip('pyarrow.parquet')

        class StatusModel(BaseModel):
            status: Status

        path = tmp_path / 'str_enum.parquet'
        exported_runtime.to_parquet(
            [StatusModel(status=Status.INACTIVE)], path, model=StatusModel
        )
        table = pq.read_table(path)
        assert str(table.schema.field('status').type) == 'string'
        assert table.column('status')[0].as_py() == 'inactive'

    def test_nested_model_written_as_struct(self, exported_runtime, tmp_path):
        pytest.importorskip('pyarrow')
        pq = pytest.importorskip('pyarrow.parquet')

        class Outer(BaseModel):
            name: str
            addr: Address

        path = tmp_path / 'nested.parquet'
        exported_runtime.to_parquet(
            [Outer(name='alice', addr=Address(street='1 Main St', city='Springfield'))],
            path,
            model=Outer,
        )
        table = pq.read_table(path)
        assert 'struct' in str(table.schema.field('addr').type)

    def test_dict_rows_accepted(self, exported_runtime, tmp_path):
        pytest.importorskip('pyarrow')
        pq = pytest.importorskip('pyarrow.parquet')

        path = tmp_path / 'from_dicts.parquet'
        written = exported_runtime.to_parquet(
            [{'id': 1, 'name': 'a'}, {'id': 2, 'name': 'b'}],
            path,
            model=User,
        )
        assert written == 2
        assert pq.read_table(path).num_rows == 2

    def test_no_compression(self, exported_runtime, tmp_path):
        pytest.importorskip('pyarrow')
        pq = pytest.importorskip('pyarrow.parquet')

        path = tmp_path / 'uncompressed.parquet'
        exported_runtime.to_parquet(_sample_users(2), path, model=User, compression=None)
        assert pq.read_table(path).num_rows == 2

    def test_bool_field_round_trip(self, exported_runtime, tmp_path):
        pytest.importorskip('pyarrow')
        pq = pytest.importorskip('pyarrow.parquet')

        class BoolModel(BaseModel):
            active: bool

        path = tmp_path / 'bools.parquet'
        exported_runtime.to_parquet(
            [BoolModel(active=True), BoolModel(active=False)], path, model=BoolModel
        )
        table = pq.read_table(path)
        assert str(table.schema.field('active').type) == 'bool'
        assert table.column('active')[0].as_py() is True
        assert table.column('active')[1].as_py() is False

    def test_date_field_round_trip(self, exported_runtime, tmp_path):
        pytest.importorskip('pyarrow')
        pq = pytest.importorskip('pyarrow.parquet')

        class DateModel(BaseModel):
            born: date

        path = tmp_path / 'dates.parquet'
        exported_runtime.to_parquet(
            [DateModel(born=date(1990, 6, 15))], path, model=DateModel
        )
        table = pq.read_table(path)
        assert str(table.schema.field('born').type) == 'date32[day]'


# -----------------------------------------------------------------------------
# Extended dispatch tests (sync + async)
# -----------------------------------------------------------------------------


class TestExportDispatchExtended:
    def test_dispatch_parquet(self, exported_runtime, tmp_path):
        pytest.importorskip('pyarrow')
        pq = pytest.importorskip('pyarrow.parquet')

        path = tmp_path / 'users.parquet'
        written = exported_runtime.export(
            _sample_users(3), path, model=User, format='parquet'
        )
        assert written == 3
        assert pq.read_table(path).num_rows == 3

    def test_export_async_csv(self, exported_runtime, tmp_path):
        path = tmp_path / 'async_dispatch.csv'

        async def _aiter(items):
            for it in items:
                yield it

        async def run():
            return await exported_runtime.export_async(
                _aiter(_sample_users(2)), path, model=User, format='csv'
            )

        written = asyncio.run(run())
        assert written == 2
        assert 'user-0' in path.read_text(encoding='utf-8')

    def test_export_async_jsonl(self, exported_runtime, tmp_path):
        path = tmp_path / 'async_dispatch.jsonl'

        async def _aiter(items):
            for it in items:
                yield it

        async def run():
            return await exported_runtime.export_async(
                _aiter(_sample_users(2)), path, model=User, format='jsonl'
            )

        asyncio.run(run())
        assert len(path.read_text(encoding='utf-8').splitlines()) == 2

    def test_export_async_parquet(self, exported_runtime, tmp_path):
        pytest.importorskip('pyarrow')
        pq = pytest.importorskip('pyarrow.parquet')

        path = tmp_path / 'async_dispatch.parquet'

        async def _aiter(items):
            for it in items:
                yield it

        async def run():
            return await exported_runtime.export_async(
                _aiter(_sample_users(3)), path, model=User, format='parquet'
            )

        written = asyncio.run(run())
        assert written == 3
        assert pq.read_table(path).num_rows == 3

    def test_export_async_rejects_unknown_format(self, exported_runtime, tmp_path):
        async def _aiter(items):
            for it in items:
                yield it

        async def run():
            await exported_runtime.export_async(
                _aiter([]), tmp_path / 'x', model=User, format='excel'
            )

        with pytest.raises(ValueError, match='Unsupported export format'):
            asyncio.run(run())


# -----------------------------------------------------------------------------
# Per-endpoint AST builders
# -----------------------------------------------------------------------------


def _make_param(name: str, *, required: bool = False, location: str = 'query'):
    """Build a minimal Parameter object for AST builder tests."""
    from otterapi.codegen.types import Parameter, Type

    int_type = Type(
        reference=None,
        name='int',
        type='primitive',
        annotation_ast=ast.Name(id='int', ctx=ast.Load()),
        annotation_imports={},
    )
    return Parameter(
        name=name,
        name_sanitized=name,
        location=location,
        required=required,
        type=int_type,
    )


def _kwarg_names(fn_ast):
    return [a.arg for a in fn_ast.args.kwonlyargs]


class TestBuildStandaloneExportFn:
    def test_sync_returns_int_with_export_kwargs(self):
        from otterapi.codegen.export import build_standalone_export_fn

        fn_ast, imports = build_standalone_export_fn(
            fn_name='list_pets_export',
            target_fn_name='list_pets',
            parameters=[_make_param('limit')],
            request_body_info=None,
            item_type_ast=ast.Name(id='Pet', ctx=ast.Load()),
            item_type_imports={'.models': {'Pet'}},
            docs='List pets.',
            is_async=False,
        )

        assert isinstance(fn_ast, ast.FunctionDef)
        assert fn_ast.name == 'list_pets_export'
        assert isinstance(fn_ast.returns, ast.Name)
        assert fn_ast.returns.id == 'int'

        kwargs = _kwarg_names(fn_ast)
        assert 'output_path' in kwargs
        assert 'format' in kwargs
        assert 'batch_size' in kwargs
        # Mirrors the underlying optional parameter
        assert 'limit' in kwargs
        # Forwards a generic kwargs catcher for format-specific options
        assert fn_ast.args.kwarg is not None
        assert fn_ast.args.kwarg.arg == 'format_kwargs'

        # Body delegates to the underlying function and pipes to ``export``
        body_calls = [
            stmt
            for stmt in fn_ast.body
            if isinstance(stmt, ast.Assign)
            and isinstance(stmt.value, ast.Call)
            and isinstance(stmt.value.func, ast.Name)
        ]
        assert any(call.value.func.id == 'list_pets' for call in body_calls)

        return_stmt = fn_ast.body[-1]
        assert isinstance(return_stmt, ast.Return)
        assert isinstance(return_stmt.value, ast.Call)
        assert isinstance(return_stmt.value.func, ast.Name)
        assert return_stmt.value.func.id == 'export'

        assert imports['._export'] == {'export'}
        assert imports['pathlib'] == {'Path'}
        assert 'Literal' in imports['typing']
        assert imports['.models'] == {'Pet'}

    def test_async_uses_await_and_sync_writer(self):
        from otterapi.codegen.export import build_standalone_export_fn

        fn_ast, imports = build_standalone_export_fn(
            fn_name='list_pets_async_export',
            target_fn_name='list_pets_async',
            parameters=None,
            request_body_info=None,
            item_type_ast=ast.Name(id='Pet', ctx=ast.Load()),
            item_type_imports={'.models': {'Pet'}},
            docs=None,
            is_async=True,
        )

        assert isinstance(fn_ast, ast.AsyncFunctionDef)
        # Underlying call is awaited (the list materializes before writing).
        assign = next(stmt for stmt in fn_ast.body if isinstance(stmt, ast.Assign))
        assert isinstance(assign.value, ast.Await)
        # Non-paginated → still uses the sync ``export`` writer.
        return_stmt = fn_ast.body[-1]
        assert isinstance(return_stmt, ast.Return)
        assert not isinstance(return_stmt.value, ast.Await)
        assert imports['._export'] == {'export'}

    def test_default_format_propagates_to_signature(self):
        from otterapi.codegen.export import build_standalone_export_fn

        fn_ast, _ = build_standalone_export_fn(
            fn_name='list_pets_export',
            target_fn_name='list_pets',
            parameters=None,
            request_body_info=None,
            item_type_ast=ast.Name(id='Pet', ctx=ast.Load()),
            item_type_imports=None,
            docs=None,
            is_async=False,
            default_format='parquet',
            default_batch_size=500,
        )
        defaults = dict(zip(_kwarg_names(fn_ast), fn_ast.args.kw_defaults))
        assert defaults['format'].value == 'parquet'
        assert defaults['batch_size'].value == 500


class TestBuildStandalonePaginatedExportFn:
    def test_sync_targets_iter_and_adds_pagination_knobs(self):
        from otterapi.codegen.export import build_standalone_paginated_export_fn

        fn_ast, imports = build_standalone_paginated_export_fn(
            fn_name='get_users_export',
            target_iter_fn_name='get_users_iter',
            parameters=[_make_param('search')],
            request_body_info=None,
            item_type_ast=ast.Name(id='User', ctx=ast.Load()),
            item_type_imports={'.models': {'User'}},
            docs=None,
            is_async=False,
        )

        assert isinstance(fn_ast, ast.FunctionDef)
        kwargs = _kwarg_names(fn_ast)
        for expected in (
            'output_path',
            'format',
            'batch_size',
            'page_size',
            'max_items',
        ):
            assert expected in kwargs

        # Forwards page_size and max_items into the iter call.
        rows_assign = next(
            stmt
            for stmt in fn_ast.body
            if isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Call)
        )
        call = rows_assign.value
        assert isinstance(call.func, ast.Name)
        assert call.func.id == 'get_users_iter'
        forwarded = {kw.arg for kw in call.keywords}
        assert {'search', 'page_size', 'max_items', 'client'}.issubset(forwarded)

        # Sync paginated → still uses the sync ``export`` writer.
        assert imports['._export'] == {'export'}

    def test_async_uses_export_async_and_awaits(self):
        from otterapi.codegen.export import build_standalone_paginated_export_fn

        fn_ast, imports = build_standalone_paginated_export_fn(
            fn_name='get_users_async_export',
            target_iter_fn_name='get_users_async_iter',
            parameters=None,
            request_body_info=None,
            item_type_ast=ast.Name(id='User', ctx=ast.Load()),
            item_type_imports={'.models': {'User'}},
            docs=None,
            is_async=True,
        )

        assert isinstance(fn_ast, ast.AsyncFunctionDef)
        # The async-iter call returns an AsyncIterator → assigned, not awaited.
        rows_assign = next(stmt for stmt in fn_ast.body if isinstance(stmt, ast.Assign))
        assert not isinstance(rows_assign.value, ast.Await)
        # Return statement awaits the async writer.
        return_stmt = fn_ast.body[-1]
        assert isinstance(return_stmt, ast.Return)
        assert isinstance(return_stmt.value, ast.Await)
        inner = return_stmt.value.value
        assert isinstance(inner.func, ast.Name)
        assert inner.func.id == 'export_async'
        assert imports['._export'] == {'export_async'}


class TestGeneratedFunctionsAreSyntacticallyValid:
    """Round-trip the generated AST through ast.unparse + ast.parse."""

    def _module(self, fn_ast):
        return ast.Module(body=[fn_ast], type_ignores=[])

    def test_standalone_export_unparses(self):
        from otterapi.codegen.export import build_standalone_export_fn

        fn_ast, _ = build_standalone_export_fn(
            fn_name='list_pets_export',
            target_fn_name='list_pets',
            parameters=[_make_param('limit')],
            request_body_info=None,
            item_type_ast=ast.Name(id='Pet', ctx=ast.Load()),
            item_type_imports=None,
            docs=None,
            is_async=False,
        )
        source = ast.unparse(ast.fix_missing_locations(self._module(fn_ast)))
        ast.parse(source)
        assert 'def list_pets_export' in source
        assert 'export(' in source

    def test_paginated_export_unparses(self):
        from otterapi.codegen.export import build_standalone_paginated_export_fn

        fn_ast, _ = build_standalone_paginated_export_fn(
            fn_name='get_users_async_export',
            target_iter_fn_name='get_users_async_iter',
            parameters=None,
            request_body_info=None,
            item_type_ast=ast.Name(id='User', ctx=ast.Load()),
            item_type_imports=None,
            docs=None,
            is_async=True,
        )
        source = ast.unparse(ast.fix_missing_locations(self._module(fn_ast)))
        ast.parse(source)
        assert 'async def get_users_async_export' in source
        assert 'await export_async' in source
