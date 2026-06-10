"""Streaming file export utilities for OtterAPI generated clients.

These helpers accept any iterable (or async iterable) of Pydantic models or
dicts and write them to a file path in the chosen format. Schemas are
derived from the provided Pydantic model class so that Parquet column
types stay correct even with empty or null-heavy batches.

Paths may be local strings / ``pathlib.Path`` / ``upath.UPath`` -- cloud
URIs like ``s3://bucket/key.parquet`` work out of the box when the
matching fsspec backend is installed.

CSV / TSV / JSONL only require the Python standard library. Parquet
requires ``pyarrow`` (install via ``pip install otterapi[parquet]``).
"""

from __future__ import annotations

import csv
import json
import types
from collections.abc import AsyncIterable, Iterable, Iterator
from datetime import date, datetime, time
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, TypeAlias, Union, get_args, get_origin
from uuid import UUID

from pydantic import BaseModel
from upath import UPath

if TYPE_CHECKING:
    import pyarrow as pa

Row: TypeAlias = BaseModel | dict
PathLike: TypeAlias = str | Path | UPath


# -----------------------------------------------------------------------------
# Shared helpers
# -----------------------------------------------------------------------------


def _as_path(path: PathLike) -> UPath:
    return UPath(path)


def _model_fields(model: type[BaseModel]) -> list[str]:
    """Return field names in declaration order."""
    return list(model.model_fields.keys())


def _as_model_instance(row: Row, model: type[BaseModel]) -> BaseModel:
    if isinstance(row, model):
        return row
    return model.model_validate(row)


def _row_to_dict(row: Row, model: type[BaseModel]) -> dict[str, Any]:
    """Normalize a row to a JSON-ready dict via the given Pydantic model.

    Datetimes / UUIDs / enums are rendered in their JSON-safe form,
    suitable for CSV / TSV / JSONL output.
    """
    return _as_model_instance(row, model).model_dump(mode='json')


def _arrow_sanitize(value: Any) -> Any:
    """Convert a Python-mode Pydantic value into a pyarrow-native value.

    ``model_dump(mode='python')`` keeps datetimes, Decimals, etc. as native
    types (which pyarrow handles) but leaves Enum instances and UUIDs as
    objects pyarrow doesn't know how to encode. This walker normalizes
    them without touching datetimes / Decimals / bytes.
    """
    if value is None:
        return None
    if isinstance(value, Enum):
        return _arrow_sanitize(value.value)
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, dict):
        return {k: _arrow_sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_arrow_sanitize(v) for v in value]
    return value


def _row_to_arrow_dict(
    row: Row,
    model: type[BaseModel],
    *,
    _string_fields: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    """Normalize a row to a pyarrow-native dict for Parquet writes.

    ``_string_fields`` names fields whose schema type is ``pa.string()`` but
    whose Python value may not be a ``str`` (e.g. ``Union[int, str]`` falls
    back to string in the schema).  Those values are coerced with ``str()``.
    """
    raw = _as_model_instance(row, model).model_dump(mode='python')
    result = {key: _arrow_sanitize(value) for key, value in raw.items()}
    for name in _string_fields:
        val = result.get(name)
        if val is not None and not isinstance(val, str):
            result[name] = str(val)
    return result


def _chunks(iterable: Iterable[Row], size: int) -> Iterator[list[Row]]:
    batch: list[Row] = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


async def _achunks(iterable: AsyncIterable[Row], size: int) -> AsyncIterable[list[Row]]:
    batch: list[Row] = []
    async for item in iterable:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


# -----------------------------------------------------------------------------
# CSV / TSV
# -----------------------------------------------------------------------------


def _stringify_cell(value: Any) -> Any:
    """CSV cells must be scalar; nested containers are JSON-encoded."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, dict)):
        return json.dumps(value, default=str, separators=(',', ':'))
    return str(value)


def to_csv(
    rows: Iterable[Row],
    path: PathLike,
    *,
    model: type[BaseModel],
    delimiter: str = ',',
    header: bool = True,
    batch_size: int = 1000,
) -> int:
    """Stream ``rows`` to a CSV file using the model's field order.

    Returns the number of rows written.
    """
    fieldnames = _model_fields(model)
    target = _as_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with target.open('w', encoding='utf-8', newline='') as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=fieldnames,
            delimiter=delimiter,
            extrasaction='ignore',
        )
        if header:
            writer.writeheader()
        for batch in _chunks(rows, batch_size):
            for row in batch:
                record = _row_to_dict(row, model)
                writer.writerow(
                    {key: _stringify_cell(record.get(key)) for key in fieldnames}
                )
                written += 1
    return written


def to_tsv(
    rows: Iterable[Row],
    path: PathLike,
    *,
    model: type[BaseModel],
    header: bool = True,
    batch_size: int = 1000,
) -> int:
    """Stream ``rows`` to a TSV file (CSV with a tab delimiter)."""
    return to_csv(
        rows,
        path,
        model=model,
        delimiter='\t',
        header=header,
        batch_size=batch_size,
    )


async def to_csv_async(
    rows: AsyncIterable[Row],
    path: PathLike,
    *,
    model: type[BaseModel],
    delimiter: str = ',',
    header: bool = True,
    batch_size: int = 1000,
) -> int:
    """Async variant of :func:`to_csv`. Disk I/O remains synchronous."""
    fieldnames = _model_fields(model)
    target = _as_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with target.open('w', encoding='utf-8', newline='') as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=fieldnames,
            delimiter=delimiter,
            extrasaction='ignore',
        )
        if header:
            writer.writeheader()
        async for batch in _achunks(rows, batch_size):
            for row in batch:
                record = _row_to_dict(row, model)
                writer.writerow(
                    {key: _stringify_cell(record.get(key)) for key in fieldnames}
                )
                written += 1
    return written


async def to_tsv_async(
    rows: AsyncIterable[Row],
    path: PathLike,
    *,
    model: type[BaseModel],
    header: bool = True,
    batch_size: int = 1000,
) -> int:
    """Async variant of :func:`to_tsv`."""
    return await to_csv_async(
        rows,
        path,
        model=model,
        delimiter='\t',
        header=header,
        batch_size=batch_size,
    )


# -----------------------------------------------------------------------------
# JSONL
# -----------------------------------------------------------------------------


def to_jsonl(
    rows: Iterable[Row],
    path: PathLike,
    *,
    model: type[BaseModel],
    batch_size: int = 1000,
) -> int:
    """Stream ``rows`` to a JSON Lines file.

    Returns the number of rows written.
    """
    target = _as_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with target.open('w', encoding='utf-8') as fh:
        for batch in _chunks(rows, batch_size):
            for row in batch:
                if isinstance(row, model):
                    line = row.model_dump_json()
                else:
                    line = model.model_validate(row).model_dump_json()
                fh.write(line)
                fh.write('\n')
                written += 1
    return written


async def to_jsonl_async(
    rows: AsyncIterable[Row],
    path: PathLike,
    *,
    model: type[BaseModel],
    batch_size: int = 1000,
) -> int:
    """Async variant of :func:`to_jsonl`."""
    target = _as_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with target.open('w', encoding='utf-8') as fh:
        async for batch in _achunks(rows, batch_size):
            for row in batch:
                if isinstance(row, model):
                    line = row.model_dump_json()
                else:
                    line = model.model_validate(row).model_dump_json()
                fh.write(line)
                fh.write('\n')
                written += 1
    return written


# -----------------------------------------------------------------------------
# Parquet (pyarrow, lazy import)
# -----------------------------------------------------------------------------

# Mapping from Pydantic sentinel types (AwareDatetime, NaiveDatetime, etc.) to a
# simple kind string used in _python_type_to_arrow.  These are plain classes, not
# Annotated[...] wrappers, so _strip_annotated cannot unwrap them.
_PYDANTIC_SENTINELS: dict[type, str] = {}
try:
    import pydantic.types as _pydantic_types

    for _attr, _kind in (
        ('AwareDatetime', 'timestamp_tz'),
        ('NaiveDatetime', 'timestamp'),
        ('PastDatetime', 'timestamp'),
        ('FutureDatetime', 'timestamp'),
        ('PastDate', 'date'),
        ('FutureDate', 'date'),
    ):
        _t = getattr(_pydantic_types, _attr, None)
        if _t is not None:
            _PYDANTIC_SENTINELS[_t] = _kind
except ImportError:
    pass


def _require_pyarrow():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - exercised via tests
        raise ImportError(
            'pyarrow is required for Parquet export. '
            "Install with: pip install 'otterapi[parquet]'"
        ) from exc
    return pa, pq


def _strip_annotated(annotation: Any) -> Any:
    """Unwrap ``Annotated[T, ...]`` to its base type ``T``.

    Handles constrained Pydantic types like ``PositiveInt``, ``StrictStr``, etc.
    that are expressed as ``Annotated[base, ...]``.  Plain sentinel classes such
    as ``AwareDatetime`` are handled separately via ``_PYDANTIC_SENTINELS``.
    """
    while get_origin(annotation) is Annotated:
        annotation = get_args(annotation)[0]
    return annotation


def _unwrap_optional(annotation: Any) -> tuple[Any, bool]:
    """Return ``(inner, nullable)`` for Optional / ``T | None`` annotations."""
    annotation = _strip_annotated(annotation)
    origin = get_origin(annotation)
    if origin is Union or origin is types.UnionType:
        args = [a for a in get_args(annotation) if a is not type(None)]
        nullable = type(None) in get_args(annotation)
        if len(args) == 1:
            return args[0], nullable
        # Multi-type union -- opaque, fall back to string.
        return str, nullable
    return annotation, False


def _python_type_to_arrow(annotation: Any, pa) -> pa.DataType:
    """Map a Python/Pydantic annotation to a pyarrow DataType."""
    inner, _nullable = _unwrap_optional(annotation)
    # Strip any remaining Annotated wrapper (e.g. inside Optional[AwareDatetime]).
    inner = _strip_annotated(inner)
    origin = get_origin(inner)

    if isinstance(inner, type) and issubclass(inner, BaseModel):
        model_cls: type[BaseModel] = inner
        return pa.struct(
            [
                pa.field(
                    name,
                    _python_type_to_arrow(field.annotation, pa),
                    nullable=_is_nullable(field.annotation),
                )
                for name, field in model_cls.model_fields.items()
            ]
        )

    if isinstance(inner, type) and issubclass(inner, Enum):
        members = [m.value for m in inner]
        if members and all(isinstance(m, int) for m in members):
            return pa.int64()
        if members and all(isinstance(m, float) for m in members):
            return pa.float64()
        return pa.string()

    if origin in (list, tuple, set, frozenset):
        args = get_args(inner)
        if not args:
            return pa.list_(pa.string())
        if origin is tuple:
            if len(args) == 2 and args[1] is Ellipsis:
                # tuple[T, ...] - homogeneous variable-length
                return pa.list_(_python_type_to_arrow(args[0], pa))
            if len(args) > 1:
                # tuple[A, B, ...] - heterogeneous, no native Parquet type
                return pa.string()
        return pa.list_(_python_type_to_arrow(args[0], pa))

    if origin is dict:
        return pa.string()  # JSON-encoded fallback (keeps schema simple).

    if inner is str or inner is UUID:
        return pa.string()
    if inner is bytes:
        return pa.binary()
    if inner is bool:
        return pa.bool_()
    if inner is int:
        return pa.int64()
    if inner is float:
        return pa.float64()
    if inner is Decimal:
        return pa.decimal128(38, 18)
    if inner is datetime:
        # Use no-tz timestamp so both naive and aware datetimes write without error.
        return pa.timestamp('us')
    if inner is date:
        return pa.date32()
    if inner is time:
        return pa.time64('us')

    # Pydantic sentinel types (AwareDatetime, NaiveDatetime, PastDate, etc.) are
    # plain classes, not Annotated wrappers, so _strip_annotated can't resolve them.
    _kind = _PYDANTIC_SENTINELS.get(inner)
    if _kind == 'timestamp_tz':
        return pa.timestamp('us', tz='UTC')
    if _kind == 'timestamp':
        return pa.timestamp('us')
    if _kind == 'date':
        return pa.date32()

    return pa.string()


def _is_nullable(annotation: Any) -> bool:
    _, nullable = _unwrap_optional(annotation)
    return nullable


def pydantic_to_arrow_schema(model: type[BaseModel]) -> pa.Schema:
    """Derive a pyarrow schema from a Pydantic model class.

    Field order matches declaration order. Optional fields are marked
    nullable. Unknown/opaque types fall back to string.
    """
    pa, _pq = _require_pyarrow()
    fields = [
        pa.field(
            name,
            _python_type_to_arrow(field.annotation, pa),
            nullable=_is_nullable(field.annotation),
        )
        for name, field in model.model_fields.items()
    ]
    return pa.schema(fields)


def _parquet_sink(target: UPath):
    """Choose a sink suitable for ``pq.ParquetWriter``.

    For local filesystem paths we hand pyarrow the path string directly --
    this avoids platform-specific quirks with passing Python file handles
    on Windows. For remote / fsspec-backed paths we open in binary mode
    and let pyarrow write into the resulting handle.

    Returns ``(sink, file_handle_or_none)``; the caller must close the
    handle if it's not ``None`` (the path-string case manages its own
    file lifetime inside pyarrow).
    """
    protocol = getattr(target, 'protocol', '') or ''
    if protocol in ('', 'file'):
        return str(target), None
    fh = target.open('wb')
    return fh, fh


def to_parquet(
    rows: Iterable[Row],
    path: PathLike,
    *,
    model: type[BaseModel],
    compression: str | None = 'snappy',
    batch_size: int = 1000,
) -> int:
    """Stream ``rows`` to a Parquet file using the model's schema.

    The schema is derived once from ``model`` before the first write and
    reused for every row group; rows are never used to infer types.

    Returns the number of rows written.
    """
    pa, pq = _require_pyarrow()
    schema = pydantic_to_arrow_schema(model)
    string_fields = frozenset(f.name for f in schema if pa.types.is_string(f.type))
    target = _as_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    sink, fh = _parquet_sink(target)
    try:
        with pq.ParquetWriter(sink, schema, compression=compression) as writer:
            for batch in _chunks(rows, batch_size):
                records = [
                    _row_to_arrow_dict(row, model, _string_fields=string_fields)
                    for row in batch
                ]
                table = pa.Table.from_pylist(records, schema=schema)
                writer.write_table(table)
                written += len(records)
            if written == 0:
                # Preserve a valid, empty-but-typed Parquet file.
                writer.write_table(pa.Table.from_pylist([], schema=schema))
    finally:
        if fh is not None:
            fh.close()
    return written


async def to_parquet_async(
    rows: AsyncIterable[Row],
    path: PathLike,
    *,
    model: type[BaseModel],
    compression: str | None = 'snappy',
    batch_size: int = 1000,
) -> int:
    """Async variant of :func:`to_parquet`."""
    pa, pq = _require_pyarrow()
    schema = pydantic_to_arrow_schema(model)
    string_fields = frozenset(f.name for f in schema if pa.types.is_string(f.type))
    target = _as_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    sink, fh = _parquet_sink(target)
    try:
        with pq.ParquetWriter(sink, schema, compression=compression) as writer:
            async for batch in _achunks(rows, batch_size):
                records = [
                    _row_to_arrow_dict(row, model, _string_fields=string_fields)
                    for row in batch
                ]
                table = pa.Table.from_pylist(records, schema=schema)
                writer.write_table(table)
                written += len(records)
            if written == 0:
                writer.write_table(pa.Table.from_pylist([], schema=schema))
    finally:
        if fh is not None:
            fh.close()
    return written


# -----------------------------------------------------------------------------
# Unified entry point
# -----------------------------------------------------------------------------


_WRITERS = {
    'csv': to_csv,
    'tsv': to_tsv,
    'jsonl': to_jsonl,
    'parquet': to_parquet,
}

_ASYNC_WRITERS = {
    'csv': to_csv_async,
    'tsv': to_tsv_async,
    'jsonl': to_jsonl_async,
    'parquet': to_parquet_async,
}


def export(
    rows: Iterable[Row],
    path: PathLike,
    *,
    model: type[BaseModel],
    format: str,
    batch_size: int = 1000,
    **kwargs: Any,
) -> int:
    """Dispatch to the writer for ``format`` (``csv``/``tsv``/``jsonl``/``parquet``)."""
    writer = _WRITERS.get(format)
    if writer is None:
        raise ValueError(
            f'Unsupported export format: {format!r}. '
            f'Expected one of: {sorted(_WRITERS)}'
        )
    return writer(rows, path, model=model, batch_size=batch_size, **kwargs)


async def export_async(
    rows: AsyncIterable[Row],
    path: PathLike,
    *,
    model: type[BaseModel],
    format: str,
    batch_size: int = 1000,
    **kwargs: Any,
) -> int:
    """Async dispatch to the writer for ``format``."""
    writer = _ASYNC_WRITERS.get(format)
    if writer is None:
        raise ValueError(
            f'Unsupported export format: {format!r}. '
            f'Expected one of: {sorted(_ASYNC_WRITERS)}'
        )
    return await writer(rows, path, model=model, batch_size=batch_size, **kwargs)
