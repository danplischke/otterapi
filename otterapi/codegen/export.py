"""File export utilities for OtterAPI code generation.

This module provides utilities for:
- Generating the ``_export.py`` runtime module in the output directory.
- Resolving per-endpoint export configuration.
- Checking whether an endpoint is eligible for export helper generation.

The generated ``_export.py`` exposes streaming writers for CSV, TSV, JSONL,
and Parquet. Schemas are derived from the endpoint's Pydantic item model
(never inferred from data) so types and nullability stay correct even for
empty results.

CSV / TSV / JSONL use only the Python standard library. Parquet support is
an optional extra (``pip install otterapi[parquet]``) that imports
``pyarrow`` lazily at call time, mirroring the pandas/polars pattern in
:mod:`otterapi.codegen.dataframes`.
"""

import ast
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from upath import UPath

from otterapi.codegen.ast_utils import (
    _argument,
    _assign,
    _async_func,
    _call,
    _func,
    _name,
    _subscript,
    _union_expr,
)

if TYPE_CHECKING:
    from otterapi.codegen.types import Endpoint, Parameter, RequestBodyInfo, Type
    from otterapi.config import ExportConfig, ExportFormat

ImportDict = dict[str, set[str]]

__all__ = [
    'EXPORT_MODULE_CONTENT',
    'ExportMethodConfig',
    'generate_export_module',
    'get_export_config_for_endpoint',
    'get_export_config_from_parts',
    'endpoint_is_exportable',
    'build_standalone_export_fn',
    'build_standalone_paginated_export_fn',
]


# =============================================================================
# Runtime module content (written verbatim as ``_export.py``)
# =============================================================================

EXPORT_MODULE_CONTENT = '''\
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
from typing import TYPE_CHECKING, Any, Union, get_args, get_origin
from uuid import UUID

from pydantic import BaseModel
from upath import UPath

if TYPE_CHECKING:
    import pyarrow as pa

Row = BaseModel | dict
PathLike = str | Path | UPath


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


def _row_to_arrow_dict(row: Row, model: type[BaseModel]) -> dict[str, Any]:
    """Normalize a row to a pyarrow-native dict for Parquet writes."""
    raw = _as_model_instance(row, model).model_dump(mode='python')
    return {key: _arrow_sanitize(value) for key, value in raw.items()}


def _chunks(iterable: Iterable[Row], size: int) -> Iterator[list[Row]]:
    batch: list[Row] = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


async def _achunks(
    iterable: AsyncIterable[Row], size: int
) -> 'AsyncIterable[list[Row]]':
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
        delimiter='\\t',
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
        delimiter='\\t',
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
                fh.write('\\n')
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
                fh.write('\\n')
                written += 1
    return written


# -----------------------------------------------------------------------------
# Parquet (pyarrow, lazy import)
# -----------------------------------------------------------------------------


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


def _unwrap_optional(annotation: Any) -> tuple[Any, bool]:
    """Return ``(inner, nullable)`` for Optional / ``T | None`` annotations."""
    origin = get_origin(annotation)
    if origin is Union or origin is types.UnionType:
        args = [a for a in get_args(annotation) if a is not type(None)]
        nullable = type(None) in get_args(annotation)
        if len(args) == 1:
            return args[0], nullable
        # Multi-type union -- opaque, fall back to string.
        return str, nullable
    return annotation, False


def _python_type_to_arrow(annotation: Any, pa) -> 'pa.DataType':
    """Map a Python/Pydantic annotation to a pyarrow DataType."""
    inner, _nullable = _unwrap_optional(annotation)
    origin = get_origin(inner)

    if isinstance(inner, type) and issubclass(inner, BaseModel):
        return pa.struct(
            [
                pa.field(
                    name,
                    _python_type_to_arrow(field.annotation, pa),
                    nullable=_is_nullable(field.annotation),
                )
                for name, field in inner.model_fields.items()
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
        (item_annotation,) = get_args(inner) or (Any,)
        return pa.list_(_python_type_to_arrow(item_annotation, pa))

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
        return pa.timestamp('us', tz='UTC')
    if inner is date:
        return pa.date32()
    if inner is time:
        return pa.time64('us')

    return pa.string()


def _is_nullable(annotation: Any) -> bool:
    _, nullable = _unwrap_optional(annotation)
    return nullable


def pydantic_to_arrow_schema(model: type[BaseModel]) -> 'pa.Schema':
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
    target = _as_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    sink, fh = _parquet_sink(target)
    try:
        with pq.ParquetWriter(sink, schema, compression=compression) as writer:
            for batch in _chunks(rows, batch_size):
                records = [_row_to_arrow_dict(row, model) for row in batch]
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
    target = _as_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    sink, fh = _parquet_sink(target)
    try:
        with pq.ParquetWriter(sink, schema, compression=compression) as writer:
            async for batch in _achunks(rows, batch_size):
                records = [_row_to_arrow_dict(row, model) for row in batch]
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
            f"Unsupported export format: {format!r}. "
            f"Expected one of: {sorted(_WRITERS)}"
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
            f"Unsupported export format: {format!r}. "
            f"Expected one of: {sorted(_ASYNC_WRITERS)}"
        )
    return await writer(rows, path, model=model, batch_size=batch_size, **kwargs)
'''


# =============================================================================
# Module generation
# =============================================================================


def generate_export_module(output_dir: Path | UPath) -> Path | UPath:
    """Generate the ``_export.py`` utility module in the output directory."""
    output_dir = UPath(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / '_export.py'
    file_path.write_text(EXPORT_MODULE_CONTENT, encoding='utf-8')
    return file_path


# =============================================================================
# Configuration resolution
# =============================================================================


@dataclass
class ExportMethodConfig:
    """Resolved per-endpoint export configuration.

    Attributes:
        enabled: Whether export helpers should be generated.
        formats: Formats the helpers should support.
        path: JSON path for extracting list data from the response.
        batch_size: Streaming batch size.
    """

    enabled: bool = False
    formats: 'list[ExportFormat]' = None  # type: ignore[assignment]
    path: str | None = None
    batch_size: int = 1000

    def __post_init__(self) -> None:
        if self.formats is None:
            self.formats = []


def _returns_list(response_type: 'Type | None') -> bool:
    # Local import avoids a circular dependency at module load.
    from otterapi.codegen.dataframes import response_type_returns_list

    return response_type_returns_list(response_type)


def get_export_config_for_endpoint(
    endpoint: 'Endpoint',
    export_config: 'ExportConfig',
) -> ExportMethodConfig:
    """Resolve export configuration for a fully-constructed endpoint."""
    if not export_config.enabled:
        return ExportMethodConfig()

    returns_list = _returns_list(endpoint.response_type)
    should_generate, formats, path = export_config.should_generate_for_endpoint(
        endpoint_name=endpoint.fn.name,
        returns_list=returns_list,
    )
    return ExportMethodConfig(
        enabled=should_generate,
        formats=formats,
        path=path,
        batch_size=export_config.batch_size,
    )


def get_export_config_from_parts(
    endpoint_name: str,
    response_type: 'Type | None',
    export_config: 'ExportConfig',
) -> ExportMethodConfig:
    """Resolve export configuration from individual endpoint parts."""
    if not export_config.enabled:
        return ExportMethodConfig()

    returns_list = _returns_list(response_type)
    should_generate, formats, path = export_config.should_generate_for_endpoint(
        endpoint_name=endpoint_name,
        returns_list=returns_list,
    )
    return ExportMethodConfig(
        enabled=should_generate,
        formats=formats,
        path=path,
        batch_size=export_config.batch_size,
    )


def endpoint_is_exportable(
    endpoint_name: str,
    response_type: 'Type | None',
    export_config: 'ExportConfig',
) -> bool:
    """Return whether export helpers would be generated for this endpoint."""
    resolved = get_export_config_from_parts(endpoint_name, response_type, export_config)
    return resolved.enabled


# =============================================================================
# Per-endpoint AST builders
# =============================================================================
#
# These wrap an existing endpoint (or its ``_iter`` paginated variant) in a
# small forwarder that pipes results into the runtime ``export()``. The
# wrapper signature mirrors the underlying function's parameters exactly,
# adds export-specific kw-only options (``output_path``, ``format``,
# ``batch_size``), and returns the row count from the writer.
#
# The body is intentionally thin -- pagination, request handling, response
# parsing all stay in the underlying function we delegate to. Keeps the
# generated diff small and avoids re-implementing pagination AST.

_EXPORT_FORMAT_LITERAL_VALUES = ('csv', 'tsv', 'jsonl', 'parquet')


def _format_literal_annotation() -> ast.expr:
    """Build the AST for ``Literal['csv', 'tsv', 'jsonl', 'parquet']``."""
    return _subscript(
        'Literal',
        ast.Tuple(
            elts=[ast.Constant(value=v) for v in _EXPORT_FORMAT_LITERAL_VALUES],
            ctx=ast.Load(),
        ),
    )


def _output_path_annotation() -> ast.expr:
    """Build the AST for ``str | Path`` (the wrapper's path argument type)."""
    return _union_expr([_name('str'), _name('Path')])


def _add_export_kwonly_args(
    kwonlyargs: list[ast.arg],
    kw_defaults: list[ast.expr],
    default_format: str,
    default_batch_size: int,
) -> None:
    """Append the export-specific keyword-only args + defaults in place."""
    kwonlyargs.append(_argument('output_path', _output_path_annotation()))
    kw_defaults.append(ast.Constant(value=None))  # required at call time

    kwonlyargs.append(_argument('format', _format_literal_annotation()))
    kw_defaults.append(ast.Constant(value=default_format))

    kwonlyargs.append(
        _argument('batch_size', _name('int')),
    )
    kw_defaults.append(ast.Constant(value=default_batch_size))


def _forward_call_keywords(
    parameters: 'list[Parameter] | None',
    request_body_info: 'RequestBodyInfo | None',
    extra_kw_names: list[str],
) -> list[ast.keyword]:
    """Build keyword arguments for forwarding to the underlying function.

    Forwards every cloned parameter and any extra kwargs (``client``, the
    pagination knobs, etc.) by name, leaving the values bound to the
    same-named locals in the wrapper's scope.
    """
    keywords: list[ast.keyword] = []
    for param in parameters or []:
        name = param.name_sanitized
        keywords.append(ast.keyword(arg=name, value=_name(name)))
    if request_body_info is not None:
        keywords.append(ast.keyword(arg='body', value=_name('body')))
    for kw_name in extra_kw_names:
        keywords.append(ast.keyword(arg=kw_name, value=_name(kw_name)))
    return keywords


def _build_export_dispatch(
    *,
    target_fn_name: str,
    target_call_keywords: list[ast.keyword],
    item_type_ast: ast.expr,
    is_async: bool,
    consumes_async_iter: bool,
) -> list[ast.stmt]:
    """Build the function body for an export wrapper.

    Body shape:
        rows = <target>(...)            # or `await ...`
        return export(rows, output_path, model=<Item>, format=format,
                      batch_size=batch_size)

    For async + paginated case, ``rows`` is an AsyncIterator and we route
    through ``export_async`` (and ``await`` it).
    """
    target_call: ast.expr = _call(
        func=_name(target_fn_name),
        args=[],
        keywords=target_call_keywords,
    )
    if is_async and not consumes_async_iter:
        # ``await get_users_async(...)`` returns the materialized list.
        target_call = ast.Await(value=target_call)

    rows_assign = _assign(_name('rows'), target_call)

    writer_name = 'export_async' if consumes_async_iter else 'export'
    writer_call: ast.expr = _call(
        func=_name(writer_name),
        args=[
            _name('rows'),
            _name('output_path'),
        ],
        keywords=[
            ast.keyword(arg='model', value=copy.deepcopy(item_type_ast)),
            ast.keyword(arg='format', value=_name('format')),
            ast.keyword(arg='batch_size', value=_name('batch_size')),
        ],
    )
    if consumes_async_iter:
        writer_call = ast.Await(value=writer_call)

    return [rows_assign, ast.Return(value=writer_call)]


def _build_export_function(
    *,
    fn_name: str,
    target_fn_name: str,
    parameters: 'list[Parameter] | None',
    request_body_info: 'RequestBodyInfo | None',
    item_type_ast: ast.expr,
    item_type_imports: ImportDict,
    docs: str | None,
    is_async: bool,
    is_paginated: bool,
    default_format: str,
    default_batch_size: int,
) -> tuple[ast.FunctionDef | ast.AsyncFunctionDef, ImportDict]:
    """Shared builder body for both standalone + paginated export wrappers."""
    # Local import avoids a circular dependency at module load.
    from otterapi.codegen.endpoints import FunctionSignatureBuilder

    builder = FunctionSignatureBuilder()
    builder.add_parameters(parameters)
    builder.add_request_body(request_body_info)
    builder.add_client_parameter()
    signature = builder.build()

    extra_kw_forwards: list[str] = ['client']

    # Mirror pagination's wrapper params so they end up forwarded to the
    # underlying ``_iter`` function. We add them here on the wrapper so that
    # they're locally bound names; the actual pagination logic stays in
    # ``_iter`` itself.
    if is_paginated:
        for kw_name, annotation, default in (
            (
                'page_size',
                _name('int'),
                ast.Constant(value=100),
            ),
            (
                'max_items',
                _union_expr([_name('int'), ast.Constant(value=None)]),
                ast.Constant(value=None),
            ),
        ):
            signature.kwonlyargs.append(_argument(kw_name, annotation))
            signature.kw_defaults.append(default)
            extra_kw_forwards.append(kw_name)

    _add_export_kwonly_args(
        signature.kwonlyargs,
        signature.kw_defaults,
        default_format=default_format,
        default_batch_size=default_batch_size,
    )

    target_call_keywords = _forward_call_keywords(
        parameters=parameters,
        request_body_info=request_body_info,
        extra_kw_names=extra_kw_forwards,
    )

    body = _build_export_dispatch(
        target_fn_name=target_fn_name,
        target_call_keywords=target_call_keywords,
        item_type_ast=item_type_ast,
        is_async=is_async,
        consumes_async_iter=is_async and is_paginated,
    )

    docstring = (
        f'Export results of ``{target_fn_name}`` to a file. '
        f'Returns the number of rows written.'
    )
    if docs:
        docstring = f'{docstring}\n\n{docs.strip()}'
    body = [ast.Expr(value=ast.Constant(value=docstring))] + body

    builder_fn = _async_func if is_async else _func
    func_ast = builder_fn(
        name=fn_name,
        args=signature.args,
        body=body,
        kwargs=_argument('format_kwargs', _name('Any')),
        kwonlyargs=signature.kwonlyargs,
        kw_defaults=signature.kw_defaults,
        returns=_name('int'),
    )

    imports: ImportDict = {
        'pathlib': {'Path'},
        'typing': {'Any', 'Literal'},
        '._export': {'export_async' if (is_async and is_paginated) else 'export'},
    }
    for module, names in (signature.imports or {}).items():
        imports.setdefault(module, set()).update(names)
    for module, names in (item_type_imports or {}).items():
        imports.setdefault(module, set()).update(names)

    return func_ast, imports


def build_standalone_export_fn(
    *,
    fn_name: str,
    target_fn_name: str,
    parameters: 'list[Parameter] | None',
    request_body_info: 'RequestBodyInfo | None',
    item_type_ast: ast.expr,
    item_type_imports: ImportDict | None,
    docs: str | None,
    is_async: bool,
    default_format: str = 'csv',
    default_batch_size: int = 1000,
) -> tuple[ast.FunctionDef | ast.AsyncFunctionDef, ImportDict]:
    """Build an export wrapper around a non-paginated list endpoint.

    The wrapper mirrors ``target_fn_name``'s signature, adds keyword-only
    ``output_path`` / ``format`` / ``batch_size`` arguments, calls the
    underlying endpoint, and pipes the resulting list into the runtime
    ``export(...)`` writer. Returns the row count.
    """
    return _build_export_function(
        fn_name=fn_name,
        target_fn_name=target_fn_name,
        parameters=parameters,
        request_body_info=request_body_info,
        item_type_ast=item_type_ast,
        item_type_imports=item_type_imports or {},
        docs=docs,
        is_async=is_async,
        is_paginated=False,
        default_format=default_format,
        default_batch_size=default_batch_size,
    )


def build_standalone_paginated_export_fn(
    *,
    fn_name: str,
    target_iter_fn_name: str,
    parameters: 'list[Parameter] | None',
    request_body_info: 'RequestBodyInfo | None',
    item_type_ast: ast.expr,
    item_type_imports: ImportDict | None,
    docs: str | None,
    is_async: bool,
    default_format: str = 'csv',
    default_batch_size: int = 1000,
) -> tuple[ast.FunctionDef | ast.AsyncFunctionDef, ImportDict]:
    """Build an export wrapper around a paginated endpoint's ``_iter`` variant.

    The wrapper mirrors the paginated endpoint's signature plus
    ``page_size``/``max_items`` (forwarded to the iter function), drives the
    iterator, and pipes items into ``export(...)`` (sync) or
    ``export_async(...)`` (async). Memory stays bounded by ``batch_size``.
    """
    return _build_export_function(
        fn_name=fn_name,
        target_fn_name=target_iter_fn_name,
        parameters=parameters,
        request_body_info=request_body_info,
        item_type_ast=item_type_ast,
        item_type_imports=item_type_imports or {},
        docs=docs,
        is_async=is_async,
        is_paginated=True,
        default_format=default_format,
        default_batch_size=default_batch_size,
    )
