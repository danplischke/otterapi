# 🦦 OtterAPI

> *A cute otter that turns OpenAPI & Swagger specs into type-safe, async Python clients — powered by [httpx2](https://github.com/pydantic/httpx2) and Pydantic.*

**OtterAPI** is a sleek Python library that transforms OpenAPI specifications into clean, type-safe client code with Pydantic models and httpx2-based HTTP clients.

## ✨ Features

- **Type-Safe Code Generation** - Generates Pydantic models and fully typed endpoint functions
- **Sync & Async Support** - Generate both synchronous and asynchronous API clients
- **OpenAPI 3.x Support** - Full support for OpenAPI 3.0, 3.1, and 3.2 specifications
- **Strict or Lenient Validation** - Fail fast on malformed specs, or drop stray fields with `--lenient` (vendor `x-*` extensions always preserved)
- **Module Splitting** - Organize large APIs into multiple organized files
- **Pagination** - Auto-detect or configure offset, cursor, page, and link-header pagination
- **DataFrame Conversion** - Generate pandas/polars DataFrame methods for list endpoints
- **File Export** - Generate CSV, TSV, JSONL, and Parquet streaming helpers
- **Response Unwrapping** - Transparently unwrap envelope-style responses
- **Customizable Client** - Generated client class with configurable base URL, timeout, and headers
- **Environment Variable Support** - Use `${VAR}` or `${VAR:-default}` syntax in config files

## 🚀 Quick Start

### Installation

```bash
pip install otterapi
```

Generated clients use [httpx2](https://github.com/pydantic/httpx2), pydantic's
maintained fork of httpx (same API, package renamed), installed alongside
otterapi. A custom `http_client` / `async_http_client` you pass in must be an
`httpx2.Client` / `httpx2.AsyncClient`. Note that httpx2 verifies TLS against
the operating system's trust store (`truststore`) rather than `certifi`. To
generate clients that use the original `httpx` package instead, set
`http_library: httpx` (see [Document Options](#document-options)).

### Basic Usage

1. Create an `otter.yml` configuration file:

```yaml
documents:
  - source: https://petstore3.swagger.io/api/v3/openapi.json
    output: petstore_client
```

2. Generate the client:

```bash
otter generate
```

3. Use the generated code:

```python
from petstore_client import get_pet_by_id, async_get_pet_by_id

# Synchronous usage
pet = get_pet_by_id(pet_id=123)

# Asynchronous usage
import asyncio
pet = asyncio.run(async_get_pet_by_id(pet_id=123))
```

---

## 📝 Configuration

### Config File Locations

OtterAPI looks for configuration in this order:

1. Path passed via `otter generate -c <path>`
2. `otter.yaml` or `otter.yml` in the current directory
3. `otter.json` in the current directory
4. `[tool.otterapi]` section in `pyproject.toml`
5. `OTTER_SOURCE` and `OTTER_OUTPUT` environment variables

### Config File Formats

**YAML (recommended):**

```yaml
documents:
  - source: https://api.example.com/openapi.json
    output: ./client
```

**pyproject.toml:**

```toml
[tool.otterapi]
[[tool.otterapi.documents]]
source = "https://api.example.com/openapi.json"
output = "./client"
```

**JSON:**

```json
{
  "documents": [
    { "source": "https://api.example.com/openapi.json", "output": "./client" }
  ]
}
```

### Top-Level Options

These sit at the root of your config file, outside of `documents:`.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `documents` | list | — | List of OpenAPI documents to process (required) |
| `generate_endpoints` | bool | `true` | Whether to generate endpoint functions |
| `format_output` | bool | `true` | Format generated code with ruff/black |
| `validate_output` | bool | `true` | Validate generated code: syntax, plus every referenced name resolving to an import or definition |
| `create_py_typed` | bool | `true` | Create `py.typed` marker files |
| `lenient` | bool | `false` | Drop unrecognized, invalid fields instead of failing (see [Strict vs Lenient Validation](#-strict-vs-lenient-validation)) |

```yaml
format_output: true
validate_output: true
create_py_typed: true

documents:
  - source: https://api.example.com/openapi.json
    output: ./client
```

> `format_output` only controls *cosmetic* formatting (line wrapping, quote
> style). The import block of every generated module is computed from the
> names the module actually references, so generated code never carries unused
> imports — with or without ruff/black installed. `validate_output` checks the
> other direction: generation fails rather than emitting a module that
> references a name nothing imports or defines.

### Document Options

Each entry under `documents:` supports these fields:

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `source` | string | — | URL or file path to the OpenAPI spec (required) |
| `output` | string | — | Output directory for generated code (required) |
| `base_url` | string | from spec | Override the base URL defined in the spec |
| `models_file` | string | `models.py` | Filename for generated models |
| `endpoints_file` | string | `endpoints.py` | Filename for generated endpoints (no-split mode) |
| `models_import_path` | string | `null` | Override the import path used for models in endpoints |
| `generate_async` | bool | `true` | Generate async endpoint functions |
| `generate_sync` | bool | `true` | Generate sync endpoint functions |
| `client_class_name` | string | from API title | Override the generated client class name |
| `client_style` | `functions` \| `client` \| `resource` | `functions` | Shape of the public API. `functions` exposes a standalone function per endpoint/variant; `client` adds `Client` / `AsyncClient` classes with a flat method per endpoint; `resource` groups those methods into resource sub-clients (see [Client Style](#-client-style)) |
| `resource_naming` | `tag` \| `path` \| `operation_id` | `tag` | How resource sub-clients are derived for `client_style: resource`. `tag` uses the module-split strategy (one level); `path` nests by URL segments; `operation_id` nests by the dotted operationId hierarchy |
| `result_objects` | bool | `false` | With `client_style: client` or `resource`, a list endpoint returns a deferred `Query` object instead of separate `_df`/`_pl`/`_iter`/`_export` methods (see [Client Style](#-client-style)) |
| `function_naming` | `operation_id` \| `path` | `operation_id` | How to name endpoint functions. `path` derives names from the HTTP method and URL path — use it for specs that reuse one `operationId` across many paths |
| `include_paths` | list | `null` | Glob patterns — only matching paths are generated |
| `exclude_paths` | list | `null` | Glob patterns — matching paths are skipped (applied after `include_paths`) |
| `target_python` | `3.11` … `3.14` | `3.11` | Oldest Python the generated code has to run on. `3.12`+ emits PEP 695 type parameters in the runtime helpers |
| `http_library` | `httpx2` \| `httpx` | `httpx2` | HTTP library the generated client imports. Use `httpx` if your project already depends on the original package (install it yourself; otterapi only ships `httpx2`) |
| `pydantic_version` | `1` \| `2` | `2` | Target Pydantic version for generated models and helpers |

#### Function Naming

By default, endpoint function names come from each operation's `operationId`. Some
specs (notably Swashbuckle/.NET output) reuse a single `operationId` across many
distinct paths, which produces a pile of numerically-suffixed duplicates like
`feed_changes`, `feed_changes_2`, … `feed_changes_43`. Set `function_naming: path`
to derive names from the HTTP method and URL path instead, giving distinct,
descriptive names:

```yaml
documents:
  - source: https://api.example.com/swagger.json
    output: ./client
    function_naming: path   # get_v1_feed_trial_changes, get_v1_feed_hcp_profile_changes, ...
```

#### Renaming Functions

`naming.rules` rewrites every operation's name with regular expressions, in
order; `naming.overrides` pins single operations (keyed by operationId) to an
exact name. Every variant (`_df`, `_iter`, `_page`, `async_*`, ...), `__all__`
and the class-client methods follow the new name.

```yaml
naming:
  rules:
    - pattern: '_v\d+(?=_|$)'     # get_widgets_by_id_v2_count -> get_widgets_by_id_count
      replace: ''
  overrides:
    legacyGetThing: get_thing
```

A rename that gives two operations the same name fails generation (naming one
of them `_2` would hide which one you meant) -- unless `versioning` resolves it.
Generation also fails when two generated functions in one module would share
a name, such as an operation called `list_users_page` next to the `_page`
variant of a paginated `list_users`.

Per-endpoint settings (`pagination.endpoints`, `dataframe.endpoints`,
`export.endpoints`, `response_unwrap.endpoints`) are keyed by the final
function name; an operationId works too. A key that matches no endpoint is
reported as a warning.

#### Path-Versioned APIs

For specs that version endpoints in the path (`/widgets/v1/...` next to
`/widgets/v2/...`), `versioning` keeps the newest version under the plain name
and moves older ones out of the way:

```yaml
module_split: {enabled: true, strategy: path}
naming:
  rules: [{pattern: '_v\d+(?=_|$)', replace: ''}]
versioning:
  path_pattern: "/v(?P<version>\\d+)/"   # default: /v(?P<version>\d+)(?=/|$)
  latest: unsuffixed
  legacy: submodule                      # or: suffix
```

```python
from client.widgets import get_widgets_by_id       # v2, the newest
from client.widgets.v1 import get_widgets_by_id    # v1
```

Versioning applies to operations that end up with the same name and differ
only in their path version; an operation that exists in one version stays
where it is. With `legacy: submodule` older versions go to
`<module>/v<N>.py` (the module becomes a package) and are not re-exported from
the package root; this needs `module_split` and `client_style: functions`.
`legacy: suffix` works with every layout and client style: older versions
stay next to the newest as `<name>_v<N>`. Per-endpoint settings for a
submodule version are keyed `<name>_v<N>` (e.g. `get_widgets_by_id_v1`).

#### Path Filtering

```yaml
documents:
  - source: https://api.example.com/openapi.json
    output: ./client
    include_paths:
      - /api/v2/**       # only v2 endpoints
    exclude_paths:
      - /internal/*      # skip internal endpoints
      - /admin/**        # skip admin endpoints
```

Patterns follow standard glob syntax (`*` = single segment, `**` = any depth).

#### Target Python Version

Generated code defaults to running on Python 3.11, so the runtime helpers
(`_pagination.py`, `_concurrency.py`) declare their generics with module-level
`TypeVar`s:

```python
T = TypeVar('T')

def paginate_offset(..., extract_items: Callable[[Any], list[T]]) -> list[T]:
```

That is the only spelling that works before 3.12, but a project whose linter is
configured for a newer Python flags it as legacy (ruff's
[UP047](https://docs.astral.sh/ruff/rules/non-pep695-generic-function/)). If the
generated client only has to run on 3.12+, say so and the same helpers are
emitted with [PEP 695](https://peps.python.org/pep-0695/) type parameters:

```yaml
documents:
  - source: https://api.example.com/openapi.json
    output: ./client
    target_python: "3.12"
```

```python
def paginate_offset[T](..., extract_items: Callable[[Any], list[T]]) -> list[T]:
```

The setting describes the *generated* code — OtterAPI itself still runs on any
supported interpreter, including when it emits 3.12-only syntax.

### Environment Variable Support

Any string value in a config file can reference environment variables:

```yaml
documents:
  - source: ${API_SPEC_URL}
    output: ${OUTPUT_DIR:-./client}
    base_url: ${BASE_URL:-https://api.example.com}
```

You can also configure a single document entirely via environment variables (no config file needed):

| Variable | Description |
|----------|-------------|
| `OTTER_SOURCE` | Path or URL to the OpenAPI spec |
| `OTTER_OUTPUT` | Output directory |
| `OTTER_BASE_URL` | Base URL override |
| `OTTER_MODELS_FILE` | Models filename |
| `OTTER_ENDPOINTS_FILE` | Endpoints filename |

### 🔎 Strict vs Lenient Validation

By default, OtterAPI validates specs **strictly**: an unrecognized or
structurally-invalid field aborts generation with a clear error, catching
malformed specs early.

Some real-world specs (especially auto-generated Swagger 2.0 documents) carry
stray, non-standard keys. Set `lenient: true` — or pass `--lenient` — to **drop**
those fields instead of failing. Each dropped key is reported as a warning.

```yaml
lenient: true

documents:
  - source: https://api.example.com/openapi.json
    output: ./client
```

```bash
otter generate --lenient
```

**Vendor extensions are always preserved.** OpenAPI specification extensions
(`x-*`) are kept in both strict and lenient mode. The only exception is the
type-discriminated security-scheme objects (`apiKey`, `http`, `oauth2`,
`openIdConnect`), which still reject unknown keys so their variant stays
unambiguous.

---

## 📦 Module Splitting

For large APIs, OtterAPI can split generated code into multiple organized modules.

### Enabling Module Splitting

```yaml
documents:
  - source: https://api.example.com/openapi.json
    output: ./client
    module_split:
      enabled: true
      strategy: tag
```

### Splitting Strategies

#### `tag` — Split by OpenAPI Tags

```yaml
module_split:
  enabled: true
  strategy: tag
  min_endpoints: 1
```

Endpoints tagged `["Users"]` go to `users.py`, `["Orders"]` to `orders.py`, etc.

#### `path` — Split by URL Path

```yaml
module_split:
  enabled: true
  strategy: path
  path_depth: 1
  global_strip_prefixes:
    - /api/v1
    - /api/v2
```

`/api/v1/users/123` → `users.py`, `/api/v1/orders/456` → `orders.py`.

#### `custom` — Explicit Module Mapping

```yaml
module_split:
  enabled: true
  strategy: custom
  module_map:
    users:
      - /users
      - /users/*
      - /users/**
    orders:
      - /orders/*
    health:
      - /health
      - /ready
```

#### `hybrid` — Combined Strategy (Default)

Tries `module_map` first, then tags, then path:

```yaml
module_split:
  enabled: true
  strategy: hybrid
  module_map:
    health:
      - /health
      - /ready
```

#### `none` — All to Fallback

```yaml
module_split:
  enabled: true
  strategy: none
  fallback_module: api
```

### Module Map Patterns

| Pattern | Matches |
|---------|---------|
| `/users` | Exact path only |
| `/users/*` | One additional segment |
| `/users/**` | Any depth below `/users` |
| `/v?/users` | Single wildcard character |

### Nested Module Maps

```yaml
module_split:
  enabled: true
  strategy: custom
  module_map:
    identity:
      users:
        - /users/*
      auth:
        - /auth/*
        - /login
        - /logout
    billing:
      invoices:
        - /invoices/*
      payments:
        - /payments/*
```

### Advanced Module Definition

```yaml
module_split:
  enabled: true
  strategy: custom
  module_map:
    v2_api:
      paths:
        - /v2/**
      strip_prefix: /v2
      description: "API v2 endpoints"
      file_name: v2.py          # override the generated filename
      modules:
        users:
          paths:
            - /users/*
```

### Module Split Options Reference

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enabled` | bool | `false` | Enable module splitting |
| `strategy` | string | `hybrid` | `none`, `path`, `tag`, `hybrid`, `custom` |
| `fallback_module` | string | `common` | Module for unmatched endpoints |
| `min_endpoints` | int | `2` | Minimum endpoints per module before consolidating into fallback |
| `flat_structure` | bool | `false` | Flat files instead of nested directories |
| `path_depth` | int | `1` | Path segments used for `path` strategy (1–5) |
| `global_strip_prefixes` | list | `/api`, `/api/v1`, `/api/v2`, `/api/v3` | Prefixes stripped from all paths before matching (whole segments only: `/api` strips `/api/users` but leaves `/apikeys` alone) |
| `module_map` | object | `{}` | Custom module-to-path mappings |
| `split_models` | bool | `false` | Generate per-module model files instead of one shared `models.py` |
| `shared_models_module` | string | `_models` | Module name for shared models when `split_models` is `true` |

### Output Structure

**Flat (`flat_structure: false`, default):**

```
client/
├── __init__.py
├── models.py
├── _client.py
├── client.py
├── users.py
├── orders.py
└── health.py
```

**Nested (`flat_structure: false` with nested `module_map`):**

```
client/
├── __init__.py
├── models.py
├── _client.py
├── client.py
├── identity/
│   ├── __init__.py
│   ├── users.py
│   └── auth.py
└── billing/
    ├── __init__.py
    └── invoices.py
```

---

## 📄 Pagination

OtterAPI generates iterator-based pagination methods alongside standard endpoint functions. Paginated endpoints get an `_iter` suffix variant that yields items one by one, handling page-fetching automatically.

### Enabling Pagination

```yaml
documents:
  - source: https://api.example.com/openapi.json
    output: ./client
    pagination:
      enabled: true
```

### Auto-Detection

When `auto_detect: true` (the default), OtterAPI inspects each endpoint's parameters. If it finds a matching pair for any pagination style, it generates pagination methods automatically — no per-endpoint config needed.

```yaml
pagination:
  enabled: true
  auto_detect: true           # default
  default_style: offset       # default; used when auto-detected
```

### Pagination Styles

| Style | Required Parameters | Description |
|-------|--------------------|-|
| `offset` | `offset` + `limit` | Offset/limit (e.g. `?offset=0&limit=100`) |
| `cursor` | `cursor` + `limit` | Cursor-based (e.g. `?cursor=abc&limit=100`) |
| `page` | `page` + `per_page` | Page number (e.g. `?page=2&per_page=50`) |
| `link` | — | RFC 5988 `Link` header |

### Global Pagination Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enabled` | bool | `false` | Enable pagination generation |
| `auto_detect` | bool | `true` | Auto-detect pagination from parameter names |
| `default_style` | string | `offset` | Fallback style when auto-detecting |
| `default_page_size` | int | `100` | Default page size for iteration |
| `default_data_path` | string | `null` | Default JSON path to items array in response |
| `default_total_path` | string | `null` | Default JSON path to total count in response |
| `default_offset_param` | string | `offset` | Param name used for offset detection |
| `default_limit_param` | string | `limit` | Param name used for limit detection |
| `default_cursor_param` | string | `cursor` | Param name used for cursor detection |
| `default_page_param` | string | `page` | Param name used for page detection |
| `default_per_page_param` | string | `per_page` | Param name used for per-page detection |
| `endpoints` | object | `{}` | Per-endpoint overrides |

### Per-Endpoint Pagination Options

```yaml
pagination:
  enabled: true
  endpoints:
    list_users:
      style: cursor
      cursor_param: next_token
      limit_param: max_results
      data_path: data.users
      total_path: meta.total
      next_cursor_path: meta.next_token
      default_page_size: 50
      max_page_size: 200
    get_orders:
      enabled: false            # disable pagination for this endpoint
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enabled` | bool\|null | inherits | Override whether to generate pagination |
| `style` | string | inherits | `offset`, `cursor`, `page`, `link` |
| `offset_param` | string | inherits | Name of offset parameter |
| `limit_param` | string | inherits | Name of limit parameter |
| `cursor_param` | string | inherits | Name of cursor parameter |
| `page_param` | string | inherits | Name of page parameter |
| `per_page_param` | string | inherits | Name of per-page parameter |
| `data_path` | string | inherits | JSON path to items array in response |
| `total_path` | string | inherits | JSON path to total count |
| `next_cursor_path` | string | `null` | JSON path to next cursor value |
| `total_pages_path` | string | `null` | JSON path to total page count |
| `default_page_size` | int | inherits | Default page size |
| `max_page_size` | int | `null` | Maximum allowed page size |
| `cursor_location` | `query`\|`body` | `query` | Where the position parameter (offset/cursor/page) is sent |
| `limit_location` | `query`\|`body` | `cursor_location` | Where the page-size parameter is sent |
| `next_cursor_from` | `response`\|`last_item` | `response` | Read the next cursor from `next_cursor_path`, or from the last item on the page |
| `next_cursor_item_path` | string | `cursor` | Path of the cursor on an item, with `next_cursor_from: last_item` |

### Paginating Through the Request Body

Some APIs page `POST` endpoints through the JSON body instead of the query
string, and expect the last row's cursor back as the next `after`. Configure
them explicitly -- auto-detection only looks at query parameters, and never
turns on body pagination by itself, because it re-sends the request once per
page:

```yaml
pagination:
  enabled: true
  default_data_path: data
  default_total_path: total
  endpoints:
    search_widgets:
      style: cursor
      cursor_param: after        # body field names, as on the wire
      limit_param: limit
      cursor_location: body      # limit_location defaults to the same
      next_cursor_from: last_item
      next_cursor_item_path: cursor
```

With `request_body.flatten: true` the paginated functions (`_iter`, `_df`,
`_export`, `_page`, ...) do not take `after` and `limit` as arguments at all,
the same way the query-string `offset` / `limit` are hidden; set them with
`cursor=` and `page_size=`.

The paginator overlays `after` and `limit` onto the dumped body for each
page; everything else you put in the body is sent unchanged. `page_size`
takes precedence over a `limit` set on the body, and an `after` set on the
body is used as the starting position unless `cursor=` is passed. Paging
stops on an empty page, once `total` items have been fetched (when the
response has a total), or when the server returns the cursor it was just
sent. It does not stop on a short page, since a server that caps the page
size would otherwise cut the results off.

When pagination is enabled, code generation warns about endpoints that return
the configured items list and total count but are not paginated, since their
DataFrame and export helpers would return only the first page.

### Usage

```python
from client import list_users_iter, async_list_users_iter

# Iterate all users (handles pagination automatically)
for user in list_users_iter():
    print(user.name)

# Async variant
async for user in async_list_users_iter():
    print(user.name)
```

### Page Objects

Every paginated endpoint also gets `<name>_page` (one page) and `<name>_pages`
(an iterator of pages). A page keeps what the item-level helpers discard:

```python
from client import list_users_page, list_users_pages

for page in list_users_pages(page_size=100):
    print(page.total, page.next_cursor)   # or next_offset / next_page
    print(page.data.rate_limit)           # the validated response envelope
    for user in page.items:
        ...

# Checkpoint a long iteration and resume it later
page = list_users_page(cursor=saved_cursor)
if page.has_next_page():
    page = page.get_next_page()
```

| Attribute | Description |
|-----------|-------------|
| `items` | The items on the page (`Page` also iterates over them and has a `len`) |
| `data` | The parsed response envelope, typed as the endpoint's response model |
| `total` | The total item count (`total_path`), or `None` |
| `total_pages` | The total page count (`total_pages_path`, page style), or `None` |
| `next_cursor` / `next_offset` / `next_page` | Where the next page starts, or `None` on the last page |
| `has_next_page()` / `get_next_page()` | Whether another page follows / fetch it |
| `iter_pages(max_pages=None)` | This page and every page after it |

The async variants (`async_list_users_page`, `async_list_users_pages`) return
`AsyncPage` objects, whose `get_next_page()` is awaited and whose
`iter_pages()` is consumed with `async for`. With `result_objects: true`,
`Query.page()` and `Query.pages()` do the same. Turn the variants off with
`pagination.page_variants: false`.

---

## 📊 DataFrame Conversion

When enabled, list-returning endpoints get additional methods that return pandas or polars DataFrames directly.

### Enabling DataFrame Methods

```yaml
documents:
  - source: https://api.example.com/openapi.json
    output: ./client
    dataframe:
      enabled: true
      pandas: true      # generate _df methods (default: true)
      polars: true      # generate _pl methods (default: false)
```

### Generated Methods

| Original | Pandas | Polars |
|----------|--------|--------|
| `get_users()` | `get_users_df()` | `get_users_pl()` |
| `async_get_users()` | `async_get_users_df()` | `async_get_users_pl()` |

### Usage

```python
from client import list_pets_df, list_pets_pl

pdf = list_pets_df("available")
plf = list_pets_pl("available")

# Override the data extraction path at call time
df = list_pets_df("available", path="response.data.pets")
```

### Nested Response Paths

```yaml
dataframe:
  enabled: true
  pandas: true
  default_path: data.items       # default for all endpoints
  endpoints:
    get_users:
      path: data.users           # override for this endpoint
    get_analytics:
      path: response.events
      polars: true
      pandas: false
```

### DataFrame Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enabled` | bool | `false` | Enable DataFrame generation |
| `pandas` | bool | `true` | Generate `_df` (pandas) methods |
| `polars` | bool | `false` | Generate `_pl` (polars) methods |
| `default_path` | string | `null` | Default JSON path to extract data |
| `include_all` | bool | `true` | Generate for all list-returning endpoints |
| `endpoints` | object | `{}` | Per-endpoint overrides |

### Per-Endpoint DataFrame Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enabled` | bool\|null | inherits | Override whether to generate methods |
| `path` | string | inherits | JSON path to extract data |
| `pandas` | bool\|null | inherits | Override pandas generation |
| `polars` | bool\|null | inherits | Override polars generation |

---

## 💾 File Export

When enabled, list-returning endpoints get streaming export helpers for writing responses directly to files (local or remote via UPath).

### Enabling Export

```yaml
documents:
  - source: https://api.example.com/openapi.json
    output: ./client
    export:
      enabled: true
      formats:
        - csv
        - jsonl
```

Parquet support requires the `pyarrow` extra:

```bash
pip install otterapi[parquet]
```

### Export Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enabled` | bool | `false` | Enable export helper generation |
| `formats` | list | `[csv, jsonl]` | Default formats: `csv`, `tsv`, `jsonl`, `parquet` |
| `default_path` | string | `null` | Default JSON path to extract list data |
| `include_all` | bool | `true` | Generate helpers for all list-returning endpoints |
| `batch_size` | int | `1000` | Batch size used when streaming pages to disk |
| `endpoints` | object | `{}` | Per-endpoint overrides |

### Per-Endpoint Export Options

```yaml
export:
  enabled: true
  formats: [csv, jsonl]
  endpoints:
    list_users:
      formats: [parquet]        # only parquet for this endpoint
      path: data.users
    get_events:
      enabled: false            # disable export for this endpoint
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enabled` | bool\|null | inherits | Override whether to generate helpers |
| `path` | string | inherits | JSON path to extract data |
| `formats` | list | inherits | Override formats for this endpoint |

### Usage

Every endpoint returning a list of models gets an `<endpoint>_export` function
(and an `async_<endpoint>_export` twin) that mirrors the endpoint's own
parameters and adds `output_path` first, plus keyword-only `format` and
`batch_size`. Extra keyword arguments are passed through to the file writer.
A list of a union of models -- such as a `oneOf` with a `discriminator` --
exports too: the file has every variant's columns, and each row fills its own
(CSV / TSV cells and Parquet values of the others stay empty; JSONL keeps each
row's shape). Lists of plain values (`list[str]`) have no columns, so they get
no export function.

```python
from client import list_users_export, async_list_users_export

rows = list_users_export("output/users.csv", status="active")          # returns the row count
list_users_export("s3://my-bucket/users.parquet", format="parquet")   # UPath remote targets work too
list_users_export("out.csv", format="csv", delimiter=";")             # writer options pass through
```

For a [paginated](#-pagination) endpoint the export wrapper streams through the
endpoint's `_iter` variant and declares exactly its knobs -- the starting
position (`offset` / `cursor` / `page`), `page_size` and `max_items` -- so
`list_users_export("out.csv", offset=100)` starts where
`list_users_iter(offset=100)` does.

If an endpoint has its own parameter named `format`, `output_path` or
`batch_size`, the wrapper exposes it as `format_` (etc.) and forwards it to the
API under its real name; with `result_objects`, `Query.export()` applies the
same aliasing for you.

---

## 🔓 Response Unwrapping

For APIs that wrap all responses in an envelope (e.g. `{"data": {...}, "meta": {...}}`), response unwrapping makes endpoints return just the inner data automatically.

### Enabling Response Unwrap

```yaml
documents:
  - source: https://api.example.com/openapi.json
    output: ./client
    response_unwrap:
      enabled: true
      data_path: data          # default path (default: "data")
```

### Per-Endpoint Overrides

```yaml
response_unwrap:
  enabled: true
  data_path: data
  endpoints:
    get_user:
      data_path: result.user   # this endpoint uses a different path
    get_raw_data:
      enabled: false           # don't unwrap this endpoint
```

The document-wide `data_path` only applies to endpoints whose response
actually has that field: an endpoint returning `{"total": 3}`, a plain object
or no body at all returns the parsed response unchanged. A `data_path` set
for one endpoint must exist on its response, or generation fails. Use one to
unwrap a scalar, e.g. `count_users: {data_path: total}` returns an `int`.

### Response Unwrap Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enabled` | bool | `false` | Enable response unwrapping |
| `data_path` | string | `data` | Default JSON path to extract from all responses |
| `endpoints` | object | `{}` | Per-endpoint overrides |

### Per-Endpoint Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enabled` | bool\|null | inherits | Override whether to unwrap |
| `data_path` | string | inherits | JSON path for this endpoint |

---

## 📖 Using Generated Code

### Direct Function Imports

```python
from client import get_user, create_user, list_orders

# Async versions are prefixed with 'async_'
from client import async_get_user, async_create_user, async_list_orders

# Sync
user = get_user(user_id=123)

# Async
import asyncio

async def main():
    user = await async_get_user(user_id=123)

asyncio.run(main())
```

### Module-Specific Imports (with splitting)

```python
from client.users import get_user, create_user
from client.orders import list_orders, get_order
```

### Using the Client Class

In the default `functions` style the `Client` carries configuration and the
connection pool; you pass it to each endpoint function via `client=`:

```python
from client import Client, get_user, async_get_user

client = Client(
    base_url='https://api.example.com',
    timeout=30.0,
    headers={'Authorization': 'Bearer your-token'},
)

user = get_user(user_id=123, client=client)

async def main():
    user = await async_get_user(user_id=123, client=client)
```

Prefer a method-based API (`client.get_user(...)`)? Set
[`client_style`](#-client-style).

### 🎛 Client Style

`client_style` controls the shape of the generated public API.

**`functions` (default)** — a standalone function per endpoint and variant, plus
an infrastructure `Client` you pass in:

```python
from client import get_user, async_get_user, list_users_df

user = get_user(user_id=123)
df = list_users_df()
```

**`client`** — additionally generates `Client` and `AsyncClient` classes whose
methods delegate to those functions, giving a class-namespaced surface with clean
names (no `async_` prefix):

```yaml
documents:
  - source: https://api.example.com/openapi.json
    output: ./client
    client_style: client
```

```python
from client import Client, AsyncClient

user = Client(base_url='https://api.example.com').get_user(user_id=123)

async def main():
    async with AsyncClient() as api:
        user = await api.get_user(user_id=123)          # no async_ prefix
        df = await api.list_users_df()
        async for u in api.list_users_iter():
            ...
```

**`resource`** — groups the methods into resource sub-clients, derived from the
same strategy as [module splitting](#-module-splitting) (tags/path/custom). Only
the grouping changes: every method keeps its function's name (or, with
`resource_naming: operation_id`, the name its operationId declares), so adding
an operation to the spec never renames or reshapes a method you already call:

```yaml
documents:
  - source: https://api.example.com/openapi.json
    output: ./client
    client_style: resource
```

```python
from client import Client, AsyncClient

client = Client(base_url='https://api.example.com')
user = client.users.get_user(user_id=123)
users = client.users.list_users()
order = client.orders.get_order(order_id=7)

async with AsyncClient() as api:
    user = await api.users.get_user(user_id=123)
```

`resource_naming` controls how the sub-clients are derived and can **nest**:

```yaml
    client_style: resource
    resource_naming: path          # or: operation_id
```

- `tag` (default) — one level, from the module-split strategy.
- `path` — nests by URL segments, one sub-client per segment:
  `GET /identity/users/{id}` (operationId `getUser`) →
  `client.identity.users.get_user(id)`, and `/pet/findByStatus` →
  `client.pet.find_by_status.find_pets_by_status(...)`.
- `operation_id` — nests by a dotted (or `/`-separated) operationId, whose last
  part is the method name the spec declares: `identity.users.get` →
  `client.identity.users.get()` (variants keep their suffix: `get_df`). If two
  operations in one resource declare the same name, both keep their full
  function names. A flat operationId (no separator) stays on the client.

Segments are made into valid attribute names (`/2fa` → `client.n2fa`, a keyword
gets a trailing underscore: `/import` → `client.import_`).

```python
user = client.identity.users.get_user(user_id=123)
invoices = client.billing.invoices.list_invoices()
```

#### Result objects

With `result_objects: true` (with `client_style: client` or `resource`), every
**list** endpoint (one that returns `list[X]`) returns a deferred `Query` instead
of exposing separate `_df` / `_pl` / `_iter` / `_export` methods. Whether a method
returns a `Query` depends only on the endpoint's return type, never on which
features are enabled, so switching on DataFrames or export only adds terminals. The endpoint's arguments are captured
once; a terminal method materializes it. `Query` / `AsyncQuery` are exported from
the package for type hints:

```yaml
    client_style: resource
    result_objects: true
```

```python
q = client.users.list_users(status="active")   # nothing sent yet
rows = q.all()                                  # list[User]
df   = q.to_pandas()                            # pandas DataFrame
for u in q.iter(): ...                          # streamed, page by page
q.export("users.csv")                           # write to a file

async with AsyncClient() as api:
    rows = await api.users.list_users().all()
    async for u in api.users.list_users().iter(): ...
```

Terminals whose feature wasn't enabled at generation time raise a clear error;
every other endpoint keeps returning its model directly, with its variants as
separate methods. Iterating a
result object directly (`for row in q` / `async for row in q`) streams pages
when pagination is enabled and otherwise walks the fetched list.

#### Composing your own SDK

To ship a user-facing SDK of your own, **compose** over a generated `Client`
rather than subclassing it: hold the generated client privately and expose
only the methods you choose. You own the public names, so a renamed
`operationId` in the spec stays an internal change instead of a breaking one,
and nothing the generator emits leaks into your API by accident.

The `client` style is the layout to compose over. Its flat `Client` /
`AsyncClient` reach every operation the `resource` style does, on both the sync
and the async side (the test suite asserts this), and with `result_objects: true`
each list endpoint hands you a `Query` whose terminals cover every enabled
feature -- so the facade loses nothing.

```yaml
documents:
  - source: https://api.example.com/openapi.json
    output: ./mysdk/_generated        # private to your package
    client_style: client
    result_objects: true
    pagination: { enabled: true }
    dataframe: { enabled: true }
    export: { enabled: true }
```

```python
from ._generated import AsyncClient, BaseAPIError, Client, Query, User


class DirectoryError(Exception): ...


class UserDirectory:
    """Your public API. Only what is defined here is visible to users."""

    def __init__(self, base_url: str, token: str):
        self._api = Client(base_url=base_url, headers={"Authorization": f"Bearer {token}"})

    def active(self) -> Query[User]:            # .all() / .iter() / .to_pandas() / .export()
        return self._api.list_users(status="active")

    def by_id(self, user_id: int) -> User:
        try:
            return self._api.get_user(user_id)
        except BaseAPIError as e:               # translate, don't leak
            raise DirectoryError(e.status_code) from e
```

The async facade is the same shape around `AsyncClient`, returning
`AsyncQuery[User]`. Inject an `httpx2` client (`Client(http_client=...)`) to test
the facade against a mock transport.

Two things to know when composing: `resource` is not a good base for a facade
(its sub-client classes are private and rebuilt on every access, so they cannot
be extended or injected), and a generated method whose name collides with a
base-client member such as `close` is renamed with a trailing underscore --
check the generated names before delegating to them.

The free functions remain as the implementation the methods call. `client` and
`resource` compose with [module splitting](#-module-splitting): the functions are
split across module files and each method routes to its function's module, while
`Client` / `AsyncClient` stay the single public entry point. With `resource` the
resources mirror the split modules.

### Working with Models

```python
from client.models import User, CreateUserRequest

new_user = CreateUserRequest(name='John Doe', email='john@example.com')
user = create_user(body=new_user)
print(user.id, user.email)
```

---

## 🔐 Authentication

Every scheme under `components.securitySchemes` becomes a keyword-only
parameter on the generated client, applied automatically to every request.

```yaml
# spec
components:
  securitySchemes:
    api_key:
      type: apiKey
      name: X-API-Key
      in: header
    bearerAuth:
      type: http
      scheme: bearer
```

```python
from client import Client

client = Client(api_key='...', bearer_auth='...')
user = get_user(user_id=123, client=client)
```

| Scheme | Parameter type | Sent as |
|--------|----------------|---------|
| `apiKey` in `header` | `str` | the header named in the spec |
| `apiKey` in `query` | `str` | the query parameter named in the spec |
| `apiKey` in `cookie` | `str` | appended to the `Cookie` header |
| `http` / `bearer` | `str` | `Authorization: Bearer <token>` |
| `http` / `basic` | `tuple[str, str]` | `Authorization: Basic <base64>` |
| `oauth2`, `openIdConnect` | `str` | `Authorization: Bearer <token>` — you supply an already-obtained token |

Schemes OtterAPI does not recognize are skipped with a warning rather than
failing generation.

### Credentials from the environment

Omit a credential and the client reads it from `<env_prefix>_<SCHEME_NAME>`,
upper-cased — so the `api_key` scheme above falls back to `OTTER_API_KEY`.
Set `env_prefix` per document when one project generates several clients:

```yaml
documents:
  - source: https://api.example.com/openapi.json
    output: ./client
    auth:
      env_prefix: MYAPI    # -> MYAPI_API_KEY
```

When a credential already lives in a variable the prefix rule would never
derive, name it outright with `env_vars`. Keys are scheme names from
`components.securitySchemes` (the generated parameter name works too), and
only the schemes you name are affected — the rest keep the derived default:

```yaml
documents:
  - source: https://api.example.com/openapi.json
    output: ./client
    auth:
      env_prefix: MYAPI
      env_vars:
        api_key: STRIPE_SECRET_KEY   # this one exactly
        # bearerAuth still falls back to MYAPI_BEARER_AUTH
```

A constructor argument always beats the environment. Basic auth needs two
values, so it has no environment fallback at all.

### Auth Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enabled` | bool | `true` | Generate credential parameters from `securitySchemes` |
| `env_prefix` | string | `OTTER` | Prefix for the environment variables consulted when an argument is omitted. Empty string drops the prefix |
| `env_vars` | map | `{}` | Exact environment variable per scheme, keyed by scheme name. Overrides `env_prefix` for the schemes it names |

### Customizing

`_apply_auth` runs before the `_before_request` hook, so the hook can still
override or refresh whatever it set — useful for rotating an OAuth token:

```python
class MyClient(Client):
    def _before_request(self, request):
        if self._token_expired():
            request['headers']['Authorization'] = f'Bearer {self._refresh()}'
        return request
```

Set `auth.enabled: false` to opt out entirely and wire authentication by hand
in the user-owned `client.py`.

---

## 📤 Request Bodies

Bodies are serialized with `model_dump(mode='json', by_alias=True,
exclude_unset=True)`: JSON mode so `datetime`, `UUID` and `Decimal` fields
reach the wire as strings rather than raising, and aliases so fields whose
spec name is not a Python identifier keep their wire name.

`exclude_unset` means only the fields you actually set are sent:

```python
place_order(body=Order(id=7))     # -> {"id": 7}
```

Most APIs read an explicit `null` as "clear this field", so omitting the
fields you never touched is the safer default — and it is what makes partial
updates work. If an endpoint genuinely needs those nulls, a PUT that replaces
a whole resource for instance, turn it off:

```yaml
documents:
  - source: https://api.example.com/openapi.json
    output: ./client
    request_body:
      exclude_unset: false
```

```python
place_order(body=Order(id=7))     # -> {"id": 7, "shipDate": null, "note": null}
```

### Multipart bodies

A `multipart/form-data` body is sent one part per property, following the
OpenAPI Encoding Object defaults: `bytes` or file-like values become file
uploads, arrays become one part per item under the same name, objects are sent
as `application/json` parts, and everything else is sent as a plain text
field. `None` values are left out, since multipart cannot express null.

```python
upload(body=Upload(name='a', tags=['x', 'y'], file=b'...'))
# -> parts: name=a, tags=x, tags=y, file (upload)
```

### Flattening the body into parameters

By default a POST/PUT/PATCH takes the body as one model:

```python
place_order(body=Order(id=7, quantity=2))
```

Set `flatten` and the model's fields become parameters instead:

```yaml
documents:
  - source: https://api.example.com/openapi.json
    output: ./client
    request_body:
      flatten: true
```

```python
place_order(id=7, quantity=2)
```

Required fields stay positional so you cannot forget them; the rest are
keyword-only and default to `NOTSET`. Fields inherited through `allOf` are
included. The request that goes on the wire is identical either way —
flattening changes the signature, not the protocol.

### `NOTSET` vs `None`

A nullable field has two distinct "empty" states, and one default cannot carry
both. Omitted fields therefore default to `NOTSET`, which leaves `None`
meaning what it says:

```python
place_order(id=7)                 # -> {"id": 7}
place_order(id=7, note=None)      # -> {"id": 7, "note": null}
place_order(id=7, note=NOTSET)    # -> {"id": 7}   (same as omitting)
```

`NOTSET` is exported from the generated package, so you can pass it from a
conditional rather than branching on the call itself:

```python
from myclient import NOTSET, place_order

place_order(id=7, note=note if include_note else NOTSET)
```

It is a single-member enum, so type checkers treat it as a literal: inside the
client, `value is not NOTSET` narrows `int | None | NotSet` to `int | None`.
It is also falsy, so `if value:` reads as "was anything given".

Two more details worth knowing:

- **A body field can collide with a path or query parameter.** The parameter
  keeps its name and the body field gains a `_body` suffix, so a spec with
  both a `status` query param and a `status` body field generates
  `add_note(status_body, *, status=None)`. The payload still uses `status`.
- **Only object-shaped JSON bodies can flatten.** An array body, a bare
  string, or a multipart upload has nothing to spread; those keep the single
  `body=` parameter and log why.

### Per-path overrides

`paths` overrides the document setting for matching URL paths, using the same
glob syntax as `include_paths`. When several patterns match, the longest wins,
so a specific path beats the wildcard covering it:

```yaml
request_body:
  flatten: true
  paths:
    /reports/**:                 # a big nested body reads better as a model
      flatten: false
    /reports/summary:            # …except this one
      flatten: true
    /users/{id}:
      exclude_unset: false       # this PUT replaces the whole resource
```

### Request Body Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `exclude_unset` | bool | `true` | Send only the fields the caller set, rather than an explicit null for every untouched optional field |
| `flatten` | bool | `false` | Spread the body model's fields into individual parameters instead of one `body=` argument |
| `paths` | map | `{}` | Per-path overrides keyed by a glob over the URL path; longest match wins |

Each path override accepts `exclude_unset` and `flatten`; anything omitted
inherits the document setting.

---

## 🔧 CLI Reference

```bash
# Generate from default config (otter.yml, otter.yaml, or pyproject.toml)
otter generate

# Generate from specific config file
otter generate -c my-config.yml

# Generate without a config file
otter generate -s ./api.yaml -o ./client

# Override the base URL — bakes an absolute base into a file-loaded spec whose
# servers are relative (e.g. `url: /api/v3`); applies to every document in a
# config. Without it the relative URL stays the client default, so callers pass
# `base_url=` when constructing the client.
otter generate -s ./api.yaml -o ./client -b https://api.example.com

# Tolerate malformed specs (drop unknown/invalid fields with a warning)
otter generate --lenient

# Initialize a new config file
otter init

# Validate configuration
otter validate
```

Generation is all-or-nothing: if a run fails partway through, the output
directory is restored to exactly what it was before, so a failure never
leaves a half-written client behind. Files you own inside it — `client.py`
and anything else you keep there — survive both a failed run and a
successful regeneration.

---

## 🐍 Programmatic API

```python
from otterapi.codegen import Codegen
from otterapi.config import DocumentConfig, PaginationConfig, ExportConfig

config = DocumentConfig(
    source='https://petstore3.swagger.io/api/v3/openapi.json',
    output='./client',
    pagination=PaginationConfig(enabled=True),
    export=ExportConfig(enabled=True, formats=['csv', 'parquet']),
)
Codegen(config).generate()
```

---

## ❗ Error Handling

Generated clients raise a typed exception hierarchy rooted at `BaseAPIError`:

```python
from my_client import (
    Client, list_users,
    BaseAPIError,        # catches every API error
    ClientError,         # all 4xx
    ServerError,         # all 5xx
    NotFoundError,       # 404
    RateLimitError,      # 429
)

try:
    users = list_users(client=Client())
except NotFoundError as e:
    log.warning('not found: %s', e.detail)
except RateLimitError:
    backoff_and_retry()
except ServerError:
    page_oncall()
except BaseAPIError as e:
    log.error('unexpected %d: %s', e.status_code, e.detail)
```

Mapped status codes: 400, 401, 403, 404, 409, 422, 429, 500, 502, 503, 504. Other 4xx/5xx fall through to `ClientError`/`ServerError`. Subclass `BaseAPIError` in your `client.py` to customize error parsing.

---

## ♻️ Regenerating after spec changes

OtterAPI is **idempotent** for generated files:

- `models.py`, `endpoints.py`, `_client.py`, `_pagination.py`, `_export.py`, `_dataframe.py` are **rewritten on every run** — never edit them by hand.
- `client.py` is generated **once** and left alone — this is your customization seam.

---

## 🛠 Development

```bash
git clone https://github.com/danplischke/otterapi.git
cd otterapi
uv sync
uv run pytest
uv run pytest --cov=otterapi
uv run ruff format .
uv run ruff check .
```

---

## 📄 License

MIT License — see LICENSE for details.
