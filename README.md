# 🦦 OtterAPI

> *A cute and intelligent OpenAPI client generator that dives deep into your OpenAPIs*

**OtterAPI** is a sleek Python library that transforms OpenAPI specifications into clean, type-safe client code with Pydantic models and httpx-based HTTP clients.

## ✨ Features

- **Type-Safe Code Generation** - Generates Pydantic models and fully typed endpoint functions
- **Sync & Async Support** - Generate both synchronous and asynchronous API clients
- **OpenAPI 3.x Support** - Full support for OpenAPI 3.0, 3.1, and 3.2 specifications
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
from petstore_client import get_pet_by_id, aget_pet_by_id

# Synchronous usage
pet = get_pet_by_id(pet_id=123)

# Asynchronous usage
import asyncio
pet = asyncio.run(aget_pet_by_id(pet_id=123))
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
| `validate_output` | bool | `true` | Validate generated code syntax after writing |
| `create_py_typed` | bool | `true` | Create `py.typed` marker files |

```yaml
format_output: true
validate_output: true
create_py_typed: true

documents:
  - source: https://api.example.com/openapi.json
    output: ./client
```

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
| `include_paths` | list | `null` | Glob patterns — only matching paths are generated |
| `exclude_paths` | list | `null` | Glob patterns — matching paths are skipped (applied after `include_paths`) |

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
| `global_strip_prefixes` | list | `/api`, `/api/v1`, `/api/v2`, `/api/v3` | Prefixes stripped from all paths before matching |
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

### Usage

```python
from client import list_users_iter, alist_users_iter

# Iterate all users (handles pagination automatically)
for user in list_users_iter():
    print(user.name)

# Async variant
async for user in alist_users_iter():
    print(user.name)
```

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
| `aget_users()` | `aget_users_df()` | `aget_users_pl()` |

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

```python
from client import export_list_users_csv, export_list_users_parquet

# Export directly to a file
export_list_users_csv("output/users.csv")
export_list_users_parquet("s3://my-bucket/users.parquet")  # UPath remote targets work too
```

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

# Async versions are prefixed with 'a'
from client import aget_user, acreate_user, alist_orders

# Sync
user = get_user(user_id=123)

# Async
import asyncio

async def main():
    user = await aget_user(user_id=123)

asyncio.run(main())
```

### Module-Specific Imports (with splitting)

```python
from client.users import get_user, create_user
from client.orders import list_orders, get_order
```

### Using the Client Class

```python
from client import Client

client = Client(
    base_url='https://api.example.com',
    timeout=30.0,
    headers={'Authorization': 'Bearer your-token'},
)

user = client.get_user(user_id=123)

async def main():
    user = await client.aget_user(user_id=123)
```

### Working with Models

```python
from client.models import User, CreateUserRequest

new_user = CreateUserRequest(name='John Doe', email='john@example.com')
user = create_user(body=new_user)
print(user.id, user.email)
```

---

## 🔧 CLI Reference

```bash
# Generate from default config (otter.yml, otter.yaml, or pyproject.toml)
otter generate

# Generate from specific config file
otter generate -c my-config.yml

# Initialize a new config file
otter init

# Validate configuration
otter validate
```

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
