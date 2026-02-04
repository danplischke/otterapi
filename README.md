# 🦦 OtterAPI

> *A cute and intelligent OpenAPI client generator that dives deep into your OpenAPIs*

**OtterAPI** is a sleek Python library that transforms OpenAPI specifications into clean, type-safe client code with Pydantic models and httpx-based HTTP clients.

## ✨ Features

- **Type-Safe Code Generation** - Generates Pydantic models and fully typed endpoint functions
- **Sync & Async Support** - Generate both synchronous and asynchronous API clients
- **OpenAPI 3.x Support** - Full support for OpenAPI 3.0, 3.1, and 3.2 specifications
- **Module Splitting** - Organize large APIs into multiple organized files
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

## 📝 Configuration

### Basic Configuration

```yaml
documents:
  - source: https://petstore3.swagger.io/api/v3/openapi.json
    output: petstore_client

  - source: ./local-api.json
    output: local_client
    base_url: https://api.example.com
```

### Full Configuration Options

```yaml
documents:
  - source: https://api.example.com/openapi.json  # URL or file path (required)
    output: ./client                               # Output directory (required)
    base_url: https://api.example.com              # Override base URL from spec
    models_file: models.py                         # Models filename (default: models.py)
    endpoints_file: endpoints.py                   # Endpoints filename (default: endpoints.py)
    generate_async: true                           # Generate async functions (default: true)
    generate_sync: true                            # Generate sync functions (default: true)
    client_class_name: MyAPIClient                 # Client class name (default: from API title)
    module_split:                                  # Module splitting configuration
      enabled: false                               # Enable splitting (default: false)
      # ... see Module Splitting section below
```

---

## 📦 Module Splitting

For large APIs with many endpoints, OtterAPI can split the generated code into multiple organized modules instead of a single `endpoints.py` file.

### Why Use Module Splitting?

- **Better Organization** - Group related endpoints together
- **Easier Navigation** - Find endpoints quickly in smaller files
- **Improved IDE Performance** - Smaller files load faster
- **Cleaner Imports** - Import only what you need from specific modules

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

#### `tag` - Split by OpenAPI Tags

Uses the first tag from each operation to determine the module:

```yaml
module_split:
  enabled: true
  strategy: tag
  min_endpoints: 1
```

**Result:** Endpoints tagged with `["Users"]` go to `users.py`, `["Orders"]` go to `orders.py`, etc.

#### `path` - Split by URL Path

Uses the first segment(s) of the URL path:

```yaml
module_split:
  enabled: true
  strategy: path
  path_depth: 1                    # Number of path segments to use
  global_strip_prefixes:           # Remove these prefixes first
    - /api/v1
    - /api/v2
```

**Result:** `/api/v1/users/123` → `users.py`, `/api/v1/orders/456` → `orders.py`

#### `custom` - Explicit Module Mapping

Define exactly which paths go to which modules using glob patterns:

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
      - /orders/**
    health:
      - /health
      - /ready
      - /live
```

#### `hybrid` - Combined Strategy (Default)

Tries custom module_map first, then falls back to tags, then path:

```yaml
module_split:
  enabled: true
  strategy: hybrid
  module_map:
    health:                        # Custom mapping takes priority
      - /health
      - /ready
  # Remaining endpoints use tags if available, otherwise path
```

#### `none` - All to Fallback

All endpoints go to a single fallback module:

```yaml
module_split:
  enabled: true
  strategy: none
  fallback_module: api             # All endpoints go here
```

### Pattern Syntax

The module map supports glob patterns:

| Pattern | Matches | Example |
|---------|---------|---------|
| `/users` | Exact path | `/users` only |
| `/users/*` | Single segment | `/users/123`, `/users/abc` |
| `/users/**` | Any depth | `/users/123`, `/users/123/profile/settings` |
| `/v?/users` | Single character | `/v1/users`, `/v2/users` |

### Nested Module Maps

Create hierarchical module structures:

```yaml
module_split:
  enabled: true
  strategy: custom
  module_map:
    identity:                      # Parent module
      users:                       # Child: identity/users.py
        - /users/*
        - /users/**
      auth:                        # Child: identity/auth.py
        - /auth/*
        - /login
        - /logout
      roles:                       # Child: identity/roles.py
        - /roles/*
    billing:
      invoices:
        - /invoices/*
      payments:
        - /payments/*
```

### Advanced Module Definition

Full control over each module:

```yaml
module_split:
  enabled: true
  strategy: custom
  module_map:
    v2_api:
      paths:                       # Explicit paths key
        - /v2/**
      strip_prefix: /v2            # Strip this prefix from paths in this module
      description: "API v2 endpoints (deprecated)"  # Module docstring
      modules:                     # Nested submodules
        users:
          paths:
            - /users/*
        billing:
          paths:
            - /billing/*
```

### Module Split Options Reference

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enabled` | bool | `false` | Enable module splitting |
| `strategy` | string | `hybrid` | Strategy: `none`, `path`, `tag`, `hybrid`, `custom` |
| `fallback_module` | string | `common` | Module name for unmatched endpoints |
| `min_endpoints` | int | `2` | Minimum endpoints per module (smaller modules get consolidated) |
| `flat_structure` | bool | `false` | `true`: flat files, `false`: nested directories |
| `path_depth` | int | `1` | Path segments to use for `path` strategy (1-5) |
| `global_strip_prefixes` | list | common prefixes | Prefixes to strip from all paths before matching |
| `module_map` | object | `{}` | Custom module mappings |

### Output Structure Examples

**Flat Structure (default):**

```
client/
├── __init__.py          # Re-exports all endpoints
├── models.py            # Pydantic models
├── _client.py           # Base client class
├── client.py            # User-customizable client
├── users.py             # User endpoints
├── orders.py            # Order endpoints
└── health.py            # Health check endpoints
```

**Nested Structure** (`flat_structure: false` with nested module_map):

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

### Complete Example

```yaml
documents:
  - source: https://api.mycompany.com/openapi.json
    output: ./mycompany_client
    module_split:
      enabled: true
      strategy: custom

      # Strip API version prefixes
      global_strip_prefixes:
        - /api/v1
        - /api/v2
        - /api/v3

      # Consolidate small modules (< 3 endpoints) into fallback
      min_endpoints: 3
      fallback_module: misc

      # Custom module organization
      module_map:
        # Simple health checks
        health:
          - /health
          - /ready
          - /metrics

        # User management
        users:
          - /users
          - /users/*
          - /users/**

        # Authentication
        auth:
          - /auth/*
          - /login
          - /logout
          - /refresh

        # Nested billing module
        billing:
          paths:
            - /billing/**
          description: "Billing and payment endpoints"
          modules:
            invoices:
              - /invoices/*
            subscriptions:
              - /subscriptions/*
            payments:
              - /payments/*
```

---

## 📊 DataFrame Conversion

OtterAPI can generate additional methods that return pandas or polars DataFrames directly, making it easy to analyze API responses.

### Enabling DataFrame Methods

```yaml
documents:
  - source: https://api.example.com/openapi.json
    output: ./client
    dataframe:
      enabled: true
      pandas: true      # Generate _df methods (default: true when enabled)
      polars: true      # Generate _pl methods (default: false)
```

### Generated Methods

When enabled, endpoints that return lists get additional DataFrame methods:

| Original Method | Pandas Method | Polars Method |
|-----------------|---------------|---------------|
| `get_users()` | `get_users_df()` | `get_users_pl()` |
| `aget_users()` | `aget_users_df()` | `aget_users_pl()` |

### Basic Usage

```python
from client import find_pets_by_status, find_pets_by_status_df, find_pets_by_status_pl

# Get as Pydantic models (existing behavior)
pets = find_pets_by_status("available")
for pet in pets:
    print(f"{pet.id}: {pet.name}")

# Get as pandas DataFrame
pdf = find_pets_by_status_df("available")
print(pdf.head())
print(pdf.describe())

# Get as polars DataFrame
plf = find_pets_by_status_pl("available")
print(plf.schema)
print(plf.head())
```

### Handling Nested Responses

For APIs that return data nested under a key (e.g., `{"data": {"users": [...]}}`):

```yaml
dataframe:
  enabled: true
  pandas: true
  polars: true
  default_path: "data.items"      # Default path for all endpoints
  endpoints:
    get_users:
      path: "data.users"          # Override for specific endpoint
    get_analytics:
      path: "response.events"
```

You can also override the path at runtime:

```python
# Use configured path
df = get_users_df()

# Override path at call time
df = get_users_df(path="response.data.users")
```

### Selective Generation

Control which endpoints get DataFrame methods:

```yaml
dataframe:
  enabled: true
  pandas: true
  polars: true
  include_all: false              # Don't generate for all endpoints
  endpoints:
    list_users:
      enabled: true               # Only generate for this endpoint
    get_analytics:
      enabled: true
      path: "events"
      polars: true
      pandas: false               # Only polars for this endpoint
```

### DataFrame Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enabled` | bool | `false` | Enable DataFrame method generation |
| `pandas` | bool | `true` | Generate `_df` methods (pandas) |
| `polars` | bool | `false` | Generate `_pl` methods (polars) |
| `default_path` | string | `null` | Default JSON path for extracting data |
| `include_all` | bool | `true` | Generate for all list-returning endpoints |
| `endpoints` | object | `{}` | Per-endpoint configuration overrides |

### Per-Endpoint Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enabled` | bool | inherits | Override whether to generate methods |
| `path` | string | inherits | JSON path to extract data |
| `pandas` | bool | inherits | Override pandas generation |
| `polars` | bool | inherits | Override polars generation |

---

## 📖 Using Generated Code

### Direct Function Imports

```python
# Import specific endpoints
from client import get_user, create_user, list_orders

# Async versions are prefixed with 'a'
from client import aget_user, acreate_user, alist_orders

# Sync usage
user = get_user(user_id=123)
orders = list_orders(status="pending", limit=10)

# Async usage
import asyncio

async def main():
    user = await aget_user(user_id=123)
    orders = await alist_orders(status="pending")

asyncio.run(main())
```

### Module-Specific Imports (with splitting)

```python
# Import from specific modules
from client.users import get_user, create_user
from client.orders import list_orders, get_order
from client.auth import login, logout
```

### Using the Client Class

```python
from client import Client

# Create client with default settings
client = Client()

# Or customize the client
client = Client(
    base_url="https://api.example.com",
    timeout=30.0,
    headers={
        "Authorization": "Bearer your-token",
        "X-Custom-Header": "value"
    }
)

# Use client methods (sync)
user = client.get_user(user_id=123)
orders = client.list_orders(status="pending")

# Use async methods
import asyncio

async def main():
    user = await client.aget_user(user_id=123)

asyncio.run(main())
```

### Working with Models

```python
from client.models import User, Order, CreateUserRequest

# Models are Pydantic BaseModels
new_user = CreateUserRequest(
    name="John Doe",
    email="john@example.com"
)

# Create user
user = create_user(body=new_user)

# Access typed response
print(user.id)
print(user.name)
print(user.email)
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

## 🛠 Development

```bash
# Clone the repository
git clone https://github.com/yourusername/otterapi.git
cd otterapi

# Install dependencies with uv
uv sync

# Run tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=otterapi

# Run the generator
uv run python -m otterapi generate

# Format code
uv run ruff format .

# Lint code
uv run ruff check .
```

---

## 📄 License

MIT License - see LICENSE for details.
