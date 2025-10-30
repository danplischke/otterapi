# 🦦 OtterAPI

> *A cute and intelligent OpenAPI client generator that dives deep into your OpenAPIs*

**OtterAPI** is a sleek Python library that transforms OpenAPI specifications into clean, type-safe client code.

## 🚀 Quick Start

```bash
# Generate from a pyproject.toml or any of the default config names (otter.yml, otter.yaml)
otter generate

# Generate from an otterapi config file
otter generate -c otter.yml
```

## 📝 Example Config

```yaml
documents:
  - source: https://petstore3.swagger.io/api/v3/openapi.json
    output: petstore_client

  - source: ./local-users-api.json
    output: users_client
```