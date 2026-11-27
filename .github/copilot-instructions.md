# Copilot instructions for otterapi

otterapi is a code generator: it reads an OpenAPI 2.0/3.0/3.1/3.2 (or Swagger) document and writes a typed Python client package (httpx + Pydantic) into a user's project. Keep in mind there are always **two codebases** in play: this generator, and the code it emits into a user's tree.

## Commands

Tooling is `uv` + `just` (see `.justfile`). Python ≥ 3.11.

```bash
just dev                 # uv pip install -e . --group dev + pre-commit hooks
just format              # ruff format otterapi
just lint                # ruff check otterapi --fix
just typecheck           # uv run mypy   (library only; tests are excluded)
just deadcode            # vulture; false positives go in vulture_whitelist.py
just chore               # format + lint + typecheck
just test                # uv run pytest

# Single file / single test
uv run pytest otterapi/tests/test_pagination.py
uv run pytest otterapi/tests/test_pagination.py::TestClass::test_name
uv run pytest -k discriminator

# What CI's matrix runs (slow real-world-spec smoke tests are excluded)
uv run pytest otterapi/tests/ -m "not slow"

# Separate CI gates
uv sync --group test --group typecheck
uv run pytest otterapi/tests/test_generated_typecheck.py -m typecheck   # pyright on generated clients
uv run pytest otterapi/tests/test_real_specs_smoke.py -m slow           # Stripe/GitHub specs

# Run the CLI against workspace source
uv run otter generate | init | validate <spec> | version
```

Pytest runs with `filterwarnings = error` (except User/Deprecation warnings) and `--strict-markers`; register new markers in `pyproject.toml`.

## Architecture

Pipeline driven by `Codegen.generate()` in `otterapi/codegen/codegen.py`:

1. **Load & normalize** – `SchemaLoader` (`codegen/schema.py`) loads URL/file, resolves external `$ref`s, validates against the Pydantic spec models in `otterapi/openapi/v{2,3,3_1,3_2}/`, then chains `.upgrade()` calls (v2 → v3 → v3.1 → v3.2). **Codegen only ever sees an OpenAPI 3.2 model.** Upgrade warnings are collected, not raised.
   - Spec models share `openapi/_lenient.py` (`LenientRefModel`): `x-*` extensions are retained, undeclared `$ref` forces the `Reference` branch of `X | Reference` unions, other unknown keys raise in strict mode or are dropped when validated with `context={'lenient': True}`.
   - New codegen code should read the document through `OpenAPIAdapter` (`codegen/_openapi_adapter.py`) rather than raw v3_2 attribute access.
2. **Types** – `TypeGenerator` (`codegen/types.py`) turns schemas into Pydantic model AST; `Endpoint`/`Parameter`/`ResponseInfo`/`RequestBodyInfo` describe operations.
3. **Emit** – All generated Python modules (models, endpoints, client, `_clients.py`, `__init__.py`) are built as **`ast` nodes** (helpers in `codegen/ast_utils.py`: `_name`, `_assign`, `_class_def`, `ImportCollector`, …) and written by `utils.write_mod`, which prunes unused imports, sorts them, retargets `httpx2` imports to the configured `http_library` (`retarget_http_module`), `ast.unparse`s, compiles/validates, then formats. Do not build generated code with string concatenation, and always import the HTTP library via `HTTP_MODULE` so the retargeting applies.
   - `codegen/emit.py` is the single source of truth for per-endpoint *function families* (sync/async request fn, `paginate_*`/`_iter`, `_df`/`_pl`, `_export`) via the `EndpointFeature` registry; both the single-file writer and `splitting.SplitModuleEmitter` (module splitting by tag/path/custom/hybrid) consume it. Don't re-add per-layout logic.
   - `client.py` builds the base client / error hierarchy; `client_layout.py` builds `Client`/`AsyncClient` for `client_style: client|resource`; `endpoints.py` builds request functions; `auth.py`, `pagination.py`, `dataframes.py`, `export.py` add feature-specific AST.
4. **Runtime helpers** – Files in `otterapi/codegen/runtime/` (`_pagination.py`, `_dataframe.py`, `_export.py`, `_retry.py`, …) are **copied verbatim** into the generated package via the `FeatureModule` registry in `codegen/_features.py` (`all_features()` / `write_enabled_features`). They must run on Python 3.11 (legacy `TypeVar` generics); `_pep695.py` textually rewrites them to PEP 695 syntax when `target_python >= 3.12`. `_client_stub.py.tpl` is rendered by placeholder substitution (`__CLASS_NAME__` etc.).
5. **Atomicity** – `generate()` snapshots the output directory and restores it on any exception. The user-facing `client.py` stub is written only if it doesn't exist and is never overwritten.

Configuration lives in `otterapi/config.py` (`CodegenConfig` → list of `DocumentConfig`), discovered from `otter.yml`/`otter.yaml`/`otter.json` or `[tool.otterapi]` in `pyproject.toml`; supports env-var substitution. New options belong there, documented in `README.md`.

## Conventions

- **Generated output must be lint-clean under the user's ruff config**, not just ours. `test_generated_hygiene.py` checks runtime helpers, AST-built modules, and the stub template (e.g. SIM101, I001, PIE790, UP034). That's why `SIM101` is enabled in this repo's ruff config. `test_unused_imports.py` and `test_generated_typecheck.py` (pyright) enforce the same for imports/types.
- **Optional deps are lazy**: generated code and the codegen package must import without pandas/polars/pyarrow installed (CI `test-no-extras` job). Import them inside functions.
- Always pass `encoding='utf-8'` when reading/writing files (Windows CI has caught this repeatedly).
- `otterapi/tests/fixtures/golden/<case>/` hold **input** specs (`spec.yaml`/`spec.json`, optional `config.yaml` with `DocumentConfig` overrides), not expected output. Tests generate into `tmp_path` with `Codegen(DocumentConfig.model_validate({...})).generate()` and assert on the parsed AST (see `_generate` in `test_codegen_structure.py`) — don't add byte-exact output comparisons. The directory is excluded from ruff.
- `otterapi.codegen.__all__` is a stable public API guarded by `test_public_api.py`; adding is fine, removing/renaming is breaking.
- Style (ruff-enforced): single quotes inline, double quotes for multiline/docstrings; Google-style docstrings; isort with `combine-as-imports`; McCabe complexity ≤ 15 — extract a helper instead of suppressing.
- mypy is checked against the running interpreter (no pinned `python_version`) because `ast` node fields differ by Python version; keep `ast` construction compatible with 3.11–3.14.
- Commits: Conventional Commits (`feat:`, `fix:`, `test:`, `refactor:` …), must be signed; work on a topic branch, PR to `main`.
