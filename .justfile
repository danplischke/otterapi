set windows-shell := ["powershell.exe", "-NoLogo", "-Command"]

set dotenv-load
set shell := ['sh', '-cu']

pkg_src := "otterapi"

@_:
    just --list

[group('development'), doc('Install package to venv in dev mode')]
dev:
  uv tool install pre-commit
  pre-commit install
  uv pip install -e . --group dev

[group('development')]
format:
	uv tool run ruff format {{pkg_src}}

[group('development')]
lint:
	uv tool run ruff check {{pkg_src}} --fix --exit-zero

[group('development')]
chore: format lint

[group('package')]
build:
    uv build

[group('test')]
test:
    uv run pytest

[group('publish')]
publish:
    rm -rf dist
    uv build
    uv run twine upload dist/*  --verbose

[group('package')]
version:
    @git describe --tags --abbrev=0 2>/dev/null || echo "No tags found"

[group('package')]
versions:
    @git tag -l "v*" --sort=-v:refname | head -10

[group('package')]
release ver:
    git tag -a v{{ ver }} -m "Release v{{ ver }}"
    @echo "Created tag v{{ ver }}"
    git push origin "v{{ ver }}"

