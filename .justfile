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
    uv run pytest -n auto otter/tests -vv

[group('publish')]
publish:
    rm -rf dist
    uv build
    uv run twine upload dist/*  --verbose
