"""Tests that a failed generation leaves the output directory untouched.

Generation writes many files and can fail partway through -- a spec whose
servers cannot be narrowed to one base URL, a schema the type generator
rejects, a validation error on an emitted module. A half-written package is
worse than no package: it may still import, just wrong.

The rollback is a snapshot-and-restore rather than a generate-then-swap
because ``client.py`` is deliberately preserved across runs, as is anything
else the user keeps in the output directory.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest
import yaml

from otterapi.codegen.codegen import Codegen
from otterapi.config import DocumentConfig

# A file-loaded spec whose only server URL is relative. Generation succeeds:
# the relative URL is kept as the client default and the caller supplies an
# absolute one, either at generation time (--base-url) or per client.
RELATIVE_SERVER_SPEC = {
    'openapi': '3.1.0',
    'info': {'title': 'Rel', 'version': '1.0'},
    'servers': [{'url': '/api/v3'}],
    'paths': {
        '/thing': {
            'get': {
                'operationId': 'getThing',
                'responses': {
                    '200': {
                        'description': 'ok',
                        'content': {
                            'application/json': {
                                'schema': {'$ref': '#/components/schemas/Thing'}
                            }
                        },
                    }
                },
            }
        }
    },
    'components': {
        'schemas': {
            'Thing': {'type': 'object', 'properties': {'name': {'type': 'string'}}}
        }
    },
}

VALID_SPEC = {
    **RELATIVE_SERVER_SPEC,
    'servers': [{'url': 'https://example.test'}],
}

# Generation gets far enough to write models.py, then fails resolving the base
# URL: two servers give no single base to bake into the client.
MULTI_SERVER_SPEC = {
    **RELATIVE_SERVER_SPEC,
    'servers': [
        {'url': 'https://one.example.test'},
        {'url': 'https://two.example.test'},
    ],
}


def _write_spec(tmp_path: Path, spec: dict) -> Path:
    spec_file = tmp_path / 'openapi.json'
    spec_file.write_text(json.dumps(spec), encoding='utf-8')
    return spec_file


def _generate(spec_file: Path, output: Path, **kwargs) -> None:
    Codegen(
        DocumentConfig(source=str(spec_file), output=str(output), **kwargs)
    ).generate()


def _base_url_default(output: Path) -> str:
    """Read the base_url default baked into the generated client's __init__.

    Parsed rather than string-matched: ``ast.unparse`` writes an annotated
    default as ``base_url: str='...'`` and only a formatter adds the spaces,
    so asserting on the formatted spelling makes the test depend on whether
    ruff happens to be on PATH -- which it is not on the Windows runners.
    """
    tree = ast.parse((output / '_client.py').read_text(encoding='utf-8'))
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name != '__init__':
            continue
        args = node.args
        # Defaults align to the tail of the positional arguments.
        offset = len(args.args) - len(args.defaults)
        for index, arg in enumerate(args.args):
            if arg.arg == 'base_url' and index >= offset:
                default = args.defaults[index - offset]
                assert isinstance(default, ast.Constant)
                return str(default.value)
    raise AssertionError('generated client has no base_url default')


class TestFailedGenerationRollsBack:
    def test_new_directory_is_removed(self, tmp_path: Path):
        """The failure used to leave a lone models.py behind."""
        spec_file = _write_spec(tmp_path, MULTI_SERVER_SPEC)
        output = tmp_path / 'client'

        with pytest.raises(Exception):
            _generate(spec_file, output)

        assert not output.exists()

    def test_existing_files_survive(self, tmp_path: Path):
        spec_file = _write_spec(tmp_path, MULTI_SERVER_SPEC)
        output = tmp_path / 'client'
        output.mkdir()
        (output / 'client.py').write_text('# hand-written', encoding='utf-8')
        (output / 'notes.txt').write_text('keep me', encoding='utf-8')

        with pytest.raises(Exception):
            _generate(spec_file, output)

        assert sorted(p.name for p in output.iterdir()) == ['client.py', 'notes.txt']
        assert (output / 'client.py').read_text(encoding='utf-8') == '# hand-written'

    def test_a_previous_successful_client_is_not_corrupted(self, tmp_path: Path):
        """A failed regeneration must leave the last good client in place."""
        output = tmp_path / 'client'
        _generate(_write_spec(tmp_path, VALID_SPEC), output)
        before = {p.name: p.read_bytes() for p in output.rglob('*') if p.is_file()}

        bad_spec = tmp_path / 'bad.json'
        bad_spec.write_text(json.dumps(MULTI_SERVER_SPEC), encoding='utf-8')
        with pytest.raises(Exception):
            _generate(bad_spec, output)

        after = {p.name: p.read_bytes() for p in output.rglob('*') if p.is_file()}
        assert after == before


class TestSuccessfulGenerationIsUnaffected:
    def test_files_are_written(self, tmp_path: Path):
        output = tmp_path / 'client'
        _generate(_write_spec(tmp_path, VALID_SPEC), output)

        for name in ('models.py', 'endpoints.py', '_client.py', 'client.py'):
            assert (output / name).exists()

    def test_user_edits_to_client_py_survive_regeneration(self, tmp_path: Path):
        """client.py is generated once and never overwritten."""
        spec_file = _write_spec(tmp_path, VALID_SPEC)
        output = tmp_path / 'client'
        _generate(spec_file, output)

        marker = '\n# user customization\n'
        client_py = output / 'client.py'
        client_py.write_text(
            client_py.read_text(encoding='utf-8') + marker, encoding='utf-8'
        )

        _generate(spec_file, output)
        assert client_py.read_text(encoding='utf-8').endswith(marker)

    def test_unrelated_files_in_the_output_dir_survive(self, tmp_path: Path):
        output = tmp_path / 'client'
        output.mkdir()
        (output / 'notes.txt').write_text('keep me', encoding='utf-8')

        _generate(_write_spec(tmp_path, VALID_SPEC), output)

        assert (output / 'notes.txt').read_text(encoding='utf-8') == 'keep me'


class TestBaseUrlOption:
    """``--base-url`` bakes an absolute base into a relative-server client."""

    def test_supplying_a_base_url_makes_generation_succeed(self, tmp_path: Path):
        spec_file = _write_spec(tmp_path, RELATIVE_SERVER_SPEC)
        output = tmp_path / 'client'

        _generate(spec_file, output, base_url='https://example.test/api/v3')

        assert _base_url_default(output) == 'https://example.test/api/v3'

    def test_cli_flag_overrides_every_document_in_a_config_file(self, tmp_path: Path):
        from otterapi.cli import _resolve_codegen_config

        spec_file = _write_spec(tmp_path, RELATIVE_SERVER_SPEC)
        config_file = tmp_path / 'otter.yml'
        config_file.write_text(
            yaml.safe_dump(
                {
                    'documents': [
                        {'source': str(spec_file), 'output': str(tmp_path / 'a')},
                        {'source': str(spec_file), 'output': str(tmp_path / 'b')},
                    ]
                }
            ),
            encoding='utf-8',
        )

        resolved = _resolve_codegen_config(
            str(config_file), None, None, 'https://override.test'
        )
        assert [d.base_url for d in resolved.documents] == [
            'https://override.test',
            'https://override.test',
        ]

    def test_cli_flag_applies_to_the_source_output_form(self, tmp_path: Path):
        from otterapi.cli import _resolve_codegen_config

        resolved = _resolve_codegen_config(
            None, 'spec.json', './out', 'https://example.test'
        )
        assert resolved.documents[0].base_url == 'https://example.test'


class TestRelativeServerUrl:
    """A relative server (``/``) is valid OpenAPI, so it must not hard-fail.

    Springdoc's ``/v3/api-docs`` emits exactly ``{"url": "/"}``; downloading
    that document to a file and generating from it has to work.
    """

    def test_generation_succeeds_without_a_base_url(self, tmp_path: Path):
        spec_file = _write_spec(tmp_path, RELATIVE_SERVER_SPEC)
        output = tmp_path / 'client'

        _generate(spec_file, output)

        for name in ('models.py', 'endpoints.py', '_client.py', 'client.py'):
            assert (output / name).exists()

    def test_the_relative_url_becomes_the_client_default(self, tmp_path: Path):
        """Kept as-is: the caller passes ``base_url=`` to make it absolute."""
        spec_file = _write_spec(tmp_path, RELATIVE_SERVER_SPEC)
        output = tmp_path / 'client'

        _generate(spec_file, output)

        assert _base_url_default(output) == '/api/v3'

    def test_a_bare_slash_server_generates_a_base_less_client(self, tmp_path: Path):
        spec = {**RELATIVE_SERVER_SPEC, 'servers': [{'url': '/'}]}
        output = tmp_path / 'client'

        _generate(_write_spec(tmp_path, spec), output)

        # The generated __init__ does base_url.rstrip('/'), so '/' is no base.
        assert _base_url_default(output) == '/'

    def test_the_warning_names_the_cli_flag(self, tmp_path: Path, caplog):
        """The old error pointed only at "the otterapi config"."""
        spec_file = _write_spec(tmp_path, RELATIVE_SERVER_SPEC)

        with caplog.at_level('WARNING', logger='otterapi.codegen.codegen'):
            _generate(spec_file, tmp_path / 'client')

        assert any(
            '-b/--base-url' in record.message and 'relative' in record.message
            for record in caplog.records
        )

    def test_no_warning_when_a_base_url_is_supplied(self, tmp_path: Path, caplog):
        spec_file = _write_spec(tmp_path, RELATIVE_SERVER_SPEC)

        with caplog.at_level('WARNING', logger='otterapi.codegen.codegen'):
            _generate(spec_file, tmp_path / 'client', base_url='https://example.test')

        assert not any('relative' in record.message for record in caplog.records)

    def test_a_url_source_still_resolves_the_relative_server(self, tmp_path: Path):
        """Loaded over HTTP, ``/api/v3`` resolves against the source origin."""
        codegen = Codegen(
            DocumentConfig(
                source=str(_write_spec(tmp_path, RELATIVE_SERVER_SPEC)),
                output=str(tmp_path / 'client'),
            )
        )
        codegen._load_schema()
        codegen.config.source = 'https://api.example.test/v3/api-docs'

        assert codegen._resolve_base_url() == 'https://api.example.test/api/v3'


class TestUnresolvableServers:
    """Failures that remain, and what their messages point the user at."""

    def test_multiple_servers_names_the_cli_flag(self, tmp_path: Path):
        spec_file = _write_spec(tmp_path, MULTI_SERVER_SPEC)

        with pytest.raises(Exception, match=r'-b/--base-url'):
            _generate(spec_file, tmp_path / 'client')

    def test_no_servers_names_the_cli_flag(self, tmp_path: Path):
        spec = {k: v for k, v in RELATIVE_SERVER_SPEC.items() if k != 'servers'}
        spec_file = _write_spec(tmp_path, spec)

        with pytest.raises(Exception, match=r'-b/--base-url'):
            _generate(spec_file, tmp_path / 'client')
