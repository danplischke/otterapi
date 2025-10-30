"""Test CLI functionality."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from otterapi.cli import app
from otterapi.config import CodegenConfig, DocumentConfig


@pytest.fixture
def runner():
    """Fixture providing CLI test runner."""
    return CliRunner()


@pytest.fixture
def sample_config():
    """Fixture providing sample configuration."""
    return CodegenConfig(
        documents=[
            DocumentConfig(
                source='https://api.example.com/openapi.json', output='./generated'
            )
        ],
        generate_endpoints=True,
    )


class TestGenerateCommand:
    """Test the generate command."""

    @patch('otterapi.cli.get_config')
    @patch('otterapi.cli.Codegen')
    def test_generate_without_config_file(
        self, mock_codegen_class, mock_get_config, runner, sample_config
    ):
        """Test generate command without specifying config file."""
        mock_get_config.return_value = sample_config
        mock_codegen_instance = MagicMock()
        mock_codegen_class.return_value = mock_codegen_instance

        result = runner.invoke(app, ['generate'])

        assert result.exit_code == 0
        mock_get_config.assert_called_once_with(None)
        mock_codegen_class.assert_called_once_with(sample_config.documents[0])
        mock_codegen_instance.generate.assert_called_once()
        assert 'Successfully generated code' in result.stdout

    @patch('otterapi.cli.get_config')
    @patch('otterapi.cli.Codegen')
    def test_generate_with_config_file(
        self, mock_codegen_class, mock_get_config, runner, sample_config
    ):
        """Test generate command with config file specified."""
        mock_get_config.return_value = sample_config
        mock_codegen_instance = MagicMock()
        mock_codegen_class.return_value = mock_codegen_instance

        result = runner.invoke(app, ['generate', '--config', 'custom-config.yaml'])

        assert result.exit_code == 0
        mock_get_config.assert_called_once_with('custom-config.yaml')
        mock_codegen_instance.generate.assert_called_once()

    @patch('otterapi.cli.get_config')
    @patch('otterapi.cli.Codegen')
    def test_generate_with_short_config_option(
        self, mock_codegen_class, mock_get_config, runner, sample_config
    ):
        """Test generate command with short config option."""
        mock_get_config.return_value = sample_config
        mock_codegen_instance = MagicMock()
        mock_codegen_class.return_value = mock_codegen_instance

        result = runner.invoke(app, ['generate', '-c', 'config.json'])

        assert result.exit_code == 0
        mock_get_config.assert_called_once_with('config.json')

    @patch('otterapi.cli.get_config')
    @patch('otterapi.cli.Codegen')
    def test_generate_multiple_documents(
        self, mock_codegen_class, mock_get_config, runner
    ):
        """Test generate command with multiple documents."""
        multi_doc_config = CodegenConfig(
            documents=[
                DocumentConfig(source='api1.json', output='./gen1'),
                DocumentConfig(source='api2.json', output='./gen2'),
            ]
        )
        mock_get_config.return_value = multi_doc_config
        mock_codegen_instance = MagicMock()
        mock_codegen_class.return_value = mock_codegen_instance

        result = runner.invoke(app, ['generate'])

        assert result.exit_code == 0
        assert mock_codegen_class.call_count == 2
        assert mock_codegen_instance.generate.call_count == 2
        assert result.stdout.count('Successfully generated code') == 2

    @patch('otterapi.cli.get_config')
    def test_generate_config_error(self, mock_get_config, runner):
        """Test generate command when config loading fails."""
        mock_get_config.side_effect = FileNotFoundError('config not found')

        result = runner.invoke(app, ['generate'])

        assert result.exit_code == 1
        assert 'Error:' in result.stdout
        assert 'config not found' in result.stdout

    @patch('otterapi.cli.get_config')
    @patch('otterapi.cli.Codegen')
    def test_generate_codegen_error(
        self, mock_codegen_class, mock_get_config, runner, sample_config
    ):
        """Test generate command when code generation fails."""
        mock_get_config.return_value = sample_config
        mock_codegen_instance = MagicMock()
        mock_codegen_instance.generate.side_effect = Exception('Generation failed')
        mock_codegen_class.return_value = mock_codegen_instance

        result = runner.invoke(app, ['generate'])

        assert result.exit_code == 1
        assert 'Error:' in result.stdout
        assert 'Generation failed' in result.stdout


class TestVersionCommand:
    """Test the version command."""

    def test_version_with_importable_version(self, runner):
        """Test version command when version can be imported."""
        with patch('otterapi.cli.version', '1.0.0', create=True):
            result = runner.invoke(app, ['version'])

            assert result.exit_code == 0
            assert 'otterapi version: 1.0.0' in result.stdout

    def test_version_without_importable_version(self, runner):
        """Test version command when version cannot be imported."""
        with patch('otterapi.cli.version', side_effect=ImportError()):
            result = runner.invoke(app, ['version'])

            assert result.exit_code == 0
            assert 'otterapi version: unknown' in result.stdout


class TestCLIIntegration:
    """Integration tests for CLI."""

    def test_help_message(self, runner):
        """Test CLI help message."""
        result = runner.invoke(app, ['--help'])

        assert result.exit_code == 0
        assert (
            'Generate Python client code from OpenAPI specifications' in result.stdout
        )
        assert 'generate' in result.stdout
        assert 'version' in result.stdout

    def test_generate_help_message(self, runner):
        """Test generate command help message."""
        result = runner.invoke(app, ['generate', '--help'])

        assert result.exit_code == 0
        assert 'Generate Python client code from configuration' in result.stdout
        assert '--config' in result.stdout
        assert 'Path to configuration file' in result.stdout

    def test_invalid_command(self, runner):
        """Test CLI with invalid command."""
        result = runner.invoke(app, ['invalid-command'])

        assert result.exit_code == 2  # Typer's exit code for invalid command
        assert 'No such command' in result.stdout

    @patch('otterapi.cli.get_config')
    @patch('otterapi.cli.Codegen')
    def test_progress_display(
        self, mock_codegen_class, mock_get_config, runner, sample_config
    ):
        """Test that progress is displayed during generation."""
        mock_get_config.return_value = sample_config
        mock_codegen_instance = MagicMock()
        mock_codegen_class.return_value = mock_codegen_instance

        result = runner.invoke(app, ['generate'])

        assert result.exit_code == 0
        # The exact progress display content may vary, but we can check for key elements
        assert 'generated' in result.stdout.lower()


class TestCLIWithRealFiles:
    """Test CLI with real configuration files."""

    def test_generate_with_yaml_config_file(self, runner):
        """Test generate command with a real YAML config file."""
        yaml_content = """
documents:
  - source: "https://petstore3.swagger.io/api/v3/openapi.json"
    output: "./test_output"
generate_endpoints: true
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            f.flush()

            try:
                with patch('otterapi.cli.Codegen') as mock_codegen_class:
                    mock_codegen_instance = MagicMock()
                    mock_codegen_class.return_value = mock_codegen_instance

                    result = runner.invoke(app, ['generate', '--config', f.name])

                    assert result.exit_code == 0
                    mock_codegen_instance.generate.assert_called_once()
            finally:
                Path(f.name).unlink()

    def test_generate_with_nonexistent_config_file(self, runner):
        """Test generate command with nonexistent config file."""
        result = runner.invoke(
            app, ['generate', '--config', '/nonexistent/config.yaml']
        )

        assert result.exit_code == 1
        assert 'Error:' in result.stdout


def test_cli_main_execution():
    """Test that CLI can be executed directly."""
    # This test ensures the CLI is properly configured for direct execution
    assert app is not None
    assert hasattr(app, 'commands')
    assert 'generate' in [cmd.name for cmd in app.commands.values()]
    assert 'version' in [cmd.name for cmd in app.commands.values()]
