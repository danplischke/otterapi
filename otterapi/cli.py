"""Command-line interface for OtterAPI.

This module provides the CLI commands for generating Python client code
from OpenAPI specifications.
"""

import json
import logging
import traceback
from pathlib import Path
from typing import Annotated

import typer
import yaml
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax
from rich.table import Table

from otterapi.codegen.codegen import Codegen
from otterapi.codegen.schema import SchemaLoader
from otterapi.config import CodegenConfig, DocumentConfig, get_config
from otterapi.exceptions import OtterAPIError, SchemaLoadError, SchemaValidationError
from otterapi.openapi.constants import HTTP_METHODS

console = Console()
error_console = Console(stderr=True)


def _commented_optional_sections() -> str:
    """Return a YAML block of commented-out optional config sections.

    Appended to ``otter.yml`` by ``otterapi init`` so users discover the
    pagination / dataframe / export / module_split / response_unwrap
    knobs by editing rather than by hunting through the README.
    """
    return """
# ----------------------------------------------------------------------
# Optional sections -- uncomment to enable. See the README for details:
#   https://github.com/danplischke/otterapi
# ----------------------------------------------------------------------

# documents:
#   - source: ...
#     output: ...
#
#     # Pagination: emit *_iter() helpers that stream paginated endpoints.
#     # auto_detect=true picks up offset/limit, cursor/limit, page/per_page
#     # parameters automatically.
#     pagination:
#       enabled: true
#       auto_detect: true
#       default_page_size: 100
#
#     # File export: emit *_export() helpers that stream rows to CSV /
#     # TSV / JSONL / Parquet. Parquet support requires `pip install
#     # otterapi[parquet]`.
#     export:
#       enabled: true
#       formats: [csv, jsonl]   # csv | tsv | jsonl | parquet
#       batch_size: 1000
#
#     # DataFrame conversion: emit *_df() (pandas) / *_pl() (polars)
#     # helpers for list-returning endpoints.
#     dataframe:
#       enabled: true
#       pandas: true
#       polars: false
#
#     # Module splitting: shard a giant API into per-tag / per-path
#     # subpackages. Strategies: none | path | tag | hybrid | custom.
#     module_split:
#       enabled: false
#       strategy: hybrid
#
#     # Response unwrapping: when responses use an envelope wrapper
#     # ({"data": [...], "meta": {...}}), generated endpoints can return
#     # just the inner data.
#     response_unwrap:
#       enabled: false
#       data_path: data
"""


def _version_callback(value: bool) -> None:
    """Print version and exit if --version flag is passed."""
    if value:
        try:
            from otterapi._version import version as ver

            console.print(f'otterapi version: [bold]{ver}[/bold]')
        except ImportError:
            console.print('otterapi version: [dim]unknown (development)[/dim]')
        raise typer.Exit()


app = typer.Typer(
    name='otterapi',
    help='Generate Python client code from OpenAPI specifications',
    no_args_is_help=True,
)


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            '--version',
            '-V',
            help='Show version and exit',
            callback=_version_callback,
            is_eager=True,
        ),
    ] = False,
) -> None:
    """Generate Python client code from OpenAPI specifications."""
    pass


def setup_logging(verbose: bool = False, debug: bool = False) -> None:
    """Configure logging based on verbosity settings."""
    if debug:
        level = logging.DEBUG
    elif verbose:
        level = logging.INFO
    else:
        level = logging.WARNING

    logging.basicConfig(
        level=level,
        format='%(message)s',
        datefmt='[%X]',
        handlers=[RichHandler(console=error_console, rich_tracebacks=True)],
    )


def _resolve_codegen_config(
    config: str | None, source: str | None, output: str | None
) -> CodegenConfig:
    """Build a CodegenConfig from --source/--output or a config file."""
    if source and output:
        return CodegenConfig(documents=[DocumentConfig(source=source, output=output)])

    if source or output:
        error_console.print(
            '[red]Error:[/red] Both --source and --output must be provided together'
        )
        raise typer.Exit(1)

    try:
        return get_config(config)
    except FileNotFoundError as e:
        error_console.print(f'[red]Error:[/red] {e}')
        error_console.print(
            '\n[dim]Hint: Run [bold]otterapi init[/bold] to create a configuration file,[/dim]'
        )
        error_console.print(
            '[dim]or use [bold]--source[/bold] and [bold]--output[/bold] options.[/dim]'
        )
        raise typer.Exit(1)


def _generate_document(document_config: DocumentConfig) -> None:
    """Run code generation for a single document, with a progress spinner."""
    with Progress(
        SpinnerColumn(),
        TextColumn('[progress.description]{task.description}'),
        console=console,
    ) as progress:
        task = progress.add_task(
            f'Generating code for {document_config.source}...',
            total=None,
        )

        codegen = Codegen(document_config)
        generated_files = codegen.generate()
        progress.update(
            task,
            description=f'Code generation completed for {document_config.source}!',
        )

        console.print('[dim]Generated files:[/dim]')
        for file_path in generated_files:
            console.print(f'  - {file_path}')


@app.command()
def generate(
    config: Annotated[
        str | None,
        typer.Option(
            '--config', '-c', help='Path to configuration file (YAML or JSON)'
        ),
    ] = None,
    source: Annotated[
        str | None,
        typer.Option(
            '--source', '-s', help='Direct path or URL to OpenAPI specification'
        ),
    ] = None,
    output: Annotated[
        str | None,
        typer.Option('--output', '-o', help='Output directory for generated code'),
    ] = None,
    verbose: Annotated[
        bool, typer.Option('--verbose', '-v', help='Enable verbose output')
    ] = False,
    debug: Annotated[bool, typer.Option('--debug', help='Enable debug output')] = False,
) -> None:
    """Generate Python client code from OpenAPI specifications.

    You can either use a configuration file or specify the source and output
    directly via command-line options.

    Examples:
        otterapi generate
        otterapi generate --config my-config.yaml
        otterapi generate -c config.json
        otterapi generate --source https://api.example.com/openapi.json --output ./client
        otterapi generate -s ./api.yaml -o ./generated
    """
    setup_logging(verbose, debug)

    try:
        codegen_config = _resolve_codegen_config(config, source, output)

        for document_config in codegen_config.documents:
            _generate_document(document_config)

        console.print('\n[green]✓[/green] Code generation completed!')

    except OtterAPIError as e:
        error_console.print(f'[red]Error:[/red] {e.message}')
        if debug:
            traceback.print_exc()
        raise typer.Exit(1)
    except Exception as e:
        error_console.print(f'[red]Error:[/red] {str(e)}')
        if debug:
            traceback.print_exc()
        raise typer.Exit(1)


@app.command()
def init(
    path: Annotated[
        str, typer.Argument(help='Path for the configuration file')
    ] = 'otter.yml',
    force: Annotated[
        bool, typer.Option('--force', '-f', help='Overwrite existing file')
    ] = False,
) -> None:
    """Create a new configuration file interactively.

    This command guides you through creating an OtterAPI configuration file
    with all the necessary settings.

    Examples:
        otterapi init
        otterapi init otter.yaml
        otterapi init config.json --force
    """
    config_path = Path(path)

    # Check if file exists
    if config_path.exists() and not force:
        error_console.print(
            f'[red]Error:[/red] File {config_path} already exists. Use --force to overwrite.'
        )
        raise typer.Exit(1)

    console.print(Panel('[bold]OtterAPI Configuration Setup[/bold]'))

    # Get source
    source = typer.prompt(
        '\nOpenAPI specification source (URL or file path)',
        default='https://petstore3.swagger.io/api/v3/openapi.json',
    )

    # Get output directory
    output = typer.prompt('Output directory for generated code', default='./client')

    # Get models file name
    models_file = typer.prompt('Models file name', default='models.py')

    # Get endpoints file name
    endpoints_file = typer.prompt('Endpoints file name', default='endpoints.py')

    # Build config
    config_data = {
        'documents': [
            {
                'source': source,
                'output': output,
                'models_file': models_file,
                'endpoints_file': endpoints_file,
            }
        ]
    }

    # Ask if they want to add more documents
    while typer.confirm('\nAdd another document?', default=False):
        source = typer.prompt('OpenAPI specification source')
        output = typer.prompt('Output directory')
        config_data['documents'].append({'source': source, 'output': output})

    # Write config file
    config_path.parent.mkdir(parents=True, exist_ok=True)

    if path.endswith('.json'):
        content = json.dumps(config_data, indent=2)
    else:
        content = yaml.dump(config_data, default_flow_style=False, sort_keys=False)
        # Wave 3.17 (issue #3 item 17): append commented-out scaffolds for the
        # optional sections so users can discover them by editing instead of
        # cross-referencing the README.
        content += _commented_optional_sections()

    config_path.write_text(content)

    console.print(f'\n[green]✓[/green] Configuration saved to {config_path}')
    console.print('\n[dim]Preview:[/dim]')
    syntax = Syntax(content, 'yaml' if not path.endswith('.json') else 'json')
    console.print(syntax)

    console.print(
        f'\n[dim]Run [bold]otterapi generate -c {path}[/bold] to generate code.[/dim]'
    )


_WARNABLE_METHODS = ('get', 'post', 'put', 'patch', 'delete')


def _count_operations(paths) -> int:
    """Count HTTP operations defined across all paths in a schema."""
    operations = 0
    for path_item in paths.root.values():
        for method in HTTP_METHODS:
            if getattr(path_item, method, None):
                operations += 1
    return operations


def _summarize_schema(schema) -> dict[str, any]:
    """Collect descriptive info (title, counts, ...) about a loaded schema."""
    info: dict[str, any] = {}

    if schema.info:
        info['title'] = schema.info.title
        info['version'] = schema.info.version
        if schema.info.description:
            info['description'] = (
                schema.info.description[:200] + '...'
                if len(schema.info.description or '') > 200
                else schema.info.description
            )

    if schema.paths:
        info['paths'] = len(schema.paths.root)
        info['operations'] = _count_operations(schema.paths)

    if schema.components:
        if schema.components.schemas:
            info['schemas'] = len(schema.components.schemas)
        if schema.components.securitySchemes:
            info['security_schemes'] = len(schema.components.securitySchemes)

    return info


def _find_missing_operation_ids(paths) -> list[str]:
    """Return warnings for operations that are missing an operationId."""
    warnings: list[str] = []
    for path, path_item in paths.root.items():
        for method in _WARNABLE_METHODS:
            operation = getattr(path_item, method, None)
            if operation and not operation.operationId:
                warnings.append(f'{method.upper()} {path}: Missing operationId')
    return warnings


def _load_schema_for_validation(source: str, progress, task):
    """Load and validate a schema, exiting the CLI with a message on failure."""
    try:
        loader = SchemaLoader()
        schema = loader.load(source)
        progress.update(task, description='Validating schema...')
        return schema
    except SchemaLoadError as e:
        progress.stop()
        error_console.print(f'[red]✗ Failed to load schema:[/red] {e.message}')
        raise typer.Exit(1)
    except SchemaValidationError as e:
        progress.stop()
        error_console.print(f'[red]✗ Schema validation failed:[/red] {e.message}')
        raise typer.Exit(1)
    except Exception as e:
        progress.stop()
        error_console.print(f'[red]✗ Error:[/red] {str(e)}')
        raise typer.Exit(1)


def _print_validation_report(
    source: str, info: dict[str, any], warnings: list[str], verbose: bool
) -> None:
    """Print the schema summary table and any operationId warnings."""
    console.print(f'\n[green]✓[/green] Schema is valid: {source}\n')

    if verbose or info:
        table = Table(title='Schema Information')
        table.add_column('Property', style='cyan')
        table.add_column('Value')

        for key, value in info.items():
            table.add_row(key.replace('_', ' ').title(), str(value))

        console.print(table)

    if warnings:
        console.print(f'\n[yellow]⚠ {len(warnings)} warning(s):[/yellow]')
        for warning in warnings[:10]:  # Show first 10
            console.print(f'  - {warning}')
        if len(warnings) > 10:
            console.print(f'  ... and {len(warnings) - 10} more')


@app.command()
def validate(
    source: Annotated[str, typer.Argument(help='Path or URL to OpenAPI specification')],
    verbose: Annotated[
        bool, typer.Option('--verbose', '-v', help='Show detailed schema information')
    ] = False,
) -> None:
    """Validate an OpenAPI specification.

    This command loads and validates an OpenAPI specification without
    generating any code, reporting any errors or warnings found.

    Examples:
        otterapi validate ./api.yaml
        otterapi validate https://api.example.com/openapi.json
        otterapi validate ./api.yaml --verbose
    """
    with Progress(
        SpinnerColumn(),
        TextColumn('[progress.description]{task.description}'),
        console=console,
    ) as progress:
        task = progress.add_task(f'Loading {source}...', total=None)

        schema = _load_schema_for_validation(source, progress, task)

        info = _summarize_schema(schema)
        warnings = _find_missing_operation_ids(schema.paths) if schema.paths else []

        progress.update(task, description='Validation complete!')

    _print_validation_report(source, info, warnings, verbose)


@app.command()
def version() -> None:
    """Show the version of OtterAPI."""
    try:
        from otterapi._version import version as ver

        console.print(f'otterapi version: [bold]{ver}[/bold]')
    except ImportError:
        console.print('otterapi version: [dim]unknown (development)[/dim]')


if __name__ == '__main__':
    app()
