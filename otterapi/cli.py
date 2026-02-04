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

console = Console()
error_console = Console(stderr=True)

app = typer.Typer(
    name='otterapi',
    help='Generate Python client code from OpenAPI specifications',
    no_args_is_help=True,
)


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
    dry_run: Annotated[
        bool,
        typer.Option(
            '--dry-run', '-n', help='Preview generation without writing files'
        ),
    ] = False,
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
        otterapi generate -s ./api.yaml -o ./generated --dry-run
    """
    setup_logging(verbose, debug)

    try:
        # Build configuration from options or file
        if source and output:
            codegen_config = CodegenConfig(
                documents=[DocumentConfig(source=source, output=output)]
            )
        elif source or output:
            error_console.print(
                '[red]Error:[/red] Both --source and --output must be provided together'
            )
            raise typer.Exit(1)
        else:
            try:
                codegen_config = get_config(config)
            except FileNotFoundError as e:
                error_console.print(f'[red]Error:[/red] {e}')
                error_console.print(
                    '\n[dim]Hint: Run [bold]otterapi init[/bold] to create a configuration file,[/dim]'
                )
                error_console.print(
                    '[dim]or use [bold]--source[/bold] and [bold]--output[/bold] options.[/dim]'
                )
                raise typer.Exit(1)

        if dry_run:
            console.print(
                Panel('[yellow]DRY RUN MODE[/yellow] - No files will be written')
            )

        for document_config in codegen_config.documents:
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

                if dry_run:
                    # Load and validate schema without writing
                    codegen._load_schema()
                    progress.update(
                        task,
                        description=f'[yellow]Would generate[/yellow] code for {document_config.source}',
                    )

                    # Show what would be generated
                    console.print('\n[dim]Would generate files:[/dim]')
                    console.print(
                        f'  - {document_config.output}/{document_config.models_file}'
                    )
                    console.print(
                        f'  - {document_config.output}/{document_config.endpoints_file}'
                    )
                    console.print(f'  - {document_config.output}/__init__.py')

                    # Show schema info
                    if codegen.openapi:
                        console.print('\n[dim]Schema info:[/dim]')
                        if codegen.openapi.info:
                            console.print(f'  Title: {codegen.openapi.info.title}')
                            console.print(f'  Version: {codegen.openapi.info.version}')
                        if (
                            codegen.openapi.components
                            and codegen.openapi.components.schemas
                        ):
                            console.print(
                                f'  Schemas: {len(codegen.openapi.components.schemas)}'
                            )
                        if codegen.openapi.paths:
                            console.print(f'  Paths: {len(codegen.openapi.paths.root)}')
                else:
                    codegen.generate()
                    progress.update(
                        task,
                        description=f'Code generation completed for {document_config.source}!',
                    )

                    console.print('[dim]Generated files:[/dim]')
                    console.print(
                        f'  - {document_config.output}/{document_config.models_file}'
                    )
                    console.print(
                        f'  - {document_config.output}/{document_config.endpoints_file}'
                    )

        if not dry_run:
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

    config_path.write_text(content)

    console.print(f'\n[green]✓[/green] Configuration saved to {config_path}')
    console.print('\n[dim]Preview:[/dim]')
    syntax = Syntax(content, 'yaml' if not path.endswith('.json') else 'json')
    console.print(syntax)

    console.print(
        f'\n[dim]Run [bold]otterapi generate -c {path}[/bold] to generate code.[/dim]'
    )


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

        try:
            loader = SchemaLoader()
            schema = loader.load(source)
            progress.update(task, description='Validating schema...')

            # Collect validation info
            warnings: list[str] = []
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

                # Count operations
                operations = 0
                for path_item in schema.paths.root.values():
                    for method in [
                        'get',
                        'post',
                        'put',
                        'patch',
                        'delete',
                        'head',
                        'options',
                    ]:
                        if getattr(path_item, method, None):
                            operations += 1
                info['operations'] = operations

            if schema.components:
                if schema.components.schemas:
                    info['schemas'] = len(schema.components.schemas)
                if schema.components.securitySchemes:
                    info['security_schemes'] = len(schema.components.securitySchemes)

            # Check for potential issues
            if schema.paths:
                for path, path_item in schema.paths.root.items():
                    for method in ['get', 'post', 'put', 'patch', 'delete']:
                        operation = getattr(path_item, method, None)
                        if operation and not operation.operationId:
                            warnings.append(
                                f'{method.upper()} {path}: Missing operationId'
                            )

            progress.update(task, description='Validation complete!')

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

    # Print results
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
def version() -> None:
    """Show the version of OtterAPI."""
    try:
        from otterapi._version import version as ver

        console.print(f'otterapi version: [bold]{ver}[/bold]')
    except ImportError:
        console.print('otterapi version: [dim]unknown (development)[/dim]')

    # Show dependency versions if verbose
    console.print('\n[dim]Dependencies:[/dim]')
    try:
        import pydantic

        console.print(f'  pydantic: {pydantic.__version__}')
    except ImportError:
        pass
    try:
        import httpx

        console.print(f'  httpx: {httpx.__version__}')
    except ImportError:
        pass
    try:
        import typer

        console.print(f'  typer: {typer.__version__}')
    except ImportError:
        pass


if __name__ == '__main__':
    app()
