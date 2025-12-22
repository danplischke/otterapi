import traceback
from typing import Annotated

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from otterapi.codegen_v2.codegen import Codegen
from otterapi.config import get_config

console = Console()
app = typer.Typer(
    name='otterapi',
    help='Generate Python client code from OpenAPI specifications',
    no_args_is_help=True,
)


@app.command()
def generate(
    config: Annotated[
        str | None,
        typer.Option(
            '--config', '-c', help='Path to configuration file (YAML or JSON)'
        ),
    ] = None,
) -> None:
    """Generate Python client code from configuration.

    If no config file is specified, will look for default config files
    in the current directory or use environment variables.

    Examples:
        otterapi generate
        otterapi generate --config my-config.yaml
        otterapi generate -c config.json
    """
    config = get_config(config)

    try:

            for document_config in config.documents:
                with Progress(
                        SpinnerColumn(),
                        TextColumn('[progress.description]{task.description}'),
                        console=console,
                ) as progress:
                    task = progress.add_task(
                        f'Generating code for {document_config.source} in {document_config.output}...',
                        total=None,
                    )

                    codegen = Codegen(document_config)
                    codegen.generate()

                    progress.update(task, description=f'Code generation completed for {document_config.source}!')
                console.print('[dim]Generated files:[/dim]')
                console.print(
                    f'  - {document_config.output}/{document_config.models_file}'
                )
                console.print(
                    f'  - {document_config.output}/{document_config.endpoints_file}'
                )

            progress.update(task, description='Code generation completed!')

    except Exception as e:
        console.print(f'[red]Error:[/red] {str(e)}')
        traceback.print_exc()
        raise typer.Exit(1)


@app.command()
def version() -> None:
    """Show the version of otterapi."""
    try:
        from otterapi._version import version

        console.print(f'otterapi version: {version}')
    except ImportError:
        console.print('otterapi version: unknown')


if __name__ == '__main__':
    generate()
