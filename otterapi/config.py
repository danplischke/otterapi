import os
from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

DEFAULT_FILENAMES = ['otter.yaml', 'otter.yml']


class DocumentConfig(BaseModel):
    """Represents a single document to be processed."""

    source: str = Field(..., description='Path or URL to the OpenAPI document.')

    base_url: str | None = Field(
        None,
        description='Optional base URL to resolve paths if no servers are defined in the OpenAPI document.',
    )

    output: str = Field(..., description='Output directory for the generated code.')

    models_file: str | None = Field(
        'models.py', description='Optional file name for generated models.'
    )

    models_import_path: str | None = Field(
        None, description='Optional import path for generated models.'
    )

    endpoints_file: str | None = Field(
        'endpoints.py', description='Optional file name for generated endpoints.'
    )


class CodegenConfig(BaseSettings):
    documents: list[DocumentConfig] = Field(
        ..., description='List of OpenAPI documents to process.'
    )

    generate_endpoints: bool = Field(
        True, description='Whether to generate models from the OpenAPI schemas.'
    )


def load_yaml(path: str | Path) -> dict:
    import yaml

    return yaml.load(Path(path).read_text(), Loader=yaml.FullLoader)


def get_config(path: str | None = None) -> CodegenConfig:
    """Load configuration from a file or return default config."""
    if path:
        return CodegenConfig.model_validate(load_yaml(path))

    cwd = os.getcwd()

    for filename in DEFAULT_FILENAMES:
        path = Path(cwd) / filename
        if path.exists():
            return CodegenConfig.model_validate(load_yaml(path))

    path = Path(os.getcwd()) / 'pyproject.toml'

    if path.exists():
        import tomllib

        pyproject = tomllib.loads(path.read_text())
        tools = pyproject.get('tool', {})

        if 'otterapi' in tools:
            return CodegenConfig.model_validate(tools['otterapi'])

    raise FileNotFoundError('config not found')
