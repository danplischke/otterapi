"""Tests for module names that collide with files the generator owns.

With module splitting on, module names come from spec tags and URL path
segments, and each module is written to ``<output>/<name>.py``.  A tag called
``models`` or ``Client`` therefore used to overwrite ``models.py`` or the
user's customised ``client.py`` with endpoint code, and the package then failed
to import.  These tests cover the reserved-name set, the deterministic rename
of names derived from the spec, and the error for names the user chose.
"""

import ast
import importlib
import itertools
import json
import logging
import re
import sys
from pathlib import Path

import pytest

from otterapi.codegen import splitting
from otterapi.codegen._features import FeatureModule, all_features
from otterapi.codegen.codegen import Codegen
from otterapi.codegen.splitting import (
    ROOT_MODULE_NAME,
    ModuleMapResolver,
    ModuleTree,
    ModuleTreeBuilder,
    ResolvedModule,
    SplitModuleEmitter,
    build_module_tree,
    reserved_module_names,
    validate_user_module_names,
)
from otterapi.codegen.types import Endpoint
from otterapi.config import (
    DataFrameConfig,
    DocumentConfig,
    ExportConfig,
    ModuleSplitConfig,
    PaginationConfig,
)
from otterapi.exceptions import ConfigurationError, OtterAPIError

PACKAGE = 'reserved_pkg'

# Tags that would each overwrite a file of the package, with the operations
# tagged with them.
RESERVED_TAGS = {
    'Client': ['client_list', 'client_get'],
    'models': ['model_list', 'model_get'],
    '_retry': ['retry_list', 'retry_get'],
    '__init__': ['init_list', 'init_get'],
    'endpoints': ['endpoint_list', 'endpoint_get'],
}

# What each of them is renamed to.
RENAMED = {
    'client': 'client_',
    'models': 'models_',
    '_retry': '_retry_',
    '__init__': '__init___',
    'endpoints': 'endpoints_',
}


def make_endpoint(
    name: str,
    path: str,
    method: str = 'GET',
    tags: list[str] | None = None,
) -> Endpoint:
    """Create a minimal Endpoint for testing."""
    sync_ast = ast.FunctionDef(
        name=name,
        args=ast.arguments(
            posonlyargs=[],
            args=[],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
        ),
        body=[ast.Return(value=ast.Constant(value=None))],
        decorator_list=[],
        returns=ast.Constant(value=None),
    )
    async_ast = ast.AsyncFunctionDef(
        name=f'a{name}',
        args=ast.arguments(
            posonlyargs=[],
            args=[],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
        ),
        body=[ast.Return(value=ast.Constant(value=None))],
        decorator_list=[],
        returns=ast.Constant(value=None),
    )

    return Endpoint(
        sync_ast=sync_ast,
        async_ast=async_ast,
        sync_fn_name=name,
        async_fn_name=f'a{name}',
        method=method,
        path=path,
        tags=tags,
    )


def tagged_endpoints(*tags: str) -> list[Endpoint]:
    """Two endpoints per tag, named after it, so the module survives consolidation."""
    return [
        make_endpoint(f'{tag}.{suffix}', f'/{tag}/{suffix}', tags=[tag])
        for tag in tags
        for suffix in ('list', 'get')
    ]


def endpoint_names(node) -> list[str]:
    """Names of the endpoints a tree node holds."""
    return [endpoint.sync_fn_name for endpoint in node.endpoints]


def split_config(**overrides) -> ModuleSplitConfig:
    """A tag-based split config that keeps every module, however small."""
    options = {'enabled': True, 'strategy': 'tag', 'min_endpoints': 1, **overrides}
    return ModuleSplitConfig(**options)


def operation(operation_id: str, tag: str | None = None) -> dict:
    """An operation that returns the ``Pet`` model, so modules import ``.models``."""
    result: dict = {
        'operationId': operation_id,
        'responses': {
            '200': {
                'description': 'ok',
                'content': {
                    'application/json': {'schema': {'$ref': '#/components/schemas/Pet'}}
                },
            }
        },
    }
    if tag is not None:
        result['tags'] = [tag]
    return result


def make_spec(
    tagged: dict[str, list[str]] | None = None, paths: dict[str, str] | None = None
) -> dict:
    """Build a spec from ``{tag: [operation ids]}`` and ``{path: operation id}``."""
    spec_paths: dict = {}
    for tag, operation_ids in (tagged or {}).items():
        for operation_id in operation_ids:
            spec_paths[f'/{operation_id}'] = {'get': operation(operation_id, tag)}
    for path, operation_id in (paths or {}).items():
        item = operation(operation_id)
        # A placeholder needs a declared path parameter, or generation refuses
        # the path.
        placeholders = re.findall(r'\{([^}]+)\}', path)
        if placeholders:
            item['parameters'] = [
                {
                    'in': 'path',
                    'name': name,
                    'required': True,
                    'schema': {'type': 'string'},
                }
                for name in placeholders
            ]
        spec_paths[path] = {'get': item}
    return {
        'openapi': '3.1.0',
        'info': {'title': 'Reserved', 'version': '1.0'},
        'servers': [{'url': 'https://example.test'}],
        'paths': spec_paths,
        'components': {
            'schemas': {
                'Pet': {
                    'type': 'object',
                    'properties': {'name': {'type': 'string'}},
                }
            }
        },
    }


def generate(
    tmp_path: Path, spec: dict, module_split: ModuleSplitConfig, **document
) -> Path:
    """Generate ``spec`` into ``tmp_path / PACKAGE`` and return that directory."""
    spec_file = tmp_path / 'openapi.json'
    spec_file.write_text(json.dumps(spec), encoding='utf-8')
    output = tmp_path / PACKAGE
    Codegen(
        DocumentConfig(
            source=str(spec_file),
            output=str(output),
            module_split=module_split,
            **document,
        )
    ).generate()
    return output


def python_files(directory: Path) -> set[str]:
    """Relative POSIX paths of every ``.py`` file under ``directory``."""
    return {p.relative_to(directory).as_posix() for p in directory.rglob('*.py')}


class MetricsFeature(FeatureModule):
    """A runtime helper module that does not exist yet."""

    module_filename = '_metrics.py'
    module_content = '"""Stub."""\n'

    def is_enabled(self, config: DocumentConfig) -> bool:
        return True


def add_metrics_feature(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ``all_features()`` list one more runtime helper module."""
    real = splitting.all_features
    monkeypatch.setattr(splitting, 'all_features', lambda: [*real(), MetricsFeature()])


@pytest.fixture
def import_package(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Import a package generated under ``tmp_path``, forgetting it afterwards."""
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()

    def forget() -> None:
        for name in [m for m in sys.modules if m.split('.')[0] == PACKAGE]:
            del sys.modules[name]

    forget()
    yield lambda name=PACKAGE: importlib.import_module(name)
    forget()


class TestReservedModuleNames:
    """Tests for the single source of truth for reserved names."""

    def test_names_of_files_the_generator_owns(self):
        """The package skeleton, the models and the root module are reserved."""
        reserved = reserved_module_names()

        assert {'__init__', 'client', '_client', 'models', 'endpoints'} <= reserved
        assert ROOT_MODULE_NAME in reserved

    @pytest.mark.parametrize('feature', all_features(), ids=lambda f: type(f).__name__)
    def test_every_feature_module_is_reserved(self, feature: FeatureModule):
        """Each runtime helper the generator writes is reserved, by derivation."""
        assert Path(feature.module_filename).stem in reserved_module_names()

    def test_a_new_feature_module_is_reserved_automatically(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """The set follows ``all_features()``; no name list needs updating."""
        assert '_metrics' not in reserved_module_names()

        add_metrics_feature(monkeypatch)

        assert '_metrics' in reserved_module_names()

    def test_a_new_feature_module_renames_a_tag(self, monkeypatch: pytest.MonkeyPatch):
        """A spec tag matching a newly added feature module is renamed."""
        add_metrics_feature(monkeypatch)

        tree = ModuleTreeBuilder(split_config(), reserved_module_names()).build(
            tagged_endpoints('_metrics')
        )

        assert list(tree.children) == ['_metrics_']

    @pytest.mark.parametrize('flat_structure', [False, True])
    def test_root_level_endpoints_use_a_reserved_module_name(
        self, tmp_path: Path, flat_structure: bool
    ):
        """The module the emitter itself names is one the generator reserves."""
        tree = ModuleTree(name='__root__')
        tree.add_endpoint([], make_endpoint('ping', '/ping'))
        emitter = SplitModuleEmitter(
            ModuleSplitConfig(enabled=True, flat_structure=flat_structure),
            tmp_path,
            tmp_path / 'models.py',
        )

        (emitted,) = emitter.emit(tree)

        assert emitted.path.name == f'{ROOT_MODULE_NAME}.py'
        assert emitted.path.stem in reserved_module_names()

    def test_generator_written_modules_are_all_reserved(self, tmp_path: Path):
        """Every module the generator writes at the root is in the reserved set."""
        output = generate(
            tmp_path,
            make_spec(paths={'/pets': 'list_pets'}),
            ModuleSplitConfig(enabled=False),
            pagination=PaginationConfig(enabled=True),
            dataframe=DataFrameConfig(enabled=True, pandas=True),
            export=ExportConfig(enabled=True),
        )

        stems = {path.stem for path in output.glob('*.py')}

        assert stems <= reserved_module_names()

    def test_models_and_endpoints_file_stems_are_reserved(self):
        """The configured file names count, not just the defaults."""
        reserved = reserved_module_names('schemas.py', 'api.py')

        assert {'schemas', 'api'} <= reserved
        # Generated modules import ``.models`` whatever the file is called.
        assert 'models' in reserved

    @pytest.mark.parametrize(
        'name',
        [
            'con',
            'prn',
            'aux',
            'nul',
            *(f'com{n}' for n in range(1, 10)),
            *(f'lpt{n}' for n in range(1, 10)),
        ],
    )
    def test_windows_device_names_are_reserved(self, name: str):
        assert name in reserved_module_names()

    def test_names_are_case_folded(self):
        """Matching is case-insensitive, so the set holds lower-case names."""
        reserved = reserved_module_names('Schemas.py', 'API.py')

        assert {'schemas', 'api'} <= reserved
        assert all(name == name.casefold() for name in reserved)

    def test_default_set_uses_the_default_file_names(self):
        assert reserved_module_names() == reserved_module_names(
            'models.py', 'endpoints.py'
        )


class TestRenamingSpecNames:
    """A reserved name derived from the spec gets a trailing underscore."""

    @pytest.mark.parametrize(
        ('tag', 'renamed'),
        [
            ('Client', 'client_'),
            ('models', 'models_'),
            ('_retry', '_retry_'),
            ('__init__', '__init___'),
            ('endpoints', 'endpoints_'),
            ('_client', '_client_'),
            ('_serialization', '_serialization_'),
            ('_pagination', '_pagination_'),
            ('_export', '_export_'),
            ('_dataframe', '_dataframe_'),
            ('_concurrency', '_concurrency_'),
        ],
    )
    def test_reserved_tag_is_renamed(self, tag: str, renamed: str):
        tree = ModuleTreeBuilder(split_config()).build(tagged_endpoints(tag))

        assert list(tree.children) == [renamed]
        assert tree.children[renamed].name == renamed
        assert len(tree.children[renamed].endpoints) == 2

    def test_each_reserved_tag_gets_its_own_module(self):
        """Tags that were all reserved stay separate modules."""
        tree = ModuleTreeBuilder(split_config()).build(
            tagged_endpoints(*RESERVED_TAGS, 'Pets')
        )

        assert sorted(tree.children) == sorted([*RENAMED.values(), 'pets'])
        assert all(len(node.endpoints) == 2 for node in tree.children.values())

    @pytest.mark.parametrize(
        ('tag', 'renamed'), [('Models', 'models_'), ('CLIENT', 'client_')]
    )
    def test_matching_ignores_case(self, tag: str, renamed: str):
        tree = ModuleTreeBuilder(split_config()).build(tagged_endpoints(tag))

        assert list(tree.children) == [renamed]

    @pytest.mark.parametrize(
        ('tag', 'renamed'), [('Aux', 'aux_'), ('COM1', 'com1_'), ('nul', 'nul_')]
    )
    def test_windows_device_names_are_renamed(self, tag: str, renamed: str):
        tree = ModuleTreeBuilder(split_config()).build(tagged_endpoints(tag))

        assert list(tree.children) == [renamed]

    def test_path_segment_is_renamed(self):
        """A benign ``/models`` path needs the same care as a ``models`` tag."""
        config = split_config(strategy='path')
        endpoints = [
            make_endpoint('list_models', '/models'),
            make_endpoint('get_model', '/models/{model}'),
        ]

        tree = ModuleTreeBuilder(config).build(endpoints)

        assert list(tree.children) == ['models_']
        assert len(tree.children['models_'].endpoints) == 2

    def test_joined_path_depth_name_is_renamed(self):
        """A name built by ``path_depth`` joining can be reserved too."""
        config = split_config(strategy='path', path_depth=2, global_strip_prefixes=[])
        endpoints = [
            make_endpoint('list_models', '/api/models'),
            make_endpoint('get_model', '/api/models/{model}'),
        ]

        tree = ModuleTreeBuilder(config, reserved_module_names('api_models.py')).build(
            endpoints
        )

        assert list(tree.children) == ['api_models_']

    def test_custom_models_and_endpoints_files_are_honoured(self):
        reserved = reserved_module_names('schemas.py', 'api.py')

        tree = ModuleTreeBuilder(split_config(), reserved).build(
            tagged_endpoints('Schemas', 'API', 'Pets')
        )

        assert sorted(tree.children) == ['api_', 'pets', 'schemas_']

    def test_rename_that_collides_moves_on(self):
        """``models`` is renamed past a ``models_`` module that already exists."""
        tree = ModuleTreeBuilder(split_config()).build(
            tagged_endpoints('models', 'models_')
        )

        assert sorted(tree.children) == ['models_', 'models__']
        # The module that was not reserved keeps its own name and endpoints.
        assert endpoint_names(tree.children['models_']) == [
            'models_.list',
            'models_.get',
        ]
        assert endpoint_names(tree.children['models__']) == [
            'models.list',
            'models.get',
        ]

    def test_rename_keeps_going_until_free(self):
        tree = ModuleTreeBuilder(split_config()).build(
            tagged_endpoints('models', 'models_', 'models__')
        )

        assert sorted(tree.children) == ['models_', 'models__', 'models___']
        assert endpoint_names(tree.children['models___']) == [
            'models.list',
            'models.get',
        ]

    def test_rename_never_lands_on_another_reserved_name(self):
        """A renamed module steers clear of the reserved names as well."""
        reserved = reserved_module_names() | {'client_'}

        tree = ModuleTreeBuilder(split_config(), reserved).build(
            tagged_endpoints('client')
        )

        assert list(tree.children) == ['client__']

    def test_build_module_tree_renames_too(self):
        """The convenience function takes the reserved names like the builder."""
        endpoints = tagged_endpoints('models', 'schemas')

        default = build_module_tree(endpoints, split_config())
        custom = build_module_tree(
            endpoints, split_config(), reserved_module_names('schemas.py')
        )

        assert sorted(default.children) == ['models_', 'schemas']
        assert sorted(custom.children) == ['models_', 'schemas_']

    def test_names_sanitising_alike_still_merge(self):
        """Distinct spec names with one sanitised name share a module, as before."""
        endpoints = tagged_endpoints('Models', 'models', 'MODELS')

        tree = ModuleTreeBuilder(split_config()).build(endpoints)

        assert list(tree.children) == ['models_']
        assert len(tree.children['models_'].endpoints) == 6

    def test_unreserved_names_are_left_alone(self):
        tree = ModuleTreeBuilder(split_config()).build(
            tagged_endpoints('Users', 'User Management', 'common', 'clients')
        )

        assert sorted(tree.children) == [
            'clients',
            'common',
            'user_management',
            'users',
        ]

    def test_result_does_not_depend_on_endpoint_order(self):
        tags = ('models', 'models_', 'client', 'Pets', 'client_')
        layouts = set()

        for ordering in itertools.permutations(tags):
            tree = ModuleTreeBuilder(split_config()).build(tagged_endpoints(*ordering))
            layouts.add(
                tuple(
                    sorted(
                        (name, tuple(sorted(endpoint_names(node))))
                        for name, node in tree.children.items()
                    )
                )
            )

        assert len(layouts) == 1
        (layout,) = layouts
        assert [name for name, _ in layout] == [
            'client_',
            'client__',
            'models_',
            'models__',
            'pets',
        ]

    def test_consolidated_modules_are_not_renamed(self):
        """A reserved module that is folded into the fallback is never written."""
        config = split_config(min_endpoints=2)
        endpoints = [
            make_endpoint('only_model', '/model', tags=['models']),
            make_endpoint('list_pets', '/pets', tags=['pets']),
            make_endpoint('get_pet', '/pets/{id}', tags=['pets']),
        ]

        tree = ModuleTreeBuilder(config).build(endpoints)

        assert sorted(tree.children) == ['common', 'pets']
        assert [e.sync_fn_name for e in tree.children['common'].endpoints] == [
            'only_model'
        ]

    def test_resolver_leaves_renaming_to_the_builder(self):
        """The resolver reports the sanitised name; only the tree is final."""
        resolver = ModuleMapResolver(split_config())

        resolved = resolver.resolve('/models', 'GET', tags=['models'])

        assert resolved.module_path == ['models']

    def test_every_path_component_is_renamed(self):
        """Directory names are checked as well as the module name."""

        class NestingResolver(ModuleMapResolver):
            def resolve(self, path, method, tags=None):
                return ResolvedModule(
                    module_path=[tags[0].lower(), 'models', 'client'],
                    resolution='tag',
                )

        builder = ModuleTreeBuilder(split_config())
        builder.resolver = NestingResolver(builder.config)

        tree = builder.build(tagged_endpoints('Client', 'Store'))

        assert sorted(tree.children) == ['client_', 'store']
        for top in tree.children.values():
            assert list(top.children) == ['models_']
            assert list(top.children['models_'].children) == ['client_']
            assert top.children['models_'].children['client_'].name == 'client_'
        paths = sorted('/'.join(path) for path, _ in tree.walk_leaves())
        assert paths == ['client_/models_/client_', 'store/models_/client_']

    def test_a_directory_is_checked_on_its_own(self):
        """Names are only unique within a directory: ``a/models_`` is not ``models_``."""

        class NestingResolver(ModuleMapResolver):
            def resolve(self, path, method, tags=None):
                if tags[0] == 'Models':
                    return ResolvedModule(module_path=['models'], resolution='tag')
                return ResolvedModule(module_path=['a', 'models'], resolution='tag')

        builder = ModuleTreeBuilder(split_config())
        builder.resolver = NestingResolver(builder.config)

        tree = builder.build(tagged_endpoints('Models', 'Nested'))

        assert sorted(path for path, _ in tree.walk_leaves()) == [
            ['a', 'models_'],
            ['models_'],
        ]

    def test_rename_is_logged(self, caplog: pytest.LogCaptureFixture):
        with caplog.at_level(logging.WARNING, logger=splitting.__name__):
            ModuleTreeBuilder(split_config()).build(tagged_endpoints('models'))

        assert "Module 'models'" in caplog.text
        assert "'models_'" in caplog.text


class TestUserChosenNames:
    """A reserved name the user chose is an error, never a silent rename."""

    @pytest.mark.parametrize(
        'key',
        [
            'client',
            'Client',
            'CLIENT',
            'models',
            '_retry',
            '__init__',
            'endpoints',
            'aux',
        ],
    )
    def test_module_map_key_is_rejected(self, key: str):
        config = split_config(strategy='custom', module_map={key: ['/x/*']})

        with pytest.raises(ConfigurationError) as excinfo:
            ModuleTreeBuilder(config)

        message = str(excinfo.value)
        assert f"module_map key '{key}'" in message
        assert excinfo.value.field == 'module_split.module_map'

    def test_error_names_the_reserved_set(self):
        config = split_config(strategy='custom', module_map={'client': ['/x/*']})

        with pytest.raises(ConfigurationError) as excinfo:
            ModuleMapResolver(config)

        message = str(excinfo.value)
        for name in sorted(reserved_module_names()):
            assert name in message
        assert 'reserved' in message.lower()

    def test_error_is_a_configuration_error(self):
        config = split_config(strategy='custom', module_map={'models': ['/x/*']})

        with pytest.raises(OtterAPIError):
            ModuleMapResolver(config)

    def test_nested_key_is_checked_at_every_level(self):
        config = split_config(
            strategy='custom',
            module_map={'store': {'client': ['/c/*'], 'orders': {'_retry': ['/r/*']}}},
        )

        with pytest.raises(ConfigurationError) as excinfo:
            ModuleMapResolver(config)

        message = str(excinfo.value)
        assert "module_map key 'store.client'" in message
        assert "module_map key 'store.orders._retry'" in message

    def test_first_path_component_cannot_be_a_package_named_client(self):
        """``client/`` next to ``client.py`` would shadow the module."""
        config = split_config(
            strategy='custom', module_map={'client': {'users': ['/users/*']}}
        )

        with pytest.raises(ConfigurationError, match="module_map key 'client'"):
            ModuleTreeBuilder(config)

    def test_every_offender_is_reported(self):
        config = split_config(
            strategy='custom',
            module_map={'client': ['/c/*'], 'models': ['/m/*'], 'users': ['/u/*']},
        )

        with pytest.raises(ConfigurationError) as excinfo:
            ModuleMapResolver(config)

        message = str(excinfo.value)
        assert "module_map key 'client'" in message
        assert "module_map key 'models'" in message
        assert "'users'" not in message

    @pytest.mark.parametrize('name', ['models', 'Client', '_client'])
    def test_fallback_module_is_rejected(self, name: str):
        config = split_config(fallback_module=name)

        with pytest.raises(ConfigurationError) as excinfo:
            ModuleMapResolver(config)

        assert f"fallback_module '{name}'" in str(excinfo.value)
        assert excinfo.value.field == 'module_split.fallback_module'

    def test_configured_models_and_endpoints_files_count(self):
        config = split_config(strategy='custom', module_map={'schemas': ['/s/*']})

        ModuleMapResolver(config)  # fine with the default file names
        with pytest.raises(ConfigurationError, match='schemas'):
            ModuleMapResolver(config, reserved_module_names('schemas.py'))

    def test_flat_structure_checks_the_joined_file_name(self):
        """In the flat layout ``_`` + ``init__`` is written as ``__init__.py``."""
        module_map = {'_': {'init__': ['/i/*']}}
        nested = split_config(strategy='custom', module_map=module_map)
        flat = split_config(
            strategy='custom', module_map=module_map, flat_structure=True
        )

        validate_user_module_names(nested)
        with pytest.raises(
            ConfigurationError, match=r"'_\.init__' \(written as __init__\.py\)"
        ):
            validate_user_module_names(flat)

    def test_module_map_is_not_checked_when_the_strategy_ignores_it(self):
        """The ``tag`` and ``path`` strategies never read ``module_map``."""
        config = split_config(strategy='tag', module_map={'client': ['/c/*']})

        builder = ModuleTreeBuilder(config)
        tree = builder.build(tagged_endpoints('Pets'))

        assert list(tree.children) == ['pets']

    def test_unreserved_names_pass(self):
        config = split_config(
            strategy='custom',
            module_map={'identity': {'users': ['/users/*']}, 'clients': ['/c/*']},
            fallback_module='misc',
        )

        validate_user_module_names(config)

    def test_user_names_are_never_renamed(self):
        """The rename only ever touches names derived from the spec."""
        config = split_config(strategy='hybrid', module_map={'users': ['/users/*']})
        endpoints = [
            make_endpoint('list_users', '/users/all'),
            make_endpoint('list_models', '/m', tags=['models']),
            make_endpoint('get_model', '/m/{id}', tags=['models']),
        ]

        tree = ModuleTreeBuilder(config).build(endpoints)

        assert sorted(tree.children) == ['models_', 'users']


class TestGeneratedPackages:
    """Generate real packages and import them."""

    def test_reserved_tags_do_not_overwrite_generated_files(
        self, tmp_path: Path, import_package
    ):
        """The reproduction: operations tagged ``Client``, ``models``, ``_retry``..."""
        spec = make_spec(RESERVED_TAGS)
        # A first run creates the stub; the user then customises it.
        generate(tmp_path, make_spec(paths={'/pets': 'list_pets'}), split_config())
        output = tmp_path / PACKAGE
        custom_client = output / 'client.py'
        custom_client.write_text(
            custom_client.read_text(encoding='utf-8') + '\nCUSTOMISED = True\n',
            encoding='utf-8',
        )
        client_before = custom_client.read_bytes()
        models_before = (output / 'models.py').read_bytes()
        retry_before = (output / '_retry.py').read_bytes()

        generate(tmp_path, spec, split_config())

        # The files the generator owns, and the one it leaves alone, are intact.
        assert custom_client.read_bytes() == client_before
        assert (output / 'models.py').read_bytes() == models_before
        assert (output / '_retry.py').read_bytes() == retry_before
        assert 'class Pet' in (output / 'models.py').read_text(encoding='utf-8')
        # ``__init__.py`` is still the package's init, not an endpoint module.
        root_init = ast.parse((output / '__init__.py').read_text(encoding='utf-8'))
        assert not [n for n in ast.walk(root_init) if isinstance(n, ast.FunctionDef)]
        # Every tag got a module of its own next to them.
        assert {f'{name}.py' for name in RENAMED.values()} <= python_files(output)

        package = import_package()
        assert package.client.CUSTOMISED is True
        assert package.Client is package.client.Client
        for tag, operation_ids in RESERVED_TAGS.items():
            module = importlib.import_module(f'{PACKAGE}.{RENAMED[tag.casefold()]}')
            for operation_id in operation_ids:
                function = getattr(module, operation_id)
                assert callable(function)
                assert getattr(package, operation_id) is function
                assert callable(getattr(module, f'async_{operation_id}'))

    def test_root_init_is_not_an_endpoint_module(self, tmp_path: Path):
        output = generate(tmp_path, make_spec(RESERVED_TAGS), split_config())

        root_init = (output / '__init__.py').read_text(encoding='utf-8')
        tree = ast.parse(root_init)

        assert not [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
        assert 'from .__init___ import' in root_init
        assert 'from .client_ import' in root_init
        assert 'from .models_ import' in root_init
        assert 'from .models import' in root_init
        # The renamed modules import the real ``client`` and ``models``.
        client_module = (output / 'client_.py').read_text(encoding='utf-8')
        assert 'from .client import Client' in client_module
        assert 'from .models import Pet' in client_module

    def test_custom_client_content_is_untouched(self, tmp_path: Path):
        """Pre-created ``client.py`` content survives, whatever it contains."""
        output = tmp_path / PACKAGE
        output.mkdir()
        sentinel = '# sentinel: written by the user\nSENTINEL = object()\n'
        (output / 'client.py').write_text(sentinel, encoding='utf-8')

        generate(
            tmp_path,
            make_spec({'Client': ['client_list', 'client_get']}),
            split_config(),
        )

        assert (output / 'client.py').read_text(encoding='utf-8') == sentinel
        assert 'def client_list' in (output / 'client_.py').read_text(encoding='utf-8')

    def test_benign_models_path(self, tmp_path: Path, import_package):
        """A spec with ``/models`` paths and no tags, on the default config."""
        spec = make_spec(
            paths={'/models': 'list_models', '/models/{model}': 'get_model'}
        )

        output = generate(tmp_path, spec, ModuleSplitConfig(enabled=True))

        assert 'class Pet' in (output / 'models.py').read_text(encoding='utf-8')
        assert 'def list_models' in (output / 'models_.py').read_text(encoding='utf-8')
        package = import_package()
        module = importlib.import_module(f'{PACKAGE}.models_')
        assert package.list_models is module.list_models
        assert package.get_model is module.get_model
        assert package.Pet is importlib.import_module(f'{PACKAGE}.models').Pet

    def test_rename_that_collides_end_to_end(self, tmp_path: Path, import_package):
        spec = make_spec(
            {
                'models': ['model_list', 'model_get'],
                'models_': ['underscore_list', 'underscore_get'],
            }
        )

        output = generate(tmp_path, spec, split_config())

        assert 'def underscore_list' in (output / 'models_.py').read_text(
            encoding='utf-8'
        )
        assert 'def model_list' in (output / 'models__.py').read_text(encoding='utf-8')
        package = import_package()
        assert callable(package.model_list)
        assert callable(package.underscore_list)
        assert 'class Pet' in (output / 'models.py').read_text(encoding='utf-8')

    def test_case_insensitive_names_end_to_end(self, tmp_path: Path, import_package):
        spec = make_spec(
            {
                'Models': ['model_list', 'model_get'],
                'CLIENT': ['client_list', 'client_get'],
            }
        )

        output = generate(tmp_path, spec, split_config())

        assert {'models_.py', 'client_.py'} <= python_files(output)
        assert {'Models.py', 'CLIENT.py'}.isdisjoint(python_files(output))
        package = import_package()
        assert callable(package.model_list)
        assert callable(package.client_list)

    def test_flat_structure_end_to_end(self, tmp_path: Path, import_package):
        spec = make_spec({**RESERVED_TAGS, 'Pets': ['pet_list', 'pet_get']})

        output = generate(tmp_path, spec, split_config(flat_structure=True))

        assert {f'{name}.py' for name in RENAMED.values()} <= python_files(output)
        package = import_package()
        for operation_ids in RESERVED_TAGS.values():
            for operation_id in operation_ids:
                assert callable(getattr(package, operation_id))

    def test_nested_modules_beside_renamed_ones(self, tmp_path: Path, import_package):
        """Nested ``module_map`` packages and renamed tag modules live together."""
        spec = make_spec(
            {
                'Client': ['client_list', 'client_get'],
                'Store': ['store_list', 'store_get'],
            }
        )
        module_split = split_config(
            strategy='hybrid',
            module_map={'sales': {'stores': ['/store_*']}},
        )

        output = generate(tmp_path, spec, module_split)

        assert {'client_.py', 'sales/stores.py', 'sales/__init__.py'} <= python_files(
            output
        )
        package = import_package()
        assert callable(package.client_list)
        assert callable(package.store_list)
        assert (
            package.store_list
            is importlib.import_module(f'{PACKAGE}.sales.stores').store_list
        )

    def test_nested_directory_named_after_a_reserved_module(
        self, tmp_path: Path, import_package, monkeypatch: pytest.MonkeyPatch
    ):
        """A directory that would shadow ``client.py`` is renamed with its files."""
        resolve = ModuleMapResolver.resolve

        def nesting_resolve(self, path, method, tags=None):
            resolved = resolve(self, path, method, tags)
            resolved.module_path = [*resolved.module_path, 'v1']
            return resolved

        monkeypatch.setattr(ModuleMapResolver, 'resolve', nesting_resolve)
        spec = make_spec({'Client': ['client_list', 'client_get']})

        output = generate(tmp_path, spec, split_config())

        files = python_files(output)
        assert {'client_/v1.py', 'client_/__init__.py', 'client.py'} <= files
        assert 'client/__init__.py' not in files
        package = import_package()
        nested = importlib.import_module(f'{PACKAGE}.client_.v1')
        assert package.client_list is nested.client_list
        assert hasattr(importlib.import_module(f'{PACKAGE}.client_'), 'client_list')

    def test_regeneration_is_stable(self, tmp_path: Path):
        spec = make_spec({**RESERVED_TAGS, 'models_': ['u_list', 'u_get']})
        output = generate(tmp_path, spec, split_config())
        first = {p: (output / p).read_bytes() for p in python_files(output)}

        generate(tmp_path, spec, split_config())

        assert {p: (output / p).read_bytes() for p in python_files(output)} == first

    def test_every_generated_module_imports(self, tmp_path: Path, import_package):
        spec = make_spec({**RESERVED_TAGS, 'Pets': ['pet_list', 'pet_get']})
        output = generate(tmp_path, spec, split_config())

        import_package()
        for file in sorted(python_files(output)):
            parts = Path(file).with_suffix('').parts
            name = '.'.join([PACKAGE, *(p for p in parts if p != '__init__')])
            importlib.import_module(name)

    @pytest.mark.parametrize(
        'module_split',
        [
            {'strategy': 'custom', 'module_map': {'Client': ['/client_*']}},
            {'strategy': 'custom', 'module_map': {'client': {'users': ['/client_*']}}},
            {'strategy': 'hybrid', 'fallback_module': 'models'},
        ],
        ids=['key', 'nested-key', 'fallback'],
    )
    def test_user_chosen_name_fails_before_anything_is_written(
        self, tmp_path: Path, module_split: dict
    ):
        output = tmp_path / PACKAGE
        output.mkdir()
        (output / 'client.py').write_text('SENTINEL = 1\n', encoding='utf-8')
        spec = make_spec({'Pets': ['client_list', 'client_get']})

        with pytest.raises(ConfigurationError, match='Reserved module name'):
            generate(tmp_path, spec, split_config(**module_split))

        assert sorted(p.name for p in output.iterdir()) == ['client.py']
        assert (output / 'client.py').read_text(encoding='utf-8') == 'SENTINEL = 1\n'

    def test_user_chosen_name_fails_even_if_unused_by_the_spec(self, tmp_path: Path):
        """The name is checked up front, not when an endpoint happens to match."""
        spec = make_spec({'Pets': ['pet_list', 'pet_get']})
        module_split = split_config(
            strategy='custom', module_map={'models': ['/nothing/*']}
        )

        with pytest.raises(ConfigurationError):
            generate(tmp_path, spec, module_split)

        assert not (tmp_path / PACKAGE).exists()

    def test_user_chosen_name_with_custom_models_file(self, tmp_path: Path):
        module_split = split_config(
            strategy='custom', module_map={'schemas': ['/schema_*']}
        )
        spec = make_spec({'Pets': ['schema_list', 'schema_get']})

        with pytest.raises(ConfigurationError, match="module_map key 'schemas'"):
            generate(tmp_path, spec, module_split, models_file='schemas.py')

    def test_split_config_is_not_checked_when_splitting_is_off(self, tmp_path: Path):
        """An inactive ``module_split`` section cannot break generation."""
        module_split = ModuleSplitConfig(
            enabled=False, strategy='custom', module_map={'client': ['/c/*']}
        )

        output = generate(
            tmp_path, make_spec(paths={'/pets': 'list_pets'}), module_split
        )

        assert (output / 'endpoints.py').exists()
