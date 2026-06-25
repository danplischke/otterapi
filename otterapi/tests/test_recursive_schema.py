"""Regression tests for recursive / cyclic OpenAPI schema generation."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

from pydantic import TypeAdapter

from otterapi.codegen.codegen import Codegen
from otterapi.codegen.types import Type, TypeGenerator
from otterapi.config import DocumentConfig
from otterapi.openapi import UniversalOpenAPI
from otterapi.openapi.v3_2.v3_2 import OpenAPI


def _create_type_generator(spec: dict) -> TypeGenerator:
    openapi = TypeAdapter(UniversalOpenAPI).validate_python(spec)
    schema = openapi.root
    while not isinstance(schema, OpenAPI):
        schema, _ = schema.upgrade()
    return TypeGenerator(schema)


def _openapi_spec(*, schemas: dict, response_ref: str, operation_id: str) -> dict:
    return {
        'openapi': '3.0.0',
        'info': {'title': 'Recursive Test API', 'version': '1.0.0'},
        'paths': {
            '/items': {
                'get': {
                    'operationId': operation_id,
                    'responses': {
                        '200': {
                            'description': 'OK',
                            'content': {
                                'application/json': {'schema': {'$ref': response_ref}}
                            },
                        }
                    },
                }
            }
        },
        'components': {'schemas': schemas},
    }


def _import_generated(parent_dir: Path, package_name: str, module_name: str):
    sys.path.insert(0, str(parent_dir))
    try:
        stale = [
            mod
            for mod in list(sys.modules)
            if mod == package_name or mod.startswith(package_name + '.')
        ]
        for mod in stale:
            sys.modules.pop(mod, None)
        return importlib.import_module(f'{package_name}.{module_name}')
    finally:
        try:
            sys.path.remove(str(parent_dir))
        except ValueError:
            pass


def _generate_models_module(tmp_path: Path, spec: dict, package_name: str):
    spec_file = tmp_path / f'{package_name}.json'
    spec_file.write_text(json.dumps(spec), encoding='utf-8')

    output_dir = tmp_path / package_name
    config = DocumentConfig(
        source=str(spec_file),
        output=str(output_dir),
        base_url='https://example.test',
    )
    Codegen(config).generate()

    models_source = (output_dir / 'models.py').read_text(encoding='utf-8')
    models_module = _import_generated(tmp_path, package_name, 'models')
    return models_module, models_source


def test_direct_self_reference_generates_and_validates_nested_node(tmp_path):
    spec = _openapi_spec(
        schemas={
            'Node': {
                'type': 'object',
                'properties': {
                    'children': {
                        'type': 'array',
                        'items': {'$ref': '#/components/schemas/Node'},
                    }
                },
            }
        },
        response_ref='#/components/schemas/Node',
        operation_id='getNode',
    )

    models, _ = _generate_models_module(tmp_path, spec, 'recursive_self_pkg')

    node = models.Node(children=[models.Node()])

    assert models.Node is not None
    assert isinstance(node.children[0], models.Node)


def test_mutual_recursion_generates_imports_and_validates_nested_models(tmp_path):
    spec = _openapi_spec(
        schemas={
            'A': {
                'type': 'object',
                'properties': {'b': {'$ref': '#/components/schemas/B'}},
            },
            'B': {
                'type': 'object',
                'properties': {'a': {'$ref': '#/components/schemas/A'}},
            },
        },
        response_ref='#/components/schemas/A',
        operation_id='getA',
    )

    models, _ = _generate_models_module(tmp_path, spec, 'recursive_mutual_pkg')

    nested = models.A(b=models.B(a=models.A()))

    assert isinstance(nested, models.A)
    assert isinstance(nested.b, models.B)
    assert isinstance(nested.b.a, models.A)


def test_recursive_oneof_union_generates_imports_and_validates(tmp_path):
    spec = _openapi_spec(
        schemas={
            'LiteralExpr': {
                'type': 'object',
                'required': ['value'],
                'properties': {'value': {'type': 'string'}},
            },
            'Group': {
                'type': 'object',
                'properties': {'expr': {'$ref': '#/components/schemas/Expr'}},
            },
            'Expr': {
                'oneOf': [
                    {'$ref': '#/components/schemas/LiteralExpr'},
                    {'$ref': '#/components/schemas/Group'},
                ]
            },
        },
        response_ref='#/components/schemas/Group',
        operation_id='getGroup',
    )

    models, _ = _generate_models_module(tmp_path, spec, 'recursive_oneof_pkg')

    group = models.Group(expr=models.Group(expr=models.LiteralExpr(value='leaf')))

    assert isinstance(group, models.Group)
    assert isinstance(group.expr, models.Group)
    assert isinstance(group.expr.expr, models.LiteralExpr)


def test_allof_inheritance_cycle_emits_base_before_subclass(tmp_path):
    spec = _openapi_spec(
        schemas={
            'Base': {
                'type': 'object',
                'properties': {'child': {'$ref': '#/components/schemas/Sub'}},
            },
            'Sub': {
                'allOf': [
                    {'$ref': '#/components/schemas/Base'},
                    {
                        'type': 'object',
                        'properties': {'parent': {'$ref': '#/components/schemas/Base'}},
                    },
                ]
            },
        },
        response_ref='#/components/schemas/Base',
        operation_id='getBase',
    )

    _, models_source = _generate_models_module(tmp_path, spec, 'recursive_allof_pkg')

    # allOf subclasses may have extra bases (e.g. "class Sub(Base, UnnamedModel):");
    # match only the start of the class definition to tolerate that.
    assert models_source.index('class Base') < models_source.index('class Sub(Base')


def test_get_sorted_types_tolerates_cycles():
    spec = {
        'openapi': '3.0.0',
        'info': {'title': 'Cycle Sort Test', 'version': '1.0.0'},
        'paths': {},
    }
    typegen = _create_type_generator(spec)
    typegen.types['A'] = Type(
        reference=None,
        name='A',
        type='model',
        dependencies={'B'},
    )
    typegen.types['B'] = Type(
        reference=None,
        name='B',
        type='model',
        dependencies={'A'},
    )

    sorted_types = typegen.get_sorted_types()

    assert isinstance(sorted_types, list)
    assert {type_.name for type_ in sorted_types} == {'A', 'B'}


def test_recursive_models_source_uses_future_annotations_and_model_rebuild(tmp_path):
    spec = _openapi_spec(
        schemas={
            'A': {
                'type': 'object',
                'properties': {'b': {'$ref': '#/components/schemas/B'}},
            },
            'B': {
                'type': 'object',
                'properties': {'a': {'$ref': '#/components/schemas/A'}},
            },
        },
        response_ref='#/components/schemas/A',
        operation_id='getRecursiveA',
    )

    _, models_source = _generate_models_module(
        tmp_path, spec, 'recursive_source_guards_pkg'
    )

    assert models_source.startswith('from __future__ import annotations\n')
    assert 'A.model_rebuild()' in models_source
    assert 'B.model_rebuild()' in models_source
