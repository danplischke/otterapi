"""Tests for the OpenAPIAdapter facade between parser and codegen."""

from __future__ import annotations

from pydantic import TypeAdapter

from otterapi.codegen._openapi_adapter import OpenAPIAdapter
from otterapi.openapi import UniversalOpenAPI
from otterapi.openapi.v3_2.v3_2 import OpenAPI

SPEC = {
    'openapi': '3.0.0',
    'info': {'title': 'Petstore', 'version': '1.0.0'},
    'servers': [{'url': 'https://api.example.com'}],
    'paths': {
        '/pets': {
            'get': {
                'operationId': 'listPets',
                'responses': {
                    '200': {
                        'description': 'ok',
                        'content': {
                            'application/json': {
                                'schema': {'$ref': '#/components/schemas/Pet'},
                            },
                        },
                    },
                },
            },
            'post': {
                'operationId': 'createPet',
                'responses': {'201': {'description': 'created'}},
            },
        },
    },
    'components': {
        'schemas': {
            'Pet': {
                'type': 'object',
                'required': ['id'],
                'properties': {'id': {'type': 'integer'}},
            },
        },
    },
}


def _adapter() -> OpenAPIAdapter:
    universal = TypeAdapter(UniversalOpenAPI).validate_python(SPEC)
    doc = universal.root
    while not isinstance(doc, OpenAPI):
        doc, _ = doc.upgrade()
    return OpenAPIAdapter(doc)


class TestDocumentWideAccessors:
    def test_title_and_version(self):
        a = _adapter()
        assert a.title() == 'Petstore'
        assert a.version() == '1.0.0'

    def test_servers(self):
        a = _adapter()
        servers = a.servers()
        assert len(servers) == 1
        assert str(servers[0].url) == 'https://api.example.com'

    def test_has_paths(self):
        assert _adapter().has_paths() is True


class TestPathsAndOperations:
    def test_paths_returns_dict(self):
        paths = _adapter().paths()
        assert list(paths) == ['/pets']

    def test_operations_yields_triples(self):
        ops = list(_adapter().operations())
        assert len(ops) == 2
        methods = sorted(method for _, method, _ in ops)
        assert methods == ['get', 'post']

    def test_operations_yields_method_in_lowercase(self):
        ops = list(_adapter().operations())
        for _, method, _ in ops:
            assert method == method.lower()


class TestComponents:
    def test_schemas(self):
        schemas = _adapter().components_schemas()
        assert 'Pet' in schemas

    def test_missing_components_return_empty_or_none(self):
        spec = dict(SPEC, components=None)
        universal = TypeAdapter(UniversalOpenAPI).validate_python(spec)
        doc = universal.root
        while not isinstance(doc, OpenAPI):
            doc, _ = doc.upgrade()
        adapter = OpenAPIAdapter(doc)
        assert adapter.components_schemas() == {}
        assert adapter.components_parameter('Anything') is None
        assert adapter.components_response('Anything') is None
        assert adapter.components_request_body('Anything') is None


class TestEscapeHatch:
    def test_document_attribute_returns_raw_model(self):
        a = _adapter()
        # We deliberately expose the raw model for callers that haven't been
        # migrated yet (issue #3, item 10 follow-ups).
        from otterapi.openapi.v3_2.v3_2 import OpenAPI as OpenAPIv3_2

        assert isinstance(a.document, OpenAPIv3_2)
