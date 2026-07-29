"""Regression tests for generated-code correctness bugs surfaced by real specs.

Type-checking the *output* against the real Petstore spec (see
``test_generated_typecheck.py``) surfaced three base-codegen bugs that the small
hand-crafted fixtures never exercised. These tests pin the fixes directly, in the
normal (pyright-free) suite, so a regression is caught even when the type-check
gate is not installed:

1. An optional (not-``required``) model field is typed ``T | None`` -- not a bare
   ``T`` it can never actually hold when omitted (defaults to ``None``).
2. An optional request body is serialized with a ``None`` guard, so calling the
   endpoint without a body cannot raise ``AttributeError`` on ``.model_dump()``.
3. A scalar ``str`` / ``bytes`` response is extracted from the httpx Response
   (``.text`` / ``.content``) instead of returning the Response object while
   claiming to return ``str`` / ``bytes``.
"""

import ast
import json

from otterapi.codegen.codegen import Codegen
from otterapi.config import DocumentConfig


def _generate(tmp_path, spec: dict, **overrides) -> str:
    """Generate a functions-style client and return the output dir."""
    spec_file = tmp_path / 'openapi.json'
    spec_file.write_text(json.dumps(spec))
    output_dir = tmp_path / 'client'
    config = DocumentConfig.model_validate(
        {
            'source': str(spec_file),
            'output': str(output_dir),
            'base_url': 'https://example.test',
            **overrides,
        }
    )
    Codegen(config, format_output=False).generate()
    return output_dir


def _field_annotations(models_src: str, class_name: str) -> dict[str, str]:
    """Map ``field name -> unparsed annotation`` for a model class in the source."""
    tree = ast.parse(models_src)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return {
                stmt.target.id: ast.unparse(stmt.annotation)
                for stmt in node.body
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name)
            }
    raise AssertionError(f'class {class_name} not found in models.py')


class TestOptionalFieldsAreNullable:
    """Bug 1: a not-``required`` field defaults to None, so its type must admit None."""

    def _spec(self) -> dict:
        return {
            'openapi': '3.0.0',
            'info': {'title': 'T', 'version': '1'},
            'paths': {
                # Reference Widget from an endpoint so it is actually emitted
                # (unreferenced component schemas are pruned).
                '/widget': {
                    'get': {
                        'operationId': 'get_widget',
                        'responses': {
                            '200': {
                                'description': 'ok',
                                'content': {
                                    'application/json': {
                                        'schema': {
                                            '$ref': '#/components/schemas/Widget'
                                        }
                                    }
                                },
                            }
                        },
                    }
                }
            },
            'components': {
                'schemas': {
                    'Widget': {
                        'type': 'object',
                        'required': ['id'],
                        'properties': {
                            'id': {'type': 'integer'},  # required -> int
                            'name': {'type': 'string'},  # optional -> str | None
                            'tags': {  # optional list -> list[str] | None
                                'type': 'array',
                                'items': {'type': 'string'},
                            },
                            'nickname': {  # nullable+optional -> str | None (once)
                                'type': 'string',
                                'nullable': True,
                            },
                        },
                    },
                }
            },
        }

    def test_optional_field_is_unioned_with_none(self, tmp_path):
        out = _generate(tmp_path, self._spec())
        anns = _field_annotations((out / 'models.py').read_text(), 'Widget')
        assert anns['name'] == 'str | None'
        assert anns['tags'] == 'list[str] | None'

    def test_required_field_stays_non_optional(self, tmp_path):
        out = _generate(tmp_path, self._spec())
        anns = _field_annotations((out / 'models.py').read_text(), 'Widget')
        assert anns['id'] == 'int'

    def test_nullable_optional_field_not_double_wrapped(self, tmp_path):
        out = _generate(tmp_path, self._spec())
        anns = _field_annotations((out / 'models.py').read_text(), 'Widget')
        # Exactly one ``| None`` -- not ``str | None | None``.
        assert anns['nickname'] == 'str | None'


class TestOptionalBodyGuarded:
    """Bug 2: ``body.model_dump()`` on an omitted optional body must not raise."""

    def _spec(self, *, required: bool) -> dict:
        return {
            'openapi': '3.0.0',
            'info': {'title': 'T', 'version': '1'},
            'paths': {
                '/things': {
                    'post': {
                        'operationId': 'create_thing',
                        'requestBody': {
                            'required': required,
                            'content': {
                                'application/json': {
                                    'schema': {'$ref': '#/components/schemas/Thing'}
                                }
                            },
                        },
                        'responses': {'200': {'description': 'ok'}},
                    }
                }
            },
            'components': {
                'schemas': {
                    'Thing': {
                        'type': 'object',
                        'properties': {'name': {'type': 'string'}},
                    }
                }
            },
        }

    def test_optional_body_is_none_guarded(self, tmp_path):
        out = _generate(tmp_path, self._spec(required=False))
        src = (out / 'endpoints.py').read_text()
        assert 'json=body.model_dump() if body is not None else None' in src

    def test_required_body_is_not_guarded(self, tmp_path):
        out = _generate(tmp_path, self._spec(required=True))
        src = (out / 'endpoints.py').read_text()
        assert 'json=body.model_dump()' in src
        assert 'if body is not None else None' not in src


class TestScalarResponseExtraction:
    """Bug 3: a scalar response must be extracted, not returned as the Response."""

    def _spec(self) -> dict:
        def _op(op_id: str, content: dict) -> dict:
            return {
                'get': {
                    'operationId': op_id,
                    'responses': {'200': {'description': 'ok', 'content': content}},
                }
            }

        return {
            'openapi': '3.0.0',
            'info': {'title': 'T', 'version': '1'},
            'paths': {
                # JSON string body -> str (the login_user shape in Petstore).
                '/text': _op(
                    'get_text', {'application/json': {'schema': {'type': 'string'}}}
                ),
                # JSON binary string -> bytes.
                '/blob': _op(
                    'get_blob',
                    {
                        'application/json': {
                            'schema': {'type': 'string', 'format': 'binary'}
                        }
                    },
                ),
                # Non-JSON stream -> the httpx Response itself (returned as-is).
                '/raw': _op(
                    'get_raw',
                    {
                        'application/octet-stream': {
                            'schema': {'type': 'string', 'format': 'binary'}
                        }
                    },
                ),
            },
        }

    def test_str_response_returns_text(self, tmp_path):
        out = _generate(tmp_path, self._spec())
        src = (out / 'endpoints.py').read_text()
        assert 'def get_text(' in src
        # sync: c._request(...).text ; async: (await c._request_async(...)).text
        assert '.text' in src
        # The bare-Response anti-pattern (declared -> str, returns Response) is gone.
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.FunctionDef) and node.name == 'get_text':
                assert isinstance(node.returns, ast.Name) and node.returns.id == 'str'
                ret = node.body[-1]
                assert isinstance(ret, ast.Return)
                assert isinstance(ret.value, ast.Attribute)
                assert ret.value.attr == 'text'
                break
        else:  # pragma: no cover
            raise AssertionError('get_text not found')

    def test_bytes_response_returns_content(self, tmp_path):
        out = _generate(tmp_path, self._spec())
        src = (out / 'endpoints.py').read_text()
        assert '.content' in src
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.FunctionDef) and node.name == 'get_blob':
                ret = node.body[-1]
                assert isinstance(ret, ast.Return)
                assert isinstance(ret.value, ast.Attribute)
                assert ret.value.attr == 'content'
                break
        else:  # pragma: no cover
            raise AssertionError('get_blob not found')

    def test_response_typed_endpoint_returns_response_unchanged(self, tmp_path):
        out = _generate(tmp_path, self._spec())
        src = (out / 'endpoints.py').read_text()
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.FunctionDef) and node.name == 'get_raw':
                ret = node.body[-1]
                assert isinstance(ret, ast.Return)
                # returns the raw call, not an attribute access on it
                assert not isinstance(ret.value, ast.Attribute)
                break
        else:  # pragma: no cover
            raise AssertionError('get_raw not found')


class TestExportGuardsOptionalRows:
    """Making optional envelope data honest (``list[X] | None``) means an export
    wrapper over it must coerce None to ``[]`` before iterating."""

    def _spec(self, *, data_required: bool) -> dict:
        envelope = {
            'type': 'object',
            'properties': {
                'data': {
                    'type': 'array',
                    'items': {'$ref': '#/components/schemas/Item'},
                }
            },
        }
        if data_required:
            envelope['required'] = ['data']
        return {
            'openapi': '3.0.0',
            'info': {'title': 'T', 'version': '1'},
            'paths': {
                '/things': {
                    'get': {
                        'operationId': 'get_things',
                        'responses': {
                            '200': {
                                'description': 'ok',
                                'content': {
                                    'application/json': {
                                        'schema': {
                                            '$ref': '#/components/schemas/Envelope'
                                        }
                                    }
                                },
                            }
                        },
                    }
                }
            },
            'components': {
                'schemas': {
                    'Envelope': envelope,
                    'Item': {
                        'type': 'object',
                        'required': ['id'],
                        'properties': {
                            'id': {'type': 'integer'},
                            'name': {'type': 'string'},
                        },
                    },
                }
            },
        }

    _CFG = {
        'export': {'enabled': True, 'formats': ['csv', 'jsonl']},
        'response_unwrap': {'enabled': True, 'data_path': 'data'},
    }

    def test_optional_data_export_coerces_none(self, tmp_path):
        out = _generate(tmp_path, self._spec(data_required=False), **self._CFG)
        src = (out / 'endpoints.py').read_text()
        assert 'def get_things_export(' in src
        # rows = get_things(...) or []  -- guards None before export iterates.
        assert 'get_things(client=client) or []' in src

    def test_required_data_export_not_guarded(self, tmp_path):
        out = _generate(tmp_path, self._spec(data_required=True), **self._CFG)
        src = (out / 'endpoints.py').read_text()
        assert 'def get_things_export(' in src
        # data is required -> base return is a plain list -> no coercion added.
        assert 'or []' not in src
