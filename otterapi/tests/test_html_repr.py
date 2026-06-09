"""Tests for _HtmlReprMixin and _html_val injected into generated models."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
from pydantic import BaseModel

from otterapi.codegen.codegen import Codegen
from otterapi.config import DocumentConfig

_SPEC = Path(__file__).parent / 'fixtures' / 'golden' / 'constraints' / 'spec.yaml'


def _load_generated_models(tmp_path_factory):
    pkg = '_html_repr_models_pkg'
    parent = tmp_path_factory.mktemp('html_repr')
    Codegen(
        DocumentConfig(
            source=str(_SPEC),
            output=str(parent / pkg),
            base_url='https://example.test',
        )
    ).generate()
    models_path = parent / pkg / 'models.py'
    key = '_html_repr_generated_models'
    spec = importlib.util.spec_from_file_location(key, models_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[key] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope='module')
def mixin_ns(tmp_path_factory):
    return _load_generated_models(tmp_path_factory)


class TestHtmlVal:
    def test_none_renders_as_muted(self, mixin_ns):
        result = mixin_ns._html_val(None)
        assert '<em' in result
        assert 'None' in result

    def test_short_list_renders_inline(self, mixin_ns):
        result = mixin_ns._html_val([1, 2, 3])
        assert '[1, 2, 3]' in result
        assert '…' not in result

    def test_long_list_truncated_with_ellipsis(self, mixin_ns):
        result = mixin_ns._html_val(list(range(10)))
        assert '…' in result
        assert '(10)' in result

    def test_nested_html_repr_is_delegated(self, mixin_ns):
        class Nested:
            def _repr_html_(self):
                return '<b>nested</b>'

        assert mixin_ns._html_val(Nested()) == '<b>nested</b>'

    def test_plain_string(self, mixin_ns):
        assert mixin_ns._html_val('hello') == 'hello'

    def test_integer(self, mixin_ns):
        assert mixin_ns._html_val(42) == '42'


class TestHtmlReprMixin:
    def test_output_is_string(self, mixin_ns):
        user = mixin_ns.User(id=1, username='alice99', age=30, tags=[])
        assert isinstance(user._repr_html_(), str)

    def test_contains_details_tag(self, mixin_ns):
        user = mixin_ns.User(id=1, username='alice99', age=30, tags=[])
        html = user._repr_html_()
        assert '<details open>' in html
        assert '</details>' in html

    def test_contains_class_name(self, mixin_ns):
        user = mixin_ns.User(id=1, username='alice99', age=30, tags=[])
        assert 'User' in user._repr_html_()

    def test_contains_field_names(self, mixin_ns):
        user = mixin_ns.User(id=7, username='bob123', age=25, tags=[])
        html = user._repr_html_()
        assert 'id' in html
        assert 'username' in html
        assert 'age' in html

    def test_field_values_appear_in_output(self, mixin_ns):
        user = mixin_ns.User(id=7, username='bob123', age=25, tags=[])
        html = user._repr_html_()
        assert 'bob123' in html
        assert '25' in html

    def test_none_field_renders_muted(self, mixin_ns):
        class NullableModel(mixin_ns._HtmlReprMixin, BaseModel):
            name: str | None = None

        html = NullableModel()._repr_html_()
        assert '<em' in html

    def test_mixin_is_base_class_of_generated_model(self, mixin_ns):
        assert issubclass(mixin_ns.User, mixin_ns._HtmlReprMixin)

    def test_works_on_plain_pydantic_subclass(self, mixin_ns):
        class SimpleModel(mixin_ns._HtmlReprMixin, BaseModel):
            x: int
            y: str

        html = SimpleModel(x=1, y='hello')._repr_html_()
        assert 'SimpleModel' in html
        assert '1' in html
        assert 'hello' in html
