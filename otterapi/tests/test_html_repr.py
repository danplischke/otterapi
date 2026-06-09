"""Tests for _HtmlReprMixin and _html_val injected into generated models."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
from pydantic import BaseModel

FIXTURE_MODELS = (
    Path(__file__).parent
    / 'fixtures'
    / 'golden'
    / 'constraints'
    / 'expected'
    / 'models.py'
)


def _load_fixture_models():
    key = '_golden_constraints_models'
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, FIXTURE_MODELS)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[key] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope='module')
def mixin_ns():
    return _load_fixture_models()


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

        result = mixin_ns._html_val(Nested())
        assert '<b>nested</b>' == result

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

        obj = NullableModel()
        html = obj._repr_html_()
        assert '<em' in html

    def test_mixin_is_base_class_of_generated_model(self, mixin_ns):
        assert issubclass(mixin_ns.User, mixin_ns._HtmlReprMixin)

    def test_works_on_plain_pydantic_subclass(self, mixin_ns):
        """Mixin should work on any Pydantic model, not just the generated one."""

        class SimpleModel(mixin_ns._HtmlReprMixin, BaseModel):
            x: int
            y: str

        obj = SimpleModel(x=1, y='hello')
        html = obj._repr_html_()
        assert 'SimpleModel' in html
        assert 'x' in html
        assert 'y' in html
        assert '1' in html
        assert 'hello' in html
