"""Test suite for ast_utils module.

This module tests all AST helper functions that are used to generate
Python AST nodes for code generation.
"""

import ast

import pytest

from otterapi.codegen.ast_utils import (
    _all,
    _argument,
    _assign,
    _async_func,
    _attr,
    _call,
    _func,
    _import,
    _name,
    _optional_expr,
    _subscript,
    _union_expr,
)


class TestNameFunction:
    """Tests for _name() function."""

    def test_creates_name_node(self):
        """Test that _name creates an ast.Name node."""
        result = _name('foo')
        assert isinstance(result, ast.Name)

    def test_name_has_correct_id(self):
        """Test that the Name node has the correct id."""
        result = _name('my_variable')
        assert result.id == 'my_variable'

    def test_name_has_load_context(self):
        """Test that the Name node has Load context."""
        result = _name('bar')
        assert isinstance(result.ctx, ast.Load)

    @pytest.mark.parametrize(
        'name',
        ['a', 'variable_name', 'CamelCase', 'snake_case', '_private', '__dunder__'],
    )
    def test_various_valid_names(self, name):
        """Test _name with various valid Python identifiers."""
        result = _name(name)
        assert result.id == name
        assert isinstance(result, ast.Name)


class TestAttrFunction:
    """Tests for _attr() function."""

    def test_creates_attribute_node(self):
        """Test that _attr creates an ast.Attribute node."""
        result = _attr('obj', 'attr')
        assert isinstance(result, ast.Attribute)

    def test_attr_with_string_value(self):
        """Test _attr with a string value (gets converted to Name)."""
        result = _attr('myobj', 'myattr')
        assert isinstance(result.value, ast.Name)
        assert result.value.id == 'myobj'
        assert result.attr == 'myattr'

    def test_attr_with_ast_expr_value(self):
        """Test _attr with an existing ast.expr value."""
        name_node = _name('base')
        result = _attr(name_node, 'property')
        assert result.value is name_node
        assert result.attr == 'property'

    def test_chained_attributes(self):
        """Test creating chained attributes like obj.attr1.attr2."""
        first = _attr('obj', 'attr1')
        second = _attr(first, 'attr2')
        assert isinstance(second, ast.Attribute)
        assert isinstance(second.value, ast.Attribute)
        assert second.attr == 'attr2'
        assert second.value.attr == 'attr1'


class TestSubscriptFunction:
    """Tests for _subscript() function."""

    def test_creates_subscript_node(self):
        """Test that _subscript creates an ast.Subscript node."""
        inner = _name('int')
        result = _subscript('list', inner)
        assert isinstance(result, ast.Subscript)

    def test_subscript_structure(self):
        """Test the structure of a subscript node."""
        inner = _name('str')
        result = _subscript('List', inner)
        assert isinstance(result.value, ast.Name)
        assert result.value.id == 'List'
        assert result.slice is inner

    def test_nested_subscript(self):
        """Test nested subscripts like List[Dict[str, int]]."""
        str_node = _name('str')
        int_node = _name('int')
        dict_node = _subscript('Dict', ast.Tuple(elts=[str_node, int_node]))
        result = _subscript('List', dict_node)
        assert isinstance(result, ast.Subscript)
        assert isinstance(result.slice, ast.Subscript)


class TestUnionExprFunction:
    """Tests for _union_expr() function."""

    def test_creates_union_subscript(self):
        """Test that _union_expr creates a Union subscript."""
        types = [_name('int'), _name('str')]
        result = _union_expr(types)
        assert isinstance(result, ast.Subscript)
        assert result.value.id == 'Union'

    def test_union_contains_tuple_of_types(self):
        """Test that Union contains a tuple of the provided types."""
        types = [_name('int'), _name('str'), _name('bool')]
        result = _union_expr(types)
        assert isinstance(result.slice, ast.Tuple)
        assert len(result.slice.elts) == 3

    def test_union_with_two_types(self):
        """Test Union with two types."""
        types = [_name('int'), _name('None')]
        result = _union_expr(types)
        assert result.value.id == 'Union'
        assert len(result.slice.elts) == 2

    def test_union_with_many_types(self):
        """Test Union with many types."""
        types = [_name(t) for t in ['int', 'str', 'float', 'bool', 'None']]
        result = _union_expr(types)
        assert len(result.slice.elts) == 5


class TestOptionalExprFunction:
    """Tests for _optional_expr() function."""

    def test_creates_optional_subscript(self):
        """Test that _optional_expr creates an Optional subscript."""
        inner = _name('str')
        result = _optional_expr(inner)
        assert isinstance(result, ast.Subscript)
        assert result.value.id == 'Optional'

    def test_optional_with_simple_type(self):
        """Test Optional with a simple type."""
        inner = _name('int')
        result = _optional_expr(inner)
        assert result.slice is inner

    def test_optional_with_complex_type(self):
        """Test Optional with a complex type like List[str]."""
        list_str = _subscript('List', _name('str'))
        result = _optional_expr(list_str)
        assert isinstance(result.slice, ast.Subscript)
        assert result.value.id == 'Optional'


class TestArgumentFunction:
    """Tests for _argument() function."""

    def test_creates_arg_node(self):
        """Test that _argument creates an ast.arg node."""
        result = _argument('param')
        assert isinstance(result, ast.arg)

    def test_argument_without_annotation(self):
        """Test argument without type annotation."""
        result = _argument('x')
        assert result.arg == 'x'
        assert result.annotation is None

    def test_argument_with_annotation(self):
        """Test argument with type annotation."""
        annotation = _name('int')
        result = _argument('x', annotation)
        assert result.arg == 'x'
        assert result.annotation is annotation

    @pytest.mark.parametrize(
        'name',
        ['arg', 'parameter', 'self', 'cls', 'value', '_private_arg'],
    )
    def test_various_argument_names(self, name):
        """Test _argument with various parameter names."""
        result = _argument(name)
        assert result.arg == name


class TestAssignFunction:
    """Tests for _assign() function."""

    def test_creates_assign_node(self):
        """Test that _assign creates an ast.Assign node."""
        target = _name('x')
        value = ast.Constant(value=42)
        result = _assign(target, value)
        assert isinstance(result, ast.Assign)

    def test_assign_structure(self):
        """Test the structure of an assignment."""
        target = _name('result')
        value = ast.Constant(value='hello')
        result = _assign(target, value)
        assert len(result.targets) == 1
        # Target gets converted to Store context, so check the id instead
        assert isinstance(result.targets[0], ast.Name)
        assert result.targets[0].id == 'result'
        assert isinstance(result.targets[0].ctx, ast.Store)
        assert result.value is value

    def test_assign_with_complex_value(self):
        """Test assignment with complex value expression."""
        target = _name('data')
        value = _call(_name('dict'))
        result = _assign(target, value)
        assert isinstance(result.value, ast.Call)

    def test_assign_with_attribute_target(self):
        """Test assignment to an attribute (e.g., obj.attr = value)."""
        target = _attr('obj', 'attr')
        value = ast.Constant(value=100)
        result = _assign(target, value)
        assert len(result.targets) == 1
        assert isinstance(result.targets[0], ast.Attribute)
        assert isinstance(result.targets[0].ctx, ast.Store)
        assert result.targets[0].attr == 'attr'


class TestImportFunction:
    """Tests for _import() function."""

    def test_creates_importfrom_node(self):
        """Test that _import creates an ast.ImportFrom node."""
        result = _import('typing', ['List'])
        assert isinstance(result, ast.ImportFrom)

    def test_import_single_name(self):
        """Test importing a single name from a module."""
        result = _import('os', ['path'])
        assert result.module == 'os'
        assert len(result.names) == 1
        assert result.names[0].name == 'path'

    def test_import_multiple_names(self):
        """Test importing multiple names from a module."""
        result = _import('typing', ['Dict', 'List', 'Optional'])
        assert result.module == 'typing'
        assert len(result.names) == 3
        assert [alias.name for alias in result.names] == ['Dict', 'List', 'Optional']

    def test_import_level_zero(self):
        """Test that import level is 0 (absolute import)."""
        result = _import('collections', ['defaultdict'])
        assert result.level == 0

    def test_import_creates_aliases(self):
        """Test that imported names are ast.alias objects."""
        result = _import('json', ['dumps', 'loads'])
        for alias in result.names:
            assert isinstance(alias, ast.alias)


class TestCallFunction:
    """Tests for _call() function."""

    def test_creates_call_node(self):
        """Test that _call creates an ast.Call node."""
        func = _name('print')
        result = _call(func)
        assert isinstance(result, ast.Call)

    def test_call_without_args(self):
        """Test function call without arguments."""
        func = _name('foo')
        result = _call(func)
        assert result.func is func
        assert result.args == []
        assert result.keywords == []

    def test_call_with_positional_args(self):
        """Test function call with positional arguments."""
        func = _name('max')
        args = [ast.Constant(value=1), ast.Constant(value=2)]
        result = _call(func, args=args)
        assert len(result.args) == 2
        assert result.args == args

    def test_call_with_keywords(self):
        """Test function call with keyword arguments."""
        func = _name('dict')
        keywords = [ast.keyword(arg='key', value=ast.Constant(value='value'))]
        result = _call(func, keywords=keywords)
        assert len(result.keywords) == 1
        assert result.keywords == keywords

    def test_call_with_args_and_keywords(self):
        """Test function call with both positional and keyword arguments."""
        func = _name('sorted')
        args = [_name('items')]
        keywords = [ast.keyword(arg='reverse', value=ast.Constant(value=True))]
        result = _call(func, args=args, keywords=keywords)
        assert len(result.args) == 1
        assert len(result.keywords) == 1

    def test_call_with_attribute_func(self):
        """Test calling a method (attribute function)."""
        func = _attr('obj', 'method')
        result = _call(func)
        assert isinstance(result.func, ast.Attribute)


class TestFuncFunction:
    """Tests for _func() function."""

    def test_creates_functiondef_node(self):
        """Test that _func creates an ast.FunctionDef node."""
        body = [ast.Pass()]
        result = _func('foo', [], body)
        assert isinstance(result, ast.FunctionDef)

    def test_func_basic_structure(self):
        """Test basic function structure."""
        body = [ast.Return(value=ast.Constant(value=42))]
        result = _func('get_answer', [], body)
        assert result.name == 'get_answer'
        assert result.body == body
        assert result.decorator_list == []

    def test_func_with_args(self):
        """Test function with arguments."""
        args = [_argument('x'), _argument('y')]
        body = [ast.Pass()]
        result = _func('add', args, body)
        assert len(result.args.args) == 2
        assert result.args.args[0].arg == 'x'
        assert result.args.args[1].arg == 'y'

    def test_func_with_return_annotation(self):
        """Test function with return type annotation."""
        body = [ast.Pass()]
        returns = _name('int')
        result = _func('compute', [], body, returns=returns)
        assert result.returns is returns

    def test_func_with_kwargs(self):
        """Test function with **kwargs parameter."""
        body = [ast.Pass()]
        kwargs = _argument('kwargs')
        result = _func('process', [], body, kwargs=kwargs)
        assert result.args.kwarg is kwargs

    def test_func_with_kwonly_args(self):
        """Test function with keyword-only arguments."""
        body = [ast.Pass()]
        kwonlyargs = [_argument('force'), _argument('verbose')]
        kw_defaults = [ast.Constant(value=False), ast.Constant(value=True)]
        result = _func('run', [], body, kwonlyargs=kwonlyargs, kw_defaults=kw_defaults)
        assert len(result.args.kwonlyargs) == 2
        assert len(result.args.kw_defaults) == 2

    def test_func_arguments_structure(self):
        """Test that function arguments have correct structure."""
        args = [_argument('a')]
        body = [ast.Pass()]
        result = _func('test', args, body)
        assert isinstance(result.args, ast.arguments)
        assert result.args.posonlyargs == []
        assert result.args.defaults == []

    def test_func_complex_signature(self):
        """Test function with complex signature."""
        args = [_argument('x', _name('int')), _argument('y', _name('str'))]
        body = [ast.Return(value=_name('x'))]
        returns = _name('int')
        kwargs = _argument('options')
        result = _func('complex_func', args, body, returns=returns, kwargs=kwargs)
        assert len(result.args.args) == 2
        assert result.returns is returns
        assert result.args.kwarg is kwargs


class TestAsyncFuncFunction:
    """Tests for _async_func() function."""

    def test_creates_async_functiondef_node(self):
        """Test that _async_func creates an ast.AsyncFunctionDef node."""
        body = [ast.Pass()]
        result = _async_func('async_foo', [], body)
        assert isinstance(result, ast.AsyncFunctionDef)

    def test_async_func_basic_structure(self):
        """Test basic async function structure."""
        body = [ast.Return(value=ast.Constant(value='result'))]
        result = _async_func('fetch_data', [], body)
        assert result.name == 'fetch_data'
        assert result.body == body
        assert result.decorator_list == []

    def test_async_func_with_args(self):
        """Test async function with arguments."""
        args = [_argument('url'), _argument('timeout')]
        body = [ast.Pass()]
        result = _async_func('fetch', args, body)
        assert len(result.args.args) == 2

    def test_async_func_with_return_annotation(self):
        """Test async function with return type annotation."""
        body = [ast.Pass()]
        returns = _subscript('Awaitable', _name('dict'))
        result = _async_func('get_json', [], body, returns=returns)
        assert result.returns is returns

    def test_async_func_with_kwargs(self):
        """Test async function with **kwargs."""
        body = [ast.Pass()]
        kwargs = _argument('kwargs')
        result = _async_func('async_process', [], body, kwargs=kwargs)
        assert result.args.kwarg is kwargs

    def test_async_func_with_kwonly_args(self):
        """Test async function with keyword-only arguments."""
        body = [ast.Pass()]
        kwonlyargs = [_argument('retry')]
        kw_defaults = [ast.Constant(value=True)]
        result = _async_func(
            'async_run', [], body, kwonlyargs=kwonlyargs, kw_defaults=kw_defaults
        )
        assert len(result.args.kwonlyargs) == 1
        assert len(result.args.kw_defaults) == 1


class TestAllFunction:
    """Tests for _all() function."""

    def test_creates_assign_node(self):
        """Test that _all creates an ast.Assign node."""
        result = _all(['foo', 'bar'])
        assert isinstance(result, ast.Assign)

    def test_all_target_is_dunder_all(self):
        """Test that __all__ is the assignment target."""
        result = _all(['module_func'])
        assert len(result.targets) == 1
        target = result.targets[0]
        assert isinstance(target, ast.Name)
        assert target.id == '__all__'

    def test_all_value_is_tuple(self):
        """Test that the value is a tuple."""
        result = _all(['func1', 'func2', 'Class1'])
        assert isinstance(result.value, ast.Tuple)
        assert isinstance(result.value.ctx, ast.Load)

    def test_all_contains_string_constants(self):
        """Test that tuple elements are string constants."""
        names = ['public_func', 'PublicClass', 'PUBLIC_CONSTANT']
        result = _all(names)
        elements = result.value.elts
        assert len(elements) == 3
        for i, name in enumerate(names):
            assert isinstance(elements[i], ast.Constant)
            assert elements[i].value == name

    def test_all_with_empty_list(self):
        """Test _all with an empty list."""
        result = _all([])
        assert len(result.value.elts) == 0

    def test_all_with_single_name(self):
        """Test _all with a single exported name."""
        result = _all(['only_export'])
        assert len(result.value.elts) == 1
        assert result.value.elts[0].value == 'only_export'

    def test_all_with_iterable(self):
        """Test _all with different iterable types."""
        # Test with generator
        result = _all(name for name in ['a', 'b'])
        assert len(result.value.elts) == 2
        
        # Test with set
        result = _all({'x', 'y', 'z'})
        assert len(result.value.elts) == 3


class TestASTNodeCompilation:
    """Integration tests to verify generated AST nodes can be compiled."""

    def test_compile_simple_function(self):
        """Test that generated function AST can be compiled."""
        func = _func(
            'simple',
            [_argument('x', _name('int'))],
            [ast.Return(value=_name('x'))],
            returns=_name('int'),
        )
        module = ast.Module(body=[func], type_ignores=[])
        ast.fix_missing_locations(module)
        code = compile(module, '<test>', 'exec')
        assert code is not None

    def test_compile_async_function(self):
        """Test that generated async function AST can be compiled."""
        func = _async_func(
            'async_simple',
            [_argument('x')],
            [ast.Return(value=_name('x'))],
        )
        module = ast.Module(body=[func], type_ignores=[])
        ast.fix_missing_locations(module)
        code = compile(module, '<test>', 'exec')
        assert code is not None

    def test_compile_import_statement(self):
        """Test that generated import AST can be compiled."""
        import_node = _import('typing', ['List', 'Dict'])
        module = ast.Module(body=[import_node], type_ignores=[])
        ast.fix_missing_locations(module)
        code = compile(module, '<test>', 'exec')
        assert code is not None

    def test_compile_assignment(self):
        """Test that generated assignment AST can be compiled."""
        assign = _assign(_name('x'), ast.Constant(value=10))
        module = ast.Module(body=[assign], type_ignores=[])
        ast.fix_missing_locations(module)
        code = compile(module, '<test>', 'exec')
        assert code is not None

    def test_compile_all_export(self):
        """Test that generated __all__ AST can be compiled."""
        all_node = _all(['func1', 'func2', 'Class1'])
        module = ast.Module(body=[all_node], type_ignores=[])
        ast.fix_missing_locations(module)
        code = compile(module, '<test>', 'exec')
        assert code is not None

    def test_compile_complex_type_annotation(self):
        """Test that complex type annotations compile correctly."""
        # Create Dict[str, Optional[List[int]]]
        list_int = _subscript('List', _name('int'))
        optional_list = _optional_expr(list_int)
        dict_type = _subscript('Dict', ast.Tuple(elts=[_name('str'), optional_list], ctx=ast.Load()))
        
        func = _func(
            'complex_type',
            [_argument('data', dict_type)],
            [ast.Pass()],
        )
        module = ast.Module(body=[func], type_ignores=[])
        ast.fix_missing_locations(module)
        code = compile(module, '<test>', 'exec')
        assert code is not None


class TestASTNodeExecution:
    """Integration tests that execute generated code."""

    def test_execute_simple_function(self):
        """Test executing a generated function."""
        func = _func(
            'add',
            [_argument('a'), _argument('b')],
            [ast.Return(value=ast.BinOp(left=_name('a'), op=ast.Add(), right=_name('b')))],
        )
        module = ast.Module(body=[func], type_ignores=[])
        ast.fix_missing_locations(module)
        
        namespace = {}
        exec(compile(module, '<test>', 'exec'), namespace)
        
        assert 'add' in namespace
        assert namespace['add'](2, 3) == 5

    def test_execute_all_export(self):
        """Test executing __all__ definition."""
        all_node = _all(['foo', 'bar', 'baz'])
        module = ast.Module(body=[all_node], type_ignores=[])
        ast.fix_missing_locations(module)
        
        namespace = {}
        exec(compile(module, '<test>', 'exec'), namespace)
        
        assert '__all__' in namespace
        assert namespace['__all__'] == ('foo', 'bar', 'baz')

    def test_execute_assignment(self):
        """Test executing an assignment."""
        assign = _assign(_name('result'), ast.Constant(value=42))
        module = ast.Module(body=[assign], type_ignores=[])
        ast.fix_missing_locations(module)
        
        namespace = {}
        exec(compile(module, '<test>', 'exec'), namespace)
        
        assert 'result' in namespace
        assert namespace['result'] == 42

    def test_execute_function_call(self):
        """Test executing a function call."""
        # Create: result = max(5, 3)
        call = _call(_name('max'), args=[ast.Constant(value=5), ast.Constant(value=3)])
        assign = _assign(_name('result'), call)
        module = ast.Module(body=[assign], type_ignores=[])
        ast.fix_missing_locations(module)
        
        namespace = {'max': max}
        exec(compile(module, '<test>', 'exec'), namespace)
        
        assert namespace['result'] == 5


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_name_with_unicode(self):
        """Test name with unicode characters (valid in Python 3)."""
        result = _name('variable_α')
        assert result.id == 'variable_α'

    def test_empty_function_body_requires_pass(self):
        """Test that empty function body with Pass statement works."""
        func = _func('empty', [], [ast.Pass()])
        assert len(func.body) == 1
        assert isinstance(func.body[0], ast.Pass)

    def test_union_with_single_type(self):
        """Test Union with a single type (unusual but valid)."""
        result = _union_expr([_name('str')])
        assert len(result.slice.elts) == 1

    def test_call_with_none_defaults(self):
        """Test _call when explicitly passing None for defaults."""
        func = _name('test')
        result = _call(func, args=None, keywords=None)
        assert result.args == []
        assert result.keywords == []

    def test_func_with_none_defaults(self):
        """Test _func when explicitly passing None for optional params."""
        result = _func('test', [], [ast.Pass()], returns=None, kwargs=None)
        assert result.returns is None
        assert result.args.kwarg is None

    def test_nested_function_definitions(self):
        """Test creating nested function definitions."""
        inner = _func('inner', [], [ast.Return(value=ast.Constant(value=1))])
        outer = _func('outer', [], [inner, ast.Return(value=_call(_name('inner')))])
        assert len(outer.body) == 2
        assert isinstance(outer.body[0], ast.FunctionDef)

