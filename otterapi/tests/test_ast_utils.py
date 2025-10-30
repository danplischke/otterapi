"""Test AST utility functions."""

import ast

from otterapi.codegen.ast_utils import (
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


class TestASTUtilities:
    """Test AST utility functions."""

    def test_name(self):
        """Test _name function."""
        name_node = _name('variable')

        assert isinstance(name_node, ast.Name)
        assert name_node.id == 'variable'
        assert isinstance(name_node.ctx, ast.Load)

    def test_attr(self):
        """Test _attr function."""
        # Test with string value
        attr_node = _attr('obj', 'method')
        assert isinstance(attr_node, ast.Attribute)
        assert isinstance(attr_node.value, ast.Name)
        assert attr_node.value.id == 'obj'
        assert attr_node.attr == 'method'

        # Test with ast.expr value
        name_node = _name('obj')
        attr_node = _attr(name_node, 'method')
        assert isinstance(attr_node, ast.Attribute)
        assert attr_node.value == name_node
        assert attr_node.attr == 'method'

    def test_subscript(self):
        """Test _subscript function."""
        inner_expr = _name('str')
        subscript_node = _subscript('List', inner_expr)

        assert isinstance(subscript_node, ast.Subscript)
        assert isinstance(subscript_node.value, ast.Name)
        assert subscript_node.value.id == 'List'
        assert subscript_node.slice == inner_expr

    def test_union_expr(self):
        """Test _union_expr function."""
        type1 = _name('str')
        type2 = _name('int')
        union_node = _union_expr([type1, type2])

        assert isinstance(union_node, ast.Subscript)
        assert isinstance(union_node.value, ast.Name)
        assert union_node.value.id == 'Union'
        assert isinstance(union_node.slice, ast.Tuple)
        assert len(union_node.slice.elts) == 2
        assert union_node.slice.elts[0] == type1
        assert union_node.slice.elts[1] == type2

    def test_optional_expr(self):
        """Test _optional_expr function."""
        inner_type = _name('str')
        optional_node = _optional_expr(inner_type)

        assert isinstance(optional_node, ast.Subscript)
        assert isinstance(optional_node.value, ast.Name)
        assert optional_node.value.id == 'Optional'
        assert optional_node.slice == inner_type

    def test_argument(self):
        """Test _argument function."""
        # Test without annotation
        arg_node = _argument('param')
        assert isinstance(arg_node, ast.arg)
        assert arg_node.arg == 'param'
        assert arg_node.annotation is None

        # Test with annotation
        annotation = _name('str')
        arg_node = _argument('param', annotation)
        assert arg_node.arg == 'param'
        assert arg_node.annotation == annotation

    def test_assign(self):
        """Test _assign function."""
        target = _name('variable')
        value = ast.Constant('value')
        assign_node = _assign(target, value)

        assert isinstance(assign_node, ast.Assign)
        assert len(assign_node.targets) == 1
        assert assign_node.targets[0] == target
        assert assign_node.value == value

    def test_import(self):
        """Test _import function."""
        import_node = _import('typing', ['List', 'Dict', 'Optional'])

        assert isinstance(import_node, ast.ImportFrom)
        assert import_node.module == 'typing'
        assert import_node.level == 0
        assert len(import_node.names) == 3
        assert all(isinstance(name, ast.alias) for name in import_node.names)
        assert [name.name for name in import_node.names] == ['List', 'Dict', 'Optional']

    def test_call(self):
        """Test _call function."""
        func = _name('print')

        # Test without arguments
        call_node = _call(func)
        assert isinstance(call_node, ast.Call)
        assert call_node.func == func
        assert call_node.args == []
        assert call_node.keywords == []

        # Test with arguments
        args = [ast.Constant('hello'), ast.Constant('world')]
        keywords = [ast.keyword(arg='sep', value=ast.Constant(' '))]
        call_node = _call(func, args, keywords)
        assert call_node.args == args
        assert call_node.keywords == keywords

    def test_func(self):
        """Test _func function."""
        args = [_argument('x'), _argument('y', _name('int'))]
        body = [ast.Return(_name('x'))]
        returns = _name('int')

        func_node = _func('add', args, body, returns)

        assert isinstance(func_node, ast.FunctionDef)
        assert func_node.name == 'add'
        assert len(func_node.args.args) == 2
        assert func_node.args.args[0].arg == 'x'
        assert func_node.args.args[1].arg == 'y'
        assert func_node.body == body
        assert func_node.returns == returns
        assert func_node.decorator_list == []

    def test_func_with_kwargs(self):
        """Test _func function with keyword arguments."""
        args = [_argument('x')]
        body = [ast.Return(_name('x'))]
        kwargs = _argument('kwargs')
        kwonlyargs = [_argument('option', _name('bool'))]
        kw_defaults = [ast.Constant(True)]

        func_node = _func(
            'test',
            args,
            body,
            kwargs=kwargs,
            kwonlyargs=kwonlyargs,
            kw_defaults=kw_defaults,
        )

        assert func_node.args.kwarg == kwargs
        assert len(func_node.args.kwonlyargs) == 1
        assert func_node.args.kwonlyargs[0].arg == 'option'
        assert len(func_node.args.kw_defaults) == 1

    def test_async_func(self):
        """Test _async_func function."""
        args = [_argument('x')]
        body = [ast.Return(_name('x'))]
        returns = _name('int')

        func_node = _async_func('async_add', args, body, returns)

        assert isinstance(func_node, ast.AsyncFunctionDef)
        assert func_node.name == 'async_add'
        assert len(func_node.args.args) == 1
        assert func_node.args.args[0].arg == 'x'
        assert func_node.body == body
        assert func_node.returns == returns

    def test_async_func_with_kwargs(self):
        """Test _async_func function with keyword arguments."""
        args = [_argument('x')]
        body = [ast.Return(_name('x'))]
        kwargs = _argument('kwargs')
        kwonlyargs = [_argument('option')]
        kw_defaults = [ast.Constant(None)]

        func_node = _async_func(
            'async_test',
            args,
            body,
            kwargs=kwargs,
            kwonlyargs=kwonlyargs,
            kw_defaults=kw_defaults,
        )

        assert func_node.args.kwarg == kwargs
        assert len(func_node.args.kwonlyargs) == 1
        assert len(func_node.args.kw_defaults) == 1


class TestASTCodeGeneration:
    """Test generating complete AST structures."""

    def test_complex_function_with_union_types(self):
        """Test creating a complex function with union type annotations."""
        # Create function: def process(data: Union[str, int]) -> Optional[str]:
        param_annotation = _union_expr([_name('str'), _name('int')])
        return_annotation = _optional_expr(_name('str'))

        args = [_argument('data', param_annotation)]
        body = [
            ast.If(
                test=_call(_name('isinstance'), [_name('data'), _name('str')]),
                body=[ast.Return(_name('data'))],
                orelse=[ast.Return(ast.Constant(None))],
            )
        ]

        func_node = _func('process', args, body, return_annotation)

        # Verify the structure
        assert func_node.name == 'process'
        assert len(func_node.args.args) == 1
        assert isinstance(func_node.args.args[0].annotation, ast.Subscript)
        assert isinstance(func_node.returns, ast.Subscript)

    def test_class_method_generation(self):
        """Test generating a class method with proper self parameter."""
        args = [_argument('self'), _argument('value', _name('str'))]
        body = [_assign(_attr('self', 'value'), _name('value'))]

        method_node = _func('set_value', args, body)

        assert method_node.args.args[0].arg == 'self'
        assert method_node.args.args[1].arg == 'value'
        assert isinstance(method_node.body[0], ast.Assign)

    def test_async_method_with_await(self):
        """Test generating async method with await calls."""
        args = [_argument('self'), _argument('url', _name('str'))]
        body = [
            _assign(
                _name('response'),
                ast.Await(_call(_attr('self', 'client'), [_name('url')])),
            ),
            ast.Return(_name('response')),
        ]

        async_method = _async_func('fetch', args, body, _name('dict'))

        assert isinstance(async_method, ast.AsyncFunctionDef)
        assert isinstance(async_method.body[0].value, ast.Await)


def test_ast_nodes_are_valid():
    """Test that generated AST nodes can be compiled."""
    # Create a simple function
    func_node = _func(
        'test_func',
        [_argument('x', _name('int'))],
        [ast.Return(_name('x'))],
        _name('int'),
    )

    # Create a module with the function
    module = ast.Module(body=[func_node], type_ignores=[])
    ast.fix_missing_locations(module)

    # Should compile without errors
    code = compile(module, '<test>', 'exec')
    assert code is not None


def test_complex_ast_structure():
    """Test creating a complex AST structure that mimics real code generation."""
    # Generate: from typing import List, Optional, Union
    import_stmt = _import('typing', ['List', 'Optional', 'Union'])

    # Generate: def complex_func(items: List[Union[str, int]]) -> Optional[List[str]]:
    param_type = _subscript('List', _union_expr([_name('str'), _name('int')]))
    return_type = _optional_expr(_subscript('List', _name('str')))

    args = [_argument('items', param_type)]
    body = [
        _assign(
            _name('result'),
            ast.ListComp(
                elt=_call(_name('str'), [_name('item')]),
                generators=[
                    ast.comprehension(
                        target=_name('item'),
                        iter=_name('items'),
                        ifs=[_call(_name('isinstance'), [_name('item'), _name('str')])],
                        is_async=0,
                    )
                ],
            ),
        ),
        ast.Return(_name('result')),
    ]

    func_node = _func('complex_func', args, body, return_type)

    # Create module and verify it compiles
    module = ast.Module(body=[import_stmt, func_node], type_ignores=[])
    ast.fix_missing_locations(module)

    code = compile(module, '<test>', 'exec')
    assert code is not None

    # Verify the generated code makes sense
    generated_code = ast.unparse(module)
    assert 'from typing import' in generated_code
    assert 'def complex_func' in generated_code
    assert 'List[Union[str, int]]' in generated_code
    assert 'Optional[List[str]]' in generated_code
