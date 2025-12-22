import ast
import keyword
from collections.abc import Iterable

PYTHON_KEYWORDS = set(keyword.kwlist)


def _name(name: str) -> ast.Name:
    return ast.Name(id=name, ctx=ast.Load())


def _attr(value: str | ast.expr, attr: str) -> ast.Attribute:
    return ast.Attribute(
        value=_name(value) if isinstance(value, str) else value, attr=attr
    )


def _subscript(generic: str, inner: ast.expr) -> ast.Subscript:
    return ast.Subscript(value=_name(generic), slice=inner, ctx=ast.Load())


def _union_expr(types: list[ast.expr]) -> ast.Subscript:
    # Union[A, B, C]
    return _subscript('Union', ast.Tuple(elts=types))


def _optional_expr(inner: ast.expr) -> ast.Subscript:
    return _subscript('Optional', inner)


def _argument(name: str, value: ast.expr | None = None) -> ast.arg:
    return ast.arg(
        arg=name,
        annotation=value,
    )


def _assign(target: ast.expr, value: ast.expr) -> ast.Assign:
    # Ensure target has Store context
    if isinstance(target, ast.Name):
        target = ast.Name(id=target.id, ctx=ast.Store())
    elif isinstance(target, ast.Attribute):
        # For attributes, only the outermost needs Store context
        target.ctx = ast.Store()
    return ast.Assign(
        targets=[target],
        value=value,
    )


def _import(module: str, names: list[str]) -> ast.ImportFrom:
    return ast.ImportFrom(
        module=module,
        names=[ast.alias(name=name) for name in names],
        level=0,
    )


def _call(
        func: ast.expr,
        args: list[ast.expr] | None = None,
        keywords: list[ast.keyword] | None = None,
) -> ast.Call:
    return ast.Call(
        func=func,
        args=args or [],
        keywords=keywords or [],
    )


def _func(
        name: str,
        args: list[ast.arg],
        body: list[ast.stmt],
        returns: ast.expr | None = None,
        kwargs: ast.arg = None,
        kwonlyargs: list[ast.arg] = None,
        kw_defaults: list[ast.expr] = None,
) -> ast.FunctionDef:
    return ast.FunctionDef(
        name=name,
        args=ast.arguments(
            posonlyargs=[],
            args=args,
            kwarg=kwargs,
            kwonlyargs=kwonlyargs or [],
            kw_defaults=kw_defaults or [],
            defaults=[],
        ),
        body=body,
        decorator_list=[],
        returns=returns,
    )


def _async_func(
        name: str,
        args: list[ast.arg],
        body: list[ast.stmt],
        returns: ast.expr | None = None,
        kwargs: ast.arg = None,
        kwonlyargs: list[ast.arg] = None,
        kw_defaults: list[ast.expr] = None,
) -> ast.AsyncFunctionDef:
    return ast.AsyncFunctionDef(
        name=name,
        args=ast.arguments(
            posonlyargs=[],
            args=args,
            kwarg=kwargs,
            kwonlyargs=kwonlyargs or [],
            kw_defaults=kw_defaults or [],
            defaults=[],
        ),
        body=body,
        decorator_list=[],
        returns=returns,
    )


def _all(names: Iterable[str]) -> ast.Assign:
    return _assign(
        target=_name('__all__'),
        value=ast.Tuple(
            elts=[ast.Constant(value=name) for name in names], ctx=ast.Load()
        ),
    )
