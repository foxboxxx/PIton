"""Python AST -> tile IR, in the style of Triton's `code_generator.py`.

The kernel body is *executed* symbolically: constexpr values are ordinary
Python objects and everything that depends on runtime data is a
`tl.tensor` wrapping an IR value.  Python control flow on constexprs is
resolved at compile time; `for`/`if` on runtime values become `scf.for` /
`scf.if` with loop-carried / yielded variables discovered from the AST.
"""

from __future__ import annotations

import ast
import builtins
import inspect
import operator
import textwrap
from typing import Any

from . import ir
from . import language as tl
from .language.core import CompilationError, _unwrap, tensor, to_tensor


def _is_language_module(obj) -> bool:
    name = getattr(obj, "__name__", "")
    return name in ("triton.language", "piton.language", "piton.language.core") or \
        name.startswith("triton.language")


def _jit_target(obj):
    """Return the python function behind a @jit object (piton or triton)."""
    if hasattr(obj, "fn") and callable(getattr(obj, "fn")) and hasattr(obj, "src"):
        return obj.fn
    if getattr(obj, "_piton_jit", False):
        return obj.fn
    return None


class _LangProxy:
    """Stand-in for `triton.language` / `piton.language` inside kernels."""

    def __getattr__(self, item):
        if hasattr(tl, item):
            return getattr(tl, item)
        raise CompilationError(f"tl.{item} is not supported by PIton")


_LANG = _LangProxy()


class _TritonProxy:
    """Stand-in for the top-level `triton` / `piton` module inside kernels."""

    def __getattr__(self, item):
        if item == "language":
            return _LANG
        if item == "cdiv":
            return lambda a, b: (a + b - 1) // b
        if item == "next_power_of_2":
            return lambda n: 1 << (int(n) - 1).bit_length()
        raise CompilationError(f"triton.{item} is not available inside a kernel")


class _Return(Exception):
    def __init__(self, value):
        self.value = value


_BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
        ast.Pow: operator.pow, ast.LShift: operator.lshift, ast.RShift: operator.rshift,
        ast.BitAnd: operator.and_, ast.BitOr: operator.or_, ast.BitXor: operator.xor}
_CMP = {ast.Lt: operator.lt, ast.LtE: operator.le, ast.Gt: operator.gt, ast.GtE: operator.ge,
        ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Is: operator.is_,
        ast.IsNot: operator.is_not, ast.In: lambda a, b: a in b,
        ast.NotIn: lambda a, b: a not in b}


def _assigned_names(stmts) -> set:
    names = set()

    class V(ast.NodeVisitor):
        def visit_Name(self, node):
            if isinstance(node.ctx, ast.Store):
                names.add(node.id)

        def visit_FunctionDef(self, node):
            pass

        def visit_Lambda(self, node):
            pass

    for s in stmts:
        V().visit(s)
    return names


_RET = "__piton_return__"


def _contains_return(stmts) -> bool:
    class V(ast.NodeVisitor):
        found = False

        def visit_Return(self, node):
            self.found = True

        def visit_FunctionDef(self, node):
            pass

        def visit_Lambda(self, node):
            pass
    v = V()
    for st in stmts:
        v.visit(st)
    return v.found


def _ends_with_return(stmts) -> bool:
    if not stmts:
        return False
    last = stmts[-1]
    if isinstance(last, ast.Return):
        return True
    if isinstance(last, ast.If):
        return _ends_with_return(last.body) and _ends_with_return(last.orelse)
    return False


def get_source(fn):
    src = textwrap.dedent(inspect.getsource(fn))
    tree = ast.parse(src)
    fdef = tree.body[0]
    if not isinstance(fdef, ast.FunctionDef):
        raise CompilationError("expected a function definition")
    fdef.decorator_list = []
    return fdef


class CodeGenerator:
    def __init__(self, fn, builder: ir.Builder, filename: str = "<kernel>"):
        self.fn = fn
        self.builder = builder
        self.globals = dict(fn.__globals__)
        if fn.__closure__:
            for name, cell in zip(fn.__code__.co_freevars, fn.__closure__):
                try:
                    self.globals[name] = cell.cell_contents
                except ValueError:
                    pass
        self.scope: dict = {}
        self.filename = filename
        self.lineno_offset = fn.__code__.co_firstlineno - 1

    # -- helpers --------------------------------------------------------------

    def error(self, node, msg) -> CompilationError:
        line = getattr(node, "lineno", 0) + self.lineno_offset
        return CompilationError(f"{self.fn.__name__} line {line}: {msg}")

    def lookup(self, name, node):
        if name in self.scope:
            return self.scope[name]
        if name in self.globals:
            return self._wrap_global(self.globals[name])
        if hasattr(builtins, name):
            return getattr(builtins, name)
        raise self.error(node, f"name '{name}' is not defined")

    def _wrap_global(self, v):
        if inspect.ismodule(v):
            if _is_language_module(v):
                return _LANG
            if v.__name__ in ("triton", "piton"):
                return _TritonProxy()
        if isinstance(v, tl.constexpr):
            return v.value
        return v

    # -- statements -------------------------------------------------------------

    def run_body(self, stmts):
        for i, st in enumerate(stmts):
            if isinstance(st, ast.If) and _contains_return(st.body + st.orelse):
                # `if c: ... return` (Triton's early exit): the statements after the
                # if belong to whichever branches do not return
                cond = _unwrap(self.eval(st.test))
                rest = list(stmts[i + 1:])
                if not isinstance(cond, tensor):
                    self.run_body((st.body if cond else st.orelse) + rest)
                    return
                if cond.handle.shape:
                    raise self.error(st, "if-condition must be a scalar")
                then = st.body if _ends_with_return(st.body) else st.body + rest
                orelse = st.orelse if _ends_with_return(st.orelse) else st.orelse + rest
                self.runtime_if(st, cond, then, orelse)
                raise _Return(self.scope.get(_RET))
            self.visit_stmt(st)

    def visit_stmt(self, node):
        method = getattr(self, "stmt_" + type(node).__name__, None)
        if method is None:
            raise self.error(node, f"unsupported statement: {type(node).__name__}")
        try:
            method(node)
        except (CompilationError, _Return):
            raise
        except Exception as e:  # attach source location to internal errors
            raise self.error(node, f"{type(e).__name__}: {e}") from e

    def stmt_Expr(self, node):
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return  # docstring
        self.eval(node.value)

    def stmt_Pass(self, node):
        pass

    def stmt_Assign(self, node):
        value = self.eval(node.value)
        for target in node.targets:
            self.assign(target, value)

    def stmt_AnnAssign(self, node):
        if node.value is None:
            return
        value = self.eval(node.value)
        ann = node.annotation
        if (isinstance(ann, ast.Attribute) and ann.attr == "constexpr") or \
                (isinstance(ann, ast.Name) and ann.id == "constexpr"):
            value = _unwrap(value)
            if isinstance(value, tensor):
                raise self.error(node, "constexpr assigned a runtime value")
        self.assign(node.target, value)

    def stmt_AugAssign(self, node):
        if not isinstance(node.target, ast.Name):
            raise self.error(node, "augmented assignment only supported on names")
        cur = self.lookup(node.target.id, node)
        rhs = self.eval(node.value)
        self.scope[node.target.id] = self.binop(type(node.op), cur, rhs, node)

    def assign(self, target, value):
        if isinstance(target, ast.Name):
            self.scope[target.id] = value
        elif isinstance(target, (ast.Tuple, ast.List)):
            vals = list(value) if not isinstance(value, tensor) else None
            if vals is None or len(vals) != len(target.elts):
                raise self.error(target, "cannot unpack value")
            for t, v in zip(target.elts, vals):
                self.assign(t, v)
        else:
            raise self.error(target, f"unsupported assignment target {type(target).__name__}")

    def stmt_Return(self, node):
        raise _Return(self.eval(node.value) if node.value is not None else None)

    def stmt_Assert(self, node):
        cond = self.eval(node.test)
        if isinstance(cond, tensor):
            return  # runtime asserts are dropped (like Triton without debug)
        if not cond:
            msg = self.eval(node.msg) if node.msg is not None else ""
            raise self.error(node, f"assertion failed: {msg}")

    def stmt_If(self, node):
        cond = self.eval(node.test)
        cond = _unwrap(cond)
        if not isinstance(cond, tensor):
            self.run_body(node.body if cond else node.orelse)
            return
        if cond.handle.shape:
            raise self.error(node, "if-condition must be a scalar")
        self.runtime_if(node, cond)

    def runtime_if(self, node, cond: tensor, then_stmts=None, else_stmts=None):
        then_stmts = node.body if then_stmts is None else then_stmts
        else_stmts = node.orelse if else_stmts is None else else_stmts
        cond = tl.cast(cond, tl.int1, _builder=self.builder)
        before = dict(self.scope)
        assigned = _assigned_names(then_stmts) | _assigned_names(else_stmts)
        returns = _contains_return(then_stmts + else_stmts)

        def run(stmts):
            blk = ir.Block()
            saved = self.builder.block
            self.builder.block = blk
            self.scope = dict(before)
            try:
                self.run_body(stmts)
            except _Return as r:
                self.scope[_RET] = r.value
            finally:
                self.builder.block = saved
            return blk, dict(self.scope)

        then_blk, then_scope = run(then_stmts)
        else_blk, else_scope = run(else_stmts)
        self.scope = before
        if returns:
            ra, rb = then_scope.get(_RET), else_scope.get(_RET)
            if (ra is None) != (rb is None):
                raise self.error(node, "function returns a value on some paths only")
            if ra is not None:
                assigned.add(_RET)
        names = []
        for name in sorted(assigned):
            a, b = then_scope.get(name), else_scope.get(name)
            if a is None or b is None:
                continue  # defined in one branch only: not visible afterwards
            if not isinstance(a, tensor) and not isinstance(b, tensor):
                if a is b or (type(a) == type(b) and a == b):
                    self.scope[name] = a
                    continue
            names.append(name)
        then_vals, else_vals = [], []
        for name in names:
            a, b = then_scope[name], else_scope[name]
            saved = self.builder.block
            self.builder.block = then_blk
            a = to_tensor(a, self.builder)
            self.builder.block = else_blk
            b = to_tensor(b, self.builder)
            self.builder.block = saved
            if a.dtype != b.dtype or a.handle.shape != b.handle.shape:
                raise self.error(node, f"'{name}' has different types in the two branches of if")
            then_vals.append(a)
            else_vals.append(b)
        ir.Builder(then_blk).create("scf.yield", [v.handle for v in then_vals])
        ir.Builder(else_blk).create("scf.yield", [v.handle for v in else_vals])
        op = self.builder.create("scf.if", [cond.handle],
                                 [(v.dtype, v.handle.shape) for v in then_vals],
                                 regions=[then_blk, else_blk])
        for name, res in zip(names, op.results):
            self.scope[name] = tensor(res, self.builder)

    def stmt_For(self, node):
        if node.orelse:
            raise self.error(node, "for/else is not supported")
        it = node.iter
        if not isinstance(it, ast.Call):
            seq = self.eval(it)
            if isinstance(seq, tensor):
                raise self.error(node, "cannot iterate over a tensor")
            for v in seq:
                self.assign(node.target, v)
                self.run_body(node.body)
            return
        func = self.eval(it.func)
        args = [self.eval(a) for a in it.args]
        kwargs = {k.arg: self.eval(k.value) for k in it.keywords}
        if func is tl.static_range:
            r = tl.static_range(*args, **kwargs)
            for v in builtins.range(r.start, r.end, r.step):
                self.assign(node.target, v)
                self.run_body(node.body)
            return
        if func is builtins.range or func is tl.range:
            r = tl.range(*args, **kwargs)
            self.runtime_for(node, r)
            return
        seq = func(*args, **kwargs)
        for v in seq:
            self.assign(node.target, v)
            self.run_body(node.body)

    def runtime_for(self, node, r):
        if not isinstance(node.target, ast.Name):
            raise self.error(node, "loop variable must be a name")
        if _contains_return(node.body):
            raise self.error(node, "return inside a runtime for loop is not supported (as in Triton)")
        b = self.builder
        lb = tl.cast(to_tensor(_unwrap(r.start), b, tl.int32), tl.int32, _builder=b)
        ub = tl.cast(to_tensor(_unwrap(r.end), b, tl.int32), tl.int32, _builder=b)
        step_val = _unwrap(r.step)
        if isinstance(step_val, int) and step_val <= 0:
            raise self.error(node, "loop step must be positive")
        step = tl.cast(to_tensor(step_val, b, tl.int32), tl.int32, _builder=b)
        carried, inits = self.loop_carried(node, b)
        body = ir.Block()
        iv = ir.Value(tl.int32, (), name=None)
        body.args.append(iv)
        for v in inits:
            body.args.append(ir.Value(v.dtype, v.handle.shape))
        before = dict(self.scope)
        saved_block = b.block
        b.block = body
        self.scope = dict(before)
        self.scope[node.target.id] = tensor(iv, b)
        for name, arg in zip(carried, body.args[1:]):
            self.scope[name] = tensor(arg, b)
        try:
            self.run_body(node.body)
        except _Return:
            b.block = saved_block
            self.scope = before
            raise self.error(node, "return inside a runtime for loop is not supported")
        yields = []
        for name, init in zip(carried, inits):
            v = self.scope[name]
            v = to_tensor(v, b, init.dtype) if not isinstance(v, tensor) else v
            if v.dtype != init.dtype or v.handle.shape != init.handle.shape:
                b.block = saved_block
                self.scope = before
                raise self.error(node, f"loop-carried '{name}' changes type from "
                                       f"{init.handle.type_str()} to {v.handle.type_str()}")
            yields.append(v)
        b.create("scf.yield", [v.handle for v in yields])
        b.block = saved_block
        self.scope = before
        op = b.create("scf.for", [lb.handle, ub.handle, step.handle] + [v.handle for v in inits],
                      [(v.dtype, v.handle.shape) for v in inits], regions=[body],
                      attrs={"num_stages": r.num_stages} if r.num_stages else {})
        for name, res in zip(carried, op.results):
            self.scope[name] = tensor(res, b)

    def loop_carried(self, node, b):
        """Find the loop-carried variables by running the body once in a
        throwaway block (Triton does the same dry run)."""
        before = dict(self.scope)
        saved_block = b.block
        b.block = ir.Block()
        self.scope = dict(before)
        self.scope[node.target.id] = tensor(ir.Value(tl.int32, ()), b)
        try:
            self.run_body(node.body)
        except _Return:
            raise self.error(node, "return inside a runtime for loop is not supported")
        finally:
            b.block = saved_block
            after = self.scope
            self.scope = before
        carried, inits = [], []
        for name in sorted(before):
            if name == node.target.id or name not in after:
                continue
            old, new = before[name], after[name]
            if old is new:
                continue
            scalar_types = (int, float, bool)
            if not isinstance(old, tensor) and not isinstance(new, tensor):
                try:
                    same = type(old) == type(new) and bool(old == new)
                except CompilationError:
                    same = False
                if same:
                    continue
                if not (isinstance(old, scalar_types) and isinstance(new, scalar_types)):
                    raise self.error(node, f"variable '{name}' ({type(old).__name__}) is re-assigned "
                                           f"inside a runtime loop; only tensors and numbers can be "
                                           f"loop-carried")
            if isinstance(old, tensor):
                init = old
            elif isinstance(old, scalar_types):
                dt = new.dtype if isinstance(new, tensor) else None
                init = to_tensor(old, b, dt)
                if dt is not None:
                    init = tl.cast(init, dt, _builder=b)
                    if new.handle.shape:
                        init = tl.broadcast_to(init, new.handle.shape, _builder=b)
            else:
                raise self.error(node, f"variable '{name}' ({type(old).__name__}) cannot be loop-carried")
            carried.append(name)
            inits.append(init)
            self.scope[name] = init
        return carried, inits

    def stmt_While(self, node):
        raise self.error(node, "while loops are not supported (use for ... in range)")

    def stmt_Break(self, node):
        raise self.error(node, "break is not supported")

    def stmt_Continue(self, node):
        raise self.error(node, "continue is not supported")

    # -- expressions ------------------------------------------------------------

    def eval(self, node):
        method = getattr(self, "expr_" + type(node).__name__, None)
        if method is None:
            raise self.error(node, f"unsupported expression: {type(node).__name__}")
        return method(node)

    def expr_Constant(self, node):
        return node.value

    def expr_Name(self, node):
        return self.lookup(node.id, node)

    def expr_Tuple(self, node):
        return tuple(self.eval(e) for e in node.elts)

    def expr_List(self, node):
        return [self.eval(e) for e in node.elts]

    def expr_Attribute(self, node):
        base = self.eval(node.value)
        try:
            val = getattr(base, node.attr)
        except AttributeError:
            raise self.error(node, f"'{type(base).__name__}' has no attribute '{node.attr}'")
        return self._wrap_global(val) if inspect.ismodule(val) else _unwrap(val) \
            if isinstance(val, tl.constexpr) else val

    def expr_Subscript(self, node):
        base = self.eval(node.value)
        idx = self.eval(node.slice)
        if isinstance(base, tensor):
            return base[idx]
        return base[idx]

    def expr_Slice(self, node):
        lo = self.eval(node.lower) if node.lower else None
        hi = self.eval(node.upper) if node.upper else None
        st = self.eval(node.step) if node.step else None
        return slice(lo, hi, st)

    def binop(self, op_type, a, b, node):
        a, b = _unwrap(a), _unwrap(b)
        if not isinstance(a, tensor) and not isinstance(b, tensor):
            return _BIN[op_type](a, b)
        if op_type is ast.Pow:
            raise self.error(node, "** is not supported on tensors")
        if not isinstance(a, tensor):
            a = to_tensor(a, self.builder) if isinstance(a, bool) else a
        fn = {ast.Add: "__add__", ast.Sub: "__sub__", ast.Mult: "__mul__", ast.Div: "__truediv__",
              ast.FloorDiv: "__floordiv__", ast.Mod: "__mod__", ast.LShift: "__lshift__",
              ast.RShift: "__rshift__", ast.BitAnd: "__and__", ast.BitOr: "__or__",
              ast.BitXor: "__xor__"}[op_type]
        if isinstance(a, tensor):
            return getattr(a, fn)(b)
        rfn = "__r" + fn[2:]
        return getattr(b, rfn)(a)

    def expr_BinOp(self, node):
        return self.binop(type(node.op), self.eval(node.left), self.eval(node.right), node)

    def expr_UnaryOp(self, node):
        v = _unwrap(self.eval(node.operand))
        if isinstance(node.op, ast.Not):
            if isinstance(v, tensor):
                return tl.core.invert(tl.cast(v, tl.int1, _builder=self.builder), self.builder)
            return not v
        if isinstance(node.op, ast.USub):
            return -v
        if isinstance(node.op, ast.UAdd):
            return +v
        if isinstance(node.op, ast.Invert):
            return ~v
        raise self.error(node, "unsupported unary op")

    def expr_BoolOp(self, node):
        is_and = isinstance(node.op, ast.And)
        vals = []
        for v in node.values:
            x = _unwrap(self.eval(v))
            if not isinstance(x, tensor) and not any(isinstance(y, tensor) for y in vals):
                # python short-circuit while everything so far is a constexpr
                if (is_and and not x) or (not is_and and x):
                    return x
            vals.append(x)
        if not any(isinstance(v, tensor) for v in vals):
            return vals[-1]
        acc = tl.cast(to_tensor(vals[0], self.builder), tl.int1, _builder=self.builder)
        for v in vals[1:]:
            v = tl.cast(to_tensor(v, self.builder), tl.int1, _builder=self.builder)
            acc = (acc & v) if isinstance(node.op, ast.And) else (acc | v)
        return acc

    def expr_Compare(self, node):
        left = _unwrap(self.eval(node.left))
        result = None
        for op, comp in zip(node.ops, node.comparators):
            right = _unwrap(self.eval(comp))
            if isinstance(left, tensor) or isinstance(right, tensor):
                if len(node.ops) > 1:
                    raise self.error(node, "chained comparisons on tensors are not supported")
                name = {ast.Lt: "__lt__", ast.LtE: "__le__", ast.Gt: "__gt__", ast.GtE: "__ge__",
                        ast.Eq: "__eq__", ast.NotEq: "__ne__"}.get(type(op))
                if name is None:
                    raise self.error(node, "unsupported tensor comparison")
                if isinstance(left, tensor):
                    return getattr(left, name)(right)
                swapped = {"__lt__": "__gt__", "__le__": "__ge__", "__gt__": "__lt__",
                           "__ge__": "__le__", "__eq__": "__eq__", "__ne__": "__ne__"}[name]
                return getattr(right, swapped)(left)
            r = _CMP[type(op)](left, right)
            result = r if result is None else (result and r)
            left = right
        return result

    def expr_IfExp(self, node):
        cond = _unwrap(self.eval(node.test))
        if not isinstance(cond, tensor):
            return self.eval(node.body) if cond else self.eval(node.orelse)
        return tl.where(cond, self.eval(node.body), self.eval(node.orelse), _builder=self.builder)

    def expr_JoinedStr(self, node):
        return "<fstring>"

    def expr_Call(self, node):
        func = self.eval(node.func)
        args = []
        for a in node.args:
            if isinstance(a, ast.Starred):
                args.extend(self.eval(a.value))
            else:
                args.append(self.eval(a))
        kwargs = {k.arg: self.eval(k.value) for k in node.keywords}
        return self.call(func, args, kwargs, node)

    def call(self, func, args, kwargs, node):
        if getattr(func, "_piton_builtin", False):
            return func(*args, _builder=self.builder, **kwargs)
        target = _jit_target(func)
        if target is not None:
            return self.inline(target, args, kwargs, node)
        if func in (builtins.min, builtins.max):
            if any(isinstance(_unwrap(a), tensor) for a in args):
                f = tl.minimum if func is builtins.min else tl.maximum
                out = args[0]
                for a in args[1:]:
                    out = f(out, a, _builder=self.builder)
                return out
            return func(*[_unwrap(a) for a in args], **kwargs)
        if func is builtins.print:
            return None
        if func in (builtins.int, builtins.float, builtins.bool):
            v = _unwrap(args[0])
            if isinstance(v, tensor):
                dt = {builtins.int: tl.int32, builtins.float: tl.float32, builtins.bool: tl.int1}[func]
                return tl.cast(v, dt, _builder=self.builder)
            return func(v)
        if inspect.ismethod(func) and isinstance(func.__self__, tensor):
            return func(*args, **kwargs)
        if func is tl.constexpr:
            return _unwrap(args[0])
        if any(isinstance(_unwrap(a), tensor) for a in list(args) + list(kwargs.values())):
            if inspect.isfunction(func):
                return self.inline(func, args, kwargs, node)
            raise self.error(node, f"cannot call {getattr(func, '__name__', func)} on tensors")
        return func(*[_unwrap(a) for a in args], **{k: _unwrap(v) for k, v in kwargs.items()})

    def inline(self, fn, args, kwargs, node):
        sig = inspect.signature(fn)
        try:
            bound = sig.bind(*args, **kwargs)
        except TypeError as e:
            raise self.error(node, f"bad call to {fn.__name__}: {e}")
        bound.apply_defaults()
        gen = CodeGenerator(fn, self.builder)
        for name, val in bound.arguments.items():
            gen.scope[name] = _unwrap(val)
        fdef = get_source(fn)
        try:
            gen.run_body(fdef.body)
        except _Return as r:
            return r.value
        return None


def generate(fn, signature: dict, constants: dict, name: str = None) -> ir.Function:
    """Build the tile IR for kernel `fn`.

    signature -- {arg_name: '*fp32' | '*i32' | 'i32' | 'fp32'} for runtime args
    constants -- {arg_name: value} for constexpr / specialised args
    """
    fdef = get_source(fn)
    params = list(inspect.signature(fn).parameters.values())
    body = ir.Block()
    builder = ir.Builder(body)
    gen = CodeGenerator(fn, builder)
    args, arg_names = [], []
    for p in params:
        ann = p.annotation
        is_constexpr = ann is tl.constexpr or (isinstance(ann, str) and "constexpr" in ann) or \
            getattr(ann, "__name__", "") == "constexpr"
        if p.name in constants:
            gen.scope[p.name] = _unwrap(constants[p.name])
        elif is_constexpr:
            if p.default is not inspect.Parameter.empty:
                gen.scope[p.name] = _unwrap(p.default)
            else:
                raise CompilationError(f"constexpr argument '{p.name}' needs a value in constants=")
        elif p.name in signature:
            dt = ir.parse_type(signature[p.name]) if isinstance(signature[p.name], str) \
                else signature[p.name]
            v = ir.Value(dt, (), name=p.name)
            args.append(v)
            arg_names.append(p.name)
            gen.scope[p.name] = tensor(v, builder)
        elif p.default is not inspect.Parameter.empty:
            gen.scope[p.name] = _unwrap(p.default)
        else:
            raise CompilationError(f"argument '{p.name}' missing from signature and constants")
    try:
        gen.run_body(fdef.body)
    except _Return as r:
        if r.value is not None:
            raise CompilationError("kernels cannot return values")
    builder.create("tt.return", [])
    return ir.Function(name or fn.__name__, args, arg_names, body,
                       {k: _unwrap(v) for k, v in constants.items()})
