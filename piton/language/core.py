"""Triton-compatible language surface (`import piton.language as tl`).

Kernels are executed symbolically by `piton.frontend.CodeGenerator`; every
`tl.*` builtin receives the active IR builder through the `_builder`
keyword, exactly like Triton's `@builtin` functions, and returns `tensor`
objects wrapping IR values.  Constexpr arithmetic stays in plain Python.
"""

from __future__ import annotations

import builtins as _py
import math
from typing import Optional, Sequence

from .. import ir
from ..ir import DType, float32, int1, int32, pointer_type, uint32

# dtype aliases (Triton spellings)
bool_ = int1
int1 = int1
int32 = int32
uint32 = uint32
float32 = float32
pi32_t = pointer_type(int32)


class _Unsupported(DType):
    def __init__(self, name):
        super().__init__(name, 0, "unsupported")


float16 = _Unsupported("fp16")
bfloat16 = _Unsupported("bf16")
float64 = _Unsupported("fp64")
int8 = _Unsupported("int8")
int16 = _Unsupported("int16")
int64 = _Unsupported("int64")
uint8 = _Unsupported("uint8")


class CompilationError(Exception):
    pass


class constexpr:
    """Marker for compile-time arguments (`BLOCK: tl.constexpr`)."""

    def __init__(self, value=None):
        self.value = value.value if isinstance(value, constexpr) else value

    def __repr__(self):
        return f"constexpr[{self.value!r}]"


def _unwrap(x):
    return x.value if isinstance(x, constexpr) else x


def builtin(fn):
    fn._piton_builtin = True
    return fn


def _check_dtype(dt: DType) -> DType:
    if isinstance(dt, _Unsupported) or dt.kind == "unsupported":
        raise CompilationError(f"dtype {dt.name} is not supported on the VideoCore IV QPU "
                               f"(supported: float32, int32, uint32, int1)")
    return dt


# ---------------------------------------------------------------------------
# tensor
# ---------------------------------------------------------------------------


class tensor:
    def __init__(self, handle: ir.Value, builder: ir.Builder):
        self.handle = handle
        self.builder = builder

    @property
    def shape(self):
        return tuple(constexpr(s) for s in self.handle.shape)

    @property
    def dtype(self) -> DType:
        return self.handle.dtype

    @property
    def numel(self):
        return constexpr(self.handle.numel)

    @property
    def T(self):
        return trans(self, _builder=self.builder)

    def __repr__(self):
        return f"tensor({self.handle.type_str()})"

    def __bool__(self):
        raise CompilationError("cannot use a runtime tensor as a Python bool; "
                               "use tl.where / masks, or make the value a tl.constexpr")

    # arithmetic -----------------------------------------------------------
    def __add__(self, o):
        return add(self, o, self.builder)

    def __radd__(self, o):
        return add(o, self, self.builder)

    def __sub__(self, o):
        return sub(self, o, self.builder)

    def __rsub__(self, o):
        return sub(o, self, self.builder)

    def __mul__(self, o):
        return mul(self, o, self.builder)

    def __rmul__(self, o):
        return mul(o, self, self.builder)

    def __truediv__(self, o):
        return truediv(self, o, self.builder)

    def __rtruediv__(self, o):
        return truediv(o, self, self.builder)

    def __floordiv__(self, o):
        return floordiv(self, o, self.builder)

    def __rfloordiv__(self, o):
        return floordiv(o, self, self.builder)

    def __mod__(self, o):
        return mod(self, o, self.builder)

    def __rmod__(self, o):
        return mod(o, self, self.builder)

    def __neg__(self):
        return neg(self, self.builder)

    def __pos__(self):
        return self

    def __invert__(self):
        return invert(self, self.builder)

    def __and__(self, o):
        return bitwise(self, o, "and", self.builder)

    def __rand__(self, o):
        return bitwise(o, self, "and", self.builder)

    def __or__(self, o):
        return bitwise(self, o, "or", self.builder)

    def __ror__(self, o):
        return bitwise(o, self, "or", self.builder)

    def __xor__(self, o):
        return bitwise(self, o, "xor", self.builder)

    def __rxor__(self, o):
        return bitwise(o, self, "xor", self.builder)

    def __lshift__(self, o):
        return shift(self, o, "shl", self.builder)

    def __rlshift__(self, o):
        return shift(o, self, "shl", self.builder)

    def __rshift__(self, o):
        return shift(self, o, "shr", self.builder)

    def __rrshift__(self, o):
        return shift(o, self, "shr", self.builder)

    def __lt__(self, o):
        return compare(self, o, "lt", self.builder)

    def __le__(self, o):
        return compare(self, o, "le", self.builder)

    def __gt__(self, o):
        return compare(self, o, "gt", self.builder)

    def __ge__(self, o):
        return compare(self, o, "ge", self.builder)

    def __eq__(self, o):
        return compare(self, o, "eq", self.builder)

    def __ne__(self, o):
        return compare(self, o, "ne", self.builder)

    __hash__ = object.__hash__

    def logical_and(self, o):
        return bitwise(self, o, "and", self.builder)

    def logical_or(self, o):
        return bitwise(self, o, "or", self.builder)

    def __getitem__(self, idx):
        if not isinstance(idx, tuple):
            idx = (idx,)
        out = self
        dim = 0
        for s in idx:
            s = _unwrap(s)
            if s is None:
                out = expand_dims(out, dim, _builder=self.builder)
                dim += 1
            elif isinstance(s, slice) and s == slice(None):
                dim += 1
            else:
                raise CompilationError(f"unsupported tensor index {s!r} (only None and ':' "
                                       f"are supported, as in Triton)")
        return out

    # methods ----------------------------------------------------------------
    def to(self, dtype, bitcast=False, **_):
        return cast(self, dtype, bitcast=bitcast, _builder=self.builder)

    def reshape(self, *shape, **_):
        if len(shape) == 1 and isinstance(_unwrap(shape[0]), (tuple, list)):
            shape = _unwrap(shape[0])
        return reshape(self, shape, _builder=self.builder)

    def broadcast_to(self, *shape):
        if len(shape) == 1 and isinstance(_unwrap(shape[0]), (tuple, list)):
            shape = _unwrap(shape[0])
        return broadcast_to(self, shape, _builder=self.builder)

    def sum(self, axis=None, keep_dims=False):
        return sum(self, axis, keep_dims, _builder=self.builder)

    def max(self, axis=None, keep_dims=False):
        return max(self, axis, keep_dims=keep_dims, _builder=self.builder)

    def min(self, axis=None, keep_dims=False):
        return min(self, axis, keep_dims=keep_dims, _builder=self.builder)

    def exp(self):
        return exp(self, _builder=self.builder)

    def log(self):
        return log(self, _builder=self.builder)

    def sqrt(self):
        return sqrt(self, _builder=self.builder)

    def abs(self):
        return abs(self, _builder=self.builder)

    def trans(self, *dims):
        return trans(self, _builder=self.builder)


# ---------------------------------------------------------------------------
# semantic helpers
# ---------------------------------------------------------------------------


def _const(value, dtype: Optional[DType], builder: ir.Builder) -> tensor:
    value = _unwrap(value)
    if dtype is None:
        if isinstance(value, bool):
            dtype = int1
        elif isinstance(value, int):
            if not -2**31 <= value < 2**32:
                raise CompilationError(f"integer constant {value} does not fit in 32 bits")
            dtype = int32 if value < 2**31 else uint32
        elif isinstance(value, float):
            dtype = float32
        else:
            raise CompilationError(f"cannot convert {value!r} to a tensor")
    if dtype.is_float:
        value = float(value)
    elif dtype.is_bool:
        value = bool(value)
    else:
        value = int(value)
    op = builder.create("arith.constant", [], [(dtype, ())], {"value": value})
    return tensor(op.result, builder)


def to_tensor(x, builder: ir.Builder, dtype: Optional[DType] = None) -> tensor:
    if isinstance(x, tensor):
        return x
    return _const(x, dtype, builder)


def splat(x: tensor, shape, builder) -> tensor:
    if tuple(x.handle.shape) == tuple(shape):
        return x
    if x.handle.shape:
        raise CompilationError("splat of a non-scalar")
    op = builder.create("tt.splat", [x.handle], [(x.dtype, tuple(shape))])
    return tensor(op.result, builder)


def _broadcast_shapes(a: tuple, b: tuple) -> tuple:
    if len(a) != len(b):
        if not a or not b:
            return a or b
        # left-pad the shorter shape with ones (numpy rules)
        n = _py.max(len(a), len(b))
        a = (1,) * (n - len(a)) + tuple(a)
        b = (1,) * (n - len(b)) + tuple(b)
    out = []
    for x, y in zip(a, b):
        if x == y or y == 1:
            out.append(x)
        elif x == 1:
            out.append(y)
        else:
            raise CompilationError(f"incompatible shapes for broadcasting: {a} and {b}")
    return tuple(out)


def broadcast_to_shape(x: tensor, shape, builder) -> tensor:
    shape = tuple(shape)
    src = tuple(x.handle.shape)
    if src == shape:
        return x
    if not src:
        return splat(x, shape, builder)
    if len(src) < len(shape):
        for _ in _py.range(len(shape) - len(src)):
            x = expand_dims(x, 0, _builder=builder)
        src = tuple(x.handle.shape)
    for s, d in zip(src, shape):
        if s != d and s != 1:
            raise CompilationError(f"cannot broadcast {src} to {shape}")
    op = builder.create("tt.broadcast", [x.handle], [(x.dtype, shape)])
    return tensor(op.result, builder)


def _binary_dtype(a: DType, b: DType, op: str) -> DType:
    if a.is_ptr or b.is_ptr:
        return a if a.is_ptr else b
    if a.is_float or b.is_float:
        return float32
    if op in ("and", "or", "xor") and a.is_bool and b.is_bool:
        return int1
    if a == uint32 or b == uint32:
        return uint32
    return int32


def _prepare(a, b, builder, op: str):
    """Convert python scalars, broadcast, and promote to a common dtype."""
    if not isinstance(a, tensor) and not isinstance(b, tensor):
        raise CompilationError("internal: binary op without tensors")
    # python scalars adopt the tensor's dtype family
    if not isinstance(a, tensor):
        a = _scalar_like(a, b, builder)
    if not isinstance(b, tensor):
        b = _scalar_like(b, a, builder)
    shape = _broadcast_shapes(a.handle.shape, b.handle.shape)
    a = broadcast_to_shape(a, shape, builder)
    b = broadcast_to_shape(b, shape, builder)
    dt = _binary_dtype(a.dtype, b.dtype, op)
    if not dt.is_ptr:
        a = cast(a, dt, _builder=builder)
        b = cast(b, dt, _builder=builder)
    return a, b, dt, shape


def _scalar_like(x, other: tensor, builder):
    x = _unwrap(x)
    if isinstance(x, float) or other.dtype.is_float:
        if isinstance(x, (int, float)) and not isinstance(x, bool):
            return _const(float(x), float32, builder) if (isinstance(x, float) or other.dtype.is_float) \
                else _const(x, None, builder)
    if isinstance(x, bool) and other.dtype.is_bool:
        return _const(x, int1, builder)
    if isinstance(x, bool):
        return _const(int(x), int32, builder)
    if isinstance(x, int) and other.dtype == uint32 and x >= 0:
        return _const(x, uint32, builder)
    return _const(x, None, builder)


def _emit(builder, name, operands, dtype, shape, attrs=None) -> tensor:
    op = builder.create(name, [o.handle for o in operands], [(dtype, shape)], attrs)
    return tensor(op.result, builder)


def add(a, b, builder) -> tensor:
    a, b, dt, shape = _prepare(a, b, builder, "add")
    if a.dtype.is_ptr and b.dtype.is_ptr:
        raise CompilationError("cannot add two pointers")
    if a.dtype.is_ptr or b.dtype.is_ptr:
        ptr, off = (a, b) if a.dtype.is_ptr else (b, a)
        if not off.dtype.is_int:
            raise CompilationError("pointer offset must be an integer")
        return _emit(builder, "tt.addptr", [ptr, off], ptr.dtype, shape)
    return _emit(builder, "arith.addf" if dt.is_float else "arith.addi", [a, b], dt, shape)


def sub(a, b, builder) -> tensor:
    if isinstance(a, tensor) and a.dtype.is_ptr and not (isinstance(b, tensor) and b.dtype.is_ptr):
        return add(a, neg(to_tensor(b, builder), builder), builder)
    a, b, dt, shape = _prepare(a, b, builder, "sub")
    if dt.is_ptr:
        raise CompilationError("pointer difference is not supported")
    return _emit(builder, "arith.subf" if dt.is_float else "arith.subi", [a, b], dt, shape)


def mul(a, b, builder) -> tensor:
    a, b, dt, shape = _prepare(a, b, builder, "mul")
    if dt.is_ptr:
        raise CompilationError("cannot multiply pointers")
    return _emit(builder, "arith.mulf" if dt.is_float else "arith.muli", [a, b], dt, shape)


def truediv(a, b, builder) -> tensor:
    a = to_tensor(a, builder, float32 if not isinstance(a, tensor) else None)
    b = to_tensor(b, builder, float32 if not isinstance(b, tensor) else None)
    a = cast(a, float32, _builder=builder)
    b = cast(b, float32, _builder=builder)
    a, b, dt, shape = _prepare(a, b, builder, "div")
    return _emit(builder, "arith.divf", [a, b], float32, shape)


def floordiv(a, b, builder) -> tensor:
    a, b, dt, shape = _prepare(a, b, builder, "floordiv")
    if not dt.is_int:
        raise CompilationError("// is only supported for integers")
    return _emit(builder, "arith.divui" if dt == uint32 else "arith.divsi", [a, b], dt, shape)


def mod(a, b, builder) -> tensor:
    a, b, dt, shape = _prepare(a, b, builder, "mod")
    if not dt.is_int:
        raise CompilationError("% is only supported for integers")
    return _emit(builder, "arith.remui" if dt == uint32 else "arith.remsi", [a, b], dt, shape)


def neg(a, builder) -> tensor:
    a = to_tensor(a, builder)
    if a.dtype.is_float:
        return _emit(builder, "arith.negf", [a], a.dtype, a.handle.shape)
    return sub(0, a, builder)


def invert(a, builder) -> tensor:
    a = to_tensor(a, builder)
    if a.dtype.is_bool:
        return bitwise(a, True, "xor", builder)
    return bitwise(a, -1, "xor", builder)


def bitwise(a, b, kind, builder) -> tensor:
    a, b, dt, shape = _prepare(a, b, builder, kind)
    if not dt.is_int:
        raise CompilationError(f"bitwise {kind} needs integer operands")
    return _emit(builder, {"and": "arith.andi", "or": "arith.ori", "xor": "arith.xori"}[kind],
                 [a, b], dt, shape)


def shift(a, b, kind, builder) -> tensor:
    a, b, dt, shape = _prepare(a, b, builder, kind)
    if not dt.is_int:
        raise CompilationError("shifts need integer operands")
    name = "arith.shli" if kind == "shl" else ("arith.shrui" if dt == uint32 else "arith.shrsi")
    return _emit(builder, name, [a, b], dt, shape)


def compare(a, b, kind, builder) -> tensor:
    a, b, dt, shape = _prepare(a, b, builder, kind)
    if dt.is_float:
        pred = {"lt": "olt", "le": "ole", "gt": "ogt", "ge": "oge", "eq": "oeq", "ne": "une"}[kind]
        return _emit(builder, "arith.cmpf", [a, b], int1, shape, {"predicate": pred})
    signed = dt.is_signed or dt.is_bool
    if dt.is_ptr:
        signed = False
    pre = "s" if signed else "u"
    pred = {"lt": pre + "lt", "le": pre + "le", "gt": pre + "gt", "ge": pre + "ge",
            "eq": "eq", "ne": "ne"}[kind]
    return _emit(builder, "arith.cmpi", [a, b], int1, shape, {"predicate": pred})


# ---------------------------------------------------------------------------
# builtins
# ---------------------------------------------------------------------------


@builtin
def program_id(axis, _builder=None):
    axis = _unwrap(axis)
    if axis not in (0, 1, 2):
        raise CompilationError("program_id axis must be 0, 1 or 2")
    op = _builder.create("tt.get_program_id", [], [(int32, ())], {"axis": axis})
    return tensor(op.result, _builder)


@builtin
def num_programs(axis, _builder=None):
    axis = _unwrap(axis)
    op = _builder.create("tt.get_num_programs", [], [(int32, ())], {"axis": axis})
    return tensor(op.result, _builder)


def _is_pow2(n):
    return n > 0 and (n & (n - 1)) == 0


@builtin
def arange(start, end, _builder=None):
    start, end = _unwrap(start), _unwrap(end)
    if not isinstance(start, int) or not isinstance(end, int):
        raise CompilationError("arange bounds must be tl.constexpr")
    n = end - start
    if not _is_pow2(n):
        raise CompilationError(f"arange range must be a power of 2, got {n}")
    op = _builder.create("tt.make_range", [], [(int32, (n,))], {"start": start, "end": end})
    return tensor(op.result, _builder)


def _shape(shape):
    shape = _unwrap(shape)
    if isinstance(shape, int):
        shape = (shape,)
    shape = tuple(_unwrap(s) for s in shape)
    for s in shape:
        if not isinstance(s, int) or not _is_pow2(s):
            raise CompilationError(f"block dimensions must be constexpr powers of 2, got {shape}")
    return shape


@builtin
def full(shape, value, dtype, _builder=None):
    shape = _shape(shape)
    dtype = _check_dtype(_unwrap(dtype))
    v = to_tensor(value, _builder, None if isinstance(value, tensor) else dtype)
    v = cast(v, dtype, _builder=_builder)
    return splat(v, shape, _builder)


@builtin
def zeros(shape, dtype, _builder=None):
    return full(shape, 0, dtype, _builder=_builder)


@builtin
def zeros_like(input, _builder=None):
    return zeros(input.handle.shape, input.dtype, _builder=_builder)


@builtin
def full_like(input, value, _builder=None):
    return full(input.handle.shape, value, input.dtype, _builder=_builder)


@builtin
def cast(input, dtype, bitcast=False, fp_downcast_rounding=None, _builder=None):
    input = to_tensor(input, _builder)
    dtype = _check_dtype(_unwrap(dtype))
    src = input.dtype
    shape = input.handle.shape
    if src == dtype:
        return input
    if bitcast or (src.is_int and dtype.is_int and not src.is_bool and not dtype.is_bool):
        return _emit(_builder, "arith.bitcast", [input], dtype, shape)
    if src.is_bool and dtype.is_int:
        return _emit(_builder, "arith.extui", [input], dtype, shape)
    if src.is_bool and dtype.is_float:
        return cast(cast(input, int32, _builder=_builder), dtype, _builder=_builder)
    if dtype.is_bool:
        return compare(input, 0.0 if src.is_float else 0, "ne", _builder)
    if src.is_int and dtype.is_float:
        return _emit(_builder, "arith.uitofp" if src == uint32 else "arith.sitofp", [input], dtype, shape)
    if src.is_float and dtype.is_int:
        return _emit(_builder, "arith.fptoui" if dtype == uint32 else "arith.fptosi", [input], dtype, shape)
    if src.is_ptr and dtype.is_ptr:
        return _emit(_builder, "tt.bitcast", [input], dtype, shape)
    if src.is_ptr and dtype.is_int:
        return _emit(_builder, "tt.ptr_to_int", [input], dtype, shape)
    if src.is_int and dtype.is_ptr:
        return _emit(_builder, "tt.int_to_ptr", [input], dtype, shape)
    raise CompilationError(f"unsupported cast {src} -> {dtype}")


@builtin
def expand_dims(input, axis, _builder=None):
    input = to_tensor(input, _builder)
    axes = _unwrap(axis)
    axes = [axes] if isinstance(axes, int) else list(axes)
    out = input
    for ax in sorted(axes):
        shape = list(out.handle.shape)
        if ax < 0:
            ax += len(shape) + 1
        shape.insert(ax, 1)
        if not out.handle.shape:
            out = splat(out, (1,), _builder)
            continue
        op = _builder.create("tt.expand_dims", [out.handle], [(out.dtype, tuple(shape))], {"axis": ax})
        out = tensor(op.result, _builder)
    return out


@builtin
def broadcast_to(input, shape, _builder=None):
    shape = tuple(_unwrap(s) for s in _unwrap(shape))
    return broadcast_to_shape(to_tensor(input, _builder), shape, _builder)


@builtin
def broadcast(input, other, _builder=None):
    shape = _broadcast_shapes(input.handle.shape, other.handle.shape)
    return (broadcast_to_shape(input, shape, _builder), broadcast_to_shape(other, shape, _builder))


@builtin
def reshape(input, shape, can_reorder=False, _builder=None):
    shape = _shape(shape)
    n = 1
    for s in shape:
        n *= s
    if n != input.handle.numel:
        raise CompilationError(f"reshape {input.handle.shape} -> {shape} changes the element count")
    op = _builder.create("tt.reshape", [input.handle], [(input.dtype, shape)])
    return tensor(op.result, _builder)


view = reshape


@builtin
def trans(input, *dims, _builder=None):
    if len(input.handle.shape) != 2:
        raise CompilationError("trans needs a 2D tensor")
    m, n = input.handle.shape
    op = _builder.create("tt.trans", [input.handle], [(input.dtype, (n, m))], {"order": (1, 0)})
    return tensor(op.result, _builder)


permute = trans


@builtin
def load(pointer, mask=None, other=None, boundary_check=(), padding_option="", cache_modifier="",
         eviction_policy="", volatile=False, _builder=None):
    pointer = to_tensor(pointer, _builder)
    if not pointer.dtype.is_ptr:
        raise CompilationError(f"tl.load expects a pointer, got {pointer.dtype}")
    elem = pointer.dtype.elem
    shape = pointer.handle.shape
    operands = [pointer.handle]
    if mask is not None:
        mask = broadcast_to_shape(to_tensor(mask, _builder), shape, _builder)
        mask = cast(mask, int1, _builder=_builder)
        operands.append(mask.handle)
        if other is not None:
            other = broadcast_to_shape(cast(to_tensor(other, _builder, elem if not isinstance(other, tensor) else None),
                                            elem, _builder=_builder), shape, _builder)
            operands.append(other.handle)
    op = _builder.create("tt.load", operands, [(elem, shape)],
                         {"has_mask": mask is not None, "has_other": mask is not None and other is not None})
    return tensor(op.result, _builder)


@builtin
def store(pointer, value, mask=None, boundary_check=(), cache_modifier="", eviction_policy="",
          _builder=None):
    pointer = to_tensor(pointer, _builder)
    if not pointer.dtype.is_ptr:
        raise CompilationError(f"tl.store expects a pointer, got {pointer.dtype}")
    elem = pointer.dtype.elem
    shape = pointer.handle.shape
    value = to_tensor(value, _builder, elem if not isinstance(value, tensor) else None)
    if value.handle.shape and not shape:
        raise CompilationError("cannot store a block through a scalar pointer")
    value = broadcast_to_shape(cast(value, elem, _builder=_builder), shape, _builder)
    operands = [pointer.handle, value.handle]
    if mask is not None:
        mask = broadcast_to_shape(cast(to_tensor(mask, _builder), int1, _builder=_builder), shape, _builder)
        operands.append(mask.handle)
    _builder.create("tt.store", operands, [], {"has_mask": mask is not None})
    return None


@builtin
def dot(input, other, acc=None, input_precision=None, allow_tf32=None, max_num_imprecise_acc=None,
        out_dtype=float32, _builder=None):
    a, b = to_tensor(input, _builder), to_tensor(other, _builder)
    if len(a.handle.shape) != 2 or len(b.handle.shape) != 2:
        raise CompilationError("tl.dot needs 2D operands")
    (m, k), (k2, n) = a.handle.shape, b.handle.shape
    if k != k2:
        raise CompilationError(f"tl.dot shape mismatch {a.handle.shape} x {b.handle.shape}")
    a = cast(a, float32, _builder=_builder)
    b = cast(b, float32, _builder=_builder)
    if acc is None:
        acc = zeros((m, n), float32, _builder=_builder)
    acc = cast(to_tensor(acc, _builder), float32, _builder=_builder)
    if tuple(acc.handle.shape) != (m, n):
        raise CompilationError("tl.dot accumulator has the wrong shape")
    op = _builder.create("tt.dot", [a.handle, b.handle, acc.handle], [(float32, (m, n))])
    return tensor(op.result, _builder)


@builtin
def where(condition, x, y, _builder=None):
    cond = cast(to_tensor(condition, _builder), int1, _builder=_builder)
    if not isinstance(x, tensor) and not isinstance(y, tensor):
        x = to_tensor(x, _builder)
    x = to_tensor(x, _builder) if isinstance(x, tensor) else x
    if not isinstance(y, tensor):
        y = _scalar_like(y, x, _builder)
    if not isinstance(x, tensor):
        x = _scalar_like(x, y, _builder)
    shape = _broadcast_shapes(_broadcast_shapes(cond.handle.shape, x.handle.shape), y.handle.shape)
    dt = _binary_dtype(x.dtype, y.dtype, "select")
    x = broadcast_to_shape(cast(x, dt, _builder=_builder), shape, _builder)
    y = broadcast_to_shape(cast(y, dt, _builder=_builder), shape, _builder)
    cond = broadcast_to_shape(cond, shape, _builder)
    return _emit(_builder, "arith.select", [cond, x, y], dt, shape)


def _binary_builtin(name_f, name_i, name_u=None):
    def fn(x, y, propagate_nan=None, _builder=None):
        if not isinstance(x, tensor) and not isinstance(y, tensor):
            x = to_tensor(x, _builder)
        a, b, dt, shape = _prepare(x, y, _builder, "minmax")
        name = name_f if dt.is_float else (name_u if dt == uint32 and name_u else name_i)
        return _emit(_builder, name, [a, b], dt, shape)
    return builtin(fn)


maximum = _binary_builtin("arith.maximumf", "arith.maxsi", "arith.maxui")
minimum = _binary_builtin("arith.minimumf", "arith.minsi", "arith.minui")


@builtin
def clamp(x, min, max, propagate_nan=None, _builder=None):
    return minimum(maximum(x, min, _builder=_builder), max, _builder=_builder)


def _float_unary(name):
    def fn(x, _builder=None):
        x = cast(to_tensor(x, _builder), float32, _builder=_builder)
        return _emit(_builder, name, [x], float32, x.handle.shape)
    fn.__name__ = name.split(".")[-1]
    return builtin(fn)


exp = _float_unary("math.exp")
exp2 = _float_unary("math.exp2")
log = _float_unary("math.log")
log2 = _float_unary("math.log2")
sqrt = _float_unary("math.sqrt")
rsqrt = _float_unary("math.rsqrt")
sqrt_rn = sqrt


@builtin
def abs(x, _builder=None):
    x = to_tensor(x, _builder)
    if x.dtype.is_float:
        return _emit(_builder, "math.absf", [x], x.dtype, x.handle.shape)
    return _emit(_builder, "math.absi", [x], x.dtype, x.handle.shape)


@builtin
def fdiv(x, y, ieee_rounding=False, _builder=None):
    return truediv(x, y, _builder)


div_rn = fdiv


@builtin
def fma(x, y, z, _builder=None):
    return add(mul(x, y, _builder), z, _builder)


@builtin
def sigmoid(x, _builder=None):
    x = cast(to_tensor(x, _builder), float32, _builder=_builder)
    return truediv(1.0, add(1.0, exp(neg(x, _builder), _builder=_builder), _builder), _builder)


@builtin
def softmax(x, ieee_rounding=False, _builder=None):
    z = sub(x, max(x, 0, keep_dims=True, _builder=_builder), _builder)
    num = exp(z, _builder=_builder)
    return truediv(num, sum(num, 0, keep_dims=True, _builder=_builder), _builder)


def _reduce(x, axis, kind, keep_dims, builder):
    x = to_tensor(x, builder)
    axis = _unwrap(axis)
    keep_dims = _unwrap(keep_dims)
    if x.dtype.is_bool:
        x = cast(x, int32, _builder=builder)
    if not x.handle.shape:
        return x
    if axis is None:
        flat = reshape(x, (x.handle.numel,), _builder=builder) if len(x.handle.shape) > 1 else x
        out = _reduce(flat, 0, kind, False, builder)
        if keep_dims:
            for _ in x.handle.shape:
                out = expand_dims(out, 0, _builder=builder)
        return out
    rank = len(x.handle.shape)
    if axis < 0:
        axis += rank
    shape = tuple(s for i, s in enumerate(x.handle.shape) if i != axis)
    op = builder.create("tt.reduce", [x.handle], [(x.dtype, shape)], {"axis": axis, "kind": kind})
    out = tensor(op.result, builder)
    if keep_dims:
        out = expand_dims(out, axis, _builder=builder)
    return out


@builtin
def sum(input, axis=None, keep_dims=False, dtype=None, _builder=None):
    return _reduce(input, axis, "sum", keep_dims, _builder)


@builtin
def max(input, axis=None, return_indices=False, return_indices_tie_break_left=True, keep_dims=False,
        _builder=None):
    if _unwrap(return_indices):
        raise CompilationError("tl.max(return_indices=True) is not supported")
    return _reduce(input, axis, "max", keep_dims, _builder)


@builtin
def min(input, axis=None, return_indices=False, return_indices_tie_break_left=True, keep_dims=False,
        _builder=None):
    if _unwrap(return_indices):
        raise CompilationError("tl.min(return_indices=True) is not supported")
    return _reduce(input, axis, "min", keep_dims, _builder)


@builtin
def cdiv(x, div, _builder=None):
    x, div = _unwrap(x), _unwrap(div)
    if not isinstance(x, tensor) and not isinstance(div, tensor):
        return (x + div - 1) // div
    return floordiv(add(x, sub(div, 1, _builder) if isinstance(div, tensor) else div - 1, _builder),
                    div, _builder)


@builtin
def multiple_of(input, values, _builder=None):
    return input


@builtin
def max_contiguous(input, values, _builder=None):
    return input


@builtin
def max_constancy(input, values, _builder=None):
    return input


@builtin
def assume(cond, _builder=None):
    return None


@builtin
def debug_barrier(_builder=None):
    return None


@builtin
def static_assert(cond, msg="", _builder=None):
    if not _unwrap(cond):
        raise CompilationError(f"static_assert failed: {msg}")


@builtin
def static_print(*values, sep=" ", end="\n", file=None, flush=False, _builder=None):
    print(*[_unwrap(v) for v in values], sep=sep, end=end)


@builtin
def device_assert(cond, msg="", _builder=None):
    return None


@builtin
def device_print(prefix, *args, hex=False, _builder=None):
    raise CompilationError("tl.device_print is not supported on the QPU")


class static_range:
    """Compile-time unrolled range (handled by the code generator)."""

    def __init__(self, arg1, arg2=None, step=None):
        if arg2 is None:
            self.start, self.end = 0, _unwrap(arg1)
        else:
            self.start, self.end = _unwrap(arg1), _unwrap(arg2)
        self.step = 1 if step is None else _unwrap(step)


class range:
    """Runtime loop (`scf.for`), like Python `range` inside a kernel."""

    def __init__(self, arg1, arg2=None, step=None, num_stages=None, loop_unroll_factor=None, **_):
        if arg2 is None:
            self.start, self.end = 0, arg1
        else:
            self.start, self.end = arg1, arg2
        self.step = 1 if step is None else step
        self.num_stages = _unwrap(num_stages)
