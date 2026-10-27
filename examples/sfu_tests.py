import piton as triton
import piton.language as tl


@triton.jit
def sfu_kernel(x_ptr, out_ptr, n, OP: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m, other=1.0)
    if OP == 0:
        y = tl.exp2(x)
    elif OP == 1:
        y = tl.log2(x)
    elif OP == 2:
        y = 1.0 / x
    elif OP == 3:
        y = tl.rsqrt(x)
    elif OP == 4:
        y = tl.exp(x)
    else:
        y = tl.sqrt(x)
    tl.store(out_ptr + offs, y, mask=m)
