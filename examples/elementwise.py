import piton as triton
import piton.language as tl


@triton.jit
def mul_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=m) * tl.load(y_ptr + offs, mask=m), mask=m)
