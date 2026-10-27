import piton as triton
import piton.language as tl


@triton.jit
def busy_store(out_ptr, ITERS: tl.constexpr):
    """Each program spins a little, then stores its program id 16 times."""
    pid = tl.program_id(0)
    acc = tl.zeros((16,), dtype=tl.float32)
    for i in range(ITERS):
        acc += 1.0
    v = tl.zeros((16,), dtype=tl.int32) + pid + (acc > 1e9).to(tl.int32)
    tl.store(out_ptr + pid * 16 + tl.arange(0, 16), v)
