# Builds the Triton tutorial matmul with PIton and a bare-metal benchmark
# against the ARM1176 (caches on, -O3, naive and blocked SGEMM).
#   make            # MM=256 by default
#   make MM=512 BK=8
# Needs CS240LX_2025_PATH (course libpi) and python3 with numpy.
.DEFAULT_GOAL := all
MM ?= 256
MM_K ?= $(MM)
BK ?= 4
PITON_ROOT := ../../..
PYTHON ?= python3

PROGS := main.c
COMMON_SRC := piton_rt.c matmul.c arm_sgemm.c arm_kernel_vfp.S
OPT_LEVEL = -O3
CFLAGS += -DMM=$(MM) -DMM_K=$(MM_K) -DBK=$(BK) -I.
BOOTLOADER = my-install
RUN = 1

matmul.c matmul.h: $(PITON_ROOT)/examples/kernels.py
	PYTHONPATH=$(PITON_ROOT) $(PYTHON) -m piton.tools.compile $(PITON_ROOT)/examples/kernels.py \
	  -n matmul_kernel --out-name matmul -o . \
	  -s "a_ptr=*fp32,b_ptr=*fp32,c_ptr=*fp32,M=i32,N=i32,stride_am=i32,stride_bk=i32,stride_cm=i32" \
	  -c "K=$(MM_K),stride_ak=1,stride_bn=1,stride_cn=1,BLOCK_SIZE_M=8,BLOCK_SIZE_N=64,BLOCK_SIZE_K=$(BK),GROUP_SIZE_M=4"

piton_rt.c: $(PITON_ROOT)/pi/runtime/piton_rt.c
	cp $(PITON_ROOT)/pi/runtime/piton_rt.c $(PITON_ROOT)/pi/runtime/piton_rt.h .

main.c: matmul.h piton_rt.c

include $(CS240LX_2025_PATH)/libpi/mk/Makefile.robust-v2

clean::
	rm -f matmul.c matmul.h matmul.qasm piton_rt.c piton_rt.h
