# Builds the Triton tutorial matmul with PIton and a bare-metal benchmark
# against the ARM1176 (caches on, -O3, naive and blocked SGEMM).
#   make            # MM=256 by default
#   make MM=512 BK=8          # GPU tile: BM, BN, BK (default 8, 64, 4)
# Needs CS240LX_2025_PATH (course libpi) and python3 with numpy.
.DEFAULT_GOAL := all
MM ?= 256
MM_K ?= $(MM)
BM ?= 8
BN ?= 64
BK ?= 4
PITON_ROOT := ../../..
PYTHON ?= python3

PROGS := main.c
COMMON_SRC := piton_rt.c matmul.c arm_sgemm.c arm_kernel_vfp.S
OPT_LEVEL = -O3
CFLAGS += -DMM=$(MM) -DMM_K=$(MM_K) -DBM=$(BM) -DBN=$(BN) -DBK=$(BK) -I.
BOOTLOADER = my-install
RUN = 1

CONFIG_STAMP := .matmul-config
CONFIG_VALUES := MM=$(MM) MM_K=$(MM_K) BM=$(BM) BN=$(BN) BK=$(BK)
PITON_COMPILER_SRC := $(wildcard $(PITON_ROOT)/piton/*.py $(PITON_ROOT)/piton/*/*.py)
DEPS += $(CONFIG_STAMP)

# Older make versions compare timestamps only to the second. A changed
# configuration must force its consumers even for back-to-back invocations.
ifneq ($(shell cat $(CONFIG_STAMP) 2>/dev/null),$(CONFIG_VALUES))
.PHONY: $(CONFIG_STAMP)
endif

.PHONY: matmul-config-force
matmul-config-force:

# Keep the timestamp stable when the command-line configuration is unchanged.
$(CONFIG_STAMP): matmul-config-force
	@printf '%s\n' 'MM=$(MM)' 'MM_K=$(MM_K)' 'BM=$(BM)' 'BN=$(BN)' 'BK=$(BK)' > $@.tmp
	@if cmp -s $@.tmp $@; then rm -f $@.tmp; else mv -f $@.tmp $@; fi

# One primary target owns both compiler outputs, including under make -j.
matmul.c: Makefile $(PITON_ROOT)/examples/kernels.py $(PITON_COMPILER_SRC) $(CONFIG_STAMP)
	PYTHONPATH=$(PITON_ROOT) $(PYTHON) -m piton.tools.compile $(PITON_ROOT)/examples/kernels.py \
	  -n matmul_kernel --out-name matmul -o . \
	  -s "a_ptr=*fp32,b_ptr=*fp32,c_ptr=*fp32,M=i32,N=i32,stride_am=i32,stride_bk=i32,stride_cm=i32" \
	  -c "K=$(MM_K),stride_ak=1,stride_bn=1,stride_cn=1,BLOCK_SIZE_M=$(BM),BLOCK_SIZE_N=$(BN),BLOCK_SIZE_K=$(BK),GROUP_SIZE_M=4"

# Recover if only the generated header was deleted. All object builds below
# wait for this target, so none can consume the C file during regeneration.
matmul.h: matmul.c
	@if test ! -f $@; then rm -f matmul.c; $(MAKE) matmul.c; fi

piton_rt.c: $(PITON_ROOT)/pi/runtime/piton_rt.c
	cp $(PITON_ROOT)/pi/runtime/piton_rt.c $(PITON_ROOT)/pi/runtime/piton_rt.h .

main.c: matmul.h piton_rt.c

include $(CS240LX_2025_PATH)/libpi/mk/Makefile.robust-v2

$(all_objs): $(CONFIG_STAMP) matmul.h

clean::
	rm -f matmul.c matmul.h matmul.qasm piton_rt.c piton_rt.h $(CONFIG_STAMP) $(CONFIG_STAMP).tmp
