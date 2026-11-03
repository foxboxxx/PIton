# Per-primitive benchmark: every supported Triton operation, PIton-compiled for
# the 12 QPUs, against every CPU baseline we can write for it on the ARM1176 of
# the same board.
#
#   make                 # compile the kernels, build, boot and run
#   make RUN=0           # build only
#   make clean
#
# Needs CS240LX_2025_PATH (course libpi) and python3 with numpy. Point PYTHON at
# the project venv if numpy is not on the default python3:
#
#   make PYTHON=../../../.venv/bin/python
.DEFAULT_GOAL := all
PITON_ROOT := ../../..
PYTHON ?= python3

PROGS := main.c
OPT_LEVEL = -O3
CFLAGS += -I.
BOOTLOADER = my-install
RUN = 1

# Kernel sources the generated code depends on, and the compiler itself.
PITON_COMPILER_SRC := $(wildcard $(PITON_ROOT)/piton/*.py $(PITON_ROOT)/piton/*/*.py)
KERNEL_SRC := $(PITON_ROOT)/examples/kernels.py \
              $(PITON_ROOT)/examples/elementwise.py \
              $(PITON_ROOT)/examples/sfu_tests.py \
              $(PITON_ROOT)/examples/primitive_kernels.py

# The generated launchers, named by the same manifest main.c is built against.
# --list-sources does not import piton, so this stays cheap on every make.
GEN_SRC := $(shell $(PYTHON) build_kernels.py --list-sources)
GEN_HDR := $(GEN_SRC:.c=.h)
GEN_QASM := $(GEN_SRC:.c=.qasm)

# arm_sgemm.c carries the MMU/cache setup, the cycle counter and the matmul
# baselines; copied rather than referenced, since every source must live in this
# directory or below it.
COPIED := piton_rt.c piton_rt.h arm_sgemm.c arm_sgemm.h arm_kernel_vfp.S

COMMON_SRC := piton_rt.c arm_sgemm.c arm_kernel_vfp.S arm_ref.c arm_vfp_ops.S $(GEN_SRC)
DEPS += ops.h

# One primary target owns every compiler output, including under make -j.
ops.h: build_kernels.py $(KERNEL_SRC) $(PITON_COMPILER_SRC) | piton_rt.h
	$(PYTHON) build_kernels.py --out .

# Recover if a generated file was deleted but ops.h survived.
$(GEN_SRC) $(GEN_HDR): ops.h
	@if test ! -f $@; then rm -f ops.h; $(MAKE) ops.h; fi

piton_rt.c piton_rt.h: $(PITON_ROOT)/pi/runtime/piton_rt.c $(PITON_ROOT)/pi/runtime/piton_rt.h
	cp $(PITON_ROOT)/pi/runtime/piton_rt.c $(PITON_ROOT)/pi/runtime/piton_rt.h .

arm_sgemm.c arm_sgemm.h arm_kernel_vfp.S: $(PITON_ROOT)/pi/examples/matmul/arm_sgemm.c \
                                          $(PITON_ROOT)/pi/examples/matmul/arm_sgemm.h \
                                          $(PITON_ROOT)/pi/examples/matmul/arm_kernel_vfp.S
	cp $(PITON_ROOT)/pi/examples/matmul/arm_sgemm.c \
	   $(PITON_ROOT)/pi/examples/matmul/arm_sgemm.h \
	   $(PITON_ROOT)/pi/examples/matmul/arm_kernel_vfp.S .

main.c: ops.h $(COPIED)

include $(CS240LX_2025_PATH)/libpi/mk/Makefile.robust-v2

$(all_objs): ops.h $(COPIED)

clean::
	rm -f $(GEN_SRC) $(GEN_HDR) $(GEN_QASM) ops.h $(COPIED)
