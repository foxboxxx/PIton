# Build directory template used by examples/attention_pi.py, which generates the
# kernels (attn_*.c/.h), the case table (attn_cases.c/.h, kernels.mk) and the
# embedded inputs (attn_data.h), and copies the runtime sources next to this file.
.DEFAULT_GOAL := all
include kernels.mk
PROGS := main.c
COMMON_SRC := piton_rt.c arm_sgemm.c arm_kernel_vfp.S arm_attention.c attn_cases.c $(ATTN_KERNEL_SRC)
OPT_LEVEL = -O3
CFLAGS += -I.
BOOTLOADER = my-install
RUN = 0

include $(CS240LX_2025_PATH)/libpi/mk/Makefile.robust-v2
