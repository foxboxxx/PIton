# Build directory template used by examples/digits_pi.py. That host driver
# generates/copies all headers and runtime sources into its isolated directory.
.DEFAULT_GOAL := all
PROGS := main.c
COMMON_SRC := piton_rt.c arm_sgemm.c arm_kernel_vfp.S dense1.c dense2.c dense3.c
OPT_LEVEL = -O3
CFLAGS += -I.
BOOTLOADER = my-install
RUN = 0

include $(CS240LX_2025_PATH)/libpi/mk/Makefile.robust-v2
