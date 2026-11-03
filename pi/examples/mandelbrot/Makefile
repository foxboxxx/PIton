# Build directory template used by examples/mandelbrot_pi.py. That host driver
# generates the kernels, headers and kernels.mk, and copies the runtime here.
.DEFAULT_GOAL := all
include kernels.mk
PROGS := main.c
COMMON_SRC := piton_rt.c arm_sgemm.c arm_kernel_vfp.S $(MANDEL_KERNEL_SRC)
OPT_LEVEL = -O3
CFLAGS += -I.
BOOTLOADER = my-install
RUN = 0

include $(CS240LX_2025_PATH)/libpi/mk/Makefile.robust-v2
