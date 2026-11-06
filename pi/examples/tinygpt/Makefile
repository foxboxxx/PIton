# Build directory template used by examples/tinygpt_pi.py, which generates
# tg_model.h, tg_kernels.h, tg_*.c and kernels.mk next to this file.
.DEFAULT_GOAL := all
include kernels.mk
PROGS := main.c
COMMON_SRC := piton_rt.c $(TG_KERNEL_SRC)
OPT_LEVEL = -O3
# VFP for the ARM baseline (softfp: link-compatible with the soft-float libpi)
CFLAGS += -I. -mfpu=vfp -mfloat-abi=softfp -fno-math-errno
# libgcc for the few helpers gcc calls (the course libpi does not link it)
LIB_POST += $(shell arm-none-eabi-gcc -print-libgcc-file-name)
BOOTLOADER = pi-install
RUN = 0

include $(CS240LX_2025_PATH)/libpi/mk/Makefile.robust-v2
