# Build directory template used by examples/image_filters_pi.py. That host
# driver generates the kernels, if_config.h, if_kernels.h and kernels.mk, and
# copies these sources and the runtime into its own build directory:
#
#   .venv/bin/python examples/image_filters_pi.py --build-only   # build, no boot
#   .venv/bin/python examples/image_filters_pi.py                # build, boot, check
.DEFAULT_GOAL := all
include kernels.mk
PROGS := main.c
COMMON_SRC := piton_rt.c arm_sgemm.c arm_kernel_vfp.S arm_filters.c $(IF_KERNEL_SRC)
OPT_LEVEL = -O3
CFLAGS += -I.
BOOTLOADER = my-install
RUN = 0

include $(CS240LX_2025_PATH)/libpi/mk/Makefile.robust-v2
