# Build directory template used by examples/life_pi.py. That host driver
# generates the kernels and life_config.h and copies the runtime sources here.
.DEFAULT_GOAL := all
PROGS := main.c
COMMON_SRC := piton_rt.c arm_sgemm.c arm_kernel_vfp.S life_arm.c life_tile.c life_rows.c life_bits.c
OPT_LEVEL = -O3
CFLAGS += -I.
BOOTLOADER = my-install
RUN = 0

include $(CS240LX_2025_PATH)/libpi/mk/Makefile.robust-v2
