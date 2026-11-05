// Host stand-in for libpi's rpi.h: examples/life_pi.py --dry-run compiles
// main.c natively with -DLIFE_HOST against this and host_shim.c.
#ifndef LIFE_HOST_RPI_H
#define LIFE_HOST_RPI_H
#include <stdint.h>
#include <stdio.h>

#define printk printf
uint32_t timer_get_usec(void);
void clean_reboot(void) __attribute__((noreturn));

#endif
