#include "rpi.h"
#include "piton_rt.h"

// ---------------------------------------------------------------------------
// Mailbox property interface (channel 8)
// ---------------------------------------------------------------------------

#define MAILBOX_BASE   0x2000B880
#define MAILBOX_READ   (MAILBOX_BASE + 0x00)
#define MAILBOX_STATUS (MAILBOX_BASE + 0x18)
#define MAILBOX_WRITE  (MAILBOX_BASE + 0x20)
#define MAILBOX_FULL   0x80000000
#define MAILBOX_EMPTY  0x40000000

static void mbox_write(uint32_t ch, uint32_t data) {
    while (GET32(MAILBOX_STATUS) & MAILBOX_FULL)
        ;
    PUT32(MAILBOX_WRITE, (data & ~0xF) | (ch & 0xF));
}

static uint32_t mbox_read(uint32_t ch) {
    while (1) {
        while (GET32(MAILBOX_STATUS) & MAILBOX_EMPTY)
            ;
        uint32_t d = GET32(MAILBOX_READ);
        if ((d & 0xF) == ch)
            return d & ~0xF;
    }
}

static uint32_t mbox_call1(uint32_t tag, uint32_t nwords, uint32_t a0, uint32_t a1, uint32_t a2) {
    static volatile uint32_t p[16] __attribute__((aligned(16)));
    p[0] = (6 + nwords) * 4;
    p[1] = 0;
    p[2] = tag;
    p[3] = nwords * 4;
    p[4] = nwords * 4;
    p[5] = a0;
    p[6] = a1;
    p[7] = a2;
    p[5 + nwords] = 0;
    dev_barrier();
    mbox_write(8, (uint32_t)p);
    while (mbox_read(8) != (uint32_t)p)
        ;
    dev_barrier();
    if (p[1] != 0x80000000)
        panic("mailbox tag %x failed: %x\n", tag, p[1]);
    return p[5];
}

// ---------------------------------------------------------------------------
// V3D registers
// ---------------------------------------------------------------------------

#define V3D_BASE     0x20C00000
#define V3D_IDENT0   (V3D_BASE + 0x000)
#define V3D_IDENT1   (V3D_BASE + 0x004)
#define V3D_L2CACTL  (V3D_BASE + 0x020)
#define V3D_SLCACTL  (V3D_BASE + 0x024)
#define V3D_SRQPC    (V3D_BASE + 0x430)
#define V3D_SRQUA    (V3D_BASE + 0x434)
#define V3D_SRQCS    (V3D_BASE + 0x43c)
#define V3D_DBCFG    (V3D_BASE + 0xe00)
#define V3D_DBQITE   (V3D_BASE + 0xe2c)
#define V3D_DBQITC   (V3D_BASE + 0xe30)

#define MEM_FLAG_L1_NONALLOCATING 0xC

void piton_init(void) {
    if (mbox_call1(0x30012, 1, 1, 0, 0) != 0)
        panic("qpu_enable failed\n");
    if (GET32(V3D_IDENT0) != 0x02443356)
        panic("V3D not powered (IDENT0=%x): use start.elf (not start_cd.elf) and gpu_mem>=64\n",
              GET32(V3D_IDENT0));
}

void piton_shutdown(void) {
    mbox_call1(0x30012, 1, 0, 0, 0);
}

uint32_t piton_qpu_count(void) {
    uint32_t id1 = GET32(V3D_IDENT1);
    return ((id1 >> 4) & 0xF) * ((id1 >> 8) & 0xF);
}

piton_buf_t piton_alloc(uint32_t size) {
    piton_buf_t b;
    size = (size + 4095) & ~4095u;
    b.size = size;
    b.handle = mbox_call1(0x3000c, 3, size, 4096, MEM_FLAG_L1_NONALLOCATING);
    if (!b.handle)
        panic("GPU mem_alloc(%d) failed\n", size);
    b.bus = mbox_call1(0x3000d, 1, b.handle, 0, 0);
    b.cpu = (volatile void *)(b.bus - PITON_GPU_BASE);
    return b;
}

void piton_free(piton_buf_t *b) {
    mbox_call1(0x3000e, 1, b->handle, 0, 0);
    mbox_call1(0x3000f, 1, b->handle, 0, 0);
}

uint32_t piton_launch(uint32_t code, const uint32_t *unifs, uint32_t n) {
    PUT32(V3D_DBCFG, 0);           // no QPU interrupts to the host
    PUT32(V3D_DBQITE, 0);
    PUT32(V3D_DBQITC, -1);
    PUT32(V3D_L2CACTL, 1 << 2);    // clear the V3D L2 cache
    PUT32(V3D_SLCACTL, -1);        // and the slice caches (TMU, uniforms, instr)
    PUT32(V3D_SRQCS, (1 << 7) | (1 << 8) | (1 << 16));   // reset counters / error

    uint32_t t0 = timer_get_usec();
    uint32_t done = 0, last = 0, issued = 0;
    while (done < n) {
        if (timer_get_usec() - t0 > PITON_LAUNCH_TIMEOUT_US) {
            // a hung kernel would otherwise wedge the board until a power cycle
            printk("PANIC: QPU launch timed out (%d of %d programs finished)\n", done, n);
            clean_reboot();
        }
        uint32_t cs = GET32(V3D_SRQCS);
        uint32_t cnt = (cs >> 16) & 0xff;
        done += (cnt - last) & 0xff;
        last = cnt;
        if (cs & (1 << 7))
            panic("V3D request queue overflow\n");
        // keep at most 15 queued requests (the FIFO holds 16)
        while (issued < n && (GET32(V3D_SRQCS) & 0x3f) < 15) {
            PUT32(V3D_SRQUA, unifs[issued]);
            PUT32(V3D_SRQPC, code);
            issued++;
        }
    }
    return timer_get_usec() - t0;
}
