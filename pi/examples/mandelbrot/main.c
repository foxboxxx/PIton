// Interactive Mandelbrot on the VideoCore IV QPUs, drawn straight into the
// HDMI framebuffer (double buffered). Keys arrive over the UART (the
// bootloader forwards the terminal): i/o or +/- zoom, w/a/s/d or the arrow
// keys pan, r resets, space starts the automatic tour, q quits. After a few
// idle seconds it tours by itself.
//
// Before the interactive loop it checks every compiled kernel bit for bit
// against counts from the PIton emulator, benchmarks each one, and compares
// the fastest against a float Mandelbrot on the ARM with caches on.
// Generated headers come from examples/mandelbrot_pi.py.
#include "rpi.h"
#include "piton_rt.h"
#include "arm_sgemm.h"
#include "mandel_config.h"
#include "mandel_kernels.h"
#include "mandel_data.h"

enum { W = MANDEL_WIDTH, H = MANDEL_HEIGHT };

#define FULL_VIEW 3.5           // width of the complex plane at zoom 1
#define KEY_ZOOM 1.2            // per key press; held keys auto-repeat
#define KEY_PAN 0.08            // fraction of the view per key press
#define EASE 3.0                // per second: how fast the view approaches its target
#define AUTO_RATE 0.35          // relative view change per second while touring
#define IDLE_US (10 * 1000 * 1000)
#define BENCH_REPEATS 5

static const double bench_zooms[] = {1.0, 100.0, 10000.0};
#define NUM_BENCH (sizeof(bench_zooms) / sizeof(bench_zooms[0]))

// ---------------------------------------------------------------------------
// Framebuffer (mailbox property interface)
// ---------------------------------------------------------------------------

static volatile uint32_t mbox[64] __attribute__((aligned(16)));
static uint32_t fb_bus, fb_pitch;   // bus address of the W x 2H buffer; pitch in bytes

static unsigned tag(unsigned i, uint32_t id, unsigned nwords, uint32_t a0, uint32_t a1) {
    mbox[i++] = id;
    mbox[i++] = nwords * 4;
    mbox[i++] = 0;
    mbox[i++] = a0;
    if (nwords > 1)
        mbox[i++] = a1;
    return i;
}

static int fb_init(void) {
    unsigned i = 2;
    i = tag(i, 0x48003, 2, W, H);          // physical size
    unsigned virt = i;
    i = tag(i, 0x48004, 2, W, 2 * H);      // virtual size: two frames
    i = tag(i, 0x48009, 2, 0, 0);          // virtual offset
    i = tag(i, 0x48005, 1, 32, 0);         // depth
    i = tag(i, 0x48006, 1, 1, 0);          // pixel order RGB
    i = tag(i, 0x48007, 1, 2, 0);          // ignore alpha
    unsigned alloc = i;
    i = tag(i, 0x40001, 2, 4096, 0);       // allocate (alignment)
    unsigned pitch = i;
    i = tag(i, 0x40008, 1, 0, 0);          // pitch
    mbox[i++] = 0;
    mbox[0] = i * 4;
    mbox[1] = 0;
    if (!piton_mbox_property(mbox) || !mbox[alloc + 3])
        return 0;
    if (mbox[virt + 3] != W || mbox[virt + 4] != 2 * H) {
        printk("MANDEL_FB virtual size %dx%d refused (got %dx%d)\n", W, 2 * H,
               mbox[virt + 3], mbox[virt + 4]);
        return 0;
    }
    fb_bus = mbox[alloc + 3];
    fb_pitch = mbox[pitch + 3];
    return 1;
}

static void fb_show(unsigned line) {
    unsigned i = tag(2, 0x48009, 2, 0, line);
    mbox[i++] = 0;
    mbox[0] = i * 4;
    mbox[1] = 0;
    piton_mbox_property(mbox);
}

static uint32_t frame_bus(unsigned frame) {
    return fb_bus + frame * H * fb_pitch;
}

// ---------------------------------------------------------------------------
// Views
// ---------------------------------------------------------------------------

static double absd(double x) { return x < 0 ? -x : x; }

// fp32 coordinates: below ~4 ulp per pixel neighbouring pixels collapse. z
// itself reaches magnitude ~1, so a centre near 0 does not help.
static double min_view(double cx, double cy) {
    double c = absd(cx) > absd(cy) ? absd(cx) : absd(cy);
    return (c > 1.0 ? c : 1.0) * 0x1p-21 * W;
}

// More iterations as the view narrows, a multiple of every kernel's UNROLL.
static unsigned iterations_for(double view) {
    unsigned iters = MANDEL_MIN_ITER;
    for (double z = FULL_VIEW / view; z >= 2.0 && iters < MANDEL_MAX_ITER; z *= 0.5)
        iters += MANDEL_ITER_PER_OCTAVE;
    if (iters > MANDEL_MAX_ITER)
        iters = MANDEL_MAX_ITER;
    return iters & ~(MANDEL_ITER_ALIGN - 1u);
}

typedef struct { float x0, y0, step; unsigned iters; } frame_args_t;

static frame_args_t frame_args(double cx, double cy, double view) {
    double step = view / W;
    return (frame_args_t){ .x0 = (float)(cx - 0.5 * view), .y0 = (float)(cy - 0.5 * step * H),
                           .step = (float)step, .iters = iterations_for(view) };
}

static frame_args_t bench_args(unsigned v) {
    return frame_args(MANDEL_TARGET_X, MANDEL_TARGET_Y, FULL_VIEW / bench_zooms[v]);
}

static uint32_t render(const mandel_config_t *k, uint32_t out, uint32_t palette,
                       unsigned pitch_words, frame_args_t a) {
    return k->run(out, palette, a.x0, a.y0, a.step, W, pitch_words, a.iters, k->grid_x, k->grid_y);
}

// ---------------------------------------------------------------------------
// Checks and benchmarks
// ---------------------------------------------------------------------------

static piton_buf_t palette, identity, counts, frame_counts[NUM_BENCH];
static uint32_t arm_counts[W * H];

static void upload_palette(void) {
    palette = piton_alloc((MANDEL_MAX_ITER + 1) * 4);
    identity = piton_alloc((MANDEL_MAX_ITER + 1) * 4);
    volatile uint32_t *p = palette.cpu, *id = identity.cpu;
    for (unsigned i = 0; i <= MANDEL_MAX_ITER; i++) {
        p[i] = mandel_palette[i];
        id[i] = i;
    }
}

// The emulator models the QPUs bit for bit, so every kernel must reproduce
// its counts exactly.
static int check_patch(const mandel_config_t *k) {
    volatile uint32_t *c = counts.cpu;
    for (unsigned i = 0; i < MANDEL_PATCH_W * MANDEL_PATCH_H; i++)
        c[i] = ~0u;
    k->run(counts.bus, identity.bus, MANDEL_PATCH_X0, MANDEL_PATCH_Y0, MANDEL_PATCH_STEP,
           MANDEL_PATCH_W, MANDEL_PATCH_W, MANDEL_PATCH_ITERS, k->patch_grid_x, k->patch_grid_y);
    unsigned bad = 0;
    for (unsigned i = 0; i < MANDEL_PATCH_W * MANDEL_PATCH_H; i++)
        bad += c[i] != mandel_patch_expected[i];
    printk("MANDEL_EXACT kernel=%s pixels=%d iters=%d mismatched=%d\n", k->name,
           MANDEL_PATCH_W * MANDEL_PATCH_H, MANDEL_PATCH_ITERS, bad);
    return bad == 0;
}

static uint32_t bench_config(const mandel_config_t *k) {
    uint32_t total = 0;
    for (unsigned v = 0; v < NUM_BENCH; v++) {
        frame_args_t a = bench_args(v);
        uint32_t best = ~0u;
        render(k, frame_bus(1), palette.bus, fb_pitch >> 2, a);   // first launch allocates
        for (unsigned r = 0; r < BENCH_REPEATS; r++) {
            uint32_t t = render(k, frame_bus(1), palette.bus, fb_pitch >> 2, a);
            if (t < best)
                best = t;
        }
        fb_show(H);
        printk("MANDEL_BENCH kernel=%s zoom=%d iters=%d best_us=%d fps=%d\n", k->name,
               (unsigned)bench_zooms[v], a.iters, best, (unsigned)(1000000.0f / (float)best));
        total += best;
    }
    return total;
}

static uint32_t arm_render(frame_args_t a) {
    uint32_t start = timer_get_usec();
    for (unsigned row = 0; row < H; row++) {
        float cy = a.y0 + (float)row * a.step;
        for (unsigned col = 0; col < W; col++) {
            float cx = a.x0 + (float)col * a.step, x = 0, y = 0;
            unsigned n = 0;
            for (; n < a.iters; n++) {
                float xx = x * x, yy = y * y;
                if (xx + yy > 4.0f)
                    break;
                float xy = x * y;
                x = xx - yy + cx;
                y = xy + xy + cy;
            }
            arm_counts[row * W + col] = n == a.iters ? 0 : n;   // the kernel's index for the set
        }
    }
    return timer_get_usec() - start;
}

// GPU counts for each benchmark view; read back after the ARM caches are on.
static void gpu_frame_counts(const mandel_config_t *k, uint32_t *gpu_us) {
    for (unsigned v = 0; v < NUM_BENCH; v++) {
        frame_counts[v] = piton_alloc(W * H * 4);
        gpu_us[v] = render(k, frame_counts[v].bus, identity.bus, W, bench_args(v));
    }
}

static void compare_with_arm(const uint32_t *gpu_us) {
    arm_caches_on();
    for (unsigned v = 0; v < NUM_BENCH; v++) {
        frame_args_t a = bench_args(v);
        uint32_t arm_us = arm_render(a);
        volatile const uint32_t *g = frame_counts[v].cpu;
        unsigned differ = 0;
        for (unsigned i = 0; i < W * H; i++)
            differ += g[i] != arm_counts[i];
        // The QPUs round toward zero and the ARM to nearest, so pixels near
        // the boundary can escape on different iterations.
        printk("MANDEL_ARM zoom=%d iters=%d arm_us=%d gpu_us=%d speedup=%f pixels_differing=%d of %d\n",
               (unsigned)bench_zooms[v], a.iters, arm_us, gpu_us[v],
               (float)arm_us / (float)gpu_us[v], differ, W * H);
    }
    arm_caches_off();
}

// ---------------------------------------------------------------------------
// Profile
// ---------------------------------------------------------------------------

#define PROFILE_FRAMES 8
static const double profile_zooms[] = {1.0, 10.0, 100.0, 1000.0, 10000.0};
#define NUM_PROFILE (sizeof(profile_zooms) / sizeof(profile_zooms[0]))
static const uint8_t perf_events[] = {
    PITON_PCTR_QPU_IDLE, PITON_PCTR_QPU_VALID, PITON_PCTR_QPU_TMU_STALL, PITON_PCTR_QPU_SCOREBOARD,
    PITON_PCTR_ICACHE_HIT, PITON_PCTR_ICACHE_MISS, PITON_PCTR_UNIF_HIT, PITON_PCTR_UNIF_MISS,
    PITON_PCTR_TMU_QUADS, PITON_PCTR_TMU_MISS, PITON_PCTR_VDW_STALL, PITON_PCTR_L2_HIT, PITON_PCTR_L2_MISS,
};
static const char *perf_names[] = {
    "qpu_idle", "qpu_valid", "tmu_stall", "sb_stall", "icache_hit", "icache_miss", "unif_hit",
    "unif_miss", "tmu_quads", "tmu_miss", "vdw_stall", "l2_hit", "l2_miss",
};
#define NUM_EVENTS sizeof(perf_events)

// Frames rendered and flipped exactly as in the interactive loop. setup_us is
// the ARM writing every program's uniforms (k->run minus the launch itself).
static void profile_frames(const mandel_config_t *k, double zoom, const char *arm) {
    frame_args_t a = frame_args(MANDEL_TARGET_X, MANDEL_TARGET_Y, FULL_VIEW / zoom);
    double frame = 0, gpu = 0, setup = 0, flip = 0, events[NUM_EVENTS] = {0};
    uint32_t c[NUM_EVENTS];
    unsigned back = 1;
    render(k, frame_bus(back), palette.bus, fb_pitch >> 2, a);
    for (unsigned f = 0; f < PROFILE_FRAMES; f++) {
        uint32_t t0 = timer_get_usec();
        piton_perf_start(perf_events, NUM_EVENTS);
        uint32_t g = render(k, frame_bus(back), palette.bus, fb_pitch >> 2, a);
        piton_perf_read(c, NUM_EVENTS);
        uint32_t t1 = timer_get_usec();
        fb_show(back * H);
        back ^= 1;
        uint32_t t2 = timer_get_usec();
        frame += t2 - t0;
        gpu += g;
        setup += (t1 - t0) - g;
        flip += t2 - t1;
        for (unsigned i = 0; i < NUM_EVENTS; i++)
            events[i] += c[i];
    }
    printk("MANDEL_PROFILE kernel=%s arm=%s zoom=%d iters=%d programs=%d frame_us=%d gpu_us=%d setup_us=%d flip_us=%d",
           k->name, arm, (unsigned)zoom, a.iters, k->grid_x * k->grid_y, (unsigned)(frame / PROFILE_FRAMES),
           (unsigned)(gpu / PROFILE_FRAMES), (unsigned)(setup / PROFILE_FRAMES), (unsigned)(flip / PROFILE_FRAMES));
    for (unsigned i = 0; i < NUM_EVENTS; i++)
        printk(" %s=%d", perf_names[i], (unsigned)(events[i] / PROFILE_FRAMES));
    printk("\n");
}

// How much of the executed work is useful: a block keeps iterating (in
// UNROLL chunks) until its slowest pixel escapes or max_iter is reached.
static void profile_work(const mandel_config_t *k, double zoom, piton_buf_t *buf) {
    frame_args_t a = frame_args(MANDEL_TARGET_X, MANDEL_TARGET_Y, FULL_VIEW / zoom);
    render(k, buf->bus, identity.bus, W, a);
    volatile const uint32_t *v = buf->cpu;
    double useful = 0, executed = 0;
    unsigned inside = 0;
    for (unsigned row = 0; row < H; row += k->tile_h)
        for (unsigned col = 0; col < W; col += k->block) {
            unsigned slowest = 0;
            for (unsigned t = 0; t < k->tile_h; t++)
                for (unsigned i = 0; i < k->block; i++) {
                    unsigned n = v[(row + t) * W + col + i];
                    // n iterations stayed bounded, the next one detected the escape
                    unsigned work = n == 0 ? a.iters : n + 1;
                    inside += n == 0;
                    useful += work;
                    if (work > slowest)
                        slowest = work;
                }
            unsigned chunks = (slowest + k->unroll - 1) & ~(k->unroll - 1);
            executed += (double)(k->block * k->tile_h) * (chunks < a.iters ? chunks : a.iters);
        }
    printk("MANDEL_WORK kernel=%s zoom=%d iters=%d inside_pixels=%d useful_miter=%f executed_miter=%f lane_efficiency=%f\n",
           k->name, (unsigned)zoom, a.iters, inside, useful * 1e-6, executed * 1e-6, useful / executed);
}

static void profile(void) {
    piton_buf_t work = piton_alloc(W * H * 4);
    for (unsigned i = 0; i < MANDEL_NUM_CONFIGS; i++)
        for (unsigned z = 0; z < NUM_PROFILE; z++) {
            profile_frames(&mandel_configs[i], profile_zooms[z], "nocache");
            profile_work(&mandel_configs[i], profile_zooms[z], &work);
        }
    // The instruction cache and branch prediction need no MMU and leave data
    // accesses uncached, so mailbox and GPU buffers are unaffected.
    caches_enable();
    for (unsigned i = 0; i < MANDEL_NUM_CONFIGS; i++)
        for (unsigned z = 0; z < NUM_PROFILE; z++)
            profile_frames(&mandel_configs[i], profile_zooms[z], "icache");
    caches_disable();
}

// ---------------------------------------------------------------------------
// Interactive loop
// ---------------------------------------------------------------------------

typedef struct { double cx, cy, view; } view_t;

static double clampd(double x, double lo, double hi) {
    return x < lo ? lo : x > hi ? hi : x;
}

// Next key from the UART, or -1. Arrow keys (ESC [ A..D) become w/s/d/a.
static int read_key(int *escape) {
    while (uart_has_data()) {
        int c = uart_get8();
        if (*escape == 1) {
            *escape = c == '[' ? 2 : 0;
        } else if (*escape == 2) {
            *escape = 0;
            if (c >= 'A' && c <= 'D')
                return "wsda"[c - 'A'];
        } else if (c == 27) {
            *escape = 1;
        } else {
            return c >= 'A' && c <= 'Z' ? c + ('a' - 'A') : c;
        }
    }
    return -1;
}

// Keys move a target view; the shown view eases toward it, so taps animate
// smoothly and held (auto-repeating) keys zoom or pan continuously.
static void interactive(const mandel_config_t *k) {
    const view_t home = { MANDEL_TARGET_X, MANDEL_TARGET_Y, FULL_VIEW };
    view_t shown = home, target = home;
    int touring = 0, tour_direction = 1, escape = 0, c;
    unsigned back = 1, frames = 0;
    uint32_t gpu_total = 0, last = timer_get_usec(), report = last, last_input = last;
    while (uart_has_data())          // keys typed during the benchmarks
        uart_get8();
    printk("MANDEL_INTERACTIVE kernel=%s keys: i/o zoom, w/a/s/d or arrows pan, "
           "r reset, space tour, q quit\n", k->name);
    while (1) {
        uint32_t now = timer_get_usec();
        double dt = clampd((double)(now - last) * 1e-6, 0.0, 0.1);
        last = now;
        while ((c = read_key(&escape)) >= 0) {
            last_input = now;
            if (c == 'q') {
                printk("MANDEL_QUIT\n");
                return;
            }
            if (c == ' ') {
                touring = !touring;
                target = shown;
                continue;
            }
            touring = 0;
            double pan = KEY_PAN * target.view;
            switch (c) {
            case 'i': case '+': case '=': target.view /= KEY_ZOOM; break;
            case 'o': case '-': case '_': target.view *= KEY_ZOOM; break;
            case 'w': target.cy -= pan; break;    // row 0 is the smallest imaginary part
            case 's': target.cy += pan; break;
            case 'a': target.cx -= pan; break;
            case 'd': target.cx += pan; break;
            case 'r': target = home; break;
            }
        }
        if (MANDEL_AUTO && !touring && now - last_input > IDLE_US)
            touring = 1;
        if (touring) {
            target.view *= tour_direction > 0 ? 1.0 - AUTO_RATE * dt : 1.0 + AUTO_RATE * dt;
            if (target.view <= min_view(target.cx, target.cy) * 1.01)
                tour_direction = -1;
            else if (target.view >= FULL_VIEW)
                tour_direction = 1;
        }
        // Keep the target in the interesting region and within reach, so
        // fast key repeat cannot run far ahead of what is on screen.
        target.cx = clampd(target.cx, -2.5, 1.5);
        target.cy = clampd(target.cy, -2.0, 2.0);
        target.cx = clampd(target.cx, shown.cx - 0.5 * shown.view, shown.cx + 0.5 * shown.view);
        target.cy = clampd(target.cy, shown.cy - 0.5 * shown.view, shown.cy + 0.5 * shown.view);
        target.view = clampd(target.view, min_view(target.cx, target.cy), FULL_VIEW);
        target.view = clampd(target.view, 0.5 * shown.view, 2.0 * shown.view);
        double f = clampd(EASE * dt, 0.0, 1.0);
        shown.cx += (target.cx - shown.cx) * f;
        shown.cy += (target.cy - shown.cy) * f;
        shown.view += (target.view - shown.view) * f;

        frame_args_t a = frame_args(shown.cx, shown.cy, shown.view);
        gpu_total += render(k, frame_bus(back), palette.bus, fb_pitch >> 2, a);
        fb_show(back * H);
        back ^= 1;
        frames++;
        if (now - report >= 1000000) {
            float seconds = (float)(now - report) * 1e-6f;
            printk("MANDEL_FPS fps=%d gpu_us=%d zoom=%d iters=%d%s\n",
                   (unsigned)((float)frames / seconds), (unsigned)((float)gpu_total / (float)frames),
                   (unsigned)(FULL_VIEW / shown.view), a.iters, touring ? " (tour)" : "");
            frames = 0;
            gpu_total = 0;
            report = now;
        }
    }
}

void notmain(void) {
    piton_init();
    printk("PIton Mandelbrot: %dx%d, %d QPUs, %d kernels\n", W, H, piton_qpu_count(), MANDEL_NUM_CONFIGS);
    static const unsigned clocks[] = {PITON_CLOCK_ARM, PITON_CLOCK_CORE, PITON_CLOCK_V3D, PITON_CLOCK_SDRAM};
    static const char *names[] = {"arm", "core", "v3d", "sdram"};
    for (unsigned i = 0; i < 4; i++)
        printk("MANDEL_CLOCK name=%s mhz=%d\n", names[i], piton_clock_hz(clocks[i], 0) / 1000000);
    if (!fb_init())
        panic("framebuffer allocation failed (is an HDMI monitor connected?)\n");
    printk("MANDEL_FB bus=%x pitch=%d\n", fb_bus, fb_pitch);
    upload_palette();
    counts = piton_alloc(MANDEL_PATCH_W * MANDEL_PATCH_H * 4);

    int passed = 1;
    if (MANDEL_PROFILE_ONLY) {
        for (unsigned i = 0; i < MANDEL_NUM_CONFIGS; i++)
            passed &= check_patch(&mandel_configs[i]);
        profile();
        printk(passed ? "MANDEL_CHECK_OK\n" : "MANDEL_CHECK_FAILED\n");
        printk("PITON_DONE\n");
        return;
    }
    unsigned best = 0;
    uint32_t best_total = ~0u;
    for (unsigned i = 0; i < MANDEL_NUM_CONFIGS; i++) {
        passed &= check_patch(&mandel_configs[i]);
        uint32_t total = bench_config(&mandel_configs[i]);
        if (total < best_total) {
            best_total = total;
            best = i;
        }
    }
    const mandel_config_t *k = &mandel_configs[best];
    printk("MANDEL_BEST kernel=%s\n", k->name);
    uint32_t gpu_us[NUM_BENCH];
    gpu_frame_counts(k, gpu_us);
    compare_with_arm(gpu_us);
    printk(passed ? "MANDEL_CHECK_OK\n" : "MANDEL_CHECK_FAILED\n");
    if (MANDEL_BENCH_ONLY) {
        printk("PITON_DONE\n");
        return;
    }
    interactive(k);
}
