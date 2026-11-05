// Image filters on the VideoCore IV QPUs against the ARM1176, in one boot.
//
// Built by examples/image_filters_pi.py, which generates if_config.h (sizes,
// repeats, scene constants) and if_kernels.h (the PIton-compiled filters and
// their launch wrappers) into an isolated build directory. The scene is
// generated on the board (scene.h), so nothing large is uploaded.
//
// 1. Check size (ragged, e.g. 83 x 61): every QPU filter, row hashes printed.
// 2. Full size: every QPU filter once (setup), once more (row hashes; filters
//    with the same output compared bit for bit), then IF_REPEATS timed runs
//    with every ARM cache off and IF_REPEATS with the instruction cache on
//    (the output is hashed again afterwards).
// 3. ARM caches on: every ARM baseline once (compared with the QPU output),
//    then IF_REPEATS timed runs.
// The host compares every row hash with its model (numpy + PIton's sqrt).
// All GPU/mailbox work precedes enabling the ARM data cache.
#include "rpi.h"
#include "piton_rt.h"
#include "arm_sgemm.h"
#include "arm_filters.h"
#include "scene.h"
#include "row_hash.h"
#include "if_config.h"
#include "if_kernels.h"

#define NPIX (IF_W * IF_H)

static float arm_src[NPIX], arm_tmp[NPIX], arm_dst[NPIX];
static piton_buf_t gpu_src, gpu_tmp, gpu_scratch, gpu_out[IF_NUM_OUTPUTS];
static int passed = 1;
static int computed[IF_NUM_OUTPUTS];   // some QPU filter wrote gpu_out[i]

static int streq(const char *a, const char *b) {
    while (*a && *a == *b)
        a++, b++;
    return *a == *b;
}

static unsigned output_index(const char *name) {
    for (unsigned i = 0; i < IF_NUM_OUTPUTS; i++)
        if (streq(if_outputs[i], name))
            return i;
    panic("unknown output %s\n", name);
    return 0;
}

static void fill_nan(volatile uint32_t *p, unsigned n) {
    for (unsigned i = 0; i < n; i++)
        p[i] = 0x7fc00000u;
}

static uint32_t image_hash(volatile const uint32_t *img, int h, int w) {
    uint32_t sum = 0;
    for (int y = 0; y < h; y++)
        sum += row_hash(img + y * w, w);
    return sum;
}

static void hex8(char *out, uint32_t v) {
    static const char digits[] = "0123456789abcdef";
    for (int i = 7; i >= 0; i--, v >>= 4)
        out[i] = digits[v & 15];
}

static void print_rows(const char *filter, const char *size, volatile const uint32_t *img, int h, int w) {
    char line[8 * 9 + 1];
    printk("IF_ROWS filter=%s size=%s h=%d w=%d\n", filter, size, h, w);
    for (int y0 = 0; y0 < h; y0 += 8) {
        int n = 0;
        for (int y = y0; y < y0 + 8 && y < h; y++, n += 9) {
            hex8(line + n, row_hash(img + y * w, w));
            line[n + 8] = ' ';
        }
        line[n - 1] = 0;
        printk("%s\n", line);
    }
    printk("IF_ROWS_END\n");
}

static void generate(const scene_params_t *p, volatile float *gpu, float *arm) {
    for (int y = 0; y < p->h; y++)
        for (int x = 0; x < p->w; x++) {
            float v = (float)scene_pixel(p, x, y);
            gpu[y * p->w + x] = v;
            if (arm)
                arm[y * p->w + x] = v;
        }
}

static void check_size(void) {
    static const scene_params_t scene = IF_SCENE_CHECK;
    generate(&scene, gpu_src.cpu, 0);
    printk("IF_SCENE size=check h=%d w=%d hash=%u\n", scene.h, scene.w,
           image_hash(gpu_src.cpu, scene.h, scene.w));
    for (unsigned f = 0; f < IF_NUM_FILTERS; f++) {
        fill_nan(gpu_scratch.cpu, scene.h * scene.w);
        if_filters[f].run(gpu_src.bus, gpu_scratch.bus, gpu_tmp.bus, scene.h, scene.w);
        print_rows(if_filters[f].name, "check", gpu_scratch.cpu, scene.h, scene.w);
    }
}

static void qpu_full_size(void) {
    for (unsigned f = 0; f < IF_NUM_FILTERS; f++) {
        const if_filter_t *k = &if_filters[f];
        unsigned o = output_index(k->output);
        piton_buf_t *out = k->first ? &gpu_out[o] : &gpu_scratch;
        computed[o] |= k->first;
        fill_nan(out->cpu, NPIX);
        k->run(gpu_src.bus, out->bus, gpu_tmp.bus, IF_H, IF_W);    // allocates the launcher's buffers
        fill_nan(out->cpu, NPIX);
        k->run(gpu_src.bus, out->bus, gpu_tmp.bus, IF_H, IF_W);
        print_rows(k->name, "full", out->cpu, IF_H, IF_W);
        uint32_t before = image_hash(out->cpu, IF_H, IF_W);
        if (!k->first) {
            volatile const uint32_t *a = out->cpu, *b = gpu_out[o].cpu;
            unsigned mismatches = 0;
            for (unsigned i = 0; i < NPIX; i++)
                mismatches += a[i] != b[i];
            printk("IF_SAME filter=%s output=%s mismatches=%u\n", k->name, k->output, mismatches);
            passed &= mismatches == 0;
        }
        // "qpu": the ARM side of the launch (uniforms, queueing) runs with
        // every ARM cache off, as in the other PIton benchmarks; "qpu_icache"
        // with the instruction cache and branch prediction on
        for (int icache = 0; icache <= 1; icache++) {
            if (!icache)
                caches_disable();
            for (unsigned r = 0; r < IF_REPEATS; r++) {
                uint32_t t0 = timer_get_usec();
                k->run(gpu_src.bus, out->bus, gpu_tmp.bus, IF_H, IF_W);
                uint32_t t = timer_get_usec() - t0;
                printk("IF_TIME impl=%s name=%s output=%s us=%u\n", icache ? "qpu_icache" : "qpu",
                       k->name, k->output, t);
            }
            if (!icache)
                caches_enable();
        }
        uint32_t after = image_hash(out->cpu, IF_H, IF_W);
        printk("IF_STABLE filter=%s same=%d\n", k->name, before == after);
        passed &= before == after;
    }
}

typedef struct {
    const char *name, *output;
    int border;                     // pixels at each edge not compared (two-pass vs fused)
    int max_ulp;                    // allowed difference from the QPU (0: bit for bit)
    void (*run)(void);
} arm_impl_t;

static void run_blur3(void) { arm_blur3(arm_src, arm_dst, IF_H, IF_W); }
static void run_blur5(void) { arm_blur5(arm_src, arm_dst, IF_H, IF_W); }
static void run_blur5_sep(void) { arm_blur5_separable(arm_src, arm_tmp, arm_dst, IF_H, IF_W); }
static void run_sobel(void) { arm_sobel(arm_src, arm_dst, IF_H, IF_W); }
static void run_blur_sobel(void) { arm_blur_sobel(arm_src, arm_dst, IF_H, IF_W); }
static void run_blur_sobel_2(void) { arm_blur_sobel_two_pass(arm_src, arm_tmp, arm_dst, IF_H, IF_W); }

static const arm_impl_t arm_impls[] = {
    {"arm_blur3", "blur3", 0, 0, run_blur3},
    {"arm_blur5", "blur5", 0, 0, run_blur5},
    {"arm_blur5_separable", "blur5", 0, 0, run_blur5_sep},
    {"arm_sobel", "sobel", 0, IF_MAX_ULP, run_sobel},
    {"arm_blur_sobel", "blur_sobel", 0, IF_MAX_ULP, run_blur_sobel},
    {"arm_blur_sobel_two_pass", "blur_sobel", 2, IF_MAX_ULP, run_blur_sobel_2},
};

// Both images are non-negative floats, so their bit patterns order like the values.
static void compare(const arm_impl_t *impl) {
    volatile const uint32_t *q = gpu_out[output_index(impl->output)].cpu;
    const uint32_t *a = (const uint32_t *)arm_dst;
    unsigned checked = 0, exact = 0, failures = 0, max_ulp = 0;
    for (int y = impl->border; y < IF_H - impl->border; y++)
        for (int x = impl->border; x < IF_W - impl->border; x++) {
            uint32_t qb = q[y * IF_W + x], ab = a[y * IF_W + x];
            unsigned ulp = qb > ab ? qb - ab : ab - qb;
            int finite = (qb & 0x7f800000u) != 0x7f800000u && (ab & 0x7f800000u) != 0x7f800000u;
            checked++;
            exact += ulp == 0;
            failures += !finite || qb >> 31 || ab >> 31 || ulp > (unsigned)impl->max_ulp;
            if (ulp > max_ulp)
                max_ulp = ulp;
        }
    printk("IF_CHECK arm=%s output=%s border=%d checked=%u exact=%u max_ulp=%u failures=%u\n",
           impl->name, impl->output, impl->border, checked, exact, max_ulp, failures);
    passed &= failures == 0;
}

static void arm_all(void) {
    for (unsigned i = 0; i < sizeof(arm_impls) / sizeof(arm_impls[0]); i++) {
        const arm_impl_t *impl = &arm_impls[i];
        fill_nan((volatile uint32_t *)arm_dst, NPIX);
        impl->run();
        if (computed[output_index(impl->output)])
            compare(impl);
        else
            printk("IF_CHECK arm=%s output=%s skipped=1\n", impl->name, impl->output);
        for (unsigned r = 0; r < IF_REPEATS; r++) {
            uint32_t t0 = timer_get_usec();
            impl->run();
            uint32_t t = timer_get_usec() - t0;
            printk("IF_TIME impl=arm name=%s output=%s us=%u\n", impl->name, impl->output, t);
        }
    }
}

#if IF_DUMP
// 8-bit image of an output, two hex digits per pixel, 64 pixels per line.
static void dump(const char *name, volatile const float *img, float scale) {
    static const char digits[] = "0123456789abcdef";
    char line[2 * 64 + 1];
    printk("IF_IMAGE name=%s h=%d w=%d\n", name, IF_H, IF_W);
    for (int y = 0; y < IF_H; y++)
        for (int x0 = 0; x0 < IF_W; x0 += 64) {
            int n = 0;
            for (int x = x0; x < x0 + 64 && x < IF_W; x++) {
                float v = img[y * IF_W + x] * scale + 0.5f;
                unsigned b = v <= 0.0f ? 0 : v >= 255.0f ? 255 : (unsigned)v;
                line[n++] = digits[b >> 4];
                line[n++] = digits[b & 15];
            }
            line[n] = 0;
            printk("%s\n", line);
        }
    printk("IF_IMAGE_END\n");
}
#endif

void notmain(void) {
    piton_init();
    printk("PIton image filters: %d x %d, %d QPUs, %d filters, %d repeats\n",
           IF_W, IF_H, piton_qpu_count(), IF_NUM_FILTERS, IF_REPEATS);
    static const unsigned clocks[] = {PITON_CLOCK_ARM, PITON_CLOCK_CORE, PITON_CLOCK_V3D, PITON_CLOCK_SDRAM};
    static const char *names[] = {"arm", "core", "v3d", "sdram"};
    for (unsigned i = 0; i < 4; i++)
        printk("IF_CLOCK name=%s mhz=%d\n", names[i], piton_clock_hz(clocks[i], 0) / 1000000);

    gpu_src = piton_alloc(NPIX * 4);
    gpu_tmp = piton_alloc(NPIX * 4);
    gpu_scratch = piton_alloc(NPIX * 4);
    for (unsigned i = 0; i < IF_NUM_OUTPUTS; i++)
        gpu_out[i] = piton_alloc(NPIX * 4);

    // The instruction cache and branch prediction need no MMU and leave data
    // accesses uncached, so the mailbox and the GPU buffers are unaffected;
    // they make the hashing and checking between the QPU runs ~10x faster.
    caches_enable();
    check_size();

    static const scene_params_t scene = IF_SCENE_FULL;
    generate(&scene, gpu_src.cpu, arm_src);
    printk("IF_SCENE size=full h=%d w=%d hash=%u\n", scene.h, scene.w, image_hash(gpu_src.cpu, IF_H, IF_W));
    qpu_full_size();

    // No mailbox operation or generated GPU launcher may run with these caches
    // enabled: the mailbox request buffer is in cached ARM memory.
    caches_disable();
    arm_caches_on();
    arm_all();
#if IF_DUMP
    for (unsigned i = 0; i < IF_NUM_OUTPUTS; i++)
        if (if_dump_outputs[i] && computed[i])
            dump(if_outputs[i], gpu_out[i].cpu, if_dump_scale[i]);
#endif
    arm_caches_off();
    printk(passed ? "IF_CHECK_OK\n" : "IF_CHECK_FAILED\n");
    printk("PITON_DONE\n");
}
