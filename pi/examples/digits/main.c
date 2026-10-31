// Resident, three-layer digit classifier on VideoCore IV and ARM1176.
// Generated headers hold the trained model, independent host references, and
// PIton launchers. All GPU/mailbox work precedes enabling the ARM data cache.
#include "rpi.h"
#include "piton_rt.h"
#include "arm_sgemm.h"
#include "nn_config.h"
#include "nn_data.h"
#include "dense1.h"
#include "dense2.h"
#include "dense3.h"

#if NN_PADDED_COUNT < NN_TEST_COUNT || NN_PADDED_COUNT % 4
#error "NN_PADDED_COUNT must hold the test set and be divisible by four"
#endif
#if NN_TEST_COUNT < 256
#error "The benchmark needs at least 256 held-out examples"
#endif
#if NN_REPEATS < 1
#error "NN_REPEATS must be positive"
#endif

enum { INPUTS = 64, HIDDEN1 = 128, HIDDEN2 = 64, OUTPUTS = 16, CLASSES = 10 };

static float arm_x[NN_PADDED_COUNT * INPUTS];
static float arm_h1[NN_PADDED_COUNT * HIDDEN1];
static float arm_h2[NN_PADDED_COUNT * HIDDEN2];
static float arm_y[NN_PADDED_COUNT * OUTPUTS];
// The VFP assembly kernel reads one packed K step ahead.
static float pack_a[NN_PADDED_COUNT * ARM_KC + 8];
static float pack_b[ARM_KC * ARM_NC + 8];
static uint8_t gpu_predictions[NN_TEST_COUNT];

static piton_buf_t gpu_x, gpu_h1, gpu_h2, gpu_y;
static piton_buf_t gpu_w1, gpu_b1, gpu_w2, gpu_b2, gpu_w3, gpu_b3;
static dense1_args_t args1;
static dense2_args_t args2;
static dense3_args_t args3;

static float absf(float x) { return x < 0.0f ? -x : x; }

static int finitef(float x) {
    union { float f; uint32_t u; } bits = { .f = x };
    return (bits.u & 0x7f800000u) != 0x7f800000u;
}

static piton_buf_t upload_floats(const float *source, unsigned count) {
    piton_buf_t result = piton_alloc(count * sizeof(float));
    volatile float *destination = result.cpu;
    for (unsigned i = 0; i < count; i++)
        destination[i] = source[i];
    return result;
}

static void fill_nan(volatile float *values, unsigned count) {
    union { float f; uint32_t u; } nan = { .u = 0x7fc00000u };
    for (unsigned i = 0; i < count; i++)
        values[i] = nan.f;
}

static void initialize_buffers(void) {
    gpu_x = piton_alloc(sizeof(arm_x));
    gpu_h1 = piton_alloc(sizeof(arm_h1));
    gpu_h2 = piton_alloc(sizeof(arm_h2));
    gpu_y = piton_alloc(sizeof(arm_y));
    volatile float *input = gpu_x.cpu;
    for (unsigned i = 0; i < NN_PADDED_COUNT * INPUTS; i++) {
        float pixel = i < NN_TEST_COUNT * INPUTS ? (float)nn_pixels[i] * (1.0f / 16.0f) : 0.0f;
        arm_x[i] = input[i] = pixel;
    }
    gpu_w1 = upload_floats(nn_w1, INPUTS * HIDDEN1);
    gpu_b1 = upload_floats(nn_b1, HIDDEN1);
    gpu_w2 = upload_floats(nn_w2, HIDDEN1 * HIDDEN2);
    gpu_b2 = upload_floats(nn_b2, HIDDEN2);
    gpu_w3 = upload_floats(nn_w3, HIDDEN2 * OUTPUTS);
    gpu_b3 = upload_floats(nn_b3, OUTPUTS);
    args1 = (dense1_args_t){ .x_ptr = gpu_x.bus, .w_ptr = gpu_w1.bus,
        .bias_ptr = gpu_b1.bus, .y_ptr = gpu_h1.bus };
    args2 = (dense2_args_t){ .x_ptr = gpu_h1.bus, .w_ptr = gpu_w2.bus,
        .bias_ptr = gpu_b2.bus, .y_ptr = gpu_h2.bus };
    args3 = (dense3_args_t){ .x_ptr = gpu_h2.bus, .w_ptr = gpu_w3.bus,
        .bias_ptr = gpu_b3.bus, .y_ptr = gpu_y.bus };
}

static unsigned grid_size(unsigned rows, unsigned cols, unsigned bm, unsigned bn) {
    return ((rows + bm - 1) / bm) * ((cols + bn - 1) / bn);
}

static void run_gpu(unsigned rows) {
    args1.M = args2.M = args3.M = rows;
    dense1_launch(&args1, grid_size(rows, HIDDEN1, NN_BM1, NN_BN1), 1, 1);
    dense2_launch(&args2, grid_size(rows, HIDDEN2, NN_BM2, NN_BN2), 1, 1);
    dense3_launch(&args3, grid_size(rows, OUTPUTS, NN_BM3, NN_BN3), 1, 1);
}

static void bias_activation(float *values, const float *bias, unsigned rows,
                            unsigned cols, int relu) {
    for (unsigned row = 0; row < rows; row++)
        for (unsigned col = 0; col < cols; col++) {
            float value = values[row * cols + col] + bias[col];
            values[row * cols + col] = relu && value < 0.0f ? 0.0f : value;
        }
}

static void run_arm(unsigned rows, int vector) {
    arm_matmul_blocked(arm_x, nn_w1, arm_h1, rows, HIDDEN1, INPUTS, pack_a, pack_b, vector);
    bias_activation(arm_h1, nn_b1, rows, HIDDEN1, 1);
    arm_matmul_blocked(arm_h1, nn_w2, arm_h2, rows, HIDDEN2, HIDDEN1, pack_a, pack_b, vector);
    bias_activation(arm_h2, nn_b2, rows, HIDDEN2, 1);
    arm_matmul_blocked(arm_h2, nn_w3, arm_y, rows, OUTPUTS, HIDDEN2, pack_a, pack_b, vector);
    bias_activation(arm_y, nn_b3, rows, OUTPUTS, 0);
}

// The single-image case computes four rows so that both implementations use
// the same padded matrix dimensions. Padding/preparation is outside inference.
static void prepare_batch(volatile float *input, unsigned logical, unsigned padded) {
    for (unsigned row = 1; row < 4; row++)
        for (unsigned col = 0; col < INPUTS; col++)
            input[row * INPUTS + col] = (float)nn_pixels[row * INPUTS + col] * (1.0f / 16.0f);
    for (unsigned row = logical; row < padded; row++)
        for (unsigned col = 0; col < INPUTS; col++)
            input[row * INPUTS + col] = 0.0f;
}

static int validate_rows(const char *implementation, volatile const float *logits,
                         uint8_t *predictions, unsigned rows, int batch_check) {
    unsigned failures = 0, pred_mismatch = 0, correct = 0;
    float max_abs_error = 0.0f;
    for (unsigned row = 0; row < rows; row++) {
        unsigned prediction = 0, reference_prediction = 0;
        float allowed = 1.1f * nn_error_bound[row] + 2e-5f;
        for (unsigned col = 0; col < OUTPUTS; col++) {
            unsigned index = row * OUTPUTS + col;
            float got = logits[index], reference = nn_reference[index];
            float error = absf(got - reference);
            if (!finitef(got) || !finitef(reference) || !finitef(allowed) || allowed < 0.0f || !(error <= allowed))
                failures++;
            if (error > max_abs_error)
                max_abs_error = error;
            if (col < CLASSES) {
                if (got > logits[row * OUTPUTS + prediction])
                    prediction = col;
                if (reference > nn_reference[row * OUTPUTS + reference_prediction])
                    reference_prediction = col;
            }
        }
        pred_mismatch += prediction != reference_prediction;
        correct += prediction == nn_labels[row];
        if (predictions)
            predictions[row] = prediction;
    }
    if (batch_check)
        printk("NN_BATCH_CHECK impl=%s batch=%d checked=%d failures=%d pred_mismatch=%d correct=%d max_abs_error=%f\n",
               implementation, rows, rows * OUTPUTS, failures, pred_mismatch, correct, max_abs_error);
    else
        printk("NN_VALIDATE impl=%s checked=%d failures=%d pred_mismatch=%d correct=%d max_abs_error=%f\n",
               implementation, rows * OUTPUTS, failures, pred_mismatch, correct, max_abs_error);
    return failures == 0 && pred_mismatch == 0;
}

static int validate(const char *implementation, volatile const float *logits, uint8_t *predictions) {
    return validate_rows(implementation, logits, predictions, NN_TEST_COUNT, 0);
}

static int benchmark_gpu(void) {
    static const unsigned batches[] = {1, 16, 64, 256, NN_TEST_COUNT};
    int passed = 1;
    for (unsigned b = 0; b < sizeof(batches) / sizeof(batches[0]); b++) {
        unsigned logical = batches[b], padded = (logical + 3) & ~3u;
        prepare_batch(gpu_x.cpu, logical, padded);
        fill_nan(gpu_h1.cpu, padded * HIDDEN1);
        fill_nan(gpu_h2.cpu, padded * HIDDEN2);
        fill_nan(gpu_y.cpu, padded * OUTPUTS);
        for (unsigned repeat = 0; repeat < NN_REPEATS; repeat++) {
            uint32_t start = timer_get_usec();
            run_gpu(padded);
            uint32_t elapsed = timer_get_usec() - start;
            printk("NN_TIME impl=gpu batch=%d padded=%d us=%d\n", logical, padded, elapsed);
        }
        passed &= validate_rows("gpu", gpu_y.cpu, 0, logical, 1);
    }
    return passed;
}

static int benchmark_arm(int vector) {
    static const unsigned batches[] = {1, 16, 64, 256, NN_TEST_COUNT};
    const char *implementation = vector ? "arm_vfp" : "arm_c";
    int passed = 1;
    for (unsigned b = 0; b < sizeof(batches) / sizeof(batches[0]); b++) {
        unsigned logical = batches[b], padded = (logical + 3) & ~3u;
        prepare_batch(arm_x, logical, padded);
        fill_nan(arm_h1, padded * HIDDEN1);
        fill_nan(arm_h2, padded * HIDDEN2);
        fill_nan(arm_y, padded * OUTPUTS);
        for (unsigned repeat = 0; repeat < NN_REPEATS; repeat++) {
            uint32_t start = timer_get_usec();
            run_arm(padded, vector);
            uint32_t elapsed = timer_get_usec() - start;
            printk("NN_TIME impl=%s batch=%d padded=%d us=%d\n", implementation, logical, padded, elapsed);
        }
        passed &= validate_rows(implementation, arm_y, 0, logical, 1);
    }
    return passed;
}

static void print_predictions(void) {
    static const char hex[] = "0123456789abcdef";
    char line[33];
    printk("NN_PRED %d\n", NN_TEST_COUNT);
    for (unsigned base = 0; base < NN_TEST_COUNT; base += 32) {
        unsigned count = NN_TEST_COUNT - base < 32 ? NN_TEST_COUNT - base : 32;
        for (unsigned i = 0; i < count; i++)
            line[i] = hex[gpu_predictions[base + i]];
        line[count] = 0;
        printk("%s\n", line);
    }
    printk("NN_PRED_END\n");
}

void notmain(void) {
    piton_init();
    printk("PIton neural network: 64 -> 128 ReLU -> 64 ReLU -> 10 classes (16 stored), %d QPUs\n",
           piton_qpu_count());
    printk("NN_MODEL test_count=%d padded_count=%d repeats=%d\n", NN_TEST_COUNT, NN_PADDED_COUNT, NN_REPEATS);
    static const unsigned clocks[] = {PITON_CLOCK_ARM, PITON_CLOCK_CORE, PITON_CLOCK_V3D, PITON_CLOCK_SDRAM};
    static const char *names[] = {"arm", "core", "v3d", "sdram"};
    for (unsigned i = 0; i < 4; i++)
        printk("NN_CLOCK name=%s mhz=%d\n", names[i], piton_clock_hz(clocks[i], 0) / 1000000);
    uint32_t arm_memory, gpu_base, gpu_memory;
    piton_memory_split(&arm_memory, &gpu_base, &gpu_memory);
    printk("NN_MEMORY arm_mb=%d gpu_mb=%d gpu_base=%x\n", arm_memory >> 20, gpu_memory >> 20, gpu_base);

    uint32_t setup_start = timer_get_usec();
    initialize_buffers();
    // This first call allocates/copies each launcher's code and uniform storage.
    run_gpu(NN_PADDED_COUNT);
    printk("NN_SETUP us=%d\n", timer_get_usec() - setup_start);

    fill_nan(gpu_h1.cpu, NN_PADDED_COUNT * HIDDEN1);
    fill_nan(gpu_h2.cpu, NN_PADDED_COUNT * HIDDEN2);
    fill_nan(gpu_y.cpu, NN_PADDED_COUNT * OUTPUTS);
    run_gpu(NN_PADDED_COUNT);
    int passed = validate("gpu", gpu_y.cpu, gpu_predictions);
    passed &= benchmark_gpu();
    // Validate the final timed output as well as the independent sentinel pass.
    passed &= validate("gpu_timed", gpu_y.cpu, 0);

    // No mailbox operation or generated GPU launcher may run with these caches
    // enabled: the mailbox request buffer is in cached ARM memory.
    arm_caches_on();
    run_arm(NN_PADDED_COUNT, 0);
    run_arm(NN_PADDED_COUNT, 1);
    for (int vector = 0; vector <= 1; vector++) {
        fill_nan(arm_h1, NN_PADDED_COUNT * HIDDEN1);
        fill_nan(arm_h2, NN_PADDED_COUNT * HIDDEN2);
        fill_nan(arm_y, NN_PADDED_COUNT * OUTPUTS);
        run_arm(NN_PADDED_COUNT, vector);
        passed &= validate(vector ? "arm_vfp" : "arm_c", arm_y, 0);
        passed &= benchmark_arm(vector);
        passed &= validate(vector ? "arm_vfp_timed" : "arm_c_timed", arm_y, 0);
    }
    arm_caches_off();
    print_predictions();
    printk(passed ? "NN_CHECK_OK\n" : "NN_CHECK_FAILED\n");
    printk("PITON_DONE\n");
}
