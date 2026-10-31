"""Execute generated initialization C against a portable stand-in runtime."""
import os
import shutil
import subprocess

import numpy as np
import pytest

from piton.runtime.pi import Buffer, PiRunner, lcg_floats


def test_compact_uniforms_and_packed_lcg_in_generated_c(tmp_path):
    cc = shutil.which("cc")
    if cc is None:
        pytest.skip("host C compiler is unavailable")
    runner = object.__new__(PiRunner)
    buffers = [Buffer("a", 64), Buffer("b", 4 * 7 * 12, lcg_seed=67890, column_block=(12, 4))]
    programs = [[("buf", "a", 0), ("self",), 0xFFFFFFF0 + 7 * i, (i * i + 19) % 103,
                 ("buf", "a" if i % 2 else "b", 4 * (i % 16)),
                 ("self",) if i % 3 else 0xDEAD0000 + i, -i] for i in range(65)]
    source = runner._generate([0], buffers, programs, ["b"], 1, "generation", {})
    source = source.replace('#include "rpi.h"', '''#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#define printk printf
static uint32_t timer_get_usec(void) { static uint32_t t; return ++t; }
''')
    source = source.replace('#include "piton_rt.h"', '''#include "piton_rt.h"
static uint32_t *memory;
void piton_init(void) {}
uint32_t piton_qpu_count(void) { return 12; }
piton_buf_t piton_alloc(uint32_t size) {
    memory = calloc(size, 1);
    return (piton_buf_t){.bus = 0x40000000, .cpu = memory, .size = size};
}
void piton_free(piton_buf_t *b) { free((void *)b->cpu); }
uint32_t piton_launch(uint32_t code, const uint32_t *unifs, uint32_t n) {
    for (unsigned i = 0; i < n; i++) {
        unsigned base = (unifs[i] - 0x40000000) / 4;
        printf("UNIFORM ");
        for (unsigned j = 0; j < 7; j++) printf("%08x ", memory[base + j]);
        printf("\\n");
    }
    return 1;
}
''')
    source += "\nint main(void) { notmain(); return 0; }\n"
    main = tmp_path / "main.c"
    binary = tmp_path / "run"
    main.write_text(source)
    runtime_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "pi", "runtime"))
    subprocess.run([cc, "-std=c99", "-O2", "-I", runtime_dir, str(main), "-o", str(binary)],
                   check=True, capture_output=True, text=True)
    log = subprocess.run([str(binary)], check=True, capture_output=True, text=True).stdout
    got_uniforms = [[int(word, 16) for word in line.split()[1:]]
                    for line in log.splitlines() if line.startswith("UNIFORM ")]
    expected_uniforms = []
    for i, program in enumerate(programs):
        row = []
        for value in program:
            if isinstance(value, tuple):
                if value[0] == "self":
                    word = 0x40000000 + 12288 + 4 * i * 7
                else:
                    word = 0x40000000 + {"a": 4096, "b": 8192}[value[1]] + value[2]
            else:
                word = value
            row.append(word & 0xFFFFFFFF)
        expected_uniforms.append(row)
    np.testing.assert_array_equal(got_uniforms, expected_uniforms)
    result = runner._parse(log, buffers, ["b"])
    expected_b = (lcg_floats(7 * 12, 67890).reshape(7, 3, 4)
                  .transpose(1, 0, 2).copy().ravel().view(np.uint32))
    np.testing.assert_array_equal(result.outputs["b"], expected_b)
    assert "PITON_PREP b 1\nPITON_START" in log
    assert result.times_us == [1]


def test_large_regular_uniform_grid_has_compact_source():
    runner = object.__new__(PiRunner)
    buffers = [Buffer("a", 64), Buffer("b", 128)]
    programs = [[("buf", "a", 0), ("buf", "b", 0), 512, 512, i, ("self",)]
                for i in range(4096)]
    source = runner._generate([0], buffers, programs, [], 3, "large", {})
    assert len(source) < 4000
