"""Exercise tile/config rebuilds without a cross compiler or Raspberry Pi."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import pytest


def test_matmul_rebuilds_config_and_generated_pair_in_parallel(tmp_path):
    make = shutil.which("make")
    if make is None:
        pytest.skip("make is unavailable")
    root = tmp_path / "project"
    example = root / "pi" / "examples" / "matmul"
    example.mkdir(parents=True)
    repo = Path(__file__).resolve().parents[1]
    shutil.copy(repo / "pi" / "examples" / "matmul" / "Makefile", example)
    (example / "main.c").write_text("/* test stand-in */\n")
    (root / "examples").mkdir()
    (root / "examples" / "kernels.py").write_text("# test kernel\n")
    runtime = root / "pi" / "runtime"
    runtime.mkdir()
    for name in ("piton_rt.c", "piton_rt.h"):
        (runtime / name).write_text("/* test runtime */\n")
    tools = root / "piton" / "tools"
    tools.mkdir(parents=True)
    (root / "piton" / "__init__.py").touch()
    (tools / "__init__.py").touch()
    compiler = tools / "compile.py"
    compiler.write_text('''from pathlib import Path
import sys
config = sys.argv[sys.argv.index("-c") + 1]
with Path("compiles.log").open("a") as out:
    out.write(config + "\\n")
Path("matmul.c").write_text(config + "\\n")
Path("matmul.h").write_text(config + "\\n")
''')
    course = tmp_path / "course" / "libpi" / "mk"
    course.mkdir(parents=True)
    (course / "Makefile.robust-v2").write_text('''all_objs := objs/main.o objs/matmul.o objs/arm_kernel_vfp.o
all: $(all_objs)
objs/main.o: main.c
objs/matmul.o: matmul.c
$(all_objs):
\t@mkdir -p objs
\t@printf '%s\\n' '$(CFLAGS)' > $@
clean::
\t@rm -rf objs
''')

    def build(*config):
        result = subprocess.run([make, "-j8", "RUN=0", f"PYTHON={sys.executable}",
                                 f"CS240LX_2025_PATH={course.parents[1]}", *config],
                                cwd=example, capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr
        return (example / "compiles.log").read_text().splitlines()

    initial = ("MM=512", "BM=8", "BN=64", "BK=8")
    assert len(build(*initial)) == 1
    objects = list((example / "objs").glob("*.o"))
    times = {p.name: p.stat().st_mtime_ns for p in objects}
    assert len(build(*initial)) == 1
    assert times == {p.name: p.stat().st_mtime_ns for p in objects}

    changed = ("MM=1024", "MM_K=256", "BM=4", "BN=128", "BK=4")
    assert len(build(*changed)) == 2
    assert "K=256," in (example / "matmul.c").read_text()
    assert "BLOCK_SIZE_M=4,BLOCK_SIZE_N=128,BLOCK_SIZE_K=4" in (example / "matmul.h").read_text()
    for obj in objects:
        assert "-DMM=1024 -DMM_K=256 -DBM=4 -DBN=128 -DBK=4" in obj.read_text()
        assert obj.stat().st_mtime_ns > times[obj.name]

    # A compiler edit invalidates the generated code even with identical flags.
    time.sleep(1.05)  # GNU make 3.81 compares source timestamps to whole seconds.
    os.utime(compiler, None)
    assert len(build(*changed)) == 3
    # Deleting either output rebuilds the pair once, even with parallel objects.
    (example / "matmul.h").unlink()
    assert len(build(*changed)) == 4
    (example / "matmul.c").unlink()
    assert len(build(*changed)) == 5
    assert (example / "matmul.h").exists()
    assert len(build(*initial)) == 6
    assert len(build(*initial)) == 6
