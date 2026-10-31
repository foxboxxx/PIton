"""Resume the unfinished matmul tile sweep, recording every measured case.

    python3 benchmarks/sweep_matmul.py --repeats 5 --jsonl benchmarks/results/tiles.jsonl
    python3 benchmarks/sweep_matmul.py 512 --tiles 8x64x8 --groups 1 2 4 8 --stage-counts 1 2

The default tiles include the previous best 8x64x8 and seven unfinished
configurations. Repeating the command skips recorded cases only when the
compiler/runtime/kernel/harness source fingerprint and all settings match.
Errors are recorded and skipped too; --retry-errors retries them. Ctrl-C
records the interrupted case, which is retried on the next invocation.

Only validated sampled results have status "ok". The JSONL includes every
kernel timing, both best and median throughput, error, source revision and
machine-code fingerprint. Correctness checks cover the final repetition's
sampled outputs, not every element or every repetition.
"""
from matmul_pi import main


if __name__ == "__main__":
    raise SystemExit(main(sweep=True))
