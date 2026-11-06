# TinyGPT on the Pi Zero's GPU

A character-level transformer whose every layer (embedding, LayerNorm, Q/K/V/O,
causal attention, GELU MLP, head) is a Triton kernel compiled by PIton for the
12 VideoCore IV QPUs. The program stays resident: the prompt is prefilled with
full-width launches, then each character is one decode pass against a per-layer
KV cache. The ARM only samples the next character.

| model | params | shape | val loss | binary / upload | QPU | ARM C (VFP, -O3) |
|---|---|---|---|---|---|---|
| tiny | 32K | d32, 2 layers x 2 heads, ctx 32 | 1.81 | 157 KB / ~14 s | 2,507 chars/s | 691 chars/s (3.6x) |
| small | 217K | d64, 4 layers x 4 heads, ctx 64 | 1.46 | 550 KB / ~48 s | 770 chars/s | 92 chars/s (8.4x) |

Measured 2026-11-05 on a Pi Zero W (ARM 700 MHz, V3D 250 MHz): greedy, 2*ctx
characters including the prefill and one context slide. Both implementations
pass a teacher-forced check of every logit against a float64 host reference
(QPU max error 7.9e-5 / 1.2e-4), and their greedy outputs agree on every
character. The ARM baseline is straightforward C compiled -O3 with VFP, not
hand-tuned assembly. tiny is trained on README + STATUS; small also on the
compiler's own Python source, so it writes PIton-flavoured Python.

## Run it

```bash
export PITON_TTY=/dev/ttyUSB0 PITON_BOOTLOADER=~/projects/cs140e/bin/pi-install.linux \
       PITON_LIBPI_ROOT=~/projects/cs140e CS140E_2026_PATH=~/projects/cs140e
.venv/bin/python examples/tinygpt_pi.py --size small          # build, boot, check, benchmark (JSON in benchmarks/results)
.venv/bin/python examples/tinygpt_pi.py --size small --chat   # ... then type prompts
```

`--chat --build-only` leaves `build/tinygpt/small/main.bin`; boot it again
without rebuilding with `$PITON_BOOTLOADER /dev/ttyUSB0 build/tinygpt/small/main.bin`.

In chat: type a prompt and press Enter (a new prompt starts a fresh context);
an empty line continues; `/temp 0` greedy, `/temp 80` the default; `/len N`
characters per reply; `/arm` / `/qpu` switch implementation; `/quit` exits.
Any key while it writes stops the reply.

Train the checkpoints with `examples/tinygpt_model.py --train --size small`
(~17 minutes of NumPy on 4 cores). `medium` (d128, 4 layers, ctx 128) compiles
but has no checkpoint yet.
