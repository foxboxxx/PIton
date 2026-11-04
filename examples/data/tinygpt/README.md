# TinyGPT checkpoint

`model.npz` is a character-level GPT trained by `examples/tinygpt_model.py --train`
(seed 0, 4,000 Adam steps on batches of 32 windows of 32 characters; 66 s on the
sandbox CPU used to build it).

| | |
|---|---|
| parameters | 32,476 float32 (127 KiB uncompressed) |
| shape | d_model 32, 2 layers x 2 heads (head_dim 16), MLP 128, context 32 |
| vocabulary | 92 characters (printable ASCII + newline seen in the corpus) |
| validation loss | 1.81 nats/char on 256 random 32-character windows from the held-out last 10% of the corpus (uniform: ln 92 = 4.52) |

The corpus is this repository's own `README.md` and `STATUS.md` as of training
(47,801 characters after dropping non-ASCII; later doc edits do not affect the tests), so no third-party text is redistributed. The weights
are only meaningful with the formulas in `tinygpt_model.py` (pre-LayerNorm,
GELU as `u * sigmoid(1.702 u)`), which `examples/tinygpt.py` runs on the QPUs.

The `__meta__` entry holds the JSON config, the character table and the
training settings. Retraining overwrites this file; the tests only load it.
