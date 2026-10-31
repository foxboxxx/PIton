# Handwritten digits classifier

This directory contains a trained NumPy neural network and the official UCI
Optical Recognition of Handwritten Digits data used to test it. The network is
64 → 128 → 64 → 10 with ReLU hidden layers and float32 parameters. Its last layer
is padded to 16 outputs for the QPU; the six padded outputs are not classes.
The 17,616 exported float32 parameters occupy 70,464 bytes before compression.

## Data provenance and license

E. Alpaydin and C. Kaynak (1998), *Optical Recognition of Handwritten Digits*,
UCI Machine Learning Repository,
[doi:10.24432/C50P49](https://doi.org/10.24432/C50P49).

The [official dataset page](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits)
licenses the data under
[Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/).
Please retain this attribution when redistributing the data or adaptations.
The UCI data were converted losslessly from comma-separated rows into compressed
uint8 NumPy arrays. No pixel, label, or row order was changed. A trained model
was derived from the training portion. The data describe 8×8 digit images; each
pixel is an integer count from 0 through 16 and each label is from 0 through 9.

The source files are
[`optdigits.tra`](https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra)
and
[`optdigits.tes`](https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tes).
Their SHA-256 hashes are pinned in `digits_model.py` and `metadata.json`.
The official split contains 3,823 training images from 30 writers and 1,797 test
images from 13 different writers, as documented by UCI.

## Frozen model and validation

The training script uses seed 1729, makes a stratified validation split solely
within the official training set (3,251 fitting images and 572 validation
images), then runs a fixed 160-epoch Adam training schedule. It selects the
checkpoint with the lowest validation cross-entropy: epoch 158. The official
test set is evaluated after that checkpoint is frozen. No test examples or
labels are used to fit weights, select the epoch, or choose hyperparameters.

The frozen model classifies **1,743 of 1,797 official test images correctly
(96.9950%)**. Validation accuracy is 566/572 (98.9510%). On 2026-10-30 the Pi Zero W
also matched every host prediction and passed all 28,752 full-test logit checks
against an independent float64 reference with propagated QPU rounding bounds.
Resident inference for the complete test set took 11.161 ms on the QPUs versus
133.775 ms on tuned ARM VFP (medians of five runs, 11.986× speedup). Both compute
1,800 padded rows. Timings include all three launches, bias/ReLU, and launch
overhead; upload, allocation, normalization, validation, and serial output are
outside the timer. Full timing and validation records are in
`benchmarks/results/digits-pi-2026-10-30.{json,log}` at the repository root.

`metadata.json` records the exact configuration, validation row indices, epoch
history, metrics, NumPy version, source hashes, and saved-artifact hashes.
`dataset.npz` stores `train_images` `(3823,64)`, `train_labels` `(3823,)`,
`test_images` `(1797,64)`, and `test_labels` `(1797,)`, all uint8.
`model.npz` stores C-contiguous float32 arrays `w1`, `b1`, `w2`, `b2`, `w3`, and
`b3`, with weight shapes `(64,128)`, `(128,64)`, and `(64,16)` respectively.
Biases have the corresponding output widths. Inputs are normalized by dividing
raw pixel values by 16; there is no softmax in inference, since argmax of the
first ten raw logits gives the digit class.

## Reproduce

From the repository root, with NumPy installed:

```sh
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 \
  python3 examples/digits_model.py --prepare-data --train
python3 examples/digits_model.py
```

Training and inference require only NumPy and the Python standard library.
Downloading the original files is needed only for `--prepare-data`; the checked-in
compressed dataset and model support offline inference. Raw downloaded source
files may be removed after preparation. Reproducibility is seed-controlled;
different NumPy/BLAS implementations can produce small floating-point differences.

The helper API is `load_data()`, `load_model()`, `normalize(raw_images)`,
`forward(images, model)`, and `predict(images, model)`. Integer inputs are treated
as raw pixels; floating-point inputs must already be normalized to 0..1.
`forward` returns 16 logits, while `predict` considers only the first ten.
