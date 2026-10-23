# Reproducibility protocol

## Scope

This package reproduces a controlled metric-sensitivity experiment. Blur and
shift probes intentionally use transformed reference pixels, so they must not
be described as deployable inpainting systems or model baselines.

## Environment

- Python 3.12 was used for the reference run.
- Exact direct dependency versions are in `requirements.txt`.
- CPU execution is sufficient; no GPU or model checkpoint is required.
- Randomness is derived from the base seed, image identifier, mask family,
  target coverage and operator using SHA-256.

## Data

Use the official 2,000-pair ADEChallengeData2016 validation split. The runner
matches `images/validation/*.jpg` to `annotations/validation/*.png` by stem.
Images are resized to 256 × 256 with bicubic interpolation; annotations use
nearest-neighbor interpolation. No training or hyperparameter selection occurs.

The repository does not redistribute dataset content. Every output record
contains the source-image SHA-256 digest so a rerun can be traced to the exact
local source bytes without exposing those bytes.

## Design

Each image receives 80 conditions: 4 mask families × 5 coverage targets × 4
probes. A mask is shared by all four probes within an image/family/coverage
cell. Coverage is deterministically adjusted to the exact requested number of
pixels (up to integer pixel rounding).

## Statistical unit

The primary analysis first averages repeated mask/probe conditions inside each
image-by-coverage cell. Bootstrap resampling is performed over images, not over
the 160,000 correlated records. Reported standard deviations describe
variation across images and must not be interpreted as variation across
independent model-training runs.

## Integrity criteria

A valid complete run must satisfy all of the following:

1. 2,000 image clusters and 160,000 unique condition records.
2. Exactly 80 records per image.
3. No conflicting duplicate condition keys.
4. Maximum observed-pixel absolute error equal to zero.
5. `MSE_full = h × MSE_hole` within floating-point tolerance.
6. `MAE_full = h × MAE_hole` within floating-point tolerance.
7. `PSNR_full − PSNR_hole = −10 log10(h)` within tolerance.

Run `python src/self_check.py` before accessing the dataset. After the full run,
inspect `results/ade20k_validation_analysis.json` and compare compact artifacts
against `MANIFEST.sha256` with `shasum -a 256 -c MANIFEST.sha256`.

## Expected reference values

- Mean full PSNR: 22.204 dB (10%) and 15.114 dB (50%).
- Mean hole PSNR: 12.204 dB (10%) and 12.104 dB (50%).
- Mean full SSIM: 0.8999 (10%) and 0.5652 (50%).
- Paired full-PSNR difference, 50% minus 10%: −7.090 dB, 95% CI
  [−7.124, −7.055].
- Full-versus-hole PSNR preference disagreement: 44.23%, 95% CI
  [43.39%, 45.03%].

Small last-decimal differences may occur across platforms or dependency builds;
the analytical identities should remain within the recorded numerical tolerance.
