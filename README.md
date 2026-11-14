# Mask Coverage as a Confounding Factor in Full-Image Evaluation of Image Inpainting

Reproducibility package for the study by Mahesh Patil and Vikas Tiwari.

## Why this repository exists

If an inpainting pipeline copies every observed pixel unchanged, full-image
fidelity metrics mix two quantities: error inside the missing region and the
fraction of the image that was missing. For missing-area coverage `h`:

```text
MSE_full  = h * MSE_hole
MAE_full  = h * MAE_hole
PSNR_full = PSNR_hole - 10 log10(h)
```

The study tests the practical size of this support effect with a balanced,
deterministic experiment and matched replications using public LaMa and MI-GAN
checkpoints. The learned-model experiments test metric sensitivity; they do
**not** establish architectural superiority.

## Main result

The completed run contains 160,000 records:

- 2,000 official ADE20K validation images
- 4 mask families: centered, rectangle, free-form and semantic
- 5 exact coverage targets: 10%, 20%, 30%, 40% and 50%
- 4 controlled completion probes: mean, blur, noise and shift

Mean full-image PSNR changed from 22.204 dB at 10% coverage to 15.114 dB at
50%, while mean hole PSNR changed from 12.204 to 12.104 dB. In matched
10%-versus-50% comparisons, full-image and hole PSNR preferred opposite
coverage conditions in 44.23% of pairs (image-cluster bootstrap 95% CI:
43.39%–45.03%). A counterfactual audit then held every output fixed and changed
only the coverage-stratum weights. Moving from a small-hole-heavy to a
large-hole-heavy mixture shifted pooled full-image PSNR by 3.497 dB (95% CI:
3.481–3.513), while hole PSNR shifted by only 0.051 dB (95% CI: 0.035–0.067).
Coverage explained 97.3% of the within-image variation in full-image PSNR but
only 0.8% in hole PSNR. These probes are measurement instruments, not baselines.

## Public pretrained-model replication

The replication uses the same 250 ADE20K validation images, free-form and
semantic masks, and 10%--50% coverage targets for both models (2,500 records per
model). All source-image and mask hashes match across models. From 10% to 50%
coverage, full-image PSNR decreased by 10.785 dB for LaMa and 11.935 dB for
MI-GAN; hole-only PSNR decreased by 3.796 and 4.946 dB. The MI-GAN decline was
1.150 dB larger than the LaMa decline (image-cluster bootstrap 95% CI:
0.847--1.461 dB) under this fixed protocol.

The evaluated MI-GAN ONNX pipeline is linked by the authors' official
repository and has SHA-256 digest
`6f1f3530a1a2324b19752018ce756088b07973cda8d7d890034ace5c8a48c40b`.
The model file is not redistributed here.

Each per-condition record contains the image identifier, mask family, target
and realised coverage, and mask SHA-256 digest. The deterministic generator can
recreate the masks from a licensed local ADE20K copy. The MI-GAN records are
published as `results/learned/migan_mac_250.jsonl.gz`.

The exact 2,500-mask bank used by both ADE20K public-model runs is published as
directly readable NPZ shards in
`results/learned/frozen_masks_250x2x5/`. The manifest records every shard hash
and the source-bank SHA-256
`1bdafff1901efb45997d53c5a4855f8b487508feb6f2ad1803e19cf44b8c370d`.
The files contain packed binary masks, identifiers, coverages and hashes, but
no ADE20K photographs. `FROZEN_MASKS_NOTICE.md` records the attribution and
BSD 3-Clause terms applicable to the ADE20K annotations from which the
semantic masks were derived.

## Classical reference baselines

OpenCV Telea and Navier--Stokes inpainting were evaluated with radius 3 on the
same 250 ADE20K and 250 COCO image selections and the same free-form masks.
These inference-only reference methods provide lower bounds; they are not used
to claim architectural superiority. Records, analysis code and the clustered
bootstrap summary are in `results/learned/` and `src/`.

## Independent COCO replication

The second-dataset experiment uses 250 deterministically selected COCO
val2017 images, dataset-independent free-form masks, five coverage levels and
the same LaMa and MI-GAN checkpoints. It contains 1,250 conditions per model.
All 1,250 source-image and mask hashes match across models, observed pixels are
preserved exactly and the maximum MSE-identity residual is below `2.8e-9`.

From 10% to 50% coverage, full-image PSNR decreased by 9.758 dB for LaMa and
11.204 dB for MI-GAN on COCO. The corresponding ADE20K free-form-mask decreases
were 9.644 and 11.249 dB. The COCO-minus-ADE20K differences were -0.114 dB
(95% CI: -0.653 to 0.413) for LaMa and 0.044 dB (95% CI: -0.710 to 0.804) for
MI-GAN. The model-by-coverage-by-dataset interaction was 0.160 dB (95% CI:
-0.256 to 0.583), providing no detected evidence that the coverage response
changed across these datasets.

COCO images are not redistributed. The repository contains only selected image
identifiers and hashes, deterministic mask hashes, compressed metric records,
analysis code and derived results.

## Repository map

```text
configs/study.json          Resolved experimental configuration
src/run_study.py            Deterministic mask and probe generation
src/analyze_paired.py       Primary clustered analysis
src/expanded_analysis.py    Subgroup, ranking and mixture-shift analyses
src/make_figures.py         Primary figures and compact CSV table
src/make_qualitative.py     Controlled-probe illustration
src/run_migan_study.py      MI-GAN matched inference runner
src/analyze_model_replication.py  LaMa--MI-GAN clustered analysis
src/make_lama_migan_qualitative.py  Matched qualitative figure
src/run_coco_replication.py  Resumable COCO LaMa or MI-GAN runner
src/analyze_cross_dataset_replication.py  ADE20K--COCO clustered analysis
src/run_classical_baselines.py  OpenCV Telea and Navier--Stokes runner
src/analyze_classical_baselines.py  Matched classical-baseline analysis
src/merge_shards.py         Validated shard merger
src/self_check.py           Dataset-free mathematical/determinism checks
results/                    Compact derived results (no ADE20K images)
figures/                    Publication figures
paper/                      Journal-ready manuscript
```

## Quick verification without ADE20K

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python src/self_check.py
```

The self-check generates deterministic synthetic images and verifies observed
pixel preservation plus the MSE and PSNR identities.

## Full reproduction

Download the ADEChallengeData2016 validation images and annotations from the
official ADE20K source, then run:

```bash
python src/run_study.py \
  --dataset-root /path/to/ADEChallengeData2016/images/validation \
  --annotation-root /path/to/ADEChallengeData2016/annotations/validation \
  --config configs/study.json \
  --output results/ade20k_validation_merged.jsonl

python src/analyze_paired.py \
  --input results/ade20k_validation_merged.jsonl \
  --output results/ade20k_validation_analysis.json \
  --bootstrap 10000 --seed 20261022

python src/expanded_analysis.py \
  --input results/ade20k_validation_merged.jsonl \
  --output results/expanded_analysis.json \
  --figure-dir figures --bootstrap 3000 --seed 20261022

python src/make_figures.py \
  --analysis results/ade20k_validation_analysis.json \
  --figure-dir figures \
  --table results/coverage_summary_table.csv
```

The MI-GAN replication configuration is `configs/migan_mac.json`. The compact
outputs are in `results/learned/`, and the matched quantitative and qualitative
figures are in `figures/`.

For parallel runs, use `--start-image` and `--end-image`, then combine outputs
with `src/merge_shards.py`. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the
complete protocol and verification steps.

## Interpretation boundary

The package supports claims about metric behavior under exact observed-pixel
compositing. It does not support claims of architectural novelty, model
superiority, state-of-the-art reconstruction or deployment efficiency.

## Data, citation and license

ADE20K images are not redistributed. Only configuration, code, compact derived
statistics and figures are included. Cite the software using
[`CITATION.cff`](CITATION.cff). Code is released under the MIT License; the
manuscript and figures remain copyright © 2026 Mahesh Patil and Vikas Tiwari.
