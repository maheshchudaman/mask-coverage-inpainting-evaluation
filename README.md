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
deterministic experiment and a prespecified LaMa external validation. The
learned-model experiment tests whether the support effect remains visible in
outputs from a trained model; it is not an architectural comparison.

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

## Learned-model external validation

The public LaMa evidence contains 2,500 condition records:

- 250 deterministically selected ADE20K validation images
- 2 mask families: free-form and semantic
- 5 coverage targets: 10%, 20%, 30%, 40% and 50%
- The official OpenCV Zoo `inpainting_lama_2025jan.onnx` checkpoint

Full-image PSNR decreased from 35.361 dB at 10% coverage to 24.576 dB at
50%, while hole PSNR decreased from 25.362 to 21.566 dB. Full-image and
hole-only PSNR preferred opposite coverage conditions for 11.6% of the 250
image-level comparisons. The raw records, analysis output, summary table,
configuration, scripts, figure and execution notes are included below.

The checkpoint is not redistributed. Its verified SHA-256 is
`7df918ac3921d3daf0aae1d219776cf0dc4e4935f035af81841b40adcf74fdf2`.

## Repository map

```text
configs/study.json          Resolved experimental configuration
configs/learned_models_mac.json  Resolved LaMa validation configuration
docs/MAC_EXPERIMENT.md      LaMa download, execution and resumption protocol
src/run_study.py            Deterministic mask and probe generation
src/analyze_paired.py       Primary clustered analysis
src/expanded_analysis.py    Subgroup, ranking and mixture-shift analyses
src/run_learned_model_study.py  Resumable LaMa experiment runner
src/analyze_learned_model.py    Image-clustered LaMa analysis
src/make_figures.py         Primary figures and compact CSV table
src/make_qualitative.py     Controlled-probe illustration
src/merge_shards.py         Validated shard merger
src/self_check.py           Dataset-free mathematical/determinism checks
results/                    Compact derived results (no ADE20K images)
results/learned/            LaMa raw records and derived analysis
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

For parallel runs, use `--start-image` and `--end-image`, then combine outputs
with `src/merge_shards.py`. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the
complete protocol and verification steps.

For the learned-model validation, follow
[`docs/MAC_EXPERIMENT.md`](docs/MAC_EXPERIMENT.md). To regenerate the published
analysis and figure from the included 2,500 records:

```bash
python -m pip install -r requirements-learned-mac.txt
python src/analyze_learned_model.py \
  --input results/learned/lama_mac_250.jsonl \
  --output-json results/learned/lama_mac_250_analysis.json \
  --output-csv results/learned/lama_mac_250_summary.csv \
  --figure figures/lama_learned_metric_sensitivity.png \
  --bootstrap 3000 --seed 20261022
```

## Interpretation boundary

The package supports claims about metric behavior under exact observed-pixel
compositing. It does not support claims of architectural novelty, model
superiority, state-of-the-art reconstruction or deployment efficiency.

## Data, citation and license

ADE20K images and the LaMa checkpoint are not redistributed. The LaMa
per-condition metric records contain image and mask hashes but no source or
generated images. Configuration, code, raw metric records, derived statistics
and figures are included. Cite the software using
[`CITATION.cff`](CITATION.cff). Code is released under the MIT License; the
manuscript and figures remain copyright © 2026 Mahesh Patil and Vikas Tiwari.
