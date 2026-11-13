# Learned-model experiment on the M2 MacBook Air

## Purpose

This runner adds a real LaMa baseline without exhausting the 8 GB unified
memory or filling the remaining disk space. It uses the official OpenCV Zoo
quantized ONNX graph, evaluates one condition at a time, writes each result
immediately, and stores only a small prespecified set of example images.

The local experiment is deliberately limited to 250 deterministically selected
ADE20K validation images, two mask families, and five coverage levels. This
produces 2,500 LaMa records. RePaint should be run separately on a cloud GPU
using the same selected image IDs and mask hashes.

## Install the lightweight runtime

From the repository root:

```bash
source .venv/bin/activate
python -m pip install -r requirements-learned-mac.txt
mkdir -p models results/learned examples/lama
```

Download the 92.6 MB official model and verify its published SHA-256:

```bash
curl -L \
  -o models/inpainting_lama_2025jan.onnx \
  https://huggingface.co/opencv/inpainting_lama/resolve/main/inpainting_lama_2025jan.onnx

shasum -a 256 models/inpainting_lama_2025jan.onnx
```

Expected digest:

```text
7df918ac3921d3daf0aae1d219776cf0dc4e4935f035af81841b40adcf74fdf2
```

## Run a ten-condition pilot

The pilot verifies the model, records inference time and peak memory, and gives
a hardware-specific completion estimate.

```bash
nice -n 10 .venv/bin/python src/run_learned_model_study.py \
  --dataset-root ../ADEChallengeData2016/images/validation \
  --annotation-root ../ADEChallengeData2016/annotations/validation \
  --config configs/learned_models_mac.json \
  --output results/learned/lama_pilot.jsonl \
  --example-dir examples/lama/pilot \
  --max-conditions 10 \
  --threads 2 \
  --cooldown-every 5 \
  --cooldown-seconds 10
```

## Run or resume the 2,500-condition experiment

Re-running the same command is safe. Existing condition keys are validated and
skipped, so an interrupted run resumes without duplicating results.

```bash
nice -n 10 .venv/bin/python src/run_learned_model_study.py \
  --dataset-root ../ADEChallengeData2016/images/validation \
  --annotation-root ../ADEChallengeData2016/annotations/validation \
  --config configs/learned_models_mac.json \
  --output results/learned/lama_mac_250.jsonl \
  --example-dir examples/lama/mac_250 \
  --threads 2
```

The default 15-second cooldown after every 25 outputs reduces sustained heat on
the fanless MacBook Air. The process can be stopped with Control-C and resumed
with the same command. Two inference threads are used so email and light
browsing remain usable. The `nice` prefix further lowers CPU priority when the
local shell permits it; the runner remains safe if macOS declines that request.

## Storage policy

The JSONL file stores metrics and provenance rather than generated images.
Only the first five selected images at 10% and 50% coverage are retained as
qualitative panels. This keeps the full local experiment well below 1 GB,
including the model checkpoint.

## Scientific boundary

The local runner provides one trained baseline. It does not turn LaMa into a
new method and does not replace the controlled probes. The paper should report
LaMa as an external validation of the metric-support identities. A second
learned baseline, preferably RePaint, should be evaluated on the same subset in
a cloud environment before making a two-model generalisation claim.
