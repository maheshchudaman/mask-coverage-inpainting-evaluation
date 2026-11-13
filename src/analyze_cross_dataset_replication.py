#!/usr/bin/env python3
"""Analyze matched LaMa/MI-GAN free-form-mask results on ADE20K and COCO."""

import argparse
import csv
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path

_cache = Path(tempfile.gettempdir()) / "mask_coverage_matplotlib"
_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_cache))

import matplotlib.pyplot as plt
import numpy as np


MODELS = {
    "lama_opencv_zoo_2025jan": "LaMa",
    "migan_512_places2_onnx_pipeline_v2": "MI-GAN",
}
METRICS = ["full_psnr", "hole_psnr"]


def load(path, dataset):
    records = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                item = json.loads(line)
                if item["mask_family"] == "freeform":
                    item["dataset_id"] = dataset
                    records.append(item)
    return records


def build_cube(records):
    images = sorted({r["image_id"] for r in records})
    coverages = sorted({float(r["coverage_target"]) for r in records})
    models = list(MODELS)
    index = {(r["image_id"], r["model_id"], float(r["coverage_target"])): r for r in records}
    cube = np.full((len(images), len(models), len(coverages), len(METRICS)), np.nan)
    for i, image in enumerate(images):
        for m, model in enumerate(models):
            for c, coverage in enumerate(coverages):
                record = index.get((image, model, coverage))
                if record:
                    for k, metric in enumerate(METRICS):
                        cube[i, m, c, k] = float(record[metric])
    if np.isnan(cube).any():
        raise RuntimeError("Incomplete model-by-image-by-coverage cube")
    for image in images:
        for coverage in coverages:
            rows = [index[(image, model, coverage)] for model in models]
            if len({r["image_sha256"] for r in rows}) != 1 or len({r["mask_sha256"] for r in rows}) != 1:
                raise RuntimeError(f"Unmatched input for {image} at {coverage}")
    return images, coverages, cube


def interval(values):
    return [float(x) for x in np.percentile(values, [2.5, 97.5])]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ade-lama", required=True)
    parser.add_argument("--ade-migan", required=True)
    parser.add_argument("--coco-lama", required=True)
    parser.add_argument("--coco-migan", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--figure", required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261113)
    args = parser.parse_args()

    datasets = {
        "ADE20K": load(args.ade_lama, "ADE20K") + load(args.ade_migan, "ADE20K"),
        "COCO": load(args.coco_lama, "COCO") + load(args.coco_migan, "COCO"),
    }
    cubes = {}
    images = {}
    coverages = None
    for dataset, records in datasets.items():
        images[dataset], current, cubes[dataset] = build_cube(records)
        if coverages is None:
            coverages = current
        elif current != coverages:
            raise RuntimeError("Coverage grids differ across datasets")

    rng = np.random.default_rng(args.seed)
    draws = {
        dataset: rng.integers(0, len(cube), size=(args.bootstrap, len(cube)))
        for dataset, cube in cubes.items()
    }
    boots = {dataset: cube[draws[dataset]].mean(axis=1) for dataset, cube in cubes.items()}
    summary = defaultdict(dict)
    csv_rows = []
    for dataset, cube in cubes.items():
        for m, model_id in enumerate(MODELS):
            model = MODELS[model_id]
            summary[dataset][model] = {}
            for c, coverage in enumerate(coverages):
                summary[dataset][model][str(coverage)] = {}
                for k, metric in enumerate(METRICS):
                    mean = float(cube[:, m, c, k].mean())
                    ci = interval(boots[dataset][:, m, c, k])
                    summary[dataset][model][str(coverage)][metric] = {"mean": mean, "ci95": ci}
                    csv_rows.append([dataset, model, coverage, metric, mean, *ci])

    shifts = {}
    boot_shifts = {}
    for dataset, cube in cubes.items():
        shifts[dataset] = {}
        boot_shifts[dataset] = boots[dataset][:, :, -1, :] - boots[dataset][:, :, 0, :]
        raw = cube[:, :, -1, :] - cube[:, :, 0, :]
        for m, model_id in enumerate(MODELS):
            model = MODELS[model_id]
            shifts[dataset][model] = {}
            for k, metric in enumerate(METRICS):
                shifts[dataset][model][metric] = {
                    "mean_50_minus_10": float(raw[:, m, k].mean()),
                    "ci95": interval(boot_shifts[dataset][:, m, k]),
                }

    dataset_interactions = {}
    for m, model_id in enumerate(MODELS):
        model = MODELS[model_id]
        dataset_interactions[model] = {}
        for k, metric in enumerate(METRICS):
            delta = boot_shifts["COCO"][:, m, k] - boot_shifts["ADE20K"][:, m, k]
            dataset_interactions[model][metric] = {
                "mean_coco_minus_ade": float(
                    shifts["COCO"][model][metric]["mean_50_minus_10"]
                    - shifts["ADE20K"][model][metric]["mean_50_minus_10"]
                ),
                "ci95": interval(delta),
            }

    model_interactions = {}
    boot_model_interactions = {}
    for dataset in cubes:
        boot_model_interactions[dataset] = boot_shifts[dataset][:, 1, :] - boot_shifts[dataset][:, 0, :]
        model_interactions[dataset] = {}
        for k, metric in enumerate(METRICS):
            values = boot_model_interactions[dataset][:, k]
            model_interactions[dataset][metric] = {
                "mean_migan_minus_lama": float(values.mean()),
                "ci95": interval(values),
            }
    three_way = {}
    for k, metric in enumerate(METRICS):
        values = boot_model_interactions["COCO"][:, k] - boot_model_interactions["ADE20K"][:, k]
        three_way[metric] = {"mean": float(values.mean()), "ci95": interval(values)}

    result = {
        "design": {
            "datasets": {name: len(cube) for name, cube in cubes.items()},
            "models": list(MODELS.values()),
            "mask_family": "freeform",
            "coverage_targets": coverages,
            "records": {name: int(cube.shape[0] * cube.shape[1] * cube.shape[2]) for name, cube in cubes.items()},
            "bootstrap_replicates": args.bootstrap,
        },
        "coverage_summary": summary,
        "paired_50_minus_10": shifts,
        "dataset_by_coverage": dataset_interactions,
        "model_by_coverage": model_interactions,
        "model_by_coverage_by_dataset": three_way,
    }
    Path(args.output_json).write_text(json.dumps(result, indent=2), encoding="utf-8")
    with Path(args.output_csv).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["dataset", "model", "coverage", "metric", "mean", "ci95_low", "ci95_high"])
        writer.writerows(csv_rows)

    fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.4), constrained_layout=True)
    colors = ["#1f4e79", "#e07025"]
    for row, dataset in enumerate(["ADE20K", "COCO"]):
        for column, metric in enumerate(METRICS):
            axis = axes[row, column]
            for m, model_id in enumerate(MODELS):
                model = MODELS[model_id]
                means = [summary[dataset][model][str(c)][metric]["mean"] for c in coverages]
                lows = [summary[dataset][model][str(c)][metric]["ci95"][0] for c in coverages]
                highs = [summary[dataset][model][str(c)][metric]["ci95"][1] for c in coverages]
                x = np.asarray(coverages) * 100
                axis.plot(x, means, marker="o", color=colors[m], label=model)
                axis.fill_between(x, lows, highs, color=colors[m], alpha=0.16)
            axis.set_title(f"{dataset}: {'full-image' if metric == 'full_psnr' else 'hole-only'} PSNR")
            axis.set_xlabel("Missing-area coverage (%)")
            axis.set_ylabel("PSNR (dB)")
            axis.grid(alpha=0.25)
            if row == 0 and column == 0:
                axis.legend(frameon=False)
    Path(args.figure).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure, dpi=300)
    plt.close(fig)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
