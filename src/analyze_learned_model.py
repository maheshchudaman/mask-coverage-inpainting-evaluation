#!/usr/bin/env python3
"""Summarize a completed learned-model mask-coverage experiment."""

import argparse
import csv
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path

_mpl_cache = Path(tempfile.gettempdir()) / "mask_coverage_matplotlib"
_mpl_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_cache))

import matplotlib.pyplot as plt
import numpy as np


METRICS = ["full_psnr", "hole_psnr", "full_ssim", "hole_mae"]


def percentile_interval(values):
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def load_records(path):
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSON at line {line_number}: {exc}") from exc
    if not records:
        raise SystemExit("No learned-model records were found")
    return records


def build_cube(records):
    images = sorted({r["image_id"] for r in records})
    coverages = sorted({float(r["coverage_target"]) for r in records})
    families = sorted({r["mask_family"] for r in records})
    models = sorted({r["model_id"] for r in records})
    if len(models) != 1:
        raise RuntimeError(f"Expected one model per input file, found {models}")

    keys = set()
    by_cell = defaultdict(list)
    for record in records:
        key = (
            record["image_id"],
            record["mask_family"],
            float(record["coverage_target"]),
            record["model_id"],
        )
        if key in keys:
            raise RuntimeError(f"Duplicate condition: {key}")
        keys.add(key)
        by_cell[(record["image_id"], float(record["coverage_target"]))].append(record)

    expected = len(images) * len(coverages) * len(families)
    complete = len(records) == expected
    if complete:
        for image_id in images:
            for coverage in coverages:
                cell = by_cell[(image_id, coverage)]
                if len(cell) != len(families):
                    raise RuntimeError(
                        f"Incomplete cell for {image_id} at {coverage}: {len(cell)} records"
                    )

    cube = np.full((len(images), len(coverages), len(METRICS)), np.nan, dtype=np.float64)
    for i, image_id in enumerate(images):
        for j, coverage in enumerate(coverages):
            cell = by_cell.get((image_id, coverage), [])
            if cell:
                for k, metric in enumerate(METRICS):
                    cube[i, j, k] = np.mean([float(r[metric]) for r in cell])
    return images, coverages, families, models[0], cube, complete


def bootstrap_summary(cube, coverages, bootstrap, seed):
    valid = ~np.isnan(cube).any(axis=(1, 2))
    data = cube[valid]
    if not len(data):
        raise RuntimeError("No image has a complete set of coverage cells")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(data), size=(bootstrap, len(data)))
    boot_means = data[draws].mean(axis=1)

    summary = {}
    for j, coverage in enumerate(coverages):
        summary[str(coverage)] = {}
        for k, metric in enumerate(METRICS):
            summary[str(coverage)][metric] = {
                "mean": float(data[:, j, k].mean()),
                "ci95": percentile_interval(boot_means[:, j, k]),
            }

    low = data[:, 0, :]
    high = data[:, -1, :]
    paired = high - low
    boot_paired = boot_means[:, -1, :] - boot_means[:, 0, :]
    paired_summary = {
        metric: {
            "mean_high_minus_low": float(paired[:, k].mean()),
            "ci95": percentile_interval(boot_paired[:, k]),
        }
        for k, metric in enumerate(METRICS)
    }
    disagreement = np.mean(
        np.sign(high[:, METRICS.index("full_psnr")] - low[:, METRICS.index("full_psnr")])
        != np.sign(high[:, METRICS.index("hole_psnr")] - low[:, METRICS.index("hole_psnr")])
    )
    boot_disagreement = []
    for indices in draws:
        sample_low = low[indices]
        sample_high = high[indices]
        boot_disagreement.append(
            np.mean(
                np.sign(sample_high[:, 0] - sample_low[:, 0])
                != np.sign(sample_high[:, 1] - sample_low[:, 1])
            )
        )
    return summary, paired_summary, {
        "fraction": float(disagreement),
        "ci95": percentile_interval(np.asarray(boot_disagreement)),
    }, int(valid.sum())


def write_csv(path, summary, coverages):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["coverage", "metric", "mean", "ci95_low", "ci95_high"]
        )
        for coverage in coverages:
            for metric in METRICS:
                item = summary[str(coverage)][metric]
                writer.writerow(
                    [coverage, metric, item["mean"], item["ci95"][0], item["ci95"][1]]
                )


def make_figure(path, summary, coverages, model_id):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.2), constrained_layout=True)
    labels = {
        "full_psnr": "Full-image PSNR (dB)",
        "hole_psnr": "Hole-only PSNR (dB)",
        "full_ssim": "Full-image SSIM",
        "hole_mae": "Hole-only MAE",
    }
    for axis, metric in zip(axes.ravel(), METRICS):
        means = [summary[str(c)][metric]["mean"] for c in coverages]
        lows = [summary[str(c)][metric]["ci95"][0] for c in coverages]
        highs = [summary[str(c)][metric]["ci95"][1] for c in coverages]
        axis.plot(np.asarray(coverages) * 100, means, marker="o", color="#1F4E79")
        axis.fill_between(np.asarray(coverages) * 100, lows, highs, color="#B4C7E7", alpha=0.65)
        axis.set_xlabel("Missing-area coverage (%)")
        axis.set_ylabel(labels[metric])
        axis.grid(alpha=0.25)
    fig.suptitle(f"Learned-model metric sensitivity: {model_id}", fontsize=12)
    fig.savefig(path, dpi=240)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--figure", required=True)
    parser.add_argument("--bootstrap", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=20261022)
    args = parser.parse_args()

    records = load_records(args.input)
    images, coverages, families, model_id, cube, complete = build_cube(records)
    summary, paired, disagreement, analyzed_images = bootstrap_summary(
        cube, coverages, args.bootstrap, args.seed
    )
    result = {
        "model_id": model_id,
        "record_count": len(records),
        "image_count": len(images),
        "complete_image_count": analyzed_images,
        "mask_families": families,
        "coverage_targets": coverages,
        "design_complete": complete,
        "bootstrap_replicates": args.bootstrap,
        "coverage_summary": summary,
        "paired_high_minus_low": paired,
        "full_vs_hole_psnr_preference_disagreement": disagreement,
        "integrity": {
            "max_observed_pixel_error": max(float(r["observed_max_error"]) for r in records),
            "max_mse_identity_residual": max(
                abs(
                    float(r["full_mse"])
                    - float(r["coverage_realized"]) * float(r["hole_mse"])
                )
                for r in records
            ),
        },
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_csv(args.output_csv, summary, coverages)
    make_figure(args.figure, summary, coverages, model_id)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
