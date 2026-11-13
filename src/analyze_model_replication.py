#!/usr/bin/env python3
"""Matched LaMa--MI-GAN replication analysis for the coverage study."""

import argparse
import csv
import json
import os
import tempfile
from pathlib import Path

_mpl_cache = Path(tempfile.gettempdir()) / "mask_coverage_matplotlib"
_mpl_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_cache))

import matplotlib.pyplot as plt
import numpy as np


METRICS = ("full_psnr", "hole_psnr")


def load(path):
    records = [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x]
    if not records:
        raise RuntimeError(f"No records in {path}")
    keyed = {
        (r["image_id"], r["mask_family"], float(r["coverage_target"])): r
        for r in records
    }
    if len(keyed) != len(records):
        raise RuntimeError(f"Duplicate conditions in {path}")
    return records, keyed


def percentile(values):
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lama", required=True)
    parser.add_argument("--migan", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--figure", required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261022)
    args = parser.parse_args()

    lama_records, lama = load(args.lama)
    migan_records, migan = load(args.migan)
    if set(lama) != set(migan):
        raise RuntimeError("LaMa and MI-GAN condition sets differ")
    if any(lama[k]["mask_sha256"] != migan[k]["mask_sha256"] for k in lama):
        raise RuntimeError("At least one deterministic mask differs between models")
    if any(lama[k]["image_sha256"] != migan[k]["image_sha256"] for k in lama):
        raise RuntimeError("At least one source image differs between models")

    images = sorted({k[0] for k in lama})
    families = sorted({k[1] for k in lama})
    coverages = sorted({k[2] for k in lama})
    models = {"LaMa": lama, "MI-GAN": migan}
    cube = np.empty((2, len(images), len(coverages), len(METRICS)), dtype=np.float64)
    for model_index, model_name in enumerate(models):
        records = models[model_name]
        for image_index, image_id in enumerate(images):
            for coverage_index, coverage in enumerate(coverages):
                cells = [records[(image_id, family, coverage)] for family in families]
                for metric_index, metric in enumerate(METRICS):
                    cube[model_index, image_index, coverage_index, metric_index] = np.mean(
                        [float(cell[metric]) for cell in cells]
                    )

    rng = np.random.default_rng(args.seed)
    draws = rng.integers(0, len(images), size=(args.bootstrap, len(images)))
    boot = cube[:, draws, :, :].mean(axis=2)
    means = cube.mean(axis=1)
    summary = {}
    rows = []
    for model_index, model_name in enumerate(models):
        summary[model_name] = {}
        for coverage_index, coverage in enumerate(coverages):
            summary[model_name][str(coverage)] = {}
            for metric_index, metric in enumerate(METRICS):
                ci = percentile(boot[model_index, :, coverage_index, metric_index])
                value = float(means[model_index, coverage_index, metric_index])
                summary[model_name][str(coverage)][metric] = {"mean": value, "ci95": ci}
                rows.append([model_name, coverage, metric, value, ci[0], ci[1]])

    shifts = cube[:, :, -1, :] - cube[:, :, 0, :]
    boot_shifts = boot[:, :, -1, :] - boot[:, :, 0, :]
    shift_summary = {}
    for model_index, model_name in enumerate(models):
        shift_summary[model_name] = {}
        for metric_index, metric in enumerate(METRICS):
            shift_summary[model_name][metric] = {
                "mean_50_minus_10": float(shifts[model_index, :, metric_index].mean()),
                "ci95": percentile(boot_shifts[model_index, :, metric_index]),
            }

    difference_of_shifts = {}
    for metric_index, metric in enumerate(METRICS):
        values = shifts[1, :, metric_index] - shifts[0, :, metric_index]
        boot_values = boot_shifts[1, :, metric_index] - boot_shifts[0, :, metric_index]
        difference_of_shifts[metric] = {
            "mean_migan_minus_lama": float(values.mean()),
            "ci95": percentile(boot_values),
        }

    disagreement = {}
    for model_index, model_name in enumerate(models):
        signs_differ = np.sign(shifts[model_index, :, 0]) != np.sign(shifts[model_index, :, 1])
        boot_fraction = signs_differ[draws].mean(axis=1)
        disagreement[model_name] = {
            "fraction": float(signs_differ.mean()),
            "ci95": percentile(boot_fraction),
        }

    result = {
        "design": {
            "records_per_model": len(lama_records),
            "images": len(images),
            "mask_families": families,
            "coverage_targets": coverages,
            "matched_mask_hashes": len(lama),
            "matched_image_hashes": len(lama),
            "bootstrap_replicates": args.bootstrap,
        },
        "coverage_summary": summary,
        "paired_50_minus_10": shift_summary,
        "migan_minus_lama_difference_of_shifts": difference_of_shifts,
        "full_vs_hole_preference_disagreement": disagreement,
        "checkpoint_hashes": {
            "LaMa": sorted({r["model_sha256"] for r in lama_records}),
            "MI-GAN": sorted({r["model_sha256"] for r in migan_records}),
        },
        "integrity": {
            "max_observed_pixel_error": max(
                max(float(r["observed_max_error"]) for r in lama_records),
                max(float(r["observed_max_error"]) for r in migan_records),
            ),
            "max_mse_identity_residual": max(
                abs(float(r["full_mse"]) - float(r["coverage_realized"]) * float(r["hole_mse"]))
                for r in lama_records + migan_records
            ),
        },
    }

    Path(args.output_json).write_text(json.dumps(result, indent=2), encoding="utf-8")
    with Path(args.output_csv).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "coverage", "metric", "mean", "ci95_low", "ci95_high"])
        writer.writerows(rows)

    colors = {"LaMa": "#1F4E79", "MI-GAN": "#C55A11"}
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.5), constrained_layout=True)
    for axis, metric, label in zip(axes, METRICS, ("Full-image PSNR (dB)", "Hole-only PSNR (dB)")):
        for model_name in models:
            values = [summary[model_name][str(c)][metric]["mean"] for c in coverages]
            lows = [summary[model_name][str(c)][metric]["ci95"][0] for c in coverages]
            highs = [summary[model_name][str(c)][metric]["ci95"][1] for c in coverages]
            axis.plot(np.array(coverages) * 100, values, marker="o", label=model_name, color=colors[model_name])
            axis.fill_between(np.array(coverages) * 100, lows, highs, color=colors[model_name], alpha=0.14)
        axis.set_xlabel("Missing-area coverage (%)")
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
    axes[0].legend(frameon=False)
    Path(args.figure).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure, dpi=300)
    plt.close(fig)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
