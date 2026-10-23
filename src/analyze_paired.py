#!/usr/bin/env python3
"""Paired, image-clustered analysis for the complete controlled study."""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


RAW_METRICS = [
    "full_mse",
    "hole_mse",
    "full_ssim",
    "full_mae",
    "hole_mae",
    "hole_edge_mae",
]

METRICS = ["full_psnr", "hole_psnr", "full_ssim", "full_mae", "hole_mae", "hole_edge_mae"]


def percentile_interval(values):
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def cluster_bootstrap_mean(values, repetitions, seed):
    """Bootstrap the image-level mean; rows must already be one value per image."""
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=np.float64)
    draws = np.empty(repetitions, dtype=np.float64)
    for index in range(repetitions):
        draws[index] = rng.choice(values, size=len(values), replace=True).mean()
    return percentile_interval(draws)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261022)
    args = parser.parse_args()

    records = [
        json.loads(line)
        for line in Path(args.input).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    image_ids = sorted({record["image_id"] for record in records})
    coverage_levels = sorted({float(record["coverage_target"]) for record in records})

    # Average over the four masks and four completion probes within each
    # image-by-coverage cell before making paired coverage comparisons.
    cells = defaultdict(lambda: defaultdict(list))
    observed_error = 0.0
    identity_residual = 0.0
    mae_identity_residual = 0.0
    psnr_gap_residual = 0.0
    for record in records:
        key = (record["image_id"], float(record["coverage_target"]))
        for metric in RAW_METRICS:
            cells[key][metric].append(float(record[metric]))
        observed_error = max(observed_error, float(record["observed_max_error"]))
        expected_mse = float(record["coverage_realized"]) * float(record["hole_mse"])
        identity_residual = max(identity_residual, abs(float(record["full_mse"]) - expected_mse))
        expected_mae = float(record["coverage_realized"]) * float(record["hole_mae"])
        mae_identity_residual = max(
            mae_identity_residual, abs(float(record["full_mae"]) - expected_mae)
        )
        expected_gap = -10.0 * math.log10(float(record["coverage_realized"]))
        actual_gap = float(record["full_psnr"]) - float(record["hole_psnr"])
        psnr_gap_residual = max(psnr_gap_residual, abs(actual_gap - expected_gap))

    reduced = {}
    for key, metric_values in cells.items():
        aggregate = {
            metric: float(np.mean(values)) for metric, values in metric_values.items()
        }
        aggregate["full_psnr"] = float(-10.0 * math.log10(aggregate["full_mse"]))
        aggregate["hole_psnr"] = float(-10.0 * math.log10(aggregate["hole_mse"]))
        reduced[key] = aggregate

    coverage_summary = {}
    for coverage in coverage_levels:
        coverage_summary[f"{coverage:.2f}"] = {}
        for metric in METRICS:
            values = np.asarray(
                [reduced[(image_id, coverage)][metric] for image_id in image_ids],
                dtype=np.float64,
            )
            coverage_summary[f"{coverage:.2f}"][metric] = {
                "mean": float(values.mean()),
                "image_sd": float(values.std(ddof=1)),
                "mean_bootstrap_95ci": cluster_bootstrap_mean(
                    values, args.bootstrap, args.seed + int(coverage * 1000) + len(metric)
                ),
            }

    low, high = coverage_levels[0], coverage_levels[-1]
    paired = {}
    for metric in METRICS:
        differences = np.asarray(
            [
                reduced[(image_id, high)][metric] - reduced[(image_id, low)][metric]
                for image_id in image_ids
            ],
            dtype=np.float64,
        )
        paired[metric] = {
            "direction": "high_coverage_minus_low_coverage",
            "mean_difference": float(differences.mean()),
            "image_sd": float(differences.std(ddof=1)),
            "bootstrap_95ci": cluster_bootstrap_mean(
                differences, args.bootstrap, args.seed + len(metric)
            ),
        }

    # Descriptive rank associations on image-averaged coverage cells. The
    # paired bootstrap intervals above provide the primary uncertainty result.
    coverage_vector = np.asarray(
        [coverage for image_id in image_ids for coverage in coverage_levels]
    )
    correlations = {}
    for metric in METRICS:
        values = np.asarray(
            [
                reduced[(image_id, coverage)][metric]
                for image_id in image_ids
                for coverage in coverage_levels
            ]
        )
        correlations[metric] = float(spearmanr(coverage_vector, values).statistic)

    result = {
        "design": {
            "images": len(image_ids),
            "raw_records": len(records),
            "conditions_per_image": len(records) // len(image_ids),
            "coverage_levels": coverage_levels,
            "bootstrap_repetitions": args.bootstrap,
            "bootstrap_unit": "image",
        },
        "invariants": {
            "maximum_observed_pixel_error": observed_error,
            "maximum_full_mse_identity_residual": identity_residual,
            "maximum_full_mae_identity_residual": mae_identity_residual,
            "maximum_psnr_gap_identity_residual_db": psnr_gap_residual,
        },
        "mean_by_coverage": coverage_summary,
        "paired_high_minus_low": paired,
        "descriptive_spearman_on_image_averaged_cells": correlations,
        "psnr_inflation_db_by_coverage": {
            f"{coverage:.2f}": float(-10.0 * math.log10(coverage))
            for coverage in coverage_levels
        },
        "interpretation_boundary": (
            "Controlled completions are measurement probes, not inpainting baselines. "
            "Inference is clustered by image and supports claims about metric behaviour only."
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result["design"], indent=2, sort_keys=True))
    print(json.dumps(result["paired_high_minus_low"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
