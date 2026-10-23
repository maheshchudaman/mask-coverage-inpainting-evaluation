#!/usr/bin/env python3
"""Create publication figures and a compact results table."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


LABELS = {
    "full_psnr": "Full-image PSNR (dB)",
    "hole_psnr": "Hole PSNR (dB)",
    "full_ssim": "Full-image SSIM",
    "hole_mae": "Hole MAE",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", required=True)
    parser.add_argument("--figure-dir", required=True)
    parser.add_argument("--table", required=True)
    args = parser.parse_args()

    analysis = json.loads(Path(args.analysis).read_text(encoding="utf-8"))
    figure_dir = Path(args.figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    coverage_keys = sorted(analysis["mean_by_coverage"], key=float)
    coverage = np.asarray([100.0 * float(key) for key in coverage_keys])

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.4), constrained_layout=True)
    for axis, metric in zip(axes.flat, LABELS):
        entries = [analysis["mean_by_coverage"][key][metric] for key in coverage_keys]
        means = np.asarray([entry["mean"] for entry in entries])
        lower = np.asarray([entry["mean_bootstrap_95ci"][0] for entry in entries])
        upper = np.asarray([entry["mean_bootstrap_95ci"][1] for entry in entries])
        axis.plot(coverage, means, color="#1f4e79", marker="o", linewidth=2)
        axis.fill_between(coverage, lower, upper, color="#9dc3e6", alpha=0.45)
        axis.set_xlabel("Missing-area coverage (%)")
        axis.set_ylabel(LABELS[metric])
        axis.grid(axis="y", color="#d9d9d9", linewidth=0.6)
    fig.savefig(figure_dir / "metric_sensitivity.png", dpi=300, bbox_inches="tight")
    fig.savefig(figure_dir / "metric_sensitivity.pdf", bbox_inches="tight")
    plt.close(fig)

    theoretical = analysis["psnr_inflation_db_by_coverage"]
    x = np.asarray([100.0 * float(key) for key in sorted(theoretical, key=float)])
    y = np.asarray([theoretical[key] for key in sorted(theoretical, key=float)])
    fig, axis = plt.subplots(figsize=(6.2, 4.0), constrained_layout=True)
    axis.plot(x, y, color="#9c5700", marker="o", linewidth=2)
    for xx, yy in zip(x, y):
        axis.annotate(f"{yy:.2f} dB", (xx, yy), xytext=(0, 8), textcoords="offset points", ha="center")
    axis.set_xlabel("Missing-area coverage (%)")
    axis.set_ylabel("Automatic full-image PSNR increase (dB)")
    axis.set_title("PSNR inflation from exact copying of observed pixels")
    axis.grid(axis="y", color="#d9d9d9", linewidth=0.6)
    fig.savefig(figure_dir / "psnr_inflation_identity.png", dpi=300, bbox_inches="tight")
    fig.savefig(figure_dir / "psnr_inflation_identity.pdf", bbox_inches="tight")
    plt.close(fig)

    table_path = Path(args.table)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    with table_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["coverage_percent", *LABELS.values()])
        for key in coverage_keys:
            writer.writerow(
                [
                    100.0 * float(key),
                    *[
                        analysis["mean_by_coverage"][key][metric]["mean"]
                        for metric in LABELS
                    ],
                ]
            )

    print(json.dumps({"figure_dir": str(figure_dir), "table": str(table_path)}, indent=2))


if __name__ == "__main__":
    main()
