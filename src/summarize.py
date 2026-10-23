#!/usr/bin/env python3
"""Summarize deterministic mask-coverage experiment records."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


METRICS = [
    "full_psnr",
    "hole_psnr",
    "full_ssim",
    "full_mae",
    "hole_mae",
    "hole_edge_mae",
]


def finite(values):
    return np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)


def describe(records):
    result = {"count": len(records), "metrics": {}}
    coverage = np.asarray([r["coverage_realized"] for r in records], dtype=np.float64)
    for metric in METRICS:
        values = np.asarray([r[metric] for r in records], dtype=np.float64)
        valid = np.isfinite(values)
        clean = values[valid]
        rho, pvalue = spearmanr(coverage[valid], clean)
        result["metrics"][metric] = {
            "mean": float(clean.mean()),
            "std": float(clean.std(ddof=1)),
            "minimum": float(clean.min()),
            "maximum": float(clean.max()),
            "spearman_with_coverage": float(rho),
            "spearman_pvalue": float(pvalue),
        }
    result["maximum_observed_pixel_error"] = float(
        max(r["observed_max_error"] for r in records)
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    records = [
        json.loads(line)
        for line in Path(args.input).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    groups = defaultdict(list)
    for record in records:
        groups[(record["mask_family"], record["operator"])].append(record)

    summary = {
        "overall": describe(records),
        "by_mask_and_operator": {
            f"{family}|{operator}": describe(group)
            for (family, operator), group in sorted(groups.items())
        },
        "interpretation": (
            "Correlations describe metric sensitivity within controlled repeated conditions; "
            "they do not measure the superiority or generalisation of an inpainting model."
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary["overall"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
