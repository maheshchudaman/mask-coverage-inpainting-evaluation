#!/usr/bin/env python3
"""Summarize matched OpenCV Telea and Navier--Stokes reference baselines."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


METHODS = {
    "opencv_telea_radius3_v1": "Telea",
    "opencv_navier_stokes_radius3_v1": "Navier--Stokes",
}
METRICS = ("full_psnr", "hole_psnr")


def load(path, dataset):
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                item = json.loads(line)
                if item["mask_family"] != "freeform":
                    continue
                item["dataset"] = dataset
                rows.append(item)
    return rows


def summarize(rows, bootstrap, seed):
    images = sorted({r["image_id"] for r in rows})
    families = sorted({r["mask_family"] for r in rows})
    coverages = sorted({float(r["coverage_target"]) for r in rows})
    methods = list(METHODS)
    index = {
        (r["image_id"], r["mask_family"], float(r["coverage_target"]), r["model_id"]): r
        for r in rows
    }
    cube = np.full((len(images), len(methods), len(coverages), len(METRICS)), np.nan)
    for i, image in enumerate(images):
        for m, method in enumerate(methods):
            for c, coverage in enumerate(coverages):
                values = [index[(image, family, coverage, method)] for family in families]
                if len({r["image_sha256"] for r in values}) != 1:
                    raise RuntimeError(f"Image mismatch for {image}")
                for family in families:
                    paired = [index[(image, family, coverage, method_id)] for method_id in methods]
                    if len({r["mask_sha256"] for r in paired}) != 1:
                        raise RuntimeError(f"Mask mismatch for {image}, {family}, {coverage}")
                for k, metric in enumerate(METRICS):
                    cube[i, m, c, k] = np.mean([float(r[metric]) for r in values])
    if np.isnan(cube).any():
        raise RuntimeError("Incomplete classical-baseline cube")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(images), size=(bootstrap, len(images)))
    boots = cube[draws].mean(axis=1)
    out = {
        "images": len(images),
        "mask_families": families,
        "coverage_targets": coverages,
        "records": len(rows),
        "methods": {},
    }
    for m, method_id in enumerate(methods):
        name = METHODS[method_id]
        out["methods"][name] = {"coverage": {}, "paired_50_minus_10": {}}
        for c, coverage in enumerate(coverages):
            out["methods"][name]["coverage"][str(coverage)] = {}
            for k, metric in enumerate(METRICS):
                values = boots[:, m, c, k]
                out["methods"][name]["coverage"][str(coverage)][metric] = {
                    "mean": float(cube[:, m, c, k].mean()),
                    "ci95": [float(x) for x in np.percentile(values, [2.5, 97.5])],
                }
        for k, metric in enumerate(METRICS):
            raw = cube[:, m, -1, k] - cube[:, m, 0, k]
            values = boots[:, m, -1, k] - boots[:, m, 0, k]
            out["methods"][name]["paired_50_minus_10"][metric] = {
                "mean": float(raw.mean()),
                "ci95": [float(x) for x in np.percentile(values, [2.5, 97.5])],
            }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ade-telea", required=True)
    parser.add_argument("--ade-ns", required=True)
    parser.add_argument("--coco-telea", required=True)
    parser.add_argument("--coco-ns", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261113)
    args = parser.parse_args()
    data = {
        "ADE20K": summarize(
            load(args.ade_telea, "ADE20K") + load(args.ade_ns, "ADE20K"),
            args.bootstrap,
            args.seed,
        ),
        "COCO": summarize(
            load(args.coco_telea, "COCO") + load(args.coco_ns, "COCO"),
            args.bootstrap,
            args.seed + 1,
        ),
    }
    Path(args.output_json).write_text(json.dumps(data, indent=2), encoding="utf-8")
    with Path(args.output_csv).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["dataset", "method", "metric", "mean_50_minus_10", "ci95_low", "ci95_high"])
        for dataset, result in data.items():
            for method, item in result["methods"].items():
                for metric, value in item["paired_50_minus_10"].items():
                    writer.writerow([dataset, method, metric, value["mean"], *value["ci95"]])
    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
