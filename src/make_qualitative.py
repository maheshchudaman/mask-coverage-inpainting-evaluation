#!/usr/bin/env python3
"""Visualize the controlled masks and completion probes used in the study."""

import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from run_study import (
    condition_seed,
    controlled_completion,
    make_mask,
    resize_labels,
    resize_rgb,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--annotation", required=True)
    parser.add_argument("--config", default="configs/study.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--coverage", type=float, default=0.30)
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    image_path = Path(args.image)
    image = resize_rgb(image_path, config["image_size"])
    labels = resize_labels(args.annotation, config["image_size"])
    family = "freeform"
    coverage = args.coverage
    mask_rng = random.Random(
        condition_seed(config["seed"], image_path.stem, family, coverage, "mask")
    )
    hole = make_mask(
        family,
        config["image_size"],
        coverage,
        mask_rng,
        config["coverage_tolerance"],
        labels,
    )

    masked = image.copy()
    masked[hole] = 1.0
    panels = [("Reference", image), ("30% free-form mask", masked)]
    for operator in config["operators"]:
        seed = condition_seed(config["seed"], image_path.stem, family, coverage, operator)
        completed = controlled_completion(
            image, hole, operator, random.Random(seed), config
        )
        panels.append((f"Probe: {operator}", completed))

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(2, 3, figsize=(8.2, 5.5), constrained_layout=True)
    for axis, (title, panel) in zip(axes.flat, panels):
        axis.imshow(np.clip(panel, 0.0, 1.0))
        axis.set_title(title)
        axis.axis("off")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
