#!/usr/bin/env python3
"""Evaluate OpenCV Telea or Navier--Stokes inpainting on frozen conditions."""

import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path

import cv2
import numpy as np

from run_learned_model_study import completed_keys, deterministic_subset
from run_study import (
    condition_seed,
    discover_pairs,
    make_mask,
    metrics,
    resize_labels,
    resize_rgb,
    sha256,
)


METHODS = {
    "telea": {
        "id": "opencv_telea_radius3_v1",
        "name": "OpenCV Telea",
        "flag": cv2.INPAINT_TELEA,
    },
    "ns": {
        "id": "opencv_navier_stokes_radius3_v1",
        "name": "OpenCV Navier--Stokes",
        "flag": cv2.INPAINT_NS,
    },
}


def inpaint(reference, hole, flag):
    rgb = np.uint8(np.clip(reference * 255.0, 0, 255))
    bgr = np.ascontiguousarray(rgb[..., ::-1])
    mask = np.uint8(hole) * 255
    output = cv2.inpaint(bgr, mask, 3.0, flag)
    return np.asarray(output[..., ::-1], dtype=np.float32) / 255.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["ade20k", "coco"], required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--annotation-root", default="")
    parser.add_argument("--config", required=True)
    parser.add_argument("--method", choices=sorted(METHODS), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--progress-every", type=int, default=100)
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    spec = METHODS[args.method]
    if args.dataset == "ade20k":
        if not args.annotation_root:
            raise SystemExit("--annotation-root is required for ADE20K")
        pairs = discover_pairs(args.dataset_root, args.annotation_root)
    else:
        pairs = [(path, path) for path in Path(args.dataset_root).glob("*.jpg")]
    pairs = deterministic_subset(pairs, int(config["seed"]), int(config["sample_size"]))
    if len(pairs) != int(config["sample_size"]):
        raise SystemExit(f"Expected {config['sample_size']} images, found {len(pairs)}")

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    done = completed_keys(output)
    mode = "a" if output.exists() else "w"
    written = skipped = 0
    started = time.perf_counter()
    with output.open(mode, encoding="utf-8") as handle:
        for rank, (image_path, annotation_path) in enumerate(pairs):
            image_id = image_path.stem
            reference = resize_rgb(image_path, int(config["image_size"]))
            labels = (
                resize_labels(annotation_path, int(config["image_size"]))
                if args.dataset == "ade20k"
                else None
            )
            image_hash = sha256(image_path)
            for family in config["mask_families"]:
                for target in config["coverage_targets"]:
                    key = (image_id, family, round(float(target), 8), spec["id"])
                    if key in done:
                        skipped += 1
                        continue
                    rng = random.Random(
                        condition_seed(config["seed"], image_id, family, target, "mask")
                    )
                    hole = make_mask(
                        family,
                        int(config["image_size"]),
                        float(target),
                        rng,
                        float(config["coverage_tolerance"]),
                        labels,
                    )
                    mask_hash = hashlib.sha256(np.packbits(hole).tobytes()).hexdigest()
                    tick = time.perf_counter()
                    generated = inpaint(reference, hole, spec["flag"])
                    elapsed = time.perf_counter() - tick
                    composite = generated.copy()
                    composite[~hole] = reference[~hole]
                    record = {
                        "dataset_id": "ADE20K_validation" if args.dataset == "ade20k" else config["dataset_id"],
                        "image_id": image_id,
                        "image_sha256": image_hash,
                        "selection_rank": rank,
                        "mask_family": family,
                        "coverage_target": float(target),
                        "coverage_realized": float(hole.mean()),
                        "mask_sha256": mask_hash,
                        "model_id": spec["id"],
                        "model_name": spec["name"],
                        "model_implementation": f"OpenCV {cv2.__version__}; radius 3 pixels",
                        "model_source": "https://docs.opencv.org/4.x/d7/d8b/group__photo__inpaint.html",
                        "model_sha256": None,
                        "execution_provider": "OpenCV CPU",
                        "threads": int(cv2.getNumThreads()),
                        "inference_seconds": elapsed,
                        **metrics(reference, composite, hole),
                    }
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                    done.add(key)
                    written += 1
                    if args.progress_every and written % args.progress_every == 0:
                        print(json.dumps({"method": args.method, "written": written}), flush=True)
    print(json.dumps({
        "dataset": args.dataset,
        "method": args.method,
        "selected_images": len(pairs),
        "new_records": written,
        "skipped_records": skipped,
        "output": str(output),
        "elapsed_seconds": round(time.perf_counter() - started, 2),
    }, indent=2))


if __name__ == "__main__":
    main()
