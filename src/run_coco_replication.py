#!/usr/bin/env python3
"""Run LaMa or MI-GAN on a deterministic COCO val2017 free-form-mask subset."""

import argparse
import hashlib
import json
import os
import random
import resource
import sys
import time
from pathlib import Path

import numpy as np

from run_learned_model_study import LamaOnnx, completed_keys, deterministic_subset
from run_migan_study import MiganOnnx, save_example
from run_study import condition_seed, make_mask, metrics, resize_rgb, sha256


MODELS = {
    "lama": {
        "id": "lama_opencv_zoo_2025jan",
        "name": "LaMa",
        "implementation": "OpenCV Zoo ONNX 2025jan",
        "file": "models/inpainting_lama_2025jan.onnx",
        "sha256": "7df918ac3921d3daf0aae1d219776cf0dc4e4935f035af81841b40adcf74fdf2",
        "source": "https://huggingface.co/opencv/inpainting_lama/resolve/main/inpainting_lama_2025jan.onnx",
    },
    "migan": {
        "id": "migan_512_places2_onnx_pipeline_v2",
        "name": "MI-GAN 512 Places2",
        "implementation": "Author-linked ONNX pipeline v2",
        "file": "models/migan_pipeline_v2.onnx",
        "sha256": "6f1f3530a1a2324b19752018ce756088b07973cda8d7d890034ace5c8a48c40b",
        "source": "https://huggingface.co/andraniksargsyan/migan/resolve/main/migan_pipeline_v2.onnx",
    },
}


def max_rss_mb():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return float(value / (1024 * 1024) if sys.platform == "darwin" else value / 1024)


def verify_model(path, expected):
    if not path.is_file():
        raise SystemExit(f"Model file not found: {path}")
    actual = sha256(path)
    if actual != expected:
        raise SystemExit(f"Model SHA-256 mismatch: expected {expected}, found {actual}")
    return actual


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--config", default="configs/coco2017_freeform_replication.json")
    parser.add_argument("--model", choices=sorted(MODELS), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--example-dir", default="")
    parser.add_argument("--progress-every", type=int, default=25)
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    spec = MODELS[args.model]
    model_path = Path(spec["file"]).resolve()
    model_hash = verify_model(model_path, spec["sha256"])
    images = [(path, path) for path in Path(args.dataset_root).glob("*.jpg")]
    images = deterministic_subset(images, int(config["seed"]), int(config["sample_size"]))
    if len(images) != int(config["sample_size"]):
        raise SystemExit(f"Expected {config['sample_size']} selected images, found {len(images)}")

    if args.model == "lama":
        model = LamaOnnx(model_path, input_size=512, threads=args.threads)
    else:
        model = MiganOnnx(model_path, threads=args.threads)

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    done = completed_keys(output)
    example_dir = Path(args.example_dir).resolve() if args.example_dir else None
    selected_examples = int(config.get("save_example_images", 0))
    example_coverages = {round(float(x), 8) for x in config.get("save_example_coverages", [])}
    mode = "a" if output.exists() else "w"
    written = skipped = 0
    started = time.perf_counter()
    with output.open(mode, encoding="utf-8") as handle:
        for rank, (image_path, _) in enumerate(images):
            image_id = image_path.stem
            reference = resize_rgb(image_path, int(config["image_size"]))
            image_hash = sha256(image_path)
            for family in config["mask_families"]:
                for target in config["coverage_targets"]:
                    key = (image_id, family, round(float(target), 8), spec["id"])
                    if key in done:
                        skipped += 1
                        continue
                    rng = random.Random(condition_seed(config["seed"], image_id, family, target, "mask"))
                    hole = make_mask(
                        family,
                        int(config["image_size"]),
                        float(target),
                        rng,
                        float(config["coverage_tolerance"]),
                        None,
                    )
                    mask_hash = hashlib.sha256(np.packbits(hole).tobytes()).hexdigest()
                    tick = time.perf_counter()
                    generated = model(reference, hole)
                    inference_seconds = time.perf_counter() - tick
                    composite = generated.copy()
                    composite[~hole] = reference[~hole]
                    record = {
                        "dataset_id": config["dataset_id"],
                        "image_id": image_id,
                        "image_sha256": image_hash,
                        "selection_rank": rank,
                        "mask_family": family,
                        "coverage_target": float(target),
                        "coverage_realized": float(hole.mean()),
                        "mask_sha256": mask_hash,
                        "model_id": spec["id"],
                        "model_name": spec["name"],
                        "model_implementation": spec["implementation"],
                        "model_source": spec["source"],
                        "model_sha256": model_hash,
                        "execution_provider": "CPUExecutionProvider",
                        "threads": int(args.threads),
                        "inference_seconds": inference_seconds,
                        "max_rss_mb": max_rss_mb(),
                        **metrics(reference, composite, hole),
                    }
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                    done.add(key)
                    written += 1
                    if example_dir and rank < selected_examples and round(float(target), 8) in example_coverages:
                        save_example(example_dir / f"{image_id}_{family}_{int(target * 100):02d}.jpg", reference, hole, composite)
                    if args.progress_every and written % args.progress_every == 0:
                        print(json.dumps({
                            "model": args.model,
                            "written": written,
                            "image_id": image_id,
                            "elapsed_minutes": round((time.perf_counter() - started) / 60, 2),
                        }), flush=True)
    print(json.dumps({
        "model": args.model,
        "selected_images": len(images),
        "new_records": written,
        "skipped_records": skipped,
        "output": str(output),
        "elapsed_minutes": round((time.perf_counter() - started) / 60, 2),
    }, indent=2))


if __name__ == "__main__":
    main()
