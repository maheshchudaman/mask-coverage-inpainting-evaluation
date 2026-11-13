#!/usr/bin/env python3
"""Run a low-memory, resumable MI-GAN evaluation on the LaMa study design.

The script deliberately reuses the image selection, deterministic masks,
compositing rule and metrics from the existing learned-model experiment.  The
official MI-GAN repository links the ONNX pipeline used here.  The model file
is accepted only when its SHA-256 matches the prespecified configuration.
"""

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
from PIL import Image

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


def max_rss_mb():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return float(value / (1024 * 1024) if sys.platform == "darwin" else value / 1024)


class MiganOnnx:
    """Adapter for the author-linked MI-GAN ONNX pipeline.

    The graph accepts RGB uint8 images and binary uint8 masks in NCHW form.
    Its mask polarity is 255 for known pixels and 0 for the hole.
    """

    def __init__(self, model_path, threads=2):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise SystemExit(
                "onnxruntime is required. Install requirements-learned-mac.txt"
            ) from exc

        options = ort.SessionOptions()
        options.intra_op_num_threads = max(1, int(threads))
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.enable_mem_pattern = False
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.log_severity_level = 3
        self.session = ort.InferenceSession(
            str(model_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if len(inputs) != 2 or len(outputs) != 1:
            raise RuntimeError(
                f"Unexpected MI-GAN graph signature: {len(inputs)} inputs, "
                f"{len(outputs)} outputs"
            )
        self.image_name = next((x.name for x in inputs if x.name == "image"), inputs[0].name)
        self.mask_name = next((x.name for x in inputs if x.name == "mask"), inputs[1].name)
        self.output_name = outputs[0].name

    def __call__(self, image_rgb, hole):
        height, width = image_rgb.shape[:2]
        image_u8 = np.uint8(np.clip(image_rgb * 255.0, 0, 255))
        image_blob = np.ascontiguousarray(image_u8.transpose(2, 0, 1)[None])
        known_u8 = np.where(hole, 0, 255).astype(np.uint8)
        mask_blob = np.ascontiguousarray(known_u8[None, None])
        output = self.session.run(
            [self.output_name],
            {self.image_name: image_blob, self.mask_name: mask_blob},
        )[0]
        if output.ndim != 4 or output.shape[0] != 1 or output.shape[1] != 3:
            raise RuntimeError(f"Unexpected MI-GAN output shape: {output.shape}")
        result = np.transpose(output[0], (1, 2, 0))
        if result.shape[:2] != (height, width):
            result = np.asarray(
                Image.fromarray(np.uint8(np.clip(result, 0, 255))).resize(
                    (width, height), Image.Resampling.BILINEAR
                )
            )
        return np.asarray(result, dtype=np.float32) / 255.0


def verify_model(path, expected_sha256):
    if not path.is_file():
        raise SystemExit(f"Model file not found: {path}")
    actual = sha256(path)
    if not expected_sha256:
        raise SystemExit("A prespecified model SHA-256 is required")
    if actual != expected_sha256:
        raise SystemExit(
            f"Model SHA-256 mismatch: expected {expected_sha256}, found {actual}"
        )
    return actual


def save_example(path, reference, hole, output):
    path.parent.mkdir(parents=True, exist_ok=True)
    masked = reference.copy()
    masked[hole] = 1.0
    panel = np.concatenate([reference, masked, output], axis=1)
    Image.fromarray(np.uint8(np.clip(panel * 255.0, 0, 255))).save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--annotation-root", required=True)
    parser.add_argument("--config", default="configs/migan_mac.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-path", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-conditions", type=int, default=0)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--example-dir", default="")
    parser.add_argument("--progress-every", type=int, default=25)
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    model_spec = config["model"]
    model_path = Path(args.model_path or model_spec["file"]).resolve()
    model_hash = verify_model(model_path, model_spec["sha256"])
    model_id = model_spec["id"]

    pairs = discover_pairs(args.dataset_root, args.annotation_root)
    sample_size = args.limit or int(config.get("sample_size", 0))
    pairs = deterministic_subset(pairs, int(config["seed"]), sample_size)
    if not pairs:
        raise SystemExit("No ADE20K image/annotation pairs were found")

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    done = completed_keys(output_path)
    model = MiganOnnx(model_path, threads=args.threads)
    example_dir = Path(args.example_dir).resolve() if args.example_dir else None
    example_images = int(config.get("save_example_images", 0))
    example_coverages = {
        round(float(x), 8) for x in config.get("save_example_coverages", [])
    }

    attempted = written = skipped = 0
    start = time.perf_counter()
    mode = "a" if output_path.exists() else "w"
    with output_path.open(mode, encoding="utf-8") as handle:
        for selection_rank, (image_path, annotation_path) in enumerate(pairs):
            image_id = image_path.stem
            reference = resize_rgb(image_path, int(config["image_size"]))
            labels = resize_labels(annotation_path, int(config["image_size"]))
            image_hash = sha256(image_path)
            for family in config["mask_families"]:
                for target in config["coverage_targets"]:
                    key = (image_id, family, round(float(target), 8), model_id)
                    if key in done:
                        skipped += 1
                        continue
                    if args.max_conditions and attempted >= args.max_conditions:
                        break
                    attempted += 1
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

                    inference_start = time.perf_counter()
                    generated = model(reference, hole)
                    inference_seconds = time.perf_counter() - inference_start
                    composite = generated.copy()
                    composite[~hole] = reference[~hole]
                    record = {
                        "image_id": image_id,
                        "image_sha256": image_hash,
                        "selection_rank": selection_rank,
                        "mask_family": family,
                        "coverage_target": float(target),
                        "coverage_realized": float(hole.mean()),
                        "mask_sha256": mask_hash,
                        "model_id": model_id,
                        "model_name": model_spec["name"],
                        "model_implementation": model_spec["implementation"],
                        "model_source": model_spec["source"],
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

                    if (
                        example_dir
                        and selection_rank < example_images
                        and round(float(target), 8) in example_coverages
                    ):
                        save_example(
                            example_dir / f"{image_id}_{family}_{int(target * 100):02d}.jpg",
                            reference,
                            hole,
                            composite,
                        )
                    if args.progress_every > 0 and written % args.progress_every == 0:
                        print(
                            json.dumps(
                                {
                                    "written": written,
                                    "skipped": skipped,
                                    "image_id": image_id,
                                    "mask_family": family,
                                    "coverage": target,
                                    "inference_seconds": round(inference_seconds, 3),
                                    "elapsed_seconds": round(time.perf_counter() - start, 1),
                                    "max_rss_mb": round(max_rss_mb(), 1),
                                }
                            ),
                            flush=True,
                        )
                if args.max_conditions and attempted >= args.max_conditions:
                    break
            if args.max_conditions and attempted >= args.max_conditions:
                break

    print(
        json.dumps(
            {
                "output": str(output_path),
                "written": written,
                "skipped": skipped,
                "model_sha256": model_hash,
                "elapsed_seconds": round(time.perf_counter() - start, 3),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
