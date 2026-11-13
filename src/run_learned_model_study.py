#!/usr/bin/env python3
"""Run a low-memory, resumable LaMa evaluation on Apple Silicon.

The runner reuses the deterministic masks and metrics from ``run_study.py``.
It writes one JSON record after every inference, skips completed conditions on
restart, and stores only a small, prespecified set of qualitative examples.
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

from run_study import (
    condition_seed,
    discover_pairs,
    make_mask,
    metrics,
    resize_labels,
    resize_rgb,
    sha256,
)


MODEL_ID = "lama_opencv_zoo_2025jan"


def deterministic_subset(pairs, seed, limit):
    """Select a stable, order-independent image subset."""
    ranked = sorted(
        pairs,
        key=lambda pair: hashlib.sha256(
            f"{seed}|{pair[0].stem}".encode("utf-8")
        ).digest(),
    )
    return ranked[:limit] if limit else ranked


def completed_keys(path):
    keys = set()
    if not path.is_file():
        return keys
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                keys.add(
                    (
                        record["image_id"],
                        record["mask_family"],
                        round(float(record["coverage_target"]), 8),
                        record["model_id"],
                    )
                )
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"Invalid existing record at {path}:{line_number}: {exc}"
                ) from exc
    return keys


def max_rss_mb():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes; Linux reports KiB.
    return float(value / (1024 * 1024) if sys.platform == "darwin" else value / 1024)


class LamaOnnx:
    """Minimal ONNX Runtime adapter for the official OpenCV Zoo LaMa graph."""

    def __init__(self, model_path, input_size=512, threads=4):
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
        self.input_size = int(input_size)
        input_names = {item.name for item in self.session.get_inputs()}
        if not {"image", "mask"}.issubset(input_names):
            raise RuntimeError(f"Unexpected ONNX inputs: {sorted(input_names)}")

    def __call__(self, image_rgb, hole):
        """Return RGB float32 output in [0, 1] at the reference resolution."""
        height, width = image_rgb.shape[:2]
        source = Image.fromarray(np.uint8(np.clip(image_rgb * 255.0, 0, 255)))
        source = source.resize((self.input_size, self.input_size), Image.Resampling.BILINEAR)
        source_rgb = np.asarray(source, dtype=np.float32)

        # Match the official OpenCV Zoo sample, which uses BGR without channel swap.
        source_bgr = source_rgb[..., ::-1] / 255.0
        image_blob = np.ascontiguousarray(source_bgr.transpose(2, 0, 1)[None], dtype=np.float32)

        mask = Image.fromarray(np.uint8(hole) * 255)
        mask = mask.resize((self.input_size, self.input_size), Image.Resampling.NEAREST)
        mask_blob = (np.asarray(mask, dtype=np.float32) > 0).astype(np.float32)[None, None]

        output = self.session.run(None, {"image": image_blob, "mask": mask_blob})[0]
        if output.ndim != 4 or output.shape[0] != 1 or output.shape[1] != 3:
            raise RuntimeError(f"Unexpected ONNX output shape: {output.shape}")
        result_bgr = np.transpose(output[0], (1, 2, 0))
        # The OpenCV Zoo graph returns pixel values in the 0-255 range.
        result_rgb = np.clip(result_bgr[..., ::-1], 0.0, 255.0).astype(np.uint8)
        result = Image.fromarray(result_rgb).resize((width, height), Image.Resampling.BILINEAR)
        return np.asarray(result, dtype=np.float32) / 255.0


def save_example(path, reference, hole, output):
    path.parent.mkdir(parents=True, exist_ok=True)
    masked = reference.copy()
    masked[hole] = 1.0
    panel = np.concatenate([reference, masked, output], axis=1)
    Image.fromarray(np.uint8(np.clip(panel * 255.0, 0, 255))).save(path)


def verify_model(path, expected_sha256, allow_unverified=False):
    if not path.is_file():
        raise SystemExit(f"Model file not found: {path}")
    actual = sha256(path)
    if expected_sha256 and actual != expected_sha256 and not allow_unverified:
        raise SystemExit(
            "Model SHA-256 mismatch. "
            f"Expected {expected_sha256}, found {actual}. "
            "Use --allow-unverified-model only after checking provenance."
        )
    return actual


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--annotation-root", required=True)
    parser.add_argument("--config", default="configs/learned_models_mac.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-path", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-conditions", type=int, default=0)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--cooldown-every", type=int, default=0)
    parser.add_argument("--cooldown-seconds", type=float, default=-1)
    parser.add_argument("--example-dir", default="")
    parser.add_argument("--allow-unverified-model", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    model_path = Path(args.model_path or config["model"]["file"]).resolve()
    model_hash = verify_model(
        model_path,
        config["model"].get("sha256", ""),
        args.allow_unverified_model,
    )

    pairs = discover_pairs(args.dataset_root, args.annotation_root)
    sample_size = args.limit or int(config.get("sample_size", 0))
    pairs = deterministic_subset(pairs, int(config["seed"]), sample_size)
    if not pairs:
        raise SystemExit("No ADE20K image/annotation pairs were found")

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    done = completed_keys(output_path)
    model = LamaOnnx(
        model_path,
        input_size=int(config.get("model_input_size", 512)),
        threads=args.threads,
    )

    cooldown_every = args.cooldown_every or int(config.get("cooldown_every", 0))
    cooldown_seconds = (
        float(config.get("cooldown_seconds", 0))
        if args.cooldown_seconds < 0
        else args.cooldown_seconds
    )
    example_dir = Path(args.example_dir).resolve() if args.example_dir else None
    example_images = int(config.get("save_example_images", 0))
    example_coverages = {
        round(float(x), 8) for x in config.get("save_example_coverages", [])
    }

    attempted = 0
    written = 0
    skipped = 0
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
                    key = (image_id, family, round(float(target), 8), MODEL_ID)
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
                    # Enforce the same exact observed-pixel compositing assumption
                    # used throughout the analytical study.
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
                        "model_id": MODEL_ID,
                        "model_name": config["model"]["name"],
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

                    elapsed = time.perf_counter() - start
                    print(
                        json.dumps(
                            {
                                "written": written,
                                "skipped": skipped,
                                "image_id": image_id,
                                "mask_family": family,
                                "coverage": target,
                                "inference_seconds": round(inference_seconds, 3),
                                "elapsed_seconds": round(elapsed, 1),
                                "max_rss_mb": round(max_rss_mb(), 1),
                            }
                        ),
                        flush=True,
                    )
                    if cooldown_every and written % cooldown_every == 0:
                        time.sleep(max(0.0, cooldown_seconds))
                if args.max_conditions and attempted >= args.max_conditions:
                    break
            if args.max_conditions and attempted >= args.max_conditions:
                break

    print(
        json.dumps(
            {
                "selected_images": len(pairs),
                "new_records": written,
                "skipped_records": skipped,
                "output": str(output_path),
                "elapsed_seconds": round(time.perf_counter() - start, 1),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
