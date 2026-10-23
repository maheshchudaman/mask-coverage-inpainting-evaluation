#!/usr/bin/env python3
"""Run a deterministic, model-free mask-coverage metric experiment."""

import argparse
import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy import ndimage
from skimage.metrics import structural_similarity


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def condition_seed(master_seed, image_id, mask_family, coverage, operator):
    text = f"{master_seed}|{image_id}|{mask_family}|{coverage:.4f}|{operator}"
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


def resize_rgb(path, size):
    image = Image.open(path).convert("RGB")
    return np.asarray(image.resize((size, size), Image.Resampling.BICUBIC), dtype=np.float32) / 255.0


def resize_labels(path, size):
    image = Image.open(path)
    return np.asarray(image.resize((size, size), Image.Resampling.NEAREST))


def centered_mask(size, coverage):
    aspect = 1.0
    height = max(1, min(size, round(size * math.sqrt(coverage / aspect))))
    width = max(1, min(size, round(size * math.sqrt(coverage * aspect))))
    y0 = (size - height) // 2
    x0 = (size - width) // 2
    mask = np.zeros((size, size), dtype=bool)
    mask[y0:y0 + height, x0:x0 + width] = True
    return mask


def rectangle_mask(size, coverage, rng):
    log_aspect = rng.uniform(math.log(0.5), math.log(2.0))
    aspect = math.exp(log_aspect)
    height = max(1, min(size, round(size * math.sqrt(coverage / aspect))))
    width = max(1, min(size, round(size * math.sqrt(coverage * aspect))))
    y0 = rng.randrange(size - height + 1)
    x0 = rng.randrange(size - width + 1)
    mask = np.zeros((size, size), dtype=bool)
    mask[y0:y0 + height, x0:x0 + width] = True
    return mask


def freeform_mask(size, coverage, rng, tolerance):
    canvas = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(canvas)
    lower = max(0.0, coverage - tolerance)
    upper = min(1.0, coverage + tolerance)
    for _ in range(300):
        x, y = rng.randrange(size), rng.randrange(size)
        points = [(x, y)]
        for _ in range(rng.randint(2, 6)):
            x = max(0, min(size - 1, x + rng.randint(-size // 4, size // 4)))
            y = max(0, min(size - 1, y + rng.randint(-size // 4, size // 4)))
            points.append((x, y))
        width = rng.randint(max(3, size // 50), max(5, size // 15))
        draw.line(points, fill=255, width=width, joint="curve")
        current = np.asarray(canvas, dtype=np.uint8) > 0
        realised = float(current.mean())
        if lower <= realised <= upper:
            return adjust_coverage(current, coverage, rng)
        if realised > upper:
            break
    current = np.asarray(canvas, dtype=np.uint8) > 0
    return adjust_coverage(current, coverage, rng)


def segmentation_mask(labels, coverage, rng):
    if labels.ndim == 3:
        # ADE challenge labels are normally single-channel; preserve a stable
        # integer encoding if an RGB label image is encountered.
        labels = labels[..., 0].astype(np.int64) + 256 * labels[..., 1].astype(np.int64)
    choices = [int(v) for v in np.unique(labels) if int(v) != 0]
    rng.shuffle(choices)
    mask = np.zeros(labels.shape, dtype=bool)
    for value in choices:
        mask |= labels == value
        if float(mask.mean()) >= coverage:
            break
    return adjust_coverage(mask, coverage, rng)


def adjust_coverage(mask, coverage, rng):
    """Morph a binary mask towards the target, then make a deterministic exact adjustment."""
    mask = mask.astype(bool)
    target = int(round(mask.size * coverage))
    for _ in range(64):
        count = int(mask.sum())
        if abs(count - target) <= max(4, int(mask.size * 0.002)):
            break
        mask = ndimage.binary_dilation(mask) if count < target else ndimage.binary_erosion(mask)
        if not mask.any():
            mask[rng.randrange(mask.shape[0]), rng.randrange(mask.shape[1])] = True
    flat = mask.reshape(-1)
    count = int(flat.sum())
    if count < target:
        candidates = np.flatnonzero(~flat)
        rng.shuffle(candidates)
        flat[candidates[: target - count]] = True
    elif count > target:
        candidates = np.flatnonzero(flat)
        rng.shuffle(candidates)
        flat[candidates[: count - target]] = False
    return flat.reshape(mask.shape)


def make_mask(family, size, coverage, rng, tolerance, labels=None):
    if family == "center":
        return adjust_coverage(centered_mask(size, coverage), coverage, rng)
    if family == "rectangle":
        return adjust_coverage(rectangle_mask(size, coverage, rng), coverage, rng)
    if family == "freeform":
        return freeform_mask(size, coverage, rng, tolerance)
    if family == "segmentation":
        if labels is None:
            raise ValueError("segmentation masks require an annotation image")
        return segmentation_mask(labels, coverage, rng)
    raise ValueError(f"unknown mask family: {family}")


def controlled_completion(image, hole, operator, rng, config):
    output = image.copy()
    if operator == "mean":
        observed = image[~hole]
        value = observed.mean(axis=0) if len(observed) else image.mean(axis=(0, 1))
        output[hole] = value
    elif operator == "blur":
        source = Image.fromarray(np.uint8(np.clip(image * 255.0, 0, 255)))
        blurred = np.asarray(source.filter(ImageFilter.GaussianBlur(config["blur_radius"])), dtype=np.float32) / 255.0
        output[hole] = blurred[hole]
    elif operator == "noise":
        observed = image[~hole]
        mean = observed.mean(axis=0) if len(observed) else image.mean(axis=(0, 1))
        noise = np.random.default_rng(rng.randrange(2**63)).normal(
            mean, config["noise_sigma"], size=image.shape
        )
        output[hole] = np.clip(noise[hole], 0.0, 1.0)
    elif operator == "shift":
        pixels = max(1, round(image.shape[0] * config["shift_fraction"]))
        shifted = np.roll(image, shift=(pixels, pixels), axis=(0, 1))
        output[hole] = shifted[hole]
    else:
        raise ValueError(f"unknown operator: {operator}")
    output[~hole] = image[~hole]
    return output


def psnr_from_mse(value):
    return float("inf") if value <= 0 else float(-10.0 * math.log10(value))


def sobel_magnitude(image):
    gray = image.mean(axis=2)
    gx = ndimage.sobel(gray, axis=1, mode="reflect")
    gy = ndimage.sobel(gray, axis=0, mode="reflect")
    return np.sqrt(gx * gx + gy * gy)


def metrics(reference, output, hole):
    error = output - reference
    squared = error * error
    absolute = np.abs(error)
    full_mse = float(squared.mean())
    hole_mse = float(squared[hole].mean())
    full_mae = float(absolute.mean())
    hole_mae = float(absolute[hole].mean())
    full_ssim = float(
        structural_similarity(reference, output, data_range=1.0, channel_axis=2)
    )
    ref_edge = sobel_magnitude(reference)
    out_edge = sobel_magnitude(output)
    edge_mae = float(np.abs(ref_edge - out_edge)[hole].mean())
    observed_max_error = float(absolute[~hole].max()) if (~hole).any() else 0.0
    return {
        "full_mse": full_mse,
        "hole_mse": hole_mse,
        "full_psnr": psnr_from_mse(full_mse),
        "hole_psnr": psnr_from_mse(hole_mse),
        "full_mae": full_mae,
        "hole_mae": hole_mae,
        "full_ssim": full_ssim,
        "hole_edge_mae": edge_mae,
        "observed_max_error": observed_max_error,
    }


def discover_pairs(dataset_root, annotation_root):
    images = sorted(Path(dataset_root).glob("*.jpg"))
    pairs = []
    for image in images:
        annotation = Path(annotation_root) / f"{image.stem}.png"
        if annotation.is_file():
            pairs.append((image, annotation))
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--annotation-root", required=True)
    parser.add_argument("--config", default="configs/study.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--start-image", type=int, default=0)
    parser.add_argument("--end-image", type=int, default=0)
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    pairs = discover_pairs(args.dataset_root, args.annotation_root)
    if args.end_image:
        pairs = pairs[args.start_image:args.end_image]
    elif args.start_image:
        pairs = pairs[args.start_image:]
    if args.limit:
        pairs = pairs[: args.limit]
    if not pairs:
        raise SystemExit("No ADE20K image/annotation pairs were found")

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for image_path, annotation_path in pairs:
            image_id = image_path.stem
            image = resize_rgb(image_path, config["image_size"])
            labels = resize_labels(annotation_path, config["image_size"])
            image_hash = sha256(image_path)
            for family in config["mask_families"]:
                for target in config["coverage_targets"]:
                    # Masks are shared by every completion operator.
                    mask_rng = random.Random(
                        condition_seed(config["seed"], image_id, family, target, "mask")
                    )
                    hole = make_mask(
                        family,
                        config["image_size"],
                        target,
                        mask_rng,
                        config["coverage_tolerance"],
                        labels,
                    )
                    packed_mask_hash = hashlib.sha256(np.packbits(hole).tobytes()).hexdigest()
                    for operator in config["operators"]:
                        seed = condition_seed(config["seed"], image_id, family, target, operator)
                        result = controlled_completion(
                            image, hole, operator, random.Random(seed), config
                        )
                        record = {
                            "image_id": image_id,
                            "image_sha256": image_hash,
                            "mask_family": family,
                            "coverage_target": target,
                            "coverage_realized": float(hole.mean()),
                            "mask_sha256": packed_mask_hash,
                            "operator": operator,
                            "condition_seed": seed,
                            **metrics(image, result, hole),
                        }
                        handle.write(json.dumps(record, sort_keys=True) + "\n")
                        handle.flush()

    print(json.dumps({"images": len(pairs), "records": len(pairs) * len(config["mask_families"]) * len(config["coverage_targets"]) * len(config["operators"]), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
