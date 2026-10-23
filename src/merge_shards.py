#!/usr/bin/env python3
"""Merge deterministic JSONL shards and reject missing or conflicting records."""

import argparse
import json
from collections import Counter
from pathlib import Path


def key(record):
    return (
        record["image_id"],
        record["mask_family"],
        float(record["coverage_target"]),
        record["operator"],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-images", type=int, default=2000)
    parser.add_argument("--records-per-image", type=int, default=80)
    args = parser.parse_args()

    merged = {}
    duplicate_count = 0
    for source in args.inputs:
        with Path(source).open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                record_key = key(record)
                previous = merged.get(record_key)
                if previous is not None:
                    if previous != record:
                        raise SystemExit(
                            f"Conflicting duplicate in {source}:{line_number}: {record_key}"
                        )
                    duplicate_count += 1
                else:
                    merged[record_key] = record

    counts = Counter(record["image_id"] for record in merged.values())
    expected_records = args.expected_images * args.records_per_image
    if len(counts) != args.expected_images or len(merged) != expected_records:
        raise SystemExit(
            f"Incomplete merge: {len(counts)} images, {len(merged)} records; "
            f"expected {args.expected_images} images, {expected_records} records"
        )
    bad = {image_id: count for image_id, count in counts.items() if count != args.records_per_image}
    if bad:
        raise SystemExit(f"Unexpected per-image record counts: {bad}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record_key in sorted(merged):
            handle.write(json.dumps(merged[record_key], sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(output),
                "images": len(counts),
                "records": len(merged),
                "identical_duplicates_ignored": duplicate_count,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
