#!/usr/bin/env python3
"""Split the frozen ADE20K mask bank into public family/coverage NPZ shards."""

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'results' / 'learned' / 'frozen_masks_250x2x5.npz'
OUTPUT = ROOT / 'results' / 'learned' / 'frozen_masks_250x2x5'


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


OUTPUT.mkdir(parents=True, exist_ok=True)
source = np.load(SOURCE, allow_pickle=False)
families = sorted({str(x) for x in source['mask_family']})
coverages = sorted({round(float(x), 6) for x in source['coverage_target']})
shards = []
for family in families:
    for coverage in coverages:
        select = np.array([
            str(f) == family and round(float(c), 6) == coverage
            for f, c in zip(source['mask_family'], source['coverage_target'])
        ])
        filename = f'{family}_{int(round(coverage * 100)):02d}.npz'
        path = OUTPUT / filename
        np.savez_compressed(
            path,
            masks=source['masks'][select],
            image_id=source['image_id'][select],
            mask_family=source['mask_family'][select],
            coverage_target=source['coverage_target'][select],
            coverage_realised=source['coverage_realised'][select],
            mask_sha256=source['mask_sha256'][select],
            image_size=source['image_size'],
            bitorder=source['bitorder'],
        )
        if path.stat().st_size <= 780000:
            parts = [(filename, select)]
        else:
            selected_indices = np.flatnonzero(select)
            midpoint = len(selected_indices) // 2
            parts = []
            for suffix, indices in (('a', selected_indices[:midpoint]), ('b', selected_indices[midpoint:])):
                part_select = np.zeros(len(source['masks']), dtype=bool)
                part_select[indices] = True
                part_name = f'{family}_{int(round(coverage * 100)):02d}_{suffix}.npz'
                part_path = OUTPUT / part_name
                np.savez_compressed(
                    part_path,
                    masks=source['masks'][part_select],
                    image_id=source['image_id'][part_select],
                    mask_family=source['mask_family'][part_select],
                    coverage_target=source['coverage_target'][part_select],
                    coverage_realised=source['coverage_realised'][part_select],
                    mask_sha256=source['mask_sha256'][part_select],
                    image_size=source['image_size'],
                    bitorder=source['bitorder'],
                )
                parts.append((part_name, part_select))
            path.unlink()
        for part_name, part_select in parts:
            part_path = OUTPUT / part_name
            shards.append({
                'file': part_name,
                'mask_family': family,
                'coverage_target': coverage,
                'records': int(part_select.sum()),
                'bytes': part_path.stat().st_size,
                'sha256': sha256(part_path),
            })

manifest = {
    'format': 'packed binary masks in NumPy NPZ files',
    'source_archive': SOURCE.name,
    'source_archive_sha256': sha256(SOURCE),
    'records': int(len(source['masks'])),
    'image_size': int(source['image_size'][0]),
    'bitorder': str(source['bitorder'][0]),
    'shards': shards,
}
(OUTPUT / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
print(json.dumps(manifest, indent=2))
