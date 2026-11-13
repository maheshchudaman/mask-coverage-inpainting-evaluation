# COCO Replication Status

Completed on 13 November 2026 after resuming from a clean user-requested pause.

## Completed work

- Official COCO val2017 archive downloaded and extracted: 5,000 images.
- Deterministic evaluation subset: 250 images, seed 20261022.
- Mask design: free-form masks at 10%, 20%, 30%, 40% and 50% target coverage.
- MI-GAN: 1,250 of 1,250 conditions complete.
- LaMa: 1,250 of 1,250 conditions complete.
- Both saved files have zero observed-pixel error and pass the MSE support identity within floating-point tolerance.

## Completed analysis

The 10,000-resample clustered analysis is saved as
`results/learned/ade_coco_cross_dataset_analysis.json`, with a compact CSV and
publication figure in the corresponding results and figures directories.
