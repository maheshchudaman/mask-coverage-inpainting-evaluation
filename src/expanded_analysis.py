#!/usr/bin/env python3
"""Create subgroup and ranking-disagreement analyses from the completed study."""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--input', default=str(ROOT / 'results' / 'ade20k_validation_merged.jsonl'))
parser.add_argument('--output', default=str(ROOT / 'results' / 'expanded_analysis.json'))
parser.add_argument('--figure-dir', default=str(ROOT / 'figures'))
parser.add_argument('--seed', type=int, default=20261022)
parser.add_argument('--bootstrap', type=int, default=3000)
args = parser.parse_args()

INPUT = Path(args.input)
OUTPUT = Path(args.output)
FIGURE_DIR = Path(args.figure_dir)
FIGURE = FIGURE_DIR / 'subgroup_sensitivity.png'
RANKING_FIGURE = FIGURE_DIR / 'ranking_disagreement.png'
AGGREGATION_FIGURE = FIGURE_DIR / 'aggregation_shift.png'
SEED = args.seed
BOOTSTRAP = args.bootstrap
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


records = [json.loads(line) for line in INPUT.read_text(encoding='utf-8').splitlines() if line.strip()]
images = sorted({r['image_id'] for r in records})
coverages = sorted({float(r['coverage_target']) for r in records})
families = ['center', 'rectangle', 'freeform', 'segmentation']
operators = ['mean', 'blur', 'noise', 'shift']


def bootstrap_ci(values, salt):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(SEED + salt)
    indices = rng.integers(0, len(values), size=(BOOTSTRAP, len(values)))
    means = values[indices].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


raw = defaultdict(lambda: defaultdict(list))
for r in records:
    image = r['image_id']
    coverage = float(r['coverage_target'])
    family = r['mask_family']
    operator = r['operator']
    for grouping, level in [('family', family), ('operator', operator)]:
        key = (grouping, level, image, coverage)
        raw[key]['full_mse'].append(float(r['full_mse']))
        raw[key]['hole_mse'].append(float(r['hole_mse']))
        raw[key]['full_ssim'].append(float(r['full_ssim']))
        raw[key]['hole_mae'].append(float(r['hole_mae']))
        raw[key]['hole_edge_mae'].append(float(r['hole_edge_mae']))

reduced = {}
for key, values in raw.items():
    full_mse = float(np.mean(values['full_mse']))
    hole_mse = float(np.mean(values['hole_mse']))
    reduced[key] = {
        'full_psnr': float(-10 * math.log10(full_mse)),
        'hole_psnr': float(-10 * math.log10(hole_mse)),
        'full_ssim': float(np.mean(values['full_ssim'])),
        'hole_mae': float(np.mean(values['hole_mae'])),
        'hole_edge_mae': float(np.mean(values['hole_edge_mae'])),
    }


summary = {}
salt = 0
for grouping, levels in [('family', families), ('operator', operators)]:
    summary[grouping] = {}
    for level in levels:
        summary[grouping][level] = {}
        for coverage in coverages:
            summary[grouping][level][f'{coverage:.2f}'] = {}
            for metric in ['full_psnr', 'hole_psnr', 'full_ssim', 'hole_mae', 'hole_edge_mae']:
                vals = [reduced[(grouping, level, image, coverage)][metric] for image in images]
                salt += 1
                summary[grouping][level][f'{coverage:.2f}'][metric] = {
                    'mean': float(np.mean(vals)),
                    'bootstrap_95ci': bootstrap_ci(vals, salt),
                }


# Matched 10%-versus-50% ranking disagreement. Each image contributes a rate
# over its 16 family-by-probe pairs, and images are the inferential units.
lookup = {(r['image_id'], r['mask_family'], r['operator'], float(r['coverage_target'])): r for r in records}
ranking = {'overall': {}, 'by_family': {}}
for family_scope in ['overall'] + families:
    per_image = []
    for image in images:
        outcomes = []
        scopes = families if family_scope == 'overall' else [family_scope]
        for family in scopes:
            for operator in operators:
                low = lookup[(image, family, operator, 0.1)]
                high = lookup[(image, family, operator, 0.5)]
                hole_sign = np.sign(float(high['hole_psnr']) - float(low['hole_psnr']))
                full_sign = np.sign(float(high['full_psnr']) - float(low['full_psnr']))
                outcomes.append(float(hole_sign != full_sign))
        per_image.append(float(np.mean(outcomes)))
    salt += 1
    result = {
        'mean_disagreement_rate': float(np.mean(per_image)),
        'bootstrap_95ci': bootstrap_ci(per_image, salt),
        'image_units': len(per_image),
    }
    if family_scope == 'overall':
        ranking['overall'] = result
    else:
        ranking['by_family'][family_scope] = result


# Counterfactual aggregation audit. The image outputs are held fixed; only the
# prevalence assigned to each coverage stratum changes. This directly
# quantifies how an undisclosed mask-coverage distribution can alter a pooled
# score without any change to the completion procedure.
cell_values = defaultdict(lambda: defaultdict(list))
for r in records:
    key = (r['image_id'], float(r['coverage_target']))
    for metric in ['full_mse', 'hole_mse', 'full_ssim', 'hole_mae']:
        cell_values[key][metric].append(float(r[metric]))

cell_means = {}
for key, metrics in cell_values.items():
    full_mse = float(np.mean(metrics['full_mse']))
    hole_mse = float(np.mean(metrics['hole_mse']))
    cell_means[key] = {
        'full_psnr': float(-10 * math.log10(full_mse)),
        'hole_psnr': float(-10 * math.log10(hole_mse)),
        'full_ssim': float(np.mean(metrics['full_ssim'])),
        'hole_mae': float(np.mean(metrics['hole_mae'])),
    }
mixture_weights = {
    'small_hole_heavy': [0.50, 0.20, 0.15, 0.10, 0.05],
    'balanced': [0.20, 0.20, 0.20, 0.20, 0.20],
    'large_hole_heavy': [0.05, 0.10, 0.15, 0.20, 0.50],
}
aggregation_shift = {'coverage_order': coverages, 'weights': mixture_weights, 'summary': {}, 'contrasts': {}}
for mixture, weights in mixture_weights.items():
    aggregation_shift['summary'][mixture] = {}
    for metric in ['full_psnr', 'hole_psnr', 'full_ssim', 'hole_mae']:
        per_image = [
            float(sum(w * cell_means[(image, coverage)][metric] for w, coverage in zip(weights, coverages)))
            for image in images
        ]
        salt += 1
        aggregation_shift['summary'][mixture][metric] = {
            'mean': float(np.mean(per_image)),
            'bootstrap_95ci': bootstrap_ci(per_image, salt),
        }

for metric in ['full_psnr', 'hole_psnr', 'full_ssim', 'hole_mae']:
    small = np.asarray([
        sum(w * cell_means[(image, coverage)][metric]
            for w, coverage in zip(mixture_weights['small_hole_heavy'], coverages))
        for image in images
    ])
    large = np.asarray([
        sum(w * cell_means[(image, coverage)][metric]
            for w, coverage in zip(mixture_weights['large_hole_heavy'], coverages))
        for image in images
    ])
    differences = small - large
    salt += 1
    aggregation_shift['contrasts'][metric] = {
        'direction': 'small_hole_heavy_minus_large_hole_heavy',
        'mean_difference': float(np.mean(differences)),
        'bootstrap_95ci': bootstrap_ci(differences, salt),
    }


# Descriptive repeated-measures effect size. After removing each image's mean,
# this is the fraction of within-image sum of squares captured by the common
# coverage trend. It is an effect-size summary, not a causal attribution.
within_image_coverage_r2 = {}
for metric in ['full_psnr', 'hole_psnr', 'full_ssim', 'hole_mae']:
    matrix = np.asarray([
        [cell_means[(image, coverage)][metric] for coverage in coverages]
        for image in images
    ], dtype=float)
    image_centered = matrix - matrix.mean(axis=1, keepdims=True)
    coverage_effect = matrix.mean(axis=0, keepdims=True) - matrix.mean()
    ss_total = float(np.sum(image_centered ** 2))
    ss_coverage = float(len(images) * np.sum(coverage_effect ** 2))
    within_image_coverage_r2[metric] = ss_coverage / ss_total if ss_total else float('nan')


result = {
    'design': {
        'images': len(images),
        'raw_records': len(records),
        'bootstrap_repetitions': BOOTSTRAP,
        'bootstrap_unit': 'image',
    },
    'subgroup_summary': summary,
    'ranking_disagreement_10_vs_50': ranking,
    'counterfactual_coverage_mixtures': aggregation_shift,
    'descriptive_within_image_coverage_r2': within_image_coverage_r2,
}
OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True), encoding='utf-8')


plt.rcParams.update({'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9})
colors = {'center': '#1f4e79', 'rectangle': '#2e75b6', 'freeform': '#c55a11', 'segmentation': '#548235',
          'mean': '#7f6000', 'blur': '#4472c4', 'noise': '#c00000', 'shift': '#7030a0'}
fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.2), constrained_layout=True)
panels = [
    ('family', families, 'hole_psnr', 'Hole PSNR by mask family', 'Hole PSNR (dB)'),
    ('family', families, 'full_ssim', 'Full-image SSIM by mask family', 'Full-image SSIM'),
    ('operator', operators, 'hole_psnr', 'Hole PSNR by controlled probe', 'Hole PSNR (dB)'),
    ('operator', operators, 'full_ssim', 'Full-image SSIM by controlled probe', 'Full-image SSIM'),
]
for ax, (grouping, levels, metric, title, ylabel) in zip(axes.flat, panels):
    for level in levels:
        means = [summary[grouping][level][f'{c:.2f}'][metric]['mean'] for c in coverages]
        ax.plot([c * 100 for c in coverages], means, marker='o', linewidth=1.8, label=level.capitalize(), color=colors[level])
    ax.set_title(title)
    ax.set_xlabel('Missing-area coverage (%)')
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
fig.savefig(FIGURE, dpi=300, bbox_inches='tight')
plt.close(fig)


labels = ['Overall', 'Center', 'Rectangle', 'Free-form', 'Semantic']
keys = ['overall', 'center', 'rectangle', 'freeform', 'segmentation']
vals, lo, hi = [], [], []
for key in keys:
    item = ranking['overall'] if key == 'overall' else ranking['by_family'][key]
    mean = 100 * item['mean_disagreement_rate']
    ci = [100 * x for x in item['bootstrap_95ci']]
    vals.append(mean)
    lo.append(mean - ci[0])
    hi.append(ci[1] - mean)
fig, ax = plt.subplots(figsize=(7.6, 3.7), constrained_layout=True)
bars = ax.bar(labels, vals, color=['#1f4e79', '#5b9bd5', '#70ad47', '#ed7d31', '#a5a5a5'])
ax.errorbar(range(len(vals)), vals, yerr=[lo, hi], fmt='none', ecolor='black', capsize=3, linewidth=1)
ax.set_ylabel('Opposite metric preference (%)')
ax.set_title('Disagreement between full-image and hole PSNR\nfor matched 10% versus 50% coverage pairs')
ax.grid(axis='y', alpha=0.25)
for bar, value in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(hi) + 0.3, f'{value:.1f}%', ha='center', va='bottom', fontsize=9)
fig.savefig(RANKING_FIGURE, dpi=300, bbox_inches='tight')
plt.close(fig)


metric_labels = [('full_psnr', 'Full PSNR (dB)'), ('hole_psnr', 'Hole PSNR (dB)'),
                 ('full_ssim', 'Full SSIM'), ('hole_mae', 'Hole MAE')]
mixture_order = ['small_hole_heavy', 'balanced', 'large_hole_heavy']
mixture_labels = ['Small-hole heavy', 'Balanced', 'Large-hole heavy']
fig, axes = plt.subplots(1, 4, figsize=(10.8, 3.2), constrained_layout=True)
for ax, (metric, label) in zip(axes, metric_labels):
    values = [aggregation_shift['summary'][m][metric]['mean'] for m in mixture_order]
    ax.bar(range(3), values, color=['#5b9bd5', '#a5a5a5', '#ed7d31'])
    ax.set_xticks(range(3), ['Small', 'Balanced', 'Large'], rotation=20)
    ax.set_title(label)
    ax.grid(axis='y', alpha=0.25)
fig.suptitle('Same outputs, different coverage prevalence', fontsize=11)
fig.savefig(AGGREGATION_FIGURE, dpi=300, bbox_inches='tight')
plt.close(fig)

print(json.dumps({'ranking_disagreement': ranking, 'output': str(OUTPUT)}, indent=2))
