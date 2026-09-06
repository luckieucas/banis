#!/usr/bin/env python3
"""Descriptive UroCell shape-stratified recall and split/merge diagnostics."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.prepare_external_mito_benchmarks import write_json
from src.inference.sdt_affinity_graph import build_fragment_contingency, load_volume


def object_diagnostics(contingency, attributes, matched_ids, label_lut=None, minimum_fraction=.10):
    """Use all GT objects for merge detection; do not mask away other strata."""
    if not 0 < minimum_fraction <= .5:
        raise ValueError('Expected an overlap fraction in (0, .5]')
    if set(contingency.gt_areas) != {int(row['gt_id']) for row in attributes}:
        raise ValueError('Attributes do not cover exactly the ground-truth instance IDs')
    intersections = defaultdict(int)
    for (gt, fragment), count in contingency.overlaps.items():
        pred = int(label_lut[fragment]) if label_lut is not None else fragment
        if pred > 0:
            intersections[(gt, pred)] += count
    gt_pieces, pred_objects = defaultdict(set), defaultdict(set)
    for (gt, pred), count in intersections.items():
        if count / contingency.gt_areas[gt] >= minimum_fraction:
            gt_pieces[gt].add(pred)
            pred_objects[pred].add(gt)
    rows = []
    for attribute in attributes:
        gt = int(attribute['gt_id'])
        rows.append({**attribute, 'matched_iou50': gt in matched_ids,
                     'n_significant_pred_pieces': len(gt_pieces[gt]),
                     'split': len(gt_pieces[gt]) >= 2,
                     'merge_involved': any(len(pred_objects[p]) >= 2 for p in gt_pieces[gt])})
    return rows


def summarize_groups(rows):
    result = []
    groups = [('all', rows)]
    for key in ('branched', 'contacting'):
        groups.extend((f'{key}={value}', [row for row in rows if row[key] == value]) for value in (True, False, None))
    for group, subset in groups:
        n = len(subset)
        result.append({'group': group, 'n_gt': n,
                       'matched_iou50': sum(row['matched_iou50'] for row in subset),
                       'recall_iou50': sum(row['matched_iou50'] for row in subset) / n if n else None,
                       'split_count': sum(row['split'] for row in subset),
                       'split_rate': sum(row['split'] for row in subset) / n if n else None,
                       'merge_involved_count': sum(row['merge_involved'] for row in subset),
                       'merge_involved_rate': sum(row['merge_involved'] for row in subset) / n if n else None})
    return result


def write_csv(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case-dir', type=Path, required=True)
    parser.add_argument('--attributes', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads((args.case_dir / 'graph/config.json').read_text())
    contingency = build_fragment_contingency(load_volume(config['ground_truth']), load_volume(config['fragments']),
                                             block_shape=(32, 256, 256))
    attributes = json.loads(args.attributes.read_text())
    with (args.case_dir / 'morphometry/matched_objects.csv').open(newline='') as handle:
        matches = list(csv.DictReader(handle))
    methods = {'sdt_fragments': None,
               'banis_gr': args.case_dir / 'morphometry/banis_gr_label_lut.npy',
               'banis_gr_plus': args.case_dir / 'morphometry/banis_gr_plus_label_lut.npy',
               'region_mws': args.case_dir / 'region_mws/label_lut_beta_0.50.npy'}
    object_rows, group_rows = [], []
    for method, path in methods.items():
        matched_ids = {int(row['true_label']) for row in matches if row['method'] == method}
        rows = object_diagnostics(contingency, attributes, matched_ids, np.load(path) if path else None)
        object_rows.extend({'case': config['case'], 'method': method, **row} for row in rows)
        group_rows.extend({'case': config['case'], 'method': method, **row} for row in summarize_groups(rows))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / 'objects.csv', object_rows)
    write_csv(args.output_dir / 'groups.csv', group_rows)
    write_json(args.output_dir / 'protocol.json', {
        'significant_overlap': 'intersection / GT object voxel count >= 0.10, fixed before evaluation',
        'split': 'GT object covered by at least two significant predicted pieces',
        'merge_involved': 'GT object shares a significant predicted piece with another GT object',
        'recall': 'IoU>=0.50 one-to-one matching on the FULL volume; then stratified by GT attribute',
        'caveats': ['descriptive diagnostics, not official UroCell metrics',
                    'split and merge involvement can coexist; not complete error taxonomy',
                    'five crops share a source; no five-animal confidence interval',
                    'incomplete/inconsistent object attributes remain unknown; GT is not edited',
                    'do not compute subgroup precision/F1 by masking away other GT objects']})


if __name__ == '__main__':
    main()
