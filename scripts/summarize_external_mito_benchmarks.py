#!/usr/bin/env python3
"""Summarize declared cohorts without dropping failures or pooling source crops as animals."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.prepare_external_mito_benchmarks import DEFAULT_ROOT, sha256, write_json
from scripts.run_external_mito_benchmark import locked_learned_row


def read_csv(path):
    if not path.exists():
        return []
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    if not rows:
        path.write_text('')
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def collect_case(root, spec):
    case = spec['case']
    out = root / 'by_case' / case
    rows = []
    if (out / 'ANALYSIS_SUCCESS.json').exists():
        record = json.loads((out / 'ANALYSIS_SUCCESS.json').read_text())
        if record['manifest_sha256'] != sha256(root / 'manifest.json'):
            raise ValueError(f'Stale analysis: {case}')
        graph = {x['method']: x for x in read_csv(out / 'graph/metrics.csv')}
        selected = [('sdt_fragments', graph['baseline']), ('banis_gr', graph['affinity_merge']),
                    ('banis_gr_plus', locked_learned_row(read_csv(out / 'learned_gate/per_case_metrics.csv'))),
                    ('region_mws', read_csv(out / 'region_mws/metrics.csv')[0])]
        morph = {x['method']: x for x in read_csv(out / 'morphometry/method_metrics.csv')}
        for method, row in selected:
            metrics = {key: float(row[key]) for key in ('f1', 'panoptic_quality', 'precision', 'recall')}
            metrics.update({key: int(row[key]) for key in ('tp', 'fp', 'fn', 'n_true', 'n_pred')})
            for key in ('count_absolute_relative_error', 'volume_distribution_wasserstein_log'):
                value = float(morph[method][key])
                metrics[key] = value if math.isfinite(value) else None
            rows.append({'case': case, 'cohort': spec['cohort'], 'specimen_cluster': spec['specimen_cluster'],
                         'method': method, **metrics})
    if (out / 'mitonet/evaluation/SUCCESS').exists():
        row = read_csv(out / 'mitonet/evaluation/metrics.csv')[0]
        rows.append({'case': case, 'cohort': spec['cohort'], 'specimen_cluster': spec['specimen_cluster'],
                     'method': 'mitonet_v1',
                     **{k: float(row[k]) for k in ('f1', 'panoptic_quality', 'precision', 'recall')},
                     **{k: int(row[k]) for k in ('tp', 'fp', 'fn', 'n_true', 'n_pred')}})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    manifest = json.loads((args.root / 'manifest.json').read_text())
    rows = [row for spec in manifest['cases'] for row in collect_case(args.root, spec)]
    methods = ['sdt_fragments', 'banis_gr', 'banis_gr_plus', 'region_mws', 'mitonet_v1']
    by_group = defaultdict(list)
    for row in rows:
        by_group[(row['cohort'], row['method'])].append(row)
    aggregates = []
    for cohort in dict.fromkeys(x['cohort'] for x in manifest['cases']):
        expected = [x['case'] for x in manifest['cases'] if x['cohort'] == cohort]
        for method in methods:
            group = by_group[(cohort, method)]
            seen = {row['case'] for row in group}
            if len(seen) != len(group):
                raise ValueError('Duplicate cases in summary')
            tp, fp, fn = [sum(row[key] for row in group) for key in ('tp', 'fp', 'fn')]
            aggregates.append({'cohort': cohort, 'method': method, 'expected_cases': len(expected),
                               'completed_cases': len(group), 'missing_cases': sorted(set(expected) - seen),
                               'complete': seen == set(expected),
                               'macro_f1': sum(row['f1'] for row in group) / len(group) if group else None,
                               'macro_pq': sum(row['panoptic_quality'] for row in group) / len(group) if group else None,
                               'pooled_f1': 2 * tp / (2 * tp + fp + fn) if group and (2 * tp + fp + fn) else (0 if group else None),
                               'tp': tp, 'fp': fp, 'fn': fn})
    attributes = []
    for spec in manifest['cases']:
        if (args.root / 'by_case' / spec['case'] / 'ANALYSIS_SUCCESS.json').exists():
            attributes.extend(read_csv(args.root / 'by_case' / spec['case'] / 'attributes/groups.csv'))
    output = args.root / 'summary'
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / 'per_case_metrics.csv', rows)
    write_csv(output / 'urocell_attribute_groups.csv', attributes)
    complete = all(row['complete'] for row in aggregates)
    report = {'updated_utc': datetime.now(timezone.utc).isoformat(), 'complete': complete,
              'manifest_sha256': sha256(args.root / 'manifest.json'), 'cohorts': aggregates,
              'caveats': ['Incomplete aggregates explicitly list missing cases; never final manuscript results.',
                          'UroCell crops share a source; no independent-animal confidence interval.',
                          'Native-resolution, no external threshold tuning; historical CEM prior use disclosed.',
                          'MitoNet is externally pretrained, not training-matched; Lucchi++ counted once.']}
    write_json(output / 'summary.json', report)
    lines = ['# Frozen external mitochondrial benchmarks', '', f'Complete: {complete}', '',
             '| Cohort | Method | Cases | Macro F1 | Macro PQ | Pooled F1 |', '|---|---|---:|---:|---:|---:|']
    for row in aggregates:
        values = ['—' if row[key] is None else f'{row[key]:.4f}' for key in ('macro_f1', 'macro_pq', 'pooled_f1')]
        lines.append(f'| {row["cohort"]} | {row["method"]} | {row["completed_cases"]}/{row["expected_cases"]} | ' + ' | '.join(values) + ' |')
    lines.extend(['', *report['caveats'], ''])
    (output / 'README.md').write_text('\n'.join(lines))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
