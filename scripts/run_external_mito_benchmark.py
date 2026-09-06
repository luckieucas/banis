#!/usr/bin/env python3
"""Run one declared external case with frozen inference and decoder settings."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import zarr

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'src'))
from scripts.prepare_external_mito_benchmarks import DEFAULT_ROOT, sha256, write_json


def run(*arguments, cwd=REPO, env=None):
    command = [str(x) for x in arguments]
    print('RUN ' + ' '.join(command), flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def read_csv(path):
    with Path(path).open(newline='') as handle:
        return list(csv.DictReader(handle))


def locked_learned_row(rows):
    selected = [row for row in rows if row['method'] == 'learned_edge_scorer'
                and float(row['probability_threshold']) == .7 and float(row['maximum_uncertainty']) == 1.]
    if len(selected) != 1:
        raise ValueError('Expected exactly one frozen learned-scorer row (not its baseline row)')
    return selected[0]


def locked_inputs(manifest, case):
    spec = next(x for x in manifest['cases'] if x['case'] == case)
    root = Path(manifest['root'])
    audit = json.loads((root / 'data' / case / 'audit.json').read_text())
    if audit['spec'] != spec or not audit['roundtrip_voxel_exact']:
        raise ValueError('Preparation does not match declared case')
    for key, item in manifest['frozen_files'].items():
        if sha256(Path(item['path'])) != item['sha256']:
            raise ValueError(f'Frozen input changed: {key}')
    op = json.loads(Path(manifest['frozen_files']['operating_point']['path']).read_text())['best_operating_point']
    if float(op['probability_threshold']) != 0.70 or float(op['maximum_uncertainty']) != 1.0:
        raise ValueError('Unexpected validation operating point')
    expected = {'resampling': 'none', 'normalization': 'lossless uint8 / 255',
                'graph_threshold': .4, 'minimum_evidence': 8, 'minimum_high_fraction': .5,
                'probability_threshold': .7, 'maximum_uncertainty': 1., 'region_mws_beta': .5,
                'sdt_seed_threshold': .5, 'sdt_foreground_threshold': 0., 'sdt_min_size': 100,
                'edt_downsample_factor': 2, 'iou_threshold': .5, 'attribute_overlap_fraction': .1}
    if manifest['protocol'] != expected:
        raise ValueError('This runner only implements the declared frozen protocol')
    return spec, audit


def decode(prediction: Path, out: Path):
    from src.inference.watershed_and_eval_sdt_only import apply_watershed_segmentation_sdt_only
    if (out / 'SUCCESS.json').exists():
        return
    sdt = np.asarray(zarr.open_array(str(prediction / 'channel_6.zarr'), mode='r'))
    if not np.isfinite(sdt).all():
        raise ValueError('Non-finite SDT prediction')
    _, _, fragments = apply_watershed_segmentation_sdt_only(
        sdt, skeleton_thr=.5, foreground_thr=0., min_size=100,
        edt_downsample_factor=2, edt_parallel=int(os.environ.get('SLURM_CPUS_PER_TASK', '4')))
    out.mkdir(parents=True, exist_ok=True)
    # Preserve the algorithm's IDs rather than wrapping IDs > 65535 in uint16.
    if int(fragments.max()) > np.iinfo(np.uint32).max:
        raise ValueError('Fragment IDs exceed uint32')
    target = zarr.open_array(str(out / 'final_instance_seg.zarr'), mode='w', zarr_format=2,
                             shape=fragments.shape, dtype='uint32', chunks=(32, 256, 256))
    for start in range(0, fragments.shape[0], 32):
        target[start:start + 32] = fragments[start:start + 32]
        np.testing.assert_array_equal(target[start:start + 32], fragments[start:start + 32])
    write_json(out / 'SUCCESS.json', {'shape': list(fragments.shape), 'max_id': int(fragments.max()),
                                    'dtype': 'uint32', 'algorithm': 'unchanged production SDT watershed'})


def analyze(case, spec, audit, manifest, out):
    prediction, graph = out / 'prediction', out / 'graph'
    if not (prediction / 'SUCCESS').exists():
        raise FileNotFoundError('Inference has not completed')
    decode(prediction, out / 'sdt')
    if not (graph / 'SUCCESS').exists():
        affinities = []
        for channel in range(6):
            affinities.extend(['--affinity', f'{channel}={prediction / f"channel_{channel}.zarr"}'])
        run(sys.executable, REPO / 'scripts/evaluate_sdt_affinity_graph.py', '--case', case,
            '--fragments', out / 'sdt/final_instance_seg.zarr', '--ground-truth', Path(audit['prepared_zarr']) / 'seg',
            *affinities, '--thresholds', .4, '--minimum-evidence', 8, '--minimum-high-fraction', .5,
            '--oracle-minimum-purity', .9, '--affinity-positive-threshold', .5,
            '--block-shape', 32, 256, 256, '--voxel-spacing-nm', *spec['spacing_nm_zyx'], '--output-dir', graph)
    # A single threshold was transferred; never consume a per-test selected LUT.
    config = json.loads((graph / 'config.json').read_text())
    if config['thresholds'] != [.4]:
        raise ValueError('Graph output includes unlocked thresholds')
    if not (out / 'region_mws/SUCCESS').exists():
        run('/projects/weilab/liupeng/.codex/envs/affogato/bin/python', REPO / 'scripts/evaluate_region_mutex_watershed.py',
            '--graph-audit-dir', graph, '--output-dir', out / 'region_mws', '--betas', .5,
            '--block-shape', 32, 256, 256)
    scorer = manifest['frozen_files']['edge_scorer']['path']
    if not (out / 'learned_gate/evaluation_summary.json').exists():
        run(sys.executable, REPO / 'scripts/evaluate_sdt_affinity_edge_scorer.py', '--model', scorer,
            '--graph-dir', graph, '--output-dir', out / 'learned_gate', '--operating-point-from',
            manifest['frozen_files']['operating_point']['path'], '--minimum-evidence', 8,
            '--minimum-affinity-mean', .4, '--minimum-high-fraction', .5, '--block-shape', 32, 256, 256)
    if not (out / 'morphometry/SUCCESS').exists():
        run(sys.executable, REPO / 'scripts/evaluate_instance_morphometry.py', '--graph-audit-dir', graph,
            '--region-mws-lut', out / 'region_mws/label_lut_beta_0.50.npy', '--edge-scorer', scorer,
            '--output-dir', out / 'morphometry', '--block-shape', 32, 256, 256)
    metrics = {row['method']: row for row in read_csv(graph / 'metrics.csv')}
    reference = {'sdt_fragments': metrics['baseline'], 'banis_gr': metrics['affinity_merge'],
                 'banis_gr_plus': locked_learned_row(read_csv(out / 'learned_gate/per_case_metrics.csv')),
                 'region_mws': read_csv(out / 'region_mws/metrics.csv')[0]}
    morph = {row['method']: row for row in read_csv(out / 'morphometry/method_metrics.csv')}
    for method, row in reference.items():
        for source, target in [('f1', 'instance_f1_check'), ('panoptic_quality', 'instance_panoptic_quality_check')]:
            if not np.isclose(float(row[source]), float(morph[method][target]), rtol=0, atol=1e-12):
                raise ValueError(f'Independent evaluator disagreement: {method} {source}')
    if audit.get('attributes'):
        run(sys.executable, REPO / 'scripts/evaluate_urocell_attributes.py', '--case-dir', out,
            '--attributes', audit['attributes'], '--output-dir', out / 'attributes')
    write_json(out / 'ANALYSIS_SUCCESS.json', {'case': case, 'four_decoder_metric_crosschecks': True,
                                             'manifest_sha256': sha256(Path(manifest['root']) / 'manifest.json')})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--case', required=True)
    parser.add_argument('--stage', choices=['infer', 'analyze', 'mitonet'], required=True)
    args = parser.parse_args()
    manifest = json.loads((args.root / 'manifest.json').read_text())
    spec, audit = locked_inputs(manifest, args.case)
    out = args.root / 'by_case' / args.case
    out.mkdir(parents=True, exist_ok=True)
    record = {'case': args.case, 'stage': args.stage, 'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
              'manifest_sha256': sha256(args.root / 'manifest.json'),
              'runner_sha256': sha256(Path(__file__)), 'audit_sha256': sha256(args.root / 'data' / args.case / 'audit.json')}
    previous = out / f'{args.stage}_execution.json'
    if previous.exists():
        old = json.loads(previous.read_text())
        for field in ('manifest_sha256', 'audit_sha256'):
            if old[field] != record[field]:
                raise ValueError(f'Inputs changed since earlier execution: {field}')
    write_json(previous, record)
    os.environ.update(PYTHONHASHSEED='0', CUBLAS_WORKSPACE_CONFIG=':4096:8',
                      PYTHONPATH=f'{REPO}:{REPO / "src"}')
    if args.stage == 'infer':
        if not (out / 'prediction/SUCCESS').exists():
            run(sys.executable, REPO / 'scripts/stream_predict_affinities.py', '--checkpoint-path',
                manifest['frozen_files']['checkpoint']['path'], '--input-path', audit['prepared_zarr'],
                '--zarr-input-key', 'img', '--output-dir', out / 'prediction', '--patch-size', 128,
                '--batch-size', 2, '--output-block-shape', 256, 512, 512, '--output-chunks', 64, 256, 256,
                '--padding-mode', 'edge', '--padding-position', 'center', '--seed', 0)
    elif args.stage == 'analyze':
        analyze(args.case, spec, audit, manifest, out)
    else:
        run('bash', REPO / 'scripts/run_mitonet_baseline_case.sh', args.case,
            Path(audit['prepared_zarr']) / 'img', Path(audit['prepared_zarr']) / 'seg', '-', out / 'mitonet')


if __name__ == '__main__':
    main()
