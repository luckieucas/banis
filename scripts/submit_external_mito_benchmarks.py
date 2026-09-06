#!/usr/bin/env python3
"""Submit frozen external inference -> CPU analysis, plus released MitoNet reference."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.prepare_external_mito_benchmarks import DEFAULT_ROOT, sha256, write_json

PYTHON = '/projects/weilab/liupeng/conda/envs/sdt/bin/python'


def sbatch(root, name, command, dependencies=(), gpu=None, memory='64G', hours=12, after='afterok'):
    log = root / 'logs'
    log.mkdir(exist_ok=True)
    args = ['sbatch', '--parsable', '--partition', 'short', '--cpus-per-task', '4',
            '--mem', memory, '--time', f'{hours}:00:00', '--job-name', name,
            '--output', str(log / (name + '.%j.out')), '--error', str(log / (name + '.%j.err')),
            '--kill-on-invalid-dep=yes']
    if gpu:
        args += ['--gres', f'gpu:{gpu}:1']
    if dependencies:
        args += ['--dependency', after + ':' + ':'.join(map(str, dependencies))]
    wrapper = 'cd ' + shlex.quote(str(REPO)) + ' && ' + shlex.join(
        ['env', 'PYTHONHASHSEED=0', 'OMP_NUM_THREADS=4', 'OPENBLAS_NUM_THREADS=4', 'MKL_NUM_THREADS=4', *map(str, command)])
    args += ['--wrap', wrapper]
    result = subprocess.check_output(args, text=True).strip().split(';')[0]
    if not result.isdigit():
        raise ValueError(f'Unexpected sbatch response: {result}')
    print(f'{name}: {result}', flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--prepare-job', help='Allow preparation to finish as an explicit dependency')
    parser.add_argument('--initial-infer-dependency', help='Continue an existing BANIS GPU lane')
    parser.add_argument('--initial-mitonet-dependency', help='Continue an existing MitoNet GPU lane')
    parser.add_argument('--validation-report', type=Path,
                        default=DEFAULT_ROOT / 'validation/uro_dense_stream_fixed/comparison.json')
    parser.add_argument('--submit', action='store_true', help='Without this flag, validate inputs and print plan only')
    parser.add_argument('--retry-failed-analysis', action='store_true', help='Retry only terminal failed CPU analyses in the ledger')
    args = parser.parse_args()
    manifest = json.loads((args.root / 'manifest.json').read_text())
    validation = json.loads(args.validation_report.read_text())
    if not validation['exact_after_float16'] or validation['mismatched_float16_values'] != 0:
        raise ValueError('Real-checkpoint dense/streaming equivalence has not passed')
    if not (args.root / 'PREPARATION_SUCCESS.json').exists() and not args.prepare_job:
        raise ValueError('Preparation must pass before submission, or have an explicit dependency')
    # Do not accidentally submit the whole cohort twice. Inspect a partial ledger manually.
    ledger = args.root / 'jobs.json'
    if args.retry_failed_analysis:
        record = json.loads(ledger.read_text())
        latest = {row['case']: row for row in record['jobs'] if row['stage'] == 'analyze'}
        latest.update({row['case']: row for row in record.get('retries', [])})
        ids = ','.join(row['job_id'] for row in latest.values())
        output = subprocess.check_output(['sacct', '-n', '-X', '-P', '-j', ids, '-o', 'JobID,State'], text=True)
        states = dict(line.strip().split('|')[:2] for line in output.splitlines() if line.strip())
        failed = [row for row in latest.values() if states.get(row['job_id']) in ('FAILED', 'OUT_OF_MEMORY', 'TIMEOUT')]
        print(json.dumps({'retry_failed_analysis': failed}), flush=True)
        if not args.submit:
            return
        for old in failed:
            case = old['case']
            job = sbatch(args.root, f'ext_retry_{case}', [PYTHON, REPO / 'scripts/run_external_mito_benchmark.py',
                         '--root', args.root, '--case', case, '--stage', 'analyze'], memory='128G')
            record.setdefault('retries', []).append({'case': case, 'stage': 'analyze', 'job_id': job,
                                                    'replaces_failed_job': old['job_id']})
            write_json(ledger, record)
        terminals = [x['job_id'] for x in record['jobs'] if x['stage'] in ('analyze', 'mitonet')]
        terminals += [x['job_id'] for x in record.get('retries', [])]
        # Retain earlier summary and jobs as provenance; a new summary waits on retries.
        summary = sbatch(args.root, 'ext_retry_summary',
                         [PYTHON, REPO / 'scripts/summarize_external_mito_benchmarks.py', '--root', args.root],
                         terminals, memory='8G', hours=1, after='afterany')
        record.setdefault('previous_summary_jobs', []).append(record['summary_job'])
        record['summary_job'] = summary
        write_json(ledger, record)
        return
    if ledger.exists():
        raise FileExistsError(f'Existing submission ledger: {ledger}')
    plan = {'cases': [x['case'] for x in manifest['cases']], 'gpu_lanes_per_method': 2,
            'stages': ['BANIS L40S inference', 'CPU SDT/GR/GR+/RG-MWS/morphometry/attributes', 'MitoNet V100'],
            'summary': 'afterany terminal evaluations; explicitly marks missing results'}
    print(json.dumps(plan, indent=2), flush=True)
    if not args.submit:
        return
    # Exclusive creation protects against simultaneous invocations.
    descriptor = os.open(ledger, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o640)
    os.close(descriptor)
    record = {'submitted_utc': datetime.now(timezone.utc).isoformat(), 'manifest_sha256': sha256(args.root / 'manifest.json'),
              'validation_report': str(args.validation_report), 'validation_sha256': sha256(args.validation_report),
              'prepare_job': args.prepare_job, 'jobs': [], 'submission_complete': False,
              'initial_lane_dependencies': {'infer': args.initial_infer_dependency, 'mitonet': args.initial_mitonet_dependency},
              'code_sha256': {str(path.relative_to(REPO)): sha256(path) for path in
                              [REPO / 'src/inference/inference.py', REPO / 'src/inference/sdt_affinity_edge_scorer.py',
                               REPO / 'scripts/run_external_mito_benchmark.py', REPO / 'scripts/evaluate_urocell_attributes.py']}}
    write_json(ledger, record)
    lanes = {'infer': [args.initial_infer_dependency, None], 'mitonet': [args.initial_mitonet_dependency, None]}
    terminal = []
    for index, spec in enumerate(manifest['cases']):
        case = spec['case']
        for stage, gpu, memory in [('infer', 'l40s', '128G'), ('mitonet', 'v100', '96G')]:
            dependencies = ([args.prepare_job] if args.prepare_job else [])
            predecessor = lanes[stage][index % 2]
            if predecessor:
                dependencies.append(predecessor)
            # afterany lane chaining prevents a scientifically failed case from
            # cancelling later independent cases; preparation itself is checked
            # by the runner's audit before any GPU work.
            job = sbatch(args.root, f'ext_{stage}_{case}',
                         [PYTHON, REPO / 'scripts/run_external_mito_benchmark.py', '--root', args.root,
                          '--case', case, '--stage', stage], dependencies, gpu, memory, after='afterany')
            lanes[stage][index % 2] = job
            record['jobs'].append({'case': case, 'stage': stage, 'job_id': job})
            write_json(ledger, record)
            if stage == 'infer':
                analysis = sbatch(args.root, f'ext_analysis_{case}',
                                  [PYTHON, REPO / 'scripts/run_external_mito_benchmark.py', '--root', args.root,
                                   '--case', case, '--stage', 'analyze'], [job], memory='128G')
                terminal.append(analysis)
                record['jobs'].append({'case': case, 'stage': 'analyze', 'job_id': analysis})
                write_json(ledger, record)
            else:
                terminal.append(job)
    summary = sbatch(args.root, 'ext_summary',
                     [PYTHON, REPO / 'scripts/summarize_external_mito_benchmarks.py', '--root', args.root],
                     terminal, memory='8G', hours=1, after='afterany')
    record.update(summary_job=summary, submission_complete=True)
    write_json(ledger, record)


if __name__ == '__main__':
    main()
