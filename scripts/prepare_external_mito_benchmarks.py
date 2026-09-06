#!/usr/bin/env python3
"""Audit and losslessly stage fixed external mito cohorts; never alter raw data."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import urllib.request

import numpy as np
import SimpleITK as sitk
import tifffile
import zarr

REPO = Path(__file__).resolve().parents[1]
PROJECT = Path('/projects/weilab/liupeng')
RAW = PROJECT / 'data/raw/mito'
DEFAULT_ROOT = PROJECT / 'runs/banis/mitoem2_graph_repair_stage1/external_benchmarks_20260906'
URO_COMMIT = '4bab22add57dd00dacb7ecffae2205e6d1422299'
URO_REPO = 'https://github.com/MancaZerovnikMekuc/UroCell'
CEM_API = 'https://www.ebi.ac.uk/empiar/api/entry/10982/'
STAGE1 = PROJECT / 'runs/banis/mitoem2_graph_repair_stage1'
CHECKPOINT = PROJECT / 'runs/banis/mitoem2_sdt_train/checkpoints/mitoem2_all_sdt_mednextB_lr3e-4_s0_b2_k3_ns250000/default/checkpoints/epoch=171-step=250000.ckpt'
SCORER = STAGE1 / 'learned_edge_scorer_stage1/models/seed0/edge_scorer.joblib'
OPERATING_POINT = STAGE1 / 'learned_edge_scorer_stage1/validation/seed0/evaluation_summary.json'


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def fetch_json(url: str):
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)


def download_git_blob(relative: str, entry: dict, root: Path) -> Path:
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        url = f'https://raw.githubusercontent.com/MancaZerovnikMekuc/UroCell/{URO_COMMIT}/{relative}'
        temporary = target.with_suffix(target.suffix + '.part')
        with urllib.request.urlopen(url, timeout=120) as response, temporary.open('wb') as handle:
            while block := response.read(1024 * 1024):
                handle.write(block)
        # Validate before making a downloaded file available to preparation.
        actual = git_blob_hash(temporary)
        if actual != entry['sha'] or temporary.stat().st_size != entry['size']:
            raise ValueError(f'Download checksum mismatch: {relative}')
        temporary.replace(target)
    if git_blob_hash(target) != entry['sha'] or target.stat().st_size != entry['size']:
        raise ValueError(f'Existing raw file differs from pinned upstream: {target}')
    return target


def git_blob_hash(path: Path) -> str:
    digest = hashlib.sha1(f'blob {path.stat().st_size}\0'.encode())
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_zyx(path: Path) -> np.ndarray:
    if str(path).endswith(('.tif', '.tiff')):
        return tifffile.imread(path)
    return sitk.GetArrayFromImage(sitk.ReadImage(str(path)))


def canonical_image(image: np.ndarray) -> np.ndarray:
    if image.ndim != 3 or not np.isfinite(image).all():
        raise ValueError('Image must be finite and 3-D')
    if image.min() < 0 or image.max() > 255 or not np.equal(image, np.floor(image)).all():
        raise ValueError('Input cannot be losslessly converted to uint8; do not silently rescale')
    return image.astype(np.uint8, copy=False)


def canonical_labels(labels: np.ndarray) -> np.ndarray:
    if labels.ndim != 3 or not np.isfinite(labels).all():
        raise ValueError('Labels must be finite and 3-D')
    if labels.min() < 0 or labels.max() > np.iinfo(np.uint32).max or not np.equal(labels, np.floor(labels)).all():
        raise ValueError('Labels must be nonnegative integer IDs fitting uint32')
    ids = np.unique(labels)
    if len(ids[ids > 0]) < 2:
        raise ValueError('Expected multiple instance IDs; binary/empty labels are not accepted')
    return labels.astype(np.uint32, copy=False)


def attribute_table(labels: np.ndarray, branched: np.ndarray, contacting: np.ndarray,
                    allow_incomplete: bool = False) -> list[dict]:
    """Never impute attributes: incomplete/inconsistent objects are unknown."""
    if labels.shape != branched.shape or labels.shape != contacting.shape:
        raise ValueError('Attribute shape mismatch')
    for attribute in (branched, contacting):
        if not np.isin(attribute, [0, 1, 2]).all():
            raise ValueError('Invalid attribute class')
        if not allow_incomplete and not np.array_equal(labels > 0, attribute > 0):
            raise ValueError('Attribute foreground must exactly match instance foreground')
    rows = []
    for label in np.unique(labels):
        if label == 0:
            continue
        selection = labels == label
        b, c = np.unique(branched[selection]), np.unique(contacting[selection])
        if not allow_incomplete and (len(b) != 1 or len(c) != 1):
            raise ValueError(f'Inconsistent attributes within instance {label}')
        rows.append({'gt_id': int(label), 'branched': bool(b[0] == 1) if len(b) == 1 and b[0] > 0 else None,
                     'contacting': bool(c[0] == 2) if len(c) == 1 and c[0] > 0 else None,
                     'voxels': int(np.count_nonzero(selection))})
    return rows


def stage_case(spec: dict, root: Path) -> dict:
    out = root / 'data' / spec['case']
    marker = out / 'audit.json'
    if marker.exists():
        audit = json.loads(marker.read_text())
        for key, record in audit['raw_files'].items():
            if sha256(Path(record['path'])) != record['sha256']:
                raise ValueError(f'Raw input changed: {key}')
        if audit['spec'] != spec:
            raise ValueError(f'Case specification changed: {spec["case"]}')
        return audit
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'data.zarr').exists():
        raise FileExistsError(f'Incomplete staging exists; inspect before resuming: {out}')
    image, labels = load_zyx(Path(spec['image'])), load_zyx(Path(spec['labels']))
    original_dtypes = [str(image.dtype), str(labels.dtype)]
    image, labels = canonical_image(image), canonical_labels(labels)
    if image.shape != labels.shape or tuple(image.shape) != tuple(spec['shape_zyx']):
        raise ValueError(f'Unexpected shape: {spec["case"]}: {image.shape}, {labels.shape}')
    files = {'image': spec['image'], 'labels': spec['labels']}
    objects = None
    if spec.get('branched'):
        files.update(branched=spec['branched'], contacting=spec['contacting'])
        attribute_arrays = {key: load_zyx(Path(spec[key])) for key in ('branched', 'contacting')}
        objects = attribute_table(labels, attribute_arrays['branched'], attribute_arrays['contacting'], allow_incomplete=True)
        write_json(out / 'attribute_qc.json', {
            key: {'outside_gt_voxels': int(np.count_nonzero((array > 0) & (labels == 0))),
                  'missing_gt_voxels': int(np.count_nonzero((array == 0) & (labels > 0))),
                  'unknown_gt_ids': [row['gt_id'] for row in objects if row[key] is None]}
            for key, array in attribute_arrays.items()})
        write_json(out / 'attributes.json', objects)
    store = zarr.open_group(str(out / 'data.zarr'), mode='w-', zarr_format=2)
    store.attrs.update({'axes': 'zyx', 'voxel_spacing_nm_zyx': spec['spacing_nm_zyx'],
                        'resampling': 'none', 'source': spec['source']})
    for name, array in [('img', image), ('seg', labels)]:
        dest = store.create_array(name, shape=array.shape, dtype=array.dtype, chunks=(32, 256, 256))
        for start in range(0, array.shape[0], 32):
            dest[start:start + 32] = array[start:start + 32]
            np.testing.assert_array_equal(dest[start:start + 32], array[start:start + 32])
    audit = {'spec': spec, 'raw_files': {key: {'path': value, 'sha256': sha256(Path(value))}
                                       for key, value in files.items()},
             'shape_zyx': list(image.shape), 'original_dtypes': original_dtypes,
             'image_range': [int(image.min()), int(image.max())],
             'n_instances': int(len(np.unique(labels)) - int(np.any(labels == 0))),
             'max_instance_id': int(labels.max()), 'roundtrip_voxel_exact': True,
             'prepared_zarr': str(out / 'data.zarr'), 'attributes': str(out / 'attributes.json') if objects else None}
    if str(spec['image']).endswith('.nii.gz'):
        info = sitk.ReadImage(spec['image'])
        audit['original_itk_geometry'] = {'spacing_xyz': list(info.GetSpacing()),
                                           'origin_xyz': list(info.GetOrigin()), 'direction': list(info.GetDirection())}
    write_json(marker, audit)
    print(f'{spec["case"]}: shape={image.shape}, instances={audit["n_instances"]}, exact roundtrip', flush=True)
    return audit


def build_specs(root: Path, download_urocell: bool) -> list[dict]:
    cem_metadata = fetch_json(CEM_API)
    write_json(root / 'sources/empiar10982.json', cem_metadata)
    specimens = [('c_elegans', (256, 256, 256), 24), ('fly_brain', (256, 255, 255), 12),
                 ('hela_cell', (256, 256, 256), 15), ('glycolytic_muscle', (302, 383, 765), 18),
                 ('salivary_gland', (1260, 1081, 1200), 15), ('lucchi_pp', (165, 768, 1024), 5)]
    specs = []
    for name, shape, spacing in specimens:
        image = RAW / 'mitoNet' / (f'{name}_im.tiff' if name != 'c_elegans' else 'img/c_elegans_em.tiff')
        labels = RAW / 'mitoNet' / (f'{name}_mito.tiff' if name != 'c_elegans' else 'label/c_elegans_mito.tiff')
        info = next(x for x in cem_metadata['EMPIAR-10982']['imagesets'] if x['directory'].endswith('/' + name))
        assert tuple([info['frames_per_image'], int(info['image_height']), int(info['image_width'])]) == shape
        assert float(info['pixel_width']) / 10 == spacing  # EMPIAR API uses angstroms.
        specs.append({'case': 'cem_' + name, 'cohort': 'cem_3d', 'specimen_cluster': name,
                      'source': 'https://www.ebi.ac.uk/empiar/EMPIAR-10982/', 'image': str(image), 'labels': str(labels),
                      'shape_zyx': list(shape), 'spacing_nm_zyx': [spacing] * 3,
                      'label_provenance': 'released 3D instances; Lucchi++ is derived from binary reannotation' if name == 'lucchi_pp' else 'released 3D instance benchmark',
                      'prior_use': 'local files predate current study; c_elegans has 2025 evaluation outputs; not claimed never-seen',
                      'overlap_review': 'different named source from the eight checkpoint training domains; not a voxelwise source-wide exclusion proof'})
    tree = fetch_json(f'https://api.github.com/repos/MancaZerovnikMekuc/UroCell/git/trees/{URO_COMMIT}?recursive=1')
    write_json(root / 'sources/urocell_tree.json', tree)
    entries = {x['path']: x for x in tree['tree']}
    names = sorted(Path(p).name for p in entries if p.startswith('mito/instance/') and p.endswith('.nii.gz'))
    assert len(names) == 5
    raw_root = RAW / f'UroCell_{URO_COMMIT[:12]}'
    for name in names:
        relative = {'image': f'data/{name}', 'labels': f'mito/instance/{name}',
                    'branched': f'mito/branched/{name}', 'contacting': f'mito/contacting/{name}'}
        paths = {}
        for key, path in relative.items():
            paths[key] = str(download_git_blob(path, entries[path], raw_root) if download_urocell else raw_root / path)
        specs.append({'case': 'uro_' + name.removesuffix('.nii.gz'), 'cohort': 'urocell', 'specimen_cluster': 'urocell_fib1',
                      'source': URO_REPO, 'source_revision': URO_COMMIT, 'license': 'CC-BY-NC-SA-4.0',
                      **paths, 'shape_zyx': [256] * 3, 'spacing_nm_zyx': [15, 16, 16],
                      'spacing_note': 'Approximate released spacing from author README; no reorientation/resampling; raw XY 5.49 nm binned 3x.',
                      'label_provenance': 'manual 3D instances and expert-reviewed object attributes',
                      'prior_use': 'newly acquired for this frozen external extension',
                      'overlap_review': 'mouse bladder source absent from eight checkpoint training domains; five crops are one source cluster'})
    return specs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--download-urocell', action='store_true')
    parser.add_argument('--case', action='append', help='Stage only these cases; keep full declared cohort')
    args = parser.parse_args()
    manifest_path = args.root / 'manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    else:
        # Declare the full cohort and settings before any model evaluation.
        specs = build_specs(args.root, args.download_urocell)
        paths = {'checkpoint': CHECKPOINT, 'edge_scorer': SCORER, 'operating_point': OPERATING_POINT,
                 'training_hparams': CHECKPOINT.parent.parent / 'hparams.yaml',
                 'mitonet_config': REPO / 'configs/mitonet_v1_zero_shot.yaml',
                 'mitonet_checkpoint': PROJECT / 'code/projects/mitoem2/checkpoints/MitoNet_v1.pth',
                 'mitonet_model_config': PROJECT / 'code/projects/mitoem2/checkpoints/MitoNet_v1_local.yaml'}
        manifest = {'created_utc': datetime.now(timezone.utc).isoformat(), 'root': str(args.root.resolve()),
                    'role': 'frozen-parameter external validation; not a never-seen claim for historical CEM files',
                    'protocol': {'resampling': 'none', 'normalization': 'lossless uint8 / 255',
                                 'graph_threshold': 0.40, 'minimum_evidence': 8, 'minimum_high_fraction': 0.50,
                                 'probability_threshold': 0.70, 'maximum_uncertainty': 1.0, 'region_mws_beta': 0.50,
                                 'sdt_seed_threshold': 0.50, 'sdt_foreground_threshold': 0.0, 'sdt_min_size': 100,
                                 'edt_downsample_factor': 2, 'iou_threshold': 0.50,
                                 'attribute_overlap_fraction': 0.10},
                    'frozen_files': {k: {'path': str(v), 'sha256': sha256(v)} for k, v in paths.items()},
                    'cases': specs,
                    'exclusions': ['MitoEM-H shares source with ME2-Pyra', 'Kidney duplicates ME2-Podo',
                                   'Lucchi++ counted once within CEM six, not an extra cohort'],
                    'mitoem_r': {'status': 'access and public validation-label audit pending; not included in this manifest'}}
        write_json(manifest_path, manifest)
    unknown = set(args.case or []) - {x['case'] for x in manifest['cases']}
    if unknown:
        raise ValueError(f'Unknown cases: {unknown}')
    for spec in manifest['cases']:
        if not args.case or spec['case'] in args.case:
            stage_case(spec, args.root)
    complete = all((args.root / 'data' / x['case'] / 'audit.json').exists() for x in manifest['cases'])
    print(json.dumps({'all_cases_staged': complete, 'declared_cases': len(manifest['cases'])}), flush=True)
    if complete:
        write_json(args.root / 'PREPARATION_SUCCESS.json', {'manifest_sha256': sha256(manifest_path), 'cases': len(manifest['cases'])})


if __name__ == '__main__':
    main()
