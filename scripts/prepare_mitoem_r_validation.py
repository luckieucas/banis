#!/usr/bin/env python3
"""Fetch only native-resolution public MitoEM-R v2 validation slices via ZIP ranges."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

import fsspec
import numpy as np
from PIL import Image
import tifffile
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.prepare_external_mito_benchmarks import DEFAULT_ROOT, canonical_image, sha256, write_json

IMAGE_URL = 'https://huggingface.co/datasets/pytc/EM30/resolve/63519706548316c1f00863cbea0ece40358a759a/EM30-R-im.zip'
LABEL_URL = 'https://huggingface.co/datasets/pytc/MitoEM/resolve/2105967e61abf02ae1873c514b8ea16390606384/EM30-R-mito-train-val-v2.zip'


def validation_members(image_archive, label_archive):
    images = [f'im/im{z:04d}.png' for z in range(400, 500)]
    labels = [f'mito-val-v2/seg{z:04d}.tif' for z in range(400, 500)]
    for archive, names in ((image_archive, images), (label_archive, labels)):
        for name in names:
            if name not in archive.namelist():
                raise ValueError(f'Missing declared validation slice: {name}')
    return list(zip(images, labels))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT.with_name(DEFAULT_ROOT.name + '_mitoem_r'))
    args = parser.parse_args()
    manifest_path = args.root / 'manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    else:
        base = json.loads((args.base_root / 'manifest.json').read_text())
        spec = {'case': 'mitoem_r_val_v2_z400_499', 'cohort': 'mitoem_r', 'specimen_cluster': 'mitoem_r',
                'source': 'https://mitoem.grand-challenge.org/MitoEM/',
                'image': IMAGE_URL, 'labels': LABEL_URL, 'shape_zyx': [100, 4096, 4096],
                'spacing_nm_zyx': [30, 8, 8], 'voxel_offset_zyx': [400, 0, 0],
                'label_provenance': 'official v2 public validation, native TIFF labels, not downsampled browser labels',
                'prior_use': 'new frozen external evaluation; public benchmark validation is not official hidden test',
                'overlap_review': 'rat source distinct from human ME2-Pyra; eight training domains exclude MitoEM-R'}
        manifest = {'created_utc': datetime.now(timezone.utc).isoformat(), 'root': str(args.root.resolve()),
                    'role': 'external zero-shot on the original benchmark PUBLIC VALIDATION split, z400:500',
                    'protocol': base['protocol'], 'frozen_files': base['frozen_files'], 'cases': [spec],
                    'exclusions': ['no MitoEM-H', 'no rat train slices', 'no official hidden-test claim'],
                    'download': 'HTTP range ZIP entries; CRC32 checked by zipfile; original PNG/TIFF bytes hashed per slice'}
        write_json(manifest_path, manifest)
    spec = manifest['cases'][0]
    out = args.root / 'data' / spec['case']
    marker = out / 'audit.json'
    if marker.exists():
        print('MitoEM-R public validation already prepared', flush=True)
        return
    out.mkdir(parents=True, exist_ok=True)
    store = zarr.open_group(str(out / 'data.zarr'), mode='a', zarr_format=2)
    store.attrs.update({'axes': 'zyx', 'voxel_spacing_nm_zyx': [30, 8, 8],
                        'voxel_offset_zyx': [400, 0, 0], 'resampling': 'none', 'split': 'public_validation_v2'})
    for name, dtype in [('img', 'uint8'), ('seg', 'uint32')]:
        if name not in store:
            store.create_array(name, shape=(100, 4096, 4096), dtype=dtype, chunks=(1, 512, 512))
        if tuple(store[name].shape) != (100, 4096, 4096) or str(store[name].dtype) != dtype:
            raise ValueError('Incompatible existing staging array')
    records = []
    # No full 14-GB image archive is downloaded. Original, not JPEG browser, pixels.
    with fsspec.open(IMAGE_URL, block_size=1024 * 1024).open() as image_handle, \
         fsspec.open(LABEL_URL, block_size=1024 * 1024).open() as label_handle, \
         zipfile.ZipFile(image_handle) as images, zipfile.ZipFile(label_handle) as labels:
        for index, (im_name, gt_name) in enumerate(validation_members(images, labels)):
            slice_record = out / 'slices' / f'{index + 400:04d}.json'
            if slice_record.exists():
                record = json.loads(slice_record.read_text())
                for key in ('img', 'seg'):
                    current = np.ascontiguousarray(store[key][index])
                    if hashlib.sha256(current.tobytes()).hexdigest() != record[key + '_voxel_sha256']:
                        raise ValueError(f'Previously prepared slice changed: {slice_record}, {key}')
                records.append(record)
                continue
            raw_image, raw_labels = images.read(im_name), labels.read(gt_name)
            image = canonical_image(np.asarray(Image.open(io.BytesIO(raw_image)))[None])[0]
            gt = tifffile.imread(io.BytesIO(raw_labels))
            if image.shape != (4096, 4096) or gt.shape != image.shape:
                raise ValueError('Native image/GT slice shape mismatch')
            if not np.issubdtype(gt.dtype, np.integer) or gt.min() < 0 or int(gt.max()) > np.iinfo(np.uint32).max:
                raise ValueError('Invalid instance label IDs')
            gt = gt.astype(np.uint32)
            for key, array in [('img', image), ('seg', gt)]:
                store[key][index] = array
                np.testing.assert_array_equal(store[key][index], array)
            ids, counts = np.unique(gt, return_counts=True)
            record = {'z': index + 400, 'image_member': im_name, 'label_member': gt_name,
                      'image_member_sha256': hashlib.sha256(raw_image).hexdigest(),
                      'label_member_sha256': hashlib.sha256(raw_labels).hexdigest(),
                      'img_voxel_sha256': hashlib.sha256(image.tobytes()).hexdigest(),
                      'seg_voxel_sha256': hashlib.sha256(gt.tobytes()).hexdigest(),
                      'label_counts': {str(int(i)): int(n) for i, n in zip(ids, counts)},
                      'image_range': [int(image.min()), int(image.max())]}
            write_json(slice_record, record)
            records.append(record)
            print(f'MitoEM-R validation: {index + 1}/100 native slices verified', flush=True)
    source_record = out / 'source_manifest.json'
    write_json(source_record, {'image_url': IMAGE_URL, 'label_url': LABEL_URL, 'slices': records})
    counts = Counter()
    for record in records:
        counts.update({int(k): v for k, v in record['label_counts'].items()})
    audit = {'spec': spec, 'raw_files': {'source_manifest': {'path': str(source_record), 'sha256': sha256(source_record)}},
             'shape_zyx': spec['shape_zyx'], 'original_dtypes': ['uint8', 'uint16'],
             'n_instances': len(set(counts) - {0}), 'max_instance_id': max(counts),
             'roundtrip_voxel_exact': True, 'prepared_zarr': str(out / 'data.zarr'), 'attributes': None,
             'image_range': [min(x['image_range'][0] for x in records), max(x['image_range'][1] for x in records)]}
    write_json(marker, audit)
    write_json(args.root / 'PREPARATION_SUCCESS.json', {'cases': 1, 'manifest_sha256': sha256(manifest_path)})
    print(json.dumps({'shape_zyx': audit['shape_zyx'], 'n_instances': audit['n_instances']}), flush=True)


if __name__ == '__main__':
    main()
