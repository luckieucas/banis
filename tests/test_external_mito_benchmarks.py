import hashlib
from pathlib import Path
import tempfile
import unittest

import numpy as np
import SimpleITK as sitk
import tifffile
import zarr

from scripts.prepare_external_mito_benchmarks import (
    attribute_table, canonical_image, canonical_labels, git_blob_hash, load_zyx, stage_case,
)
from scripts.evaluate_urocell_attributes import object_diagnostics, summarize_groups
from src.inference.sdt_affinity_graph import build_fragment_contingency
from scripts.run_external_mito_benchmark import locked_learned_row


class TestExternalMitoPreparation(unittest.TestCase):
    def test_learned_row_is_selected_by_method_not_position(self):
        baseline = {'method': 'baseline', 'probability_threshold': '', 'maximum_uncertainty': '', 'f1': '.1'}
        learned = {'method': 'learned_edge_scorer', 'probability_threshold': '.7', 'maximum_uncertainty': '1.0', 'f1': '.2'}
        self.assertEqual(locked_learned_row([baseline, learned]), learned)
        self.assertEqual(locked_learned_row([learned, baseline]), learned)
        for rows in ([baseline], [learned, learned]):
            with self.assertRaises(ValueError):
                locked_learned_row(rows)

    def test_lossless_image_only(self):
        image = np.arange(24, dtype=np.int16).reshape(2, 3, 4)
        np.testing.assert_array_equal(canonical_image(image), image)
        for invalid in (image.astype(float) + 0.1, image + 256, image - 1, image * np.nan):
            with self.assertRaises(ValueError):
                canonical_image(invalid)

    def test_instance_ids_not_connected_components_or_uint16(self):
        image = np.zeros((2, 3, 4), dtype=np.int64)
        image[0, 0, 0], image[1, 2, 3] = 70000, 90000
        np.testing.assert_array_equal(canonical_labels(image), image)
        self.assertEqual(canonical_labels(image).dtype, np.uint32)
        for invalid in ((image > 0).astype(np.uint8), image - 1, image.astype(float) + 0.5):
            with self.assertRaises(ValueError):
                canonical_labels(invalid)

    def test_attribute_conventions_and_errors(self):
        labels = np.array([[[0, 1, 1, 2, 2]]], dtype=np.uint32)
        branched = np.array([[[0, 1, 1, 2, 2]]])
        contacting = np.array([[[0, 1, 1, 2, 2]]])
        rows = attribute_table(labels, branched, contacting)
        self.assertTrue(rows[0]['branched'])
        self.assertFalse(rows[0]['contacting'])
        self.assertFalse(rows[1]['branched'])
        self.assertTrue(rows[1]['contacting'])
        for invalid in (np.zeros_like(branched), np.array([[[0, 1, 2, 2, 2]]])):
            with self.assertRaises(ValueError):
                attribute_table(labels, invalid, contacting)

    def test_asymmetric_tiff_and_nifti_axis_roundtrip(self):
        array = np.arange(24, dtype=np.uint8).reshape(2, 3, 4)
        with tempfile.TemporaryDirectory() as folder:
            tiff = Path(folder) / 'im.tiff'
            nifti = Path(folder) / 'im.nii.gz'
            tifffile.imwrite(tiff, array, photometric='minisblack')
            sitk.WriteImage(sitk.GetImageFromArray(array), str(nifti))
            np.testing.assert_array_equal(load_zyx(tiff), array)
            np.testing.assert_array_equal(load_zyx(nifti), array)

    def test_incomplete_attributes_remain_unknown_without_changing_gt(self):
        labels = np.array([[[0, 1, 1, 2, 2]]])
        attribute = np.array([[[1, 0, 1, 2, 2]]])
        original = labels.copy()
        rows = attribute_table(labels, attribute, attribute, allow_incomplete=True)
        self.assertIsNone(rows[0]['branched'])
        self.assertFalse(rows[1]['branched'])
        self.assertTrue(rows[1]['contacting'])
        np.testing.assert_array_equal(labels, original)

    def test_git_blob_checksum(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'file'
            path.write_bytes(b'abc')
            self.assertEqual(git_blob_hash(path), hashlib.sha1(b'blob 3\0abc').hexdigest())

    def test_staging_voxel_exact_and_refuses_changed_raw(self):
        image = np.arange(24, dtype=np.uint8).reshape(2, 3, 4)
        labels = np.zeros_like(image, dtype=np.uint32)
        labels[0, 0, 0], labels[1, 2, 3] = 70000, 90000
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            raw_image, raw_labels = root / 'im.tiff', root / 'seg.tiff'
            tifffile.imwrite(raw_image, image, photometric='minisblack')
            tifffile.imwrite(raw_labels, labels, photometric='minisblack')
            spec = {'case': 'test', 'image': str(raw_image), 'labels': str(raw_labels),
                    'source': 'synthetic', 'shape_zyx': [2, 3, 4], 'spacing_nm_zyx': [30, 8, 8]}
            audit = stage_case(spec, root)
            store = zarr.open_group(audit['prepared_zarr'], mode='r')
            np.testing.assert_array_equal(store['img'][:], image)
            np.testing.assert_array_equal(store['seg'][:], labels)
            self.assertEqual(stage_case(spec, root), audit)
            tifffile.imwrite(raw_image, image + 1, photometric='minisblack')
            with self.assertRaises(ValueError):
                stage_case(spec, root)


class TestUroCellDiagnostics(unittest.TestCase):
    def test_split_repaired_and_cross_stratum_merge_detected(self):
        gt = np.array([[[1, 1, 1, 1, 2, 2, 2, 2]]], dtype=np.uint32)
        fragments = np.array([[[1, 1, 2, 2, 3, 3, 3, 3]]], dtype=np.uint32)
        attributes = [{'gt_id': 1, 'branched': True, 'contacting': False},
                      {'gt_id': 2, 'branched': False, 'contacting': True}]
        contingency = build_fragment_contingency(gt, fragments, block_shape=(1, 1, 3))
        baseline = object_diagnostics(contingency, attributes, {1, 2})
        self.assertTrue(baseline[0]['split'])
        self.assertFalse(baseline[1]['split'])
        repaired = object_diagnostics(contingency, attributes, {1, 2}, np.array([0, 1, 1, 2]))
        self.assertFalse(any(row['split'] or row['merge_involved'] for row in repaired))
        overmerged = object_diagnostics(contingency, attributes, {1}, np.array([0, 1, 1, 1]))
        self.assertTrue(all(row['merge_involved'] for row in overmerged))
        self.assertFalse(overmerged[1]['matched_iou50'])

    def test_missing_predictions_are_not_called_splits(self):
        gt = np.array([[[1, 1, 2, 2]]], dtype=np.uint32)
        contingency = build_fragment_contingency(gt, np.zeros_like(gt))
        attributes = [{'gt_id': i, 'branched': False, 'contacting': False} for i in (1, 2)]
        rows = object_diagnostics(contingency, attributes, set())
        self.assertFalse(any(row['split'] or row['merge_involved'] or row['matched_iou50'] for row in rows))
        groups = {row['group']: row for row in summarize_groups(rows)}
        self.assertEqual(groups['all']['recall_iou50'], 0)
        self.assertIsNone(groups['branched=True']['recall_iou50'])

    def test_fraction_boundary_and_incomplete_attributes(self):
        gt = np.ones((1, 1, 10), dtype=np.uint32)
        prediction = np.ones_like(gt)
        prediction[0, 0, 0] = 2
        contingency = build_fragment_contingency(gt, prediction)
        attributes = [{'gt_id': 1, 'branched': True, 'contacting': True}]
        self.assertTrue(object_diagnostics(contingency, attributes, {1})[0]['split'])
        with self.assertRaises(ValueError):
            object_diagnostics(contingency, [], {1})


if __name__ == '__main__':
    unittest.main()
