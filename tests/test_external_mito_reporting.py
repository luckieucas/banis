import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.prepare_external_mito_benchmarks import sha256, write_json
from scripts.prepare_mitoem_r_validation import validation_members
from scripts.summarize_external_mito_benchmarks import collect_case, main


class TestExternalMitoReporting(unittest.TestCase):
    def test_mitoem_r_members_only_public_validation_in_z_order(self):
        class Archive:
            def __init__(self, names):
                self.names = names

            def namelist(self):
                return self.names

        images = Archive([f'im/im{z:04d}.png' for z in range(1000)])
        labels = Archive([f'mito-val-v2/seg{z:04d}.tif' for z in range(400, 500)] + ['mito-train-v2/seg0000.tif'])
        members = validation_members(images, labels)
        self.assertEqual(len(members), 100)
        self.assertEqual(members[0], ('im/im0400.png', 'mito-val-v2/seg0400.tif'))
        self.assertEqual(members[-1], ('im/im0499.png', 'mito-val-v2/seg0499.tif'))
        labels.names.pop(0)
        with self.assertRaises(ValueError):
            validation_members(images, labels)

    def test_missing_results_are_explicit_not_complete(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write_json(root / 'manifest.json', {'cases': [{'case': 'a', 'cohort': 'test', 'specimen_cluster': 'one'},
                                                          {'case': 'b', 'cohort': 'test', 'specimen_cluster': 'one'}]})
            with patch('sys.argv', ['summary', '--root', str(root)]), patch('builtins.print'):
                main()
            report = json.loads((root / 'summary/summary.json').read_text())
            self.assertFalse(report['complete'])
            self.assertEqual(len(report['cohorts']), 5)
            for row in report['cohorts']:
                self.assertEqual(row['missing_cases'], ['a', 'b'])
                self.assertIsNone(row['macro_f1'])

    def test_zero_f1_is_retained_and_stale_analysis_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            spec = {'case': 'a', 'cohort': 'test', 'specimen_cluster': 'one'}
            write_json(root / 'manifest.json', {'cases': [spec]})
            out = root / 'by_case/a/mitonet/evaluation'
            out.mkdir(parents=True)
            (out / 'SUCCESS').write_text('ok\n')
            fields = ['f1', 'panoptic_quality', 'precision', 'recall', 'tp', 'fp', 'fn', 'n_true', 'n_pred']
            with (out / 'metrics.csv').open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerow(dict(zip(fields, [0, 0, 0, 0, 0, 0, 10, 10, 0])))
            rows = collect_case(root, spec)
            self.assertEqual(rows[0]['f1'], 0)
            self.assertEqual(rows[0]['fn'], 10)
            write_json(root / 'by_case/a/ANALYSIS_SUCCESS.json', {'manifest_sha256': 'wrong'})
            with self.assertRaises(ValueError):
                collect_case(root, spec)


if __name__ == '__main__':
    unittest.main()
