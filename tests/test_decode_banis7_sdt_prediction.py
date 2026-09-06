from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.decode_banis7_sdt_prediction import decode_sdt, load_sdt_prediction


class TestDecodeBANIS7SDTPrediction(unittest.TestCase):
    def test_loads_named_sdt_channel(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prediction.npz"
            expected = np.linspace(-1, 1, 4 * 5 * 6, dtype=np.float32).reshape(4, 5, 6)
            np.savez_compressed(
                path,
                prediction=expected[None].astype(np.float16),
                channel_names=np.asarray(["sdt"]),
            )
            actual, names = load_sdt_prediction(path, 0)
            np.testing.assert_array_equal(actual, expected.astype(np.float16).astype(np.float32))
            self.assertEqual(names, ["sdt"])

    def test_decode_respects_mask(self) -> None:
        sdt = np.full((12, 12, 12), -0.5, dtype=np.float32)
        sdt[2:10, 2:10, 2:10] = 0.75
        valid = np.ones_like(sdt, dtype=bool)
        valid[:, :, 7:] = False
        foreground, initial, final = decode_sdt(
            sdt,
            valid,
            seed_threshold=0.5,
            foreground_threshold=0.0,
            min_size=1,
            edt_downsample_factor=1,
            edt_parallel=1,
        )
        self.assertEqual(foreground.shape, sdt.shape)
        self.assertEqual(initial.shape, sdt.shape)
        self.assertEqual(final.shape, sdt.shape)
        self.assertFalse(foreground[~valid].any())
        self.assertFalse(initial[~valid].any())
        self.assertFalse(final[~valid].any())
        self.assertGreater(int(final.max()), 0)


if __name__ == "__main__":
    unittest.main()
