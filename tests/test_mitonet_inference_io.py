from pathlib import Path
import tempfile
import unittest

import numpy as np
import SimpleITK as sitk
import torch
import zarr

from scripts.run_mitonet_inference import (
    configure_determinism,
    deterministic_median,
    load_volume,
    replace_cuda_median,
)


class TestMitoNetInferenceIO(unittest.TestCase):
    def test_configure_determinism(self) -> None:
        configure_determinism(17)
        first = torch.rand(4)
        configure_determinism(17)
        second = torch.rand(4)
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        self.assertTrue(torch.are_deterministic_algorithms_enabled())
        self.assertTrue(torch.backends.cudnn.deterministic)
        self.assertFalse(torch.backends.cudnn.benchmark)

    def test_deterministic_median_matches_torch_median_values(self) -> None:
        class Queue:
            median_queue = [
                {"sem": torch.tensor([[[[3.0, 1.0]]]])},
                {"sem": torch.tensor([[[[2.0, 5.0]]]])},
                {"sem": torch.tensor([[[[2.0, 4.0]]]])},
                {"sem": torch.tensor([[[[9.0, 2.0]]]])},
                {"sem": torch.tensor([[[[0.0, 3.0]]]])},
            ]
            mid_idx = 2

        queue = Queue()
        stacked = torch.cat([item["sem"] for item in queue.median_queue], dim=0)
        expected = torch.median(stacked, dim=0, keepdim=True).values
        actual = deterministic_median(queue, "sem")
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

        class Engine3d:
            engine = queue

        class Wrapper:
            engine = Engine3d()

        replace_cuda_median(Wrapper())
        patched = queue.get_median("sem")
        torch.testing.assert_close(patched, expected, rtol=0, atol=0)

    def test_load_nifti_preserves_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            array = np.arange(3 * 4 * 5, dtype=np.uint16).reshape(3, 4, 5)
            image = sitk.GetImageFromArray(array)
            image.SetSpacing((2.0, 3.0, 4.0))
            path = Path(directory) / "input.nii.gz"
            sitk.WriteImage(image, str(path))

            loaded, metadata = load_volume(path)

            np.testing.assert_array_equal(loaded, array)
            self.assertIsNotNone(metadata)
            self.assertEqual(metadata.GetSpacing(), (2.0, 3.0, 4.0))

    def test_load_zarr_removes_only_trailing_singleton_channel(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            array = np.arange(3 * 4 * 5, dtype=np.uint16).reshape(3, 4, 5)
            path = Path(directory) / "input.zarr"
            zarr.save(str(path), array[..., None])

            loaded, metadata = load_volume(path)

            np.testing.assert_array_equal(loaded, array)
            self.assertEqual(loaded.dtype, np.uint8)
            self.assertIsNone(metadata)

    def test_load_zarr_keeps_genuine_uint16_dynamic_range(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            array = np.array([0, 256, 4095], dtype=np.uint16).reshape(1, 1, 3)
            path = Path(directory) / "input.zarr"
            zarr.save(str(path), array)

            loaded, _ = load_volume(path)

            np.testing.assert_array_equal(loaded, array)
            self.assertEqual(loaded.dtype, np.uint16)


if __name__ == "__main__":
    unittest.main()
