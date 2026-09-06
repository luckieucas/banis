from pathlib import Path
import tempfile
import types
import unittest

import numpy as np
import torch
import zarr

from src.inference.inference import patched_inference_batch_to_zarr


class PointwiseSevenHead(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.hparams = types.SimpleNamespace(sdt=True)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return torch.cat([image + self.anchor for _ in range(7)], dim=1)


class TestStreamingInference(unittest.TestCase):
    def test_global_batch_membership_is_invariant_to_output_blocks(self) -> None:
        class BatchSensitiveHead(PointwiseSevenHead):
            def forward(self, image):
                # Deliberately expose regrouping, including partial final batches.
                offset = image[:, :, 0, 0, 0].mean() * .1
                return super().forward(image) + offset

        image = np.linspace(0, 1, 9 * 11 * 13, dtype=np.float32).reshape(9, 11, 13, 1)
        model = BatchSensitiveHead().eval()
        with tempfile.TemporaryDirectory() as directory:
            outputs = []
            for index, block in enumerate(((30, 30, 30), (3, 4, 5))):
                paths = patched_inference_batch_to_zarr(
                    image, model, Path(directory) / str(index), output_channel_indices=(0, 6),
                    small_size=6, prediction_channels=7, divide=1, batch_size=3,
                    output_block_shape=block, output_chunks=(2, 3, 4), padding_mode='edge')
                outputs.append([np.asarray(zarr.open_array(str(path), mode='r')) for path in paths])
            for full, blocked in zip(*outputs):
                np.testing.assert_array_equal(full, blocked)

    def test_streaming_overlap_add_covers_padding_and_blocks(self) -> None:
        image = np.linspace(0, 1, 5 * 9 * 10, dtype=np.float32).reshape(5, 9, 10, 1)
        model = PointwiseSevenHead().eval()
        with tempfile.TemporaryDirectory() as directory:
            paths = patched_inference_batch_to_zarr(
                image,
                model,
                Path(directory),
                output_channel_indices=(0, 6),
                small_size=6,
                prediction_channels=7,
                divide=1,
                batch_size=2,
                output_block_shape=(3, 4, 5),
                output_chunks=(2, 3, 4),
                padding_mode="edge",
                padding_position="center",
            )
            affinity = np.asarray(zarr.open_array(str(paths[0]), mode="r"))
            sdt = np.asarray(zarr.open_array(str(paths[1]), mode="r"))

        expected_affinity = torch.sigmoid(0.2 * torch.from_numpy(image[..., 0])).numpy()
        expected_sdt = torch.tanh(torch.from_numpy(image[..., 0])).numpy()
        np.testing.assert_array_equal(affinity, expected_affinity.astype(np.float16))
        np.testing.assert_array_equal(sdt, expected_sdt.astype(np.float16))
        self.assertEqual(affinity.shape, image.shape[:3])
        self.assertTrue(np.isfinite(affinity).all())
        self.assertTrue(np.isfinite(sdt).all())


if __name__ == "__main__":
    unittest.main()
