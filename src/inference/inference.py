import gc
from pathlib import Path
from typing import Union, List, Tuple

import numba
import numpy as np
import torch
from torch.nn.functional import tanh
import torch.utils
import zarr
from numba import jit
from scipy.ndimage import distance_transform_cdt
from torch import autocast
from torch.nn.functional import sigmoid
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader


def _pad_image_for_inference(
        img: np.ndarray,
        small_size: int,
        padding_mode: str,
        padding_position: str,
        padding_constant: float = 0,
) -> Tuple[np.ndarray, Tuple[slice, slice, slice], bool]:
    """Pad spatial dimensions smaller than the patch size and return crop slices."""
    original_shape = img.shape[:3]
    needs_padding = any(dim < small_size for dim in original_shape)
    crop_slices = tuple(slice(0, dim) for dim in original_shape)
    if not needs_padding:
        return img, crop_slices, False

    if padding_position not in {"end", "center"}:
        raise ValueError(f"Unsupported padding_position={padding_position!r}; expected 'end' or 'center'.")

    print(
        f"Image dimensions {original_shape} smaller than patch size {small_size}, "
        f"padding to minimum size with mode={padding_mode}, position={padding_position}"
    )
    pad_width = []
    crop = []
    for dim in original_shape:
        if dim < small_size:
            total = small_size - dim
            if padding_position == "center":
                before = total // 2
                after = total - before
            else:
                before = 0
                after = total
            pad_width.append((before, after))
            crop.append(slice(before, before + dim))
        else:
            pad_width.append((0, 0))
            crop.append(slice(0, dim))
    pad_width.append((0, 0))

    if padding_mode == "constant":
        img = np.pad(img, pad_width, mode=padding_mode, constant_values=padding_constant)
    else:
        img = np.pad(img, pad_width, mode=padding_mode)
    print(f"Padded image shape: {img.shape}")
    return img, tuple(crop), True


def scale_sigmoid(x: torch.Tensor) -> torch.Tensor:
    """Scale sigmoid to avoid numerical issues in high confidence fp16."""
    return sigmoid(0.2 * x)


@jit(nopython=True)
def compute_connected_component_segmentation(hard_aff: np.ndarray) -> np.ndarray:
    """
    Compute connected components from affinities.

    Args:
        hard_aff: The (thresholded, boolean) short range affinities. Shape: (3, x, y, z).

    Returns:
        The segmentation. Shape: (x, y, z).
    """
    visited = np.zeros(tuple(hard_aff.shape[1:]), dtype=np.uint8)
    seg = np.zeros(tuple(hard_aff.shape[1:]), dtype=np.uint32)
    cur_id = 1
    for i in range(visited.shape[0]):
        for j in range(visited.shape[1]):
            for k in range(visited.shape[2]):
                if hard_aff[:, i, j, k].any() and not visited[i, j, k]:  # If foreground
                    cur_to_visit = [(i, j, k)]
                    visited[i, j, k] = True
                    while cur_to_visit:
                        x, y, z = cur_to_visit.pop()
                        seg[x, y, z] = cur_id

                        # Check all neighbors
                        if x + 1 < visited.shape[0] and hard_aff[0, x, y, z] and not visited[x + 1, y, z]:
                            cur_to_visit.append((x + 1, y, z))
                            visited[x + 1, y, z] = True
                        if y + 1 < visited.shape[1] and hard_aff[1, x, y, z] and not visited[x, y + 1, z]:
                            cur_to_visit.append((x, y + 1, z))
                            visited[x, y + 1, z] = True
                        if z + 1 < visited.shape[2] and hard_aff[2, x, y, z] and not visited[x, y, z + 1]:
                            cur_to_visit.append((x, y, z + 1))
                            visited[x, y, z + 1] = True
                        if x - 1 >= 0 and hard_aff[0, x - 1, y, z] and not visited[x - 1, y, z]:
                            cur_to_visit.append((x - 1, y, z))
                            visited[x - 1, y, z] = True
                        if y - 1 >= 0 and hard_aff[1, x, y - 1, z] and not visited[x, y - 1, z]:
                            cur_to_visit.append((x, y - 1, z))
                            visited[x, y - 1, z] = True
                        if z - 1 >= 0 and hard_aff[2, x, y, z - 1] and not visited[x, y, z - 1]:
                            cur_to_visit.append((x, y, z - 1))
                            visited[x, y, z - 1] = True
                    cur_id += 1
    return seg


@torch.no_grad()
@autocast(device_type="cuda")
def patched_inference(
        img: Union[np.ndarray, zarr.Array],
        model: torch.nn.Module,
        small_size: int = 128,
        do_overlap: bool = True,
        prediction_channels: int = 6,
        divide: int = 1,
        padding_mode: str = "constant",
        padding_position: str = "end",
        padding_constant: float = 0,
) -> np.ndarray:
    """
    Perform patched inference with a model on an image.

    Args:
        img: The input image. Shape: (x, y, z, channel).
        model: The model to use for predictions.
        small_size: The size of the patches. Defaults to 128.
        do_overlap: Whether to perform overlapping predictions. Defaults to True:
            half of patch size for all 3 axes.
        prediction_channels: The number of channels in the output (additional model output
            dimensions are discarded). Defaults to 6 (3 short + 3 long range affinities).
        divide: The divisor for the image. Typically, 1 or 255 if img in [0, 255]

    Returns:
        The full prediction. Shape: (channel, x, y, z).
    """
    assert 3 <= prediction_channels <= 7
    calculate_sdt = prediction_channels == 7
    if calculate_sdt:
        assert model.hparams.sdt

    print(
        f"Performing patched inference with do_overlap={do_overlap} for img of shape {img.shape} and dtype {img.dtype}")
    img = img[:]  # load into memory (expensive!)
    
    img, crop_slices, needs_padding = _pad_image_for_inference(
        img,
        small_size=small_size,
        padding_mode=padding_mode,
        padding_position=padding_position,
        padding_constant=padding_constant,
    )

    patch_coordinates = get_coordinates(img.shape[:3], small_size, do_overlap)
    single_pred_weight = get_single_pred_weight(do_overlap, small_size)
    # to weight overlapping predictions lower close to the boundaries

    weight_sum = np.zeros((1, *img.shape[:3]), dtype=np.float32)
    weighted_pred = np.zeros((prediction_channels, *img.shape[:3]), dtype=np.float32)

    device = next(model.parameters()).device
    assert device.type != 'cpu'

    for x, y, z in tqdm(patch_coordinates):
        img_patch = torch.tensor(
            np.moveaxis(img[x: x + small_size, y: y + small_size, z: z + small_size], -1, 0)[None]).half().to(
            device) / divide
        if calculate_sdt:
            # use tanh for last channel
            pred = model(img_patch)[0]
            pred = torch.cat(
                [scale_sigmoid(pred[:prediction_channels - 1]), tanh(pred[prediction_channels - 1:])], dim=0
            )
        else:
            pred = scale_sigmoid(model(img_patch))[0, :prediction_channels]

        weight_sum[:, x: x + small_size, y: y + small_size,
        z: z + small_size] += single_pred_weight if do_overlap else 1
        weighted_pred[:, x: x + small_size, y: y + small_size, z: z + small_size] += pred.cpu().numpy() * (
            single_pred_weight[None] if do_overlap else 1)
    del img  # to save memory before division
    # assert np.all(weight_sum > 0)
    np.divide(weighted_pred, weight_sum, out=weighted_pred)

    # Crop back to original size if padding was applied
    if needs_padding:
        weighted_pred = weighted_pred[:, crop_slices[0], crop_slices[1], crop_slices[2]]
        print(f"Cropped prediction back to original shape: {weighted_pred.shape[1:]}")

    return weighted_pred


class PatchDataset(Dataset):
    """Dataset for loading image patches for parallel inference."""
    
    def __init__(self, img: np.ndarray, patch_coordinates: List[Tuple[int, int, int]], 
                 small_size: int, divide: int = 1):
        self.img = img
        self.patch_coordinates = patch_coordinates
        self.small_size = small_size
        self.divide = divide
    
    def __len__(self):
        return len(self.patch_coordinates)
    
    def __getitem__(self, idx):
        x, y, z = self.patch_coordinates[idx]
        img_patch = self.img[x: x + self.small_size, 
                             y: y + self.small_size, 
                             z: z + self.small_size]
        # Convert to (channel, x, y, z) format
        img_patch = np.moveaxis(img_patch, -1, 0)
        img_tensor = torch.from_numpy(img_patch).half() / self.divide
        return img_tensor, (x, y, z)


@torch.no_grad()
@autocast(device_type="cuda")
def patched_inference_batch(
        img: Union[np.ndarray, zarr.Array],
        model: torch.nn.Module,
        small_size: int = 128,
        do_overlap: bool = True,
        prediction_channels: int = 6,
        divide: int = 1,
        batch_size: int = 4,
        num_workers: int = 4,
        pin_memory: bool = True,
        padding_mode: str = "constant",
        padding_position: str = "end",
        padding_constant: float = 0,
) -> np.ndarray:
    """
    Perform patched inference with a model on an image using parallel batch processing.
    This version processes multiple patches in parallel batches for improved efficiency.

    Args:
        img: The input image. Shape: (x, y, z, channel).
        model: The model to use for predictions.
        small_size: The size of the patches. Defaults to 128.
        do_overlap: Whether to perform overlapping predictions. Defaults to True:
            half of patch size for all 3 axes.
        prediction_channels: The number of channels in the output (additional model output
            dimensions are discarded). Defaults to 6 (3 short + 3 long range affinities).
        divide: The divisor for the image. Typically, 1 or 255 if img in [0, 255]
        batch_size: Number of patches to process in parallel. Defaults to 4.
        num_workers: Number of worker threads for data loading. Defaults to 4.
        pin_memory: Whether to use pinned memory for faster GPU transfer. Defaults to True.

    Returns:
        The full prediction. Shape: (channel, x, y, z).
    """
    assert 3 <= prediction_channels <= 7
    calculate_sdt = prediction_channels == 7
    if calculate_sdt:
        assert model.hparams.sdt

    print(
        f"Performing parallel patched inference with do_overlap={do_overlap}, "
        f"batch_size={batch_size}, num_workers={num_workers} for img of shape {img.shape} and dtype {img.dtype}")
    img = img[:]  # load into memory (expensive!)
    
    img, crop_slices, needs_padding = _pad_image_for_inference(
        img,
        small_size=small_size,
        padding_mode=padding_mode,
        padding_position=padding_position,
        padding_constant=padding_constant,
    )

    patch_coordinates = get_coordinates(img.shape[:3], small_size, do_overlap)
    single_pred_weight = get_single_pred_weight(do_overlap, small_size)
    
    weight_sum = np.zeros((1, *img.shape[:3]), dtype=np.float32)
    weighted_pred = np.zeros((prediction_channels, *img.shape[:3]), dtype=np.float32)

    device = next(model.parameters()).device
    assert device.type != 'cpu'
    
    # Create dataset and dataloader for parallel processing
    dataset = PatchDataset(img, patch_coordinates, small_size, divide)
    dataloader_kwargs = {
        'batch_size': batch_size,
        'shuffle': False,
        'num_workers': num_workers,
        'pin_memory': pin_memory,
    }
    # persistent_workers is only available in PyTorch >= 1.7.0
    # Try to use it if available, otherwise fall back
    try:
        dataloader = DataLoader(dataset, persistent_workers=num_workers > 0, **dataloader_kwargs)
    except TypeError:
        # Fall back if persistent_workers is not supported
        dataloader = DataLoader(dataset, **dataloader_kwargs)
    
    # Process patches in batches
    for batch_imgs, batch_coords in tqdm(dataloader, desc="Processing patches"):
        batch_imgs = batch_imgs.to(device, non_blocking=pin_memory)
        
        # Forward pass
        if calculate_sdt:
            pred = model(batch_imgs)
            pred = torch.cat(
                [scale_sigmoid(pred[:, :prediction_channels - 1]), 
                 tanh(pred[:, prediction_channels - 1:])], dim=1
            )
        else:
            pred = scale_sigmoid(model(batch_imgs))[:, :prediction_channels]
        
        # Move predictions to CPU and accumulate
        pred_np = pred.cpu().numpy()
        
        # DataLoader returns coordinates as tuple of tensors: (x_batch, y_batch, z_batch)
        x_coords, y_coords, z_coords = batch_coords
        
        # Accumulate predictions for each patch in the batch
        for i in range(len(x_coords)):
            x, y, z = int(x_coords[i]), int(y_coords[i]), int(z_coords[i])
            weight_sum[:, x: x + small_size, y: y + small_size,
                      z: z + small_size] += single_pred_weight if do_overlap else 1
            weighted_pred[:, x: x + small_size, y: y + small_size, z: z + small_size] += \
                pred_np[i] * (single_pred_weight[None] if do_overlap else 1)
    
    del img  # to save memory before division
    del dataset, dataloader
    gc.collect()
    torch.cuda.empty_cache()
    
    # Normalize by weight sum
    np.divide(weighted_pred, weight_sum, out=weighted_pred)

    # Crop back to original size if padding was applied
    if needs_padding:
        weighted_pred = weighted_pred[:, crop_slices[0], crop_slices[1], crop_slices[2]]
        print(f"Cropped prediction back to original shape: {weighted_pred.shape[1:]}")

    return weighted_pred


@torch.no_grad()
def patched_inference_batch_to_zarr(
        img: np.ndarray,
        model: torch.nn.Module,
        output_dir: Union[str, Path],
        output_channel_indices: Tuple[int, ...] = (0, 1, 2, 3, 4, 5),
        small_size: int = 128,
        do_overlap: bool = True,
        prediction_channels: int = 7,
        divide: int = 1,
        batch_size: int = 2,
        output_block_shape: Tuple[int, int, int] = (256, 512, 512),
        output_chunks: Tuple[int, int, int] = (64, 256, 256),
        padding_mode: str = "constant",
        padding_position: str = "end",
        padding_constant: float = 0,
) -> List[Path]:
    """Run exact overlap-add inference one output block at a time.

    Unlike :func:`patched_inference_batch`, this function never allocates a
    channel-by-full-volume accumulator. Each disjoint output block gathers all
    model patches that overlap it, normalizes the block, and immediately writes
    float16 channel arrays to Zarr. Patches crossing output-block boundaries are
    recomputed, trading modest extra GPU work for bounded host memory and no
    temporary full-volume numerator arrays.
    """
    if not isinstance(img, np.ndarray) or img.ndim != 4:
        raise ValueError(f"Expected a channel-last 4-D NumPy image, got {type(img)} {getattr(img, 'shape', None)}")
    if not 3 <= prediction_channels <= 7:
        raise ValueError(f"prediction_channels must be between 3 and 7, got {prediction_channels}")
    if not output_channel_indices or len(set(output_channel_indices)) != len(output_channel_indices):
        raise ValueError("output_channel_indices must be non-empty and unique")
    if any(channel < 0 or channel >= prediction_channels for channel in output_channel_indices):
        raise ValueError(
            f"Output channels {output_channel_indices} are incompatible with {prediction_channels} predictions"
        )
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if len(output_block_shape) != 3 or any(int(value) <= 0 for value in output_block_shape):
        raise ValueError(f"Invalid output_block_shape: {output_block_shape}")
    if len(output_chunks) != 3 or any(int(value) <= 0 for value in output_chunks):
        raise ValueError(f"Invalid output_chunks: {output_chunks}")

    calculate_sdt = prediction_channels == 7
    if calculate_sdt and not bool(model.hparams.sdt):
        raise ValueError("Seven-channel inference requires a checkpoint trained with SDT")
    original_shape = tuple(int(value) for value in img.shape[:3])
    img, crop_slices, _ = _pad_image_for_inference(
        img,
        small_size=small_size,
        padding_mode=padding_mode,
        padding_position=padding_position,
        padding_constant=padding_constant,
    )
    padded_shape = tuple(int(value) for value in img.shape[:3])
    crop_offsets = tuple(int(item.start) for item in crop_slices)
    # Preserve the legacy dense implementation's coordinate order so each
    # voxel receives floating-point additions in the same sequence.
    patch_coordinates = get_coordinates(padded_shape, small_size, do_overlap)
    single_pred_weight = get_single_pred_weight(do_overlap, small_size)
    if single_pred_weight is None:
        single_pred_weight = np.ones((small_size,) * 3, dtype=np.float32)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    chunks = tuple(min(int(chunk), dim) for chunk, dim in zip(output_chunks, original_shape))
    output_paths = [output_dir / f"channel_{channel}.zarr" for channel in output_channel_indices]
    output_arrays = [
        zarr.open_array(
            str(path),
            mode="w",
            shape=original_shape,
            chunks=chunks,
            dtype=np.float16,
        )
        for path in output_paths
    ]

    device = next(model.parameters()).device
    tensor_dtype = torch.float16 if device.type == "cuda" else torch.float32
    block_starts = [range(0, dim, int(block)) for dim, block in zip(original_shape, output_block_shape)]
    blocks = [
        (
            slice(z, min(z + int(output_block_shape[0]), original_shape[0])),
            slice(y, min(y + int(output_block_shape[1]), original_shape[1])),
            slice(x, min(x + int(output_block_shape[2]), original_shape[2])),
        )
        for z in block_starts[0]
        for y in block_starts[1]
        for x in block_starts[2]
    ]
    print(
        f"Streaming overlap-add: image={original_shape}, padded={padded_shape}, "
        f"patches={len(patch_coordinates)}, output_blocks={len(blocks)}, "
        f"block_shape={tuple(output_block_shape)}, channels={output_channel_indices}",
        flush=True,
    )

    amp_context = (
        lambda: torch.autocast(device_type="cuda", dtype=torch.float16)
        if device.type == "cuda"
        else torch.autocast(device_type="cpu", enabled=False)
    )
    for block in tqdm(blocks, desc="Streaming output blocks"):
        core_start = tuple(int(item.start) + offset for item, offset in zip(block, crop_offsets))
        core_stop = tuple(int(item.stop) + offset for item, offset in zip(block, crop_offsets))
        core_shape = tuple(stop - start for start, stop in zip(core_start, core_stop))
        weighted = np.zeros((len(output_channel_indices), *core_shape), dtype=np.float32)
        weight_sum = np.zeros(core_shape, dtype=np.float32)
        # Preserve the dense predictor's GLOBAL batch membership, including the
        # final short batch. Rebatching only the patches touching this block can
        # change CUDA convolution rounding and hence threshold decisions.
        relevant_batches = []
        for batch_start in range(0, len(patch_coordinates), batch_size):
            coordinates = patch_coordinates[batch_start:batch_start + batch_size]
            if any(all(patch_start < stop and patch_start + small_size > start
                       for patch_start, start, stop in zip(coordinate, core_start, core_stop))
                   for coordinate in coordinates):
                relevant_batches.append(coordinates)
        if not relevant_batches:
            raise RuntimeError(f"No inference patches overlap output block {block}")

        for coordinates in relevant_batches:
            patch_array = np.stack([
                np.moveaxis(
                    img[
                        z:z + small_size,
                        y:y + small_size,
                        x:x + small_size,
                    ],
                    -1,
                    0,
                )
                for z, y, x in coordinates
            ])
            # Match PatchDataset: normalize in CPU tensor_dtype before transfer.
            batch_tensor = (torch.from_numpy(patch_array).to(dtype=tensor_dtype) / divide).to(
                device=device, non_blocking=False)
            with amp_context():
                raw = model(batch_tensor)
                if calculate_sdt:
                    prediction = torch.cat(
                        [scale_sigmoid(raw[:, :prediction_channels - 1]),
                         tanh(raw[:, prediction_channels - 1:prediction_channels])],
                        dim=1,
                    )
                else:
                    prediction = scale_sigmoid(raw)[:, :prediction_channels]
            prediction = prediction[:, output_channel_indices].float().cpu().numpy()

            for item, coordinate in enumerate(coordinates):
                patch_stop = tuple(value + small_size for value in coordinate)
                overlap_start = tuple(max(left, right) for left, right in zip(coordinate, core_start))
                overlap_stop = tuple(min(left, right) for left, right in zip(patch_stop, core_stop))
                if any(stop <= start for start, stop in zip(overlap_start, overlap_stop)):
                    continue  # Batch partner outside this output block.
                source = tuple(
                    slice(start - patch_start, stop - patch_start)
                    for start, stop, patch_start in zip(overlap_start, overlap_stop, coordinate)
                )
                destination = tuple(
                    slice(start - core_axis_start, stop - core_axis_start)
                    for start, stop, core_axis_start in zip(overlap_start, overlap_stop, core_start)
                )
                local_weight = single_pred_weight[source]
                weighted[(slice(None), *destination)] += prediction[(item, slice(None), *source)] * local_weight
                weight_sum[destination] += local_weight

        if not np.all(weight_sum > 0):
            missing = int(np.count_nonzero(weight_sum <= 0))
            raise RuntimeError(f"Output block {block} has {missing} voxels without prediction weight")
        weighted /= weight_sum[None]
        for channel_array, values in zip(output_arrays, weighted):
            channel_array[block] = values.astype(np.float16)

    return output_paths


def get_coordinates(
        shape: Tuple[int, int, int], small_size: int, do_overlap: bool
) -> List[Tuple[int, int, int]]:
    """
    Get coordinates for cubes to be predicted.

    Args:
        shape: The shape of the input image (x, y, z).
        small_size: The size of the patches.
        do_overlap: Whether to perform overlapping predictions.

    Returns:
        List of (x, y, z) coordinates for prediction cubes.
    """
    offsets = [get_offsets(s, small_size) for s in shape]
    xyzs = [(x, y, z) for x in offsets[0] for y in offsets[1] for z in offsets[2]]
    if do_overlap:  # Add shifted cubes (half cube overlap)
        offset = small_size // 2

        xyzs_shifted = [
            set((x + offset, y, z) for x, y, z in xyzs),
            set((x, y + offset, z) for x, y, z in xyzs),
            set((x, y, z + offset) for x, y, z in xyzs),
            set((x + offset, y + offset, z) for x, y, z in xyzs),
            set((x + offset, y, z + offset) for x, y, z in xyzs),
            set((x, y + offset, z + offset) for x, y, z in xyzs),
            set((x + offset, y + offset, z + offset) for x, y, z in xyzs),
        ]
        xyzs_shifted = set(
            (x, y, z)
            for s in xyzs_shifted
            for x, y, z in s
            if x + small_size <= shape[0]
            and y + small_size <= shape[1]
            and z + small_size <= shape[2]
        )
        xyzs = list(set.union(set(xyzs), xyzs_shifted))
    return xyzs


def get_offsets(big_size: int, small_size: int) -> List[int]:
    """
    Calculate offsets for image patching.

    Args:
        big_size: The size of the whole image.
        small_size: The size of the patches.

    Returns:
        List of offsets.
    """
    if big_size < small_size:
        # If image is smaller than patch size, we can only start at 0
        return [0]
    
    offsets = list(range(0, big_size - small_size + 1, small_size))
    if offsets[-1] != big_size - small_size:
        offsets.append(big_size - small_size)
    return offsets


def get_single_pred_weight(do_overlap: bool, small_size: int) -> Union[np.ndarray, None]:
    """
    Get the weight for a single prediction.

    Args:
        do_overlap: Whether to perform overlapping predictions.
        small_size: The size of the patches.

    Returns:
        The weight array for a single prediction, or None if no overlap.
    """
    if do_overlap:
        # The weight (confidence/expected quality) of the predictions:
        # Low at the surface of the predicted cube, high in the center
        pred_weight_helper = np.pad(np.ones((small_size,) * 3), 1, mode='constant')
        return distance_transform_cdt(pred_weight_helper).astype(np.float32)[1:-1, 1:-1, 1:-1]
    else:
        return None
