import argparse
import os
import random
from argparse import Namespace
from typing import Tuple, Union

import cc3d
import numpy as np
import torch.utils
import zarr
from monai.transforms import RandAffined
from monai.utils import set_determinism
from torch.utils.data import Dataset, ConcatDataset
from tqdm import tqdm
try:
    # Older pytorch-connectomics API.
    from connectomics.data.utils.data_transform import skeleton_aware_distance_transform
except ModuleNotFoundError:
    try:
        # Newer pytorch-connectomics API.
        from connectomics.data.processing.distance import skeleton_aware_distance_transform
    except ModuleNotFoundError:
        skeleton_aware_distance_transform = None

import warnings


def comp_affinities(
        seg: np.ndarray, labeled_mask: np.ndarray = None, long_range: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute affinities (and a loss mask) from a (ground truth) segmentation.

    Args:
        seg: The segmentation to compute affinities from. 0: background. Shape: (x, y, z).
        labeled_mask: The labeled mask, indicating whether each voxel is labeled.
            Defaults to None. Loss mask for affinities involving unlabeled voxels is set to 0.
        long_range: The voxel offset for long range for affinities. Defaults to 10.
            (Currently uniform for all axes.)

    Returns:
        The affinities. Shape: (6, x, y, z).
        The loss mask. Shape: (6, x, y, z).
    """
    affinities = np.zeros((6, *seg.shape), dtype=bool)

    # Short range affinities
    affinities[0, :-1] = seg[:-1] == seg[1:]
    affinities[1, :, :-1] = seg[:, :-1] == seg[:, 1:]
    affinities[2, :, :, :-1] = seg[:, :, :-1] == seg[:, :, 1:]

    # Long range affinities
    affinities[3, :-long_range] = seg[:-long_range] == seg[long_range:]
    affinities[4, :, :-long_range] = seg[:, :-long_range] == seg[:, long_range:]
    affinities[5, :, :, :-long_range] = seg[:, :, :-long_range] == seg[:, :, long_range:]

    affinities[:, seg == 0] = 0  # background

    loss_mask = np.zeros_like(affinities, dtype=bool)
    if labeled_mask is None:
        # Outside of seg unknown
        loss_mask[0, :-1] = 1
        loss_mask[1, :, :-1] = 1
        loss_mask[2, :, :, :-1] = 1
        loss_mask[3, :-long_range] = 1
        loss_mask[4, :, :-long_range] = 1
        loss_mask[5, :, :, :-long_range] = 1
    else:
        loss_mask[0, :-1] = labeled_mask[:-1] & labeled_mask[1:]
        loss_mask[1, :, :-1] = labeled_mask[:, :-1] & labeled_mask[:, 1:]
        loss_mask[2, :, :, :-1] = labeled_mask[:, :, :-1] & labeled_mask[:, :, 1:]
        loss_mask[3, :-long_range] = labeled_mask[:-long_range] & labeled_mask[long_range:]
        loss_mask[4, :, :-long_range] = labeled_mask[:, :-long_range] & labeled_mask[:, long_range:]
        loss_mask[5, :, :, :-long_range] = labeled_mask[:, :, :-long_range] & labeled_mask[:, :, long_range:]

    return affinities, loss_mask


class AffinityDataset(Dataset):
    """
    Dataset for neuron instance segmentation using a (large) image and corresponding
    segmentation from which smaller cubes are sampled, affinites are computed and returned.

    Args:
        seg: The (ground truth) segmentation. -1 indicates unlabeled voxels. Shape: (x, y, z).
        img: The image. Shape: (x, y, z, channel).
        long_range: The voxel offset for long range for affinities. Defaults to 10.
            (Currently uniform for all axes.)
        small_size: The size of the cubes to sample. Defaults to 128.
        size_divisor: The divisor for the size of the cubes to sample.
            Defaults to 5, leads to >= 80% labeled for one dimension -> at least 0.8³ = 0.512 labeled
        augment: Whether to perform data augmentation. Defaults to False.
        len_multiplier: Multiplier for the length of the dataset. Reduces rounding
            issues for __len__. Defaults to 10.
        augment_args: Namespace with augmentation arguments. Defaults to Namespace(
            drop_slice_prob=0, shift_slice_prob=0, intensity_aug=False, noise_scale=0,
            affine=0.0).
        divide: The divisor for the image. Typically, 255 if img in [0, 255]
            (as is the case for uint8). Defaults to 1.
        ensure_foreground: Whether to require sampled patches to contain at least
            one foreground voxel (seg > 0). Defaults to True.
        max_sampling_attempts: Maximum number of resampling attempts when
            ensure_foreground is enabled. Defaults to 100.
    """

    def __init__(
            self,
            seg: Union[np.ndarray, zarr.Array],
            img: Union[np.ndarray, zarr.Array],
            long_range: int = 10,
            small_size: int = 128,
            size_divisor: int = 5,
            augment: bool = False,
            len_multiplier: int = 1,
            augment_args: Namespace = Namespace(
                drop_slice_prob=0,
                shift_slice_prob=0,
                intensity_aug=False,
                noise_scale=0,
                affine=0.0,
                affine_scale=0.2,
                affine_shear=0.5,
                shift_magnitude=10,
                mul_int=0.1,
                add_int=0.1,
            ),
            divide: Union[int, float] = 1,
            sdt_args: Namespace = Namespace(
                sdt=False,
                resolution=(1.0, 1.0, 1.0),
            ),
            ensure_foreground: bool = True,
            max_sampling_attempts: int = 100,
    ):
        set_determinism(seed=np.random.randint(0, 2**32))
        self.size_divisor = size_divisor
        self.img = img
        self.divide = divide
        self.long_range = long_range
        self.size = small_size
        self.augment = augment
        self.len_multiplier = len_multiplier
        self.augment_args = augment_args
        self.sdt_args = sdt_args
        self.ensure_foreground = ensure_foreground
        self.max_sampling_attempts = max_sampling_attempts

        self.offset = tuple((img.shape[i] - seg.shape[i]) // 2 for i in range(3))

        # print(f"seg shape {seg.shape}, img shape {img.shape}")

        if img.shape[:3] != seg.shape:
            # Shapes don't match, pad seg (and load it into memory)
            # only for real data.
            seg_tmp = np.full_like(img[:, :, :, 0], -1, dtype=np.int64)  # -1 is unlabeled
            slices = tuple(slice(o, -o if o else None) for o in self.offset)
            seg_tmp[slices] = seg
            self.seg = seg_tmp
        else:
            self.seg = seg

        if augment and augment_args.affine > 0:
            self.affine_aug = RandAffined(
                keys=["img", "seg"],
                mode=("bilinear", "nearest"),
                prob=augment_args.affine,
                rotate_range=(np.pi, np.pi, np.pi),
                # translate_range = 0,  No translation: we sample random location before
                scale_range=(augment_args.affine_scale,) * 3,
                shear_range=(augment_args.affine_shear,) * 6,
                padding_mode="reflection"
            )

    def __getitem__(self, item):
        slices, seg = self._sample_valid_patch()
        # Easiest for affine augmentation: all dimensions same
        img = np.moveaxis(self.img[slices] / self.divide, -1, 0)
        assert len(img.shape) == 4

        seg_cc = cc3d.connected_components(
            seg,
            connectivity=6,
            out_dtype=np.uint32,
        ).astype(np.int32)
        # Relabel disconnected components of same ID (connected outside of cube) to different IDs
        seg_cc[seg == -1] = -1  # Unlabeled stays unlabeled
        seg = seg_cc

        if self.augment:
            img, seg = self._apply_augmentations(img, seg)

        labeled_mask = seg != -1
        aff, loss_mask = comp_affinities(seg, labeled_mask=labeled_mask, long_range=self.long_range)
        if self.sdt_args.sdt:
            if skeleton_aware_distance_transform is None:
                raise ModuleNotFoundError(
                    "skeleton_aware_distance_transform is unavailable. "
                    "Install a compatible pytorch-connectomics version."
                )
            sdt_result = skeleton_aware_distance_transform(
                (seg + (seg == -1)), # treat unlabeled as background
                resolution=self.sdt_args.resolution,
            )
            # Support both return styles:
            # - old API: (sdt, skeleton) tuple
            # - new API: sdt ndarray
            if isinstance(sdt_result, tuple):
                sdt = sdt_result[0]
            else:
                sdt = sdt_result
            assert sdt.shape == seg.shape

        aff = aff[:, : self.size, : self.size, : self.size]
        loss_mask = loss_mask[:, : self.size, : self.size, : self.size]
        img = img[:, : self.size, : self.size, : self.size]
        seg = seg[: self.size, : self.size, : self.size]
        labeled_mask = labeled_mask[: self.size, : self.size, : self.size]
        if self.sdt_args.sdt:
            sdt = sdt[: self.size, : self.size, : self.size]
        if loss_mask.mean() < 0.1:
            warnings.warn("Low loss mask mean, indicating many unlabeled voxels in the cube.")

        aff = aff.astype(np.int8)
        # Bits are stored as bytes anyway, save memory by also encoding loss mask
        aff[~loss_mask] = -1

        data = {
            "img": img.astype(np.float16),
            "seg": seg,
            "aff": aff,
        }
        if self.sdt_args.sdt:
            data["sdt"] = sdt.astype(np.float16)
            data["sdt_mask"] = labeled_mask.astype(np.bool_)

        for k, v in data.items():
            # To avoid issues with negative strides (e.g. from flipping)
            data[k] = v.copy()

        return data

    def _sample_valid_patch(self):
        """Sample a patch and optionally enforce foreground presence."""
        attempts = self.max_sampling_attempts if self.ensure_foreground else 1
        for _ in range(attempts):
            pos = [_sample_position(o, self.size, self.size_divisor, s) for o, s in zip(self.offset, self.seg.shape)]
            slices = tuple(slice(p, p + self.size + self.long_range) for p in pos)
            seg = self.seg[slices]
            if not self.ensure_foreground or np.any(seg[: self.size, : self.size, : self.size] > 0):
                return slices, seg.copy()

        raise RuntimeError(
            "Could not sample a foreground-containing patch after "
            f"{self.max_sampling_attempts} attempts. "
            "Check whether this volume contains foreground labels (seg > 0) "
            "or increase max_sampling_attempts."
        )

    def _apply_augmentations(self, img, seg):
        axes_shuffled = np.random.permutation(3)  # x,y,z treated same
        seg = seg.transpose(axes_shuffled)
        img = img.transpose(0, *(axes_shuffled + 1))
        for a, b in [(-1, -2), (-1, -3), (-2, -3)]:  # Rotate along different axes
            rot = random.randint(0, 3)
            seg = np.rot90(seg, rot, (a, b))  # Cheap: rot90 returns a view
            img = np.rot90(img, rot, (a, b))
        for i in range(-3, 0):  # Flip last three axes
            if random.random() < 0.5:
                seg = np.flip(seg, i)
                img = np.flip(img, i)
        if self.augment_args.drop_slice_prob > 0 and random.random() < 0.5:
            ax = random.randint(-3, -1)
            drop = np.random.rand(img.shape[ax]) < self.augment_args.drop_slice_prob
            index = [slice(None)] * img.ndim
            index[ax] = drop
            img[tuple(index)] = 0
        if self.augment_args.shift_slice_prob > 0 and random.random() < 0.5:
            ax = random.randint(-3, -1)
            other_axes = (
                (-1, -2) if ax == -3 else (-3, -1) if ax == -2 else (-3, -2)
            )
            for i in range(img.shape[ax]):
                if np.random.rand() < self.augment_args.shift_slice_prob:
                    index = [slice(None)] * img.ndim
                    index[ax] = i
                    index = tuple(index)

                    for other_ax in other_axes:
                        img[index] = np.roll(
                            img[index],
                            np.random.randint(
                                -self.augment_args.shift_magnitude,
                                self.augment_args.shift_magnitude + 1,
                            ),
                            axis=other_ax,
                        )
        if self.augment_args.intensity_aug:
            if random.random() < 0.5:
                img = img * np.random.uniform(1 - self.augment_args.mul_int, 1 + self.augment_args.mul_int)
                img = img + np.random.uniform(-self.augment_args.add_int, self.augment_args.add_int)
        if self.augment_args.noise_scale > 0 and random.random() < 0.5:
            img = img + np.random.normal(
                0, random.random() * self.augment_args.noise_scale, img.shape
            )
        if self.augment_args.affine:
            data = {
                "img": img,
                "seg": seg[None],
            }
            data = self.affine_aug(data)

            img = data["img"].astype(np.float16)
            seg = data["seg"][0].numpy().astype(np.int64)
        return img, seg

    def __len__(self):
        # #voxels in cube / #voxels per sample
        return int(
            self.len_multiplier
            * (
                    (self.seg.shape[0] - 2 * self.offset[0])
                    * (self.seg.shape[1] - 2 * self.offset[1])
                    * (self.seg.shape[2] - 2 * self.offset[2])
            )
            / (self.size ** 3)
        )


def _sample_position(offset: int, size: int, size_divisor: int, seg_shape: int) -> int:
    """
    Sample a random position for a patch.

    Args:
        offset: Offset from the edge of the image.
        size: Size of the patch.
        size_divisor: Divisor for the size of the patch.
        seg_shape: Shape of the segmentation.

    Returns:
        Sampled position.
    """
    return random.randint(
        offset + max(-size // size_divisor, -offset),
        seg_shape - size - offset + min(offset, size // size_divisor)
    )


class WeightedConcatDataset(torch.utils.data.Dataset):
    """
    A dataset that concatenates multiple datasets and samples from them according to specified weights.

    Length:  The minimum length of the individual datasets.

    Args:
        datasets (List[torch.utils.data.Dataset]): A list of datasets to concatenate.
        weights (List[float]): A list of weights corresponding to the probability of sampling from each dataset.
    """

    def __init__(self, datasets, weights):
        self.datasets = datasets
        self.weights = weights

    def __getitem__(self, index):
        dataset_idx = np.random.choice(len(self.datasets), p=self.weights)
        return self.datasets[dataset_idx][index]

    def __len__(self):
        return min(len(d) for d in self.datasets)


def get_seg_dataset(
        data_path: str,
        len_multiplier: int = 10,
        small_size: int = 128,
        augment = False,
        augment_args: Namespace = Namespace(
            drop_slice_prob=0,
            shift_slice_prob=0,
            intensity_aug=False,
            noise_scale=0,
            affine=0.0,
            erode=False,
            long_range=10,
        ),
        sdt_args: Namespace = Namespace(
            resolution=(1.0, 1.0, 1.0),
        ),
) -> ConcatDataset:
    """
    Create a dataset from segmentation data.

    Args:
        data_path: Path to the data directory.
        len_multiplier: Multiplier for the length of the dataset.
        small_size: The size of the cubes to sample. Defaults to 128.
        augment_args: Namespace with augmentation arguments.

    Returns:
        A ConcatDataset containing all the individual datasets.
    """
    names = sorted([n for n in os.listdir(data_path) if n.endswith(".zarr")])

    datasets = []
    for name in tqdm(names, desc="Loading datasets"):
        sample = zarr.open(os.path.join(data_path, name), mode="r")
        seg = sample["/volumes/labels/neuron_ids"][:]
        img = sample["/volumes/raw"][:][:, :, :, None]

        dataset = AffinityDataset(
            seg=seg.astype(np.int64),
            img=(img / 255).astype(np.float16),
            # divide = 255
            augment=augment,
            len_multiplier=len_multiplier,
            augment_args=augment_args,
            long_range=augment_args.long_range,
            small_size=small_size,
            sdt_args=sdt_args,
            ensure_foreground=getattr(augment_args, "ensure_foreground", True),
            max_sampling_attempts=getattr(augment_args, "max_sampling_attempts", 100),
        )
        assert len(dataset) > 0
        datasets.append(dataset)
    dataset_lens = [len(d) for d in datasets]
    print(f"Dataset lens: {dataset_lens}")
    return ConcatDataset(datasets)


def load_data(args: argparse.Namespace):
    """Load training and validation data for single or multiple datasets.
    
    Args:
        args: Namespace containing:
            - data_settings: list of dataset names
            - resolutions: list of resolutions for each dataset
            - other training parameters
    
    Returns:
        train_data: Training dataset (ConcatDataset if multiple datasets)
        val_data: Validation dataset (uses first dataset by default)
        n_channels: Number of input channels
    """
    # Support both single and multiple datasets
    data_settings = args.data_settings if hasattr(args, 'data_settings') else [args.data_setting]
    resolutions = args.resolutions if hasattr(args, 'resolutions') else [args.resolution]
    
    if len(data_settings) == 1:
        # Single dataset - original behavior
        train_data = get_train_data(args)
        val_data = get_val_data(args)
    else:
        # Multiple datasets - load and concatenate all
        print(f"Loading multiple datasets: {data_settings}")
        train_datasets = []
        val_datasets = []
        
        for ds, res in zip(data_settings, resolutions):
            # Create a copy of args with current dataset info
            ds_args = argparse.Namespace(**vars(args))
            ds_args.data_setting = ds
            ds_args.resolution = res
            
            print(f"Loading dataset: {ds} with resolution {res}")
            train_ds = get_train_data(ds_args)
            val_ds = get_val_data(ds_args)
            
            train_datasets.append(train_ds)
            val_datasets.append(val_ds)
        
        # Concatenate all training datasets
        train_data = ConcatDataset(train_datasets)
        
        val_data = ConcatDataset(val_datasets)
        # Keep the channel-count lookup used below.
        val_data.img = val_datasets[0].img
        print(f"Using validation data from all datasets: {data_settings}")
    
    n_channels = val_data.img.shape[-1]
    return train_data, val_data, n_channels


def get_train_data(args: argparse.Namespace):
    assert 0 <= args.synthetic <= 1
    print(f"Loading real data from {args.real_data_path}, synthetic_ratio: {args.synthetic}")

    if args.synthetic < 1:
        real_train_data = get_seg_dataset(
            args.real_data_path,
            small_size=args.small_size,
            len_multiplier=100,
            augment=args.augment,
            augment_args=args,
            sdt_args=args,
        )
    if args.synthetic > 0:
        syn_train_data = get_syn_train_data(args)

    if 0 < args.synthetic < 1:
        train_data = WeightedConcatDataset(
            [syn_train_data, real_train_data], [args.synthetic, 1 - args.synthetic])
    elif args.synthetic == 0:
        train_data = real_train_data
    else:
        train_data = syn_train_data

    return train_data


def get_syn_train_data(args: argparse.Namespace):
    """Get synthetic training data.
    
    Args:
        args: Namespace containing data_setting (single dataset name) and other parameters
    
    Returns:
        ConcatDataset of training data from all seeds of the specified dataset
    """
    # Get the data_setting - should be a single string at this point
    data_setting = args.data_setting if isinstance(args.data_setting, str) else args.data_setting[0]
    
    base_path_train = os.path.join(args.base_data_path, data_setting, "train")
    
    if not os.path.exists(base_path_train):
        raise ValueError(f"Training data path does not exist: {base_path_train}")
    
    seeds_path_train = sorted([f for f in os.listdir(base_path_train) if os.path.isdir(os.path.join(base_path_train, f))])
    assert seeds_path_train, f"No seeds found in {base_path_train}"
    seeds_train_paths = [os.path.join(base_path_train, seed) for seed in seeds_path_train]

    img_seg_paths = sorted([
        os.path.join(seed_train_path, "data.zarr")
        for seed_train_path in seeds_train_paths
    ])
    print(f"[{data_setting}] image+segmentation paths: {img_seg_paths}")
    img_segs_train = [zarr.open(path, mode="r") for path in img_seg_paths]
    segs_train = [img_seg["seg"] for img_seg in img_segs_train]
    print(f"[{data_setting}] Segmentation shapes: {[seg.shape for seg in segs_train]}")

    imgs_train = [img_seg["img"] for img_seg in img_segs_train]
    print(f"[{data_setting}] Image shapes: {[img.shape for img in imgs_train]}")
    print(f"[{data_setting}] Image dtypes: {[img.dtype for img in imgs_train]}")

    train_datasets = [
        AffinityDataset(
            seg=img_seg["seg"],
            img=img_seg["img"],
            long_range=args.long_range,
            augment=args.augment,
            augment_args=args,
            divide=255.0,
            small_size=args.small_size,
            sdt_args=args,
            ensure_foreground=getattr(args, "ensure_foreground", True),
            max_sampling_attempts=getattr(args, "max_sampling_attempts", 100),
        )
        for img_seg in img_segs_train
    ]
    return ConcatDataset(train_datasets)


def get_val_data(args: argparse.Namespace):
    """Get validation data.
    
    Args:
        args: Namespace containing data_setting (single dataset name) and other parameters
    
    Returns:
        AffinityDataset for validation
    """
    # Get the data_setting - should be a single string at this point
    data_setting = args.data_setting if isinstance(args.data_setting, str) else args.data_setting[0]
    
    base_path_val = os.path.join(args.base_data_path, data_setting, "val")
    
    if not os.path.exists(base_path_val):
        raise ValueError(f"Validation data path does not exist: {base_path_val}")
    
    seeds_path_val = sorted([f for f in os.listdir(base_path_val) if os.path.isdir(os.path.join(base_path_val, f))])
    assert seeds_path_val, f"No seeds found in {base_path_val}"
    seeds_val_paths = [os.path.join(base_path_val, seed) for seed in seeds_path_val]

    img_seg_path = os.path.join(seeds_val_paths[0], "data.zarr")
    print(f"[{data_setting}] Loading validation data from: {img_seg_path}")
    img_seg = zarr.open(img_seg_path, mode="r")

    return AffinityDataset(
        seg=img_seg["seg"],
        img=img_seg["img"],
        long_range=args.long_range,
        augment=False,
        augment_args=args,
        divide=255.0,
        small_size=args.small_size,
        sdt_args=args,
        ensure_foreground=getattr(args, "ensure_foreground", True),
        max_sampling_attempts=getattr(args, "max_sampling_attempts", 100),
    )
