import argparse
import os
import sys
import h5py
import numpy as np
import zarr
import SimpleITK as sitk
from scipy import ndimage
from skimage.segmentation import watershed
from skimage.morphology import disk, ball
from pathlib import Path
import cc3d

# Try to import edt library for faster distance transform
try:
    import edt
    EDT_AVAILABLE = True
except ImportError:
    EDT_AVAILABLE = False
    print("Note: edt library not available. Using scipy for distance transform. Install with: pip install edt")

# Try to import evaluation module
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    evaluation_dir = os.path.join(project_root, 'src', 'evaluation')
    if evaluation_dir not in sys.path:
        sys.path.insert(0, evaluation_dir)
    from evaluate_res import evaluate_single_file
    EVALUATION_AVAILABLE = True
except ImportError:
    EVALUATION_AVAILABLE = False
    print("Warning: Evaluation module not available. Install required packages to enable evaluation.")

# Try to import fast edt library
try:
    import edt
    EDT_AVAILABLE = True
    print("Fast EDT library available (5-10x faster than scipy)")
except ImportError:
    EDT_AVAILABLE = False
    print("Fast EDT library not available. Using scipy (slower). Install with: pip install edt")


def load_mask_data(file_path: str, h5_key: str = ""):
    """
    Load mask data from h5, nii.gz, tiff, or zarr files.
    
    Args:
        file_path: Path to the mask file.
        h5_key: Key/path to the data in the h5 file (only used for h5 files).
    
    Returns:
        numpy.ndarray: The loaded mask data (binary mask, values > 0 are True/valid regions)
    """
    file_ext = file_path.lower()
    
    if file_ext.endswith('.h5') or file_ext.endswith('.hdf5'):
        # Read h5 data
        print(f"Loading mask from h5 file: {file_path}")
        with h5py.File(file_path, 'r') as f:
            # If h5_key is not provided or empty, use the first key
            if not h5_key:
                available_keys = list(f.keys())
                if len(available_keys) == 0:
                    raise ValueError("No keys found in h5 file")
                h5_key = available_keys[0]
                print(f"No key provided, using first key: {h5_key}")
            else:
                print(f"Using specified key: {h5_key}")
            
            if h5_key not in f:
                print(f"Available keys in h5 file: {list(f.keys())}")
                raise KeyError(f"Key '{h5_key}' not found in h5 file")
            
            mask_data = f[h5_key][:]
    
    elif file_ext.endswith('.nii.gz') or file_ext.endswith('.nii') or file_ext.endswith('.tif') or file_ext.endswith('.tiff'):
        # Read image data using SimpleITK
        print(f"Loading mask from image file: {file_path}")
        sitk_image = sitk.ReadImage(file_path)
        mask_data = sitk.GetArrayFromImage(sitk_image)
        print(f"Loaded mask with shape: {mask_data.shape}")
    
    elif file_ext.endswith('.zarr') or os.path.isdir(file_path):
        # Read zarr data
        print(f"Loading mask from zarr file: {file_path}")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Mask zarr file/directory not found: {file_path}")
        
        try:
            zarr_data = zarr.open_array(file_path, mode='r') if file_ext.endswith('.zarr') else zarr.open(file_path, mode='r')
            
            if hasattr(zarr_data, 'shape'):
                # It's a zarr array
                mask_data = zarr_data[:]
            else:
                # It's a zarr group
                if h5_key and h5_key in zarr_data:
                    mask_data = zarr_data[h5_key][:]
                else:
                    # Use first available key
                    available_keys = list(zarr_data.keys())
                    if len(available_keys) == 0:
                        raise ValueError("No keys found in zarr group")
                    key = available_keys[0]
                    mask_data = zarr_data[key][:]
                    print(f"Using first key: {key}")
            
            # Convert to numpy if needed
            if hasattr(mask_data, 'compute'):
                mask_data = mask_data.compute()
        except Exception as e:
            raise ValueError(f"Failed to open mask zarr file at {file_path}: {e}")
    
    else:
        raise ValueError(f"Unsupported mask file format: {file_path}. Supported formats: .zarr, .h5, .hdf5, .nii, .nii.gz, .tif, .tiff")
    
    # Convert to boolean mask (values > 0 are valid regions)
    mask_data = mask_data > 0
    
    print(f"Mask shape: {mask_data.shape}, dtype: {mask_data.dtype}")
    print(f"Mask valid pixels: {np.sum(mask_data)} / {mask_data.size} ({100 * np.sum(mask_data) / mask_data.size:.2f}%)")
    
    return mask_data


def load_prediction_data(
    file_path: str = None,
    zarr_key: str = "",
    affinity_channel_idx: int = 0,
    skeleton_channel_idx: int = -1,
    affinity_file: str = None,
    skeleton_file: str = None,
):
    """
    Load prediction data from zarr, h5, nii.gz, or tiff files.
    
    Args:
        file_path: Path to the input file (zarr/h5 or multi-channel nii.gz/tiff).
        zarr_key: Key/path to the data in the zarr file (only used for zarr files). If empty, assumes root array.
        affinity_channel_idx: Index of the affinity channel (default: 0). Can be negative for reverse indexing.
        skeleton_channel_idx: Index of the skeleton channel (default: -1 for last channel). Can be negative for reverse indexing.
        affinity_file: Optional path to a standalone affinity image file.
        skeleton_file: Optional path to a standalone skeleton image file.
    
    Returns:
        tuple: (affinity_channel, skeleton_channel, sitk.Image or None) - The loaded channels and SimpleITK image object (for metadata)
    """
    def _load_scalar_image(image_path: str):
        """Load a single-channel image from zarr or SimpleITK-supported image."""
        if image_path.lower().endswith(".zarr") or os.path.isdir(image_path):
            arr = np.asarray(zarr.open_array(image_path, mode="r"))
            if arr.ndim not in (2, 3):
                raise ValueError(
                    f"Expected 2D/3D scalar zarr for '{image_path}', got shape {arr.shape}."
                )
            return arr, None

        image = sitk.ReadImage(image_path)
        arr = sitk.GetArrayFromImage(image)
        if arr.ndim not in (2, 3):
            raise ValueError(
                f"Expected 2D/3D scalar image for '{image_path}', got shape {arr.shape}. "
                "Use a scalar affinity/skeleton TIFF (or NIfTI), or provide a multi-channel prediction file."
            )
        return arr, image

    if file_path is None and not (affinity_file and skeleton_file):
        raise ValueError(
            "Provide --prediction_file, or provide both --affinity_file and --skeleton_file."
        )

    # Explicit separate-file mode (useful for TIFF outputs saved per-channel)
    if affinity_file or skeleton_file:
        aff_path = affinity_file or file_path
        ske_path = skeleton_file or file_path
        if aff_path is None or ske_path is None:
            raise ValueError(
                "Both affinity and skeleton file paths must be resolvable. "
                "Provide both --affinity_file and --skeleton_file, or use a valid --prediction_file."
            )

        print(f"Loading affinity channel from scalar file: {aff_path}")
        affinity_channel, affinity_sitk = _load_scalar_image(aff_path)
        print(f"Loading skeleton channel from scalar file: {ske_path}")
        skeleton_channel, _ = _load_scalar_image(ske_path)

        if affinity_channel.shape != skeleton_channel.shape:
            raise ValueError(
                f"Shape mismatch between affinity ({affinity_channel.shape}) and skeleton ({skeleton_channel.shape})"
            )

        print(f"Loaded separate image channels with shape: {affinity_channel.shape}")
        return affinity_channel, skeleton_channel, affinity_sitk

    file_ext = file_path.lower()
    sitk_image = None
    
    # Check if it's a zarr file
    is_zarr = False
    if file_ext.endswith('.zarr'):
        is_zarr = True
    elif os.path.isdir(file_path):
        # Check for zarr v2 (zarr.json) or zarr v3 (.zarray) markers
        if os.path.exists(os.path.join(file_path, 'zarr.json')) or \
           os.path.exists(os.path.join(file_path, '.zarray')):
            is_zarr = True
    
    if is_zarr:
        # Read zarr data
        print(f"Loading zarr data from: {file_path}")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Zarr file/directory not found: {file_path}")
        
        try:
            #zarr_data = zarr.open(file_path, mode='r')
            zarr_data = zarr.open_array(file_path, mode='r')

        except Exception as e:
            raise ValueError(f"Failed to open zarr file at {file_path}: {e}")
        
        # If zarr_key is provided, try to access it
        if zarr_key:
            if zarr_key in zarr_data:
                data = zarr_data[zarr_key][:]
                print(f"Using specified key: {zarr_key}")
            else:
                print(f"Available keys in zarr file: {list(zarr_data.keys())}")
                raise KeyError(f"Key '{zarr_key}' not found in zarr file")
        else:
            # Assume it's a zarr array (not a group) or access root array
            if hasattr(zarr_data, 'shape'):
                # It's a zarr array
                data = zarr_data[:]
                print(f"Loaded zarr array with shape: {data.shape}")
            else:
                # It's a zarr group, try to find the main array
                # Check if there's a common name like 'data', 'prediction', etc.
                possible_keys = ['data', 'prediction', 'pred', 'affinities']
                found = False
                for key in possible_keys:
                    if key in zarr_data:
                        data = zarr_data[key][:]
                        print(f"Found data at key: {key}, shape: {data.shape}")
                        found = True
                        break
                
                if not found:
                    # List all keys and use the first one
                    available_keys = list(zarr_data.keys())
                    if len(available_keys) == 0:
                        raise ValueError("No keys found in zarr group")
                    key = available_keys[0]
                    data = zarr_data[key][:]
                    print(f"No key provided, using first key: {key}, shape: {data.shape}")
        
        # Handle different array shapes
        # Expected: (channels, z, y, x) or (channels, x, y, z) or (z, y, x, channels) or (x, y, z, channels)
        print(f"Loaded data shape: {data.shape}, dtype: {data.dtype}")
        
        # Determine which dimension is channels
        if len(data.shape) == 3:
            # 3D data - assume it's a single channel (could be affinity or skeleton)
            raise ValueError(f"3D data detected. Expected 4D data with channel dimension. Shape: {data.shape}")
        elif len(data.shape) == 4:
            # 4D data - determine if channels are first or last
            # Usually channels dimension is the smallest or one of the smaller dimensions
            min_dim = min(data.shape)
            max_dim = max(data.shape[:-1])  # Exclude last dimension
            
            # If first dimension is small relative to others, likely (channels, z, y, x) or (channels, x, y, z)
            if data.shape[0] < max_dim:
                # Channels are first: (channels, z, y, x) or (channels, x, y, z)
                channels_first = True
                num_channels = data.shape[0]
                print(f"Detected channels-first format: (channels={num_channels}, spatial dims)")
            elif data.shape[-1] < max_dim:
                # Channels are last: (z, y, x, channels) or (x, y, z, channels)
                channels_first = False
                num_channels = data.shape[-1]
                print(f"Detected channels-last format: (spatial dims, channels={num_channels})")
            else:
                # Ambiguous - default to channels first
                channels_first = True
                num_channels = data.shape[0]
                print(f"Ambiguous format, assuming channels-first: (channels={num_channels}, spatial dims)")
        else:
            raise ValueError(f"Unsupported data shape: {data.shape}. Expected 3D or 4D array.")
        
        # Extract affinity and skeleton channels
        if channels_first:
            # Normalize indices
            if affinity_channel_idx < 0:
                affinity_channel_idx = num_channels + affinity_channel_idx
            if skeleton_channel_idx < 0:
                skeleton_channel_idx = num_channels + skeleton_channel_idx
            
            affinity_channel = data[affinity_channel_idx]
            skeleton_channel = data[skeleton_channel_idx]
        else:
            # Normalize indices
            if affinity_channel_idx < 0:
                affinity_channel_idx = num_channels + affinity_channel_idx
            if skeleton_channel_idx < 0:
                skeleton_channel_idx = num_channels + skeleton_channel_idx
            
            affinity_channel = data[..., affinity_channel_idx]
            skeleton_channel = data[..., skeleton_channel_idx]
        
        print(f"Extracted affinity channel {affinity_channel_idx}, shape: {affinity_channel.shape}")
        print(f"Extracted skeleton channel {skeleton_channel_idx}, shape: {skeleton_channel.shape}")
        
        # Convert to numpy if needed
        if hasattr(affinity_channel, 'compute'):
            affinity_channel = affinity_channel.compute()
        if hasattr(skeleton_channel, 'compute'):
            skeleton_channel = skeleton_channel.compute()
        
        return affinity_channel, skeleton_channel, sitk_image
    
    elif file_ext.endswith('.h5') or file_ext.endswith('.hdf5'):
        # Read h5 data
        print(f"Loading h5 data from: {file_path}")
        with h5py.File(file_path, 'r') as f:
            # If zarr_key is not provided, use the first key
            if not zarr_key:
                available_keys = list(f.keys())
                if len(available_keys) == 0:
                    raise ValueError("No keys found in h5 file")
                zarr_key = available_keys[0]
                print(f"No key provided, using first key: {zarr_key}")
            else:
                print(f"Using specified key: {zarr_key}")
            
            if zarr_key not in f:
                print(f"Available keys in h5 file: {list(f.keys())}")
                raise KeyError(f"Key '{zarr_key}' not found in h5 file")
            
            data = f[zarr_key][:]
        
        # Handle shape similarly to zarr
        print(f"Loaded data shape: {data.shape}, dtype: {data.dtype}")
        
        if len(data.shape) == 4:
            min_dim = min(data.shape)
            max_dim = max(data.shape[:-1])
            
            if data.shape[0] < max_dim:
                channels_first = True
                num_channels = data.shape[0]
            elif data.shape[-1] < max_dim:
                channels_first = False
                num_channels = data.shape[-1]
            else:
                channels_first = True
                num_channels = data.shape[0]
            
            # Extract channels
            if channels_first:
                if affinity_channel_idx < 0:
                    affinity_channel_idx = num_channels + affinity_channel_idx
                if skeleton_channel_idx < 0:
                    skeleton_channel_idx = num_channels + skeleton_channel_idx
                affinity_channel = data[affinity_channel_idx]
                skeleton_channel = data[skeleton_channel_idx]
            else:
                if affinity_channel_idx < 0:
                    affinity_channel_idx = num_channels + affinity_channel_idx
                if skeleton_channel_idx < 0:
                    skeleton_channel_idx = num_channels + skeleton_channel_idx
                affinity_channel = data[..., affinity_channel_idx]
                skeleton_channel = data[..., skeleton_channel_idx]
            
            print(f"Extracted affinity channel {affinity_channel_idx}, shape: {affinity_channel.shape}")
            print(f"Extracted skeleton channel {skeleton_channel_idx}, shape: {skeleton_channel.shape}")
            
            return affinity_channel, skeleton_channel, sitk_image
        else:
            raise ValueError(f"Unsupported data shape: {data.shape}. Expected 4D array with channels.")
    
    elif file_ext.endswith('.nii.gz') or file_ext.endswith('.nii') or file_ext.endswith('.tif') or file_ext.endswith('.tiff'):
        # Read image data (supports multi-channel image files or scalar images with explicit separate-file args)
        print(f"Loading image prediction data from: {file_path}")
        sitk_image = sitk.ReadImage(file_path)
        data = sitk.GetArrayFromImage(sitk_image)
        print(f"Loaded image data shape: {data.shape}, dtype: {data.dtype}")

        if len(data.shape) == 4:
            # 4D data - determine if channels are first or last (same heuristic as zarr/h5 branch)
            max_dim = max(data.shape[:-1])
            if data.shape[0] < max_dim:
                channels_first = True
                num_channels = data.shape[0]
                print(f"Detected channels-first format: (channels={num_channels}, spatial dims)")
            elif data.shape[-1] < max_dim:
                channels_first = False
                num_channels = data.shape[-1]
                print(f"Detected channels-last format: (spatial dims, channels={num_channels})")
            else:
                channels_first = True
                num_channels = data.shape[0]
                print(f"Ambiguous format, assuming channels-first: (channels={num_channels}, spatial dims)")

            if affinity_channel_idx < 0:
                affinity_channel_idx = num_channels + affinity_channel_idx
            if skeleton_channel_idx < 0:
                skeleton_channel_idx = num_channels + skeleton_channel_idx

            if not (0 <= affinity_channel_idx < num_channels and 0 <= skeleton_channel_idx < num_channels):
                raise ValueError(
                    f"Channel index out of range for {num_channels} channels: "
                    f"affinity_channel_idx={affinity_channel_idx}, skeleton_channel_idx={skeleton_channel_idx}"
                )

            if channels_first:
                affinity_channel = data[affinity_channel_idx]
                skeleton_channel = data[skeleton_channel_idx]
            else:
                affinity_channel = data[..., affinity_channel_idx]
                skeleton_channel = data[..., skeleton_channel_idx]

            print(f"Extracted affinity channel {affinity_channel_idx}, shape: {affinity_channel.shape}")
            print(f"Extracted skeleton channel {skeleton_channel_idx}, shape: {skeleton_channel.shape}")
            return affinity_channel, skeleton_channel, sitk_image

        if len(data.shape) in (2, 3):
            raise ValueError(
                "Single-channel image prediction provided. "
                "For TIFF/NIfTI channel files, pass --affinity_file and --skeleton_file. "
                "For multi-channel image predictions, provide a 4D image and set channel indices."
            )

        raise ValueError(
            f"Unsupported image data shape: {data.shape}. Expected 4D multi-channel image or 2D/3D scalar channel files."
        )
    
    else:
        raise ValueError(f"Unsupported file format: {file_path}. Supported formats: .zarr, .h5, .hdf5")


def _foreground_bbox(mask: np.ndarray):
    """Return minimal bounding box slices for non-zero foreground mask."""
    if mask.ndim == 2:
        y_any = np.any(mask, axis=1)
        x_any = np.any(mask, axis=0)
        if not y_any.any() or not x_any.any():
            return None
        y_idx = np.where(y_any)[0]
        x_idx = np.where(x_any)[0]
        return (slice(y_idx[0], y_idx[-1] + 1), slice(x_idx[0], x_idx[-1] + 1))
    if mask.ndim == 3:
        z_any = np.any(mask, axis=(1, 2))
        y_any = np.any(mask, axis=(0, 2))
        x_any = np.any(mask, axis=(0, 1))
        if not z_any.any() or not y_any.any() or not x_any.any():
            return None
        z_idx = np.where(z_any)[0]
        y_idx = np.where(y_any)[0]
        x_idx = np.where(x_any)[0]
        return (
            slice(z_idx[0], z_idx[-1] + 1),
            slice(y_idx[0], y_idx[-1] + 1),
            slice(x_idx[0], x_idx[-1] + 1),
        )
    return None


def apply_watershed_segmentation(affinity_channel, skeleton_data, binary_thr=0.5, skeleton_thr=0.0, min_size=100, 
                                edt_downsample_factor=1, use_fast_edt=True, edt_parallel=4, edt_anisotropy=None,
                                crop_to_foreground=True):
    """
    Apply two-stage watershed segmentation:
    1. Use skeleton distance transform to get initial instance segmentation
    2. Use affinity channel_0 to get foreground, then refine using initial instances as seeds
    
    Args:
        affinity_channel: affinity channel data (typically channel_0) for foreground detection
        skeleton_data: skeleton distance data for initial segmentation
        binary_thr: threshold for affinity channel to create foreground mask (default: 0.5)
        skeleton_thr: threshold for skeleton channel to create seeds (default: 0.0)
        min_size: minimum size of segmented regions (default: 100)
        edt_downsample_factor: downsample factor for distance transform computation (default: 1, no downsampling)
        use_fast_edt: whether to use fast edt library if available (default: True)
        edt_parallel: number of parallel threads for fast edt (default: 4)
        edt_anisotropy: anisotropy tuple for edt (e.g., (1.0, 1.0, 1.0) for isotropic). If None, auto-detect (default: None)
    
    Returns:
        tuple: (foreground_mask, skeleton_seeds, initial_instance_seg, final_instance_seg)
            - foreground_mask: binary mask from affinity channel
            - skeleton_seeds: labeled connected components from skeleton
            - initial_instance_seg: watershed result from skeleton seeds
            - final_instance_seg: refined watershed result using affinity foreground
    """
    print(f"\n=== Stage 1: Initial watershed from skeleton (skeleton_thr={skeleton_thr}) ===")
    
    # Stage 1: Use skeleton to get initial instance segmentation
    # Step 1.1: Create skeleton mask
    skeleton_mask = skeleton_data > skeleton_thr
    print(f"Skeleton mask stats: skeleton pixels {np.sum(skeleton_mask)}")
    
    # Step 1.2: Use skeleton as seeds - label connected components
    connectivity = 26 if skeleton_mask.ndim == 3 else 8
    initial_instance_seg = cc3d.connected_components(skeleton_mask, connectivity=connectivity)
    
    if np.max(initial_instance_seg) == 0:
        print("Warning: No skeleton seeds found! Returning empty segmentation.")
        return (np.zeros_like(skeleton_mask, dtype=np.uint8), 
                np.zeros_like(skeleton_mask, dtype=np.uint16), 
                np.zeros_like(skeleton_mask, dtype=np.uint16), 
                np.zeros_like(skeleton_mask, dtype=np.uint16))
    

    print(f"Initial instance segmentation: {np.max(initial_instance_seg)} regions")
    
    # Stage 2: Refine using affinity channel
    print(f"\n=== Stage 2: Refine with affinity channel (binary_thr={binary_thr}) ===")
    
    # Step 2.1: Create foreground mask from affinity channel_0
    foreground_mask = affinity_channel > binary_thr
    print(f"Foreground mask stats: foreground pixels {np.sum(foreground_mask)}, background pixels {np.sum(~foreground_mask)}")
    
    print(f"After morphological cleaning: foreground pixels {np.sum(foreground_mask)}")
    foreground_mask = foreground_mask.astype(bool)
    
    # Step 2.3: Use initial instance segmentation as seeds
    refined_seeds = initial_instance_seg.copy()
    refined_seeds[~foreground_mask] = 0
    
    # Check how many instances remain
    num_refined_seeds = int(np.max(refined_seeds))
    print(f"Refined seeds: {num_refined_seeds} regions (from {np.max(initial_instance_seg)} initial)")
    
    if num_refined_seeds == 0:
        print("Warning: No seeds remain after foreground filtering! Using initial segmentation.")
        final_instance_seg = initial_instance_seg.copy()
        final_instance_seg[~foreground_mask] = 0
    else:
        # Step 2.4: Optionally crop computation to foreground bounding box for speed.
        compute_mask = foreground_mask
        compute_seeds = refined_seeds
        compute_slices = None
        if crop_to_foreground:
            bbox = _foreground_bbox(foreground_mask)
            if bbox is not None:
                compute_slices = bbox
                compute_mask = foreground_mask[compute_slices]
                compute_seeds = refined_seeds[compute_slices]
                print(f"Cropped watershed ROI to shape: {compute_mask.shape}")

        # Step 2.5: Compute distance transform on foreground (with optional downsampling)
        print(f"foreground mask max: {np.max(foreground_mask)}, min: {np.min(foreground_mask)}")
        
        # Determine which EDT method to use
        use_edt_lib = use_fast_edt and EDT_AVAILABLE
        if use_fast_edt and not EDT_AVAILABLE:
            print("Warning: Fast edt library requested but not available. Falling back to scipy.")
        
        if use_edt_lib:
            print(f"Using fast edt library with {edt_parallel} parallel threads (5-10x faster)")
        else:
            print("Using scipy distance_transform_edt")
        
        if edt_downsample_factor > 1:
            from scipy.ndimage import zoom
            original_shape = compute_mask.shape
            print(f"EDT downsampling: factor={edt_downsample_factor}, original shape={original_shape}")
            
            # Downsample foreground mask using nearest neighbor interpolation
            downsample_zoom = 1.0 / edt_downsample_factor
            foreground_mask_ds = zoom(compute_mask.astype(np.float32), downsample_zoom, order=0) > 0.5
            print(f"Downsampled foreground mask shape: {foreground_mask_ds.shape}")
            
            # Compute distance transform on downsampled mask
            if use_edt_lib:
                # Use fast edt library
                if edt_anisotropy is None:
                    # Auto-detect anisotropy based on data dimension
                    edt_anisotropy = tuple([1.0] * len(foreground_mask_ds.shape))
                distance_fg_ds = edt.edt(
                    foreground_mask_ds.astype(np.uint8),
                    anisotropy=edt_anisotropy,
                    black_border=True,
                    parallel=edt_parallel
                )
            else:
                distance_fg_ds = ndimage.distance_transform_edt(foreground_mask_ds)
            
            # Upsample distance transform back to original size using linear interpolation
            upsample_zoom = [orig / ds for orig, ds in zip(original_shape, distance_fg_ds.shape)]
            distance_fg = zoom(distance_fg_ds, upsample_zoom, order=1)
            
            # Scale the distance values by the downsample factor (EDT values are in pixel units)
            distance_fg = distance_fg * edt_downsample_factor
            print(f"Upsampled distance transform shape: {distance_fg.shape}")
        else:
            # Compute EDT at full resolution
            if use_edt_lib:
                # Use fast edt library
                if edt_anisotropy is None:
                    # Auto-detect anisotropy based on data dimension
                    edt_anisotropy = tuple([1.0] * len(compute_mask.shape))
                distance_fg = edt.edt(
                    compute_mask.astype(np.uint8),
                    anisotropy=edt_anisotropy,
                    black_border=True,
                    parallel=edt_parallel
                )
            else:
                distance_fg = ndimage.distance_transform_edt(compute_mask)
        
        # Step 2.6: Second watershed - use initial instances as seeds on foreground
        final_roi = watershed(-distance_fg, compute_seeds, mask=compute_mask)
        if compute_slices is not None:
            final_instance_seg = np.zeros_like(refined_seeds, dtype=final_roi.dtype)
            final_instance_seg[compute_slices] = final_roi
        else:
            final_instance_seg = final_roi
        print(f"After refinement watershed: {np.max(final_instance_seg)} regions")
    
    # Step 2.7: Remove small regions (vectorized)
    unique_labels, counts = np.unique(final_instance_seg, return_counts=True)
    remove_labels = unique_labels[(unique_labels > 0) & (counts < min_size)]
    small_regions_removed = int(remove_labels.size)
    if small_regions_removed > 0:
        final_instance_seg[np.isin(final_instance_seg, remove_labels)] = 0
    
    if small_regions_removed > 0:
        print(f"Removed {small_regions_removed} small regions (< {min_size} pixels)")
    
    # Step 2.8: Relabel to have consecutive labels
    #final_instance_seg, num_final = ndimage.label(final_instance_seg > 0)
    num_final = np.max(final_instance_seg)
    print(f"Final instance segmentation: {num_final} regions")
    
    return foreground_mask, initial_instance_seg, final_instance_seg


def save_image_data(data, output_path, reference_sitk_image=None, h5_key="data", use_compression=True):
    """
    Save image data to file in the same format as input.
    
    Args:
        data: numpy array to save
        output_path: output file path
        reference_sitk_image: reference SimpleITK image for metadata (optional)
        h5_key: key to use when saving h5 files
    """
    file_ext = output_path.lower()
    
    if file_ext.endswith('.h5') or file_ext.endswith('.hdf5'):
        # Save as h5
        print(f"Saving as h5 to: {output_path}")
        with h5py.File(output_path, 'w') as f:
            if use_compression:
                f.create_dataset(h5_key, data=data, compression='gzip', compression_opts=6)
            else:
                f.create_dataset(h5_key, data=data)
    
    elif file_ext.endswith('.nii.gz') or file_ext.endswith('.nii'):
        # Save as NIfTI using SimpleITK
        print(f"Saving as NIfTI to: {output_path}")
        sitk_output = sitk.GetImageFromArray(data)
        if reference_sitk_image is not None:
            sitk_output.SetSpacing(reference_sitk_image.GetSpacing())
            sitk_output.SetOrigin(reference_sitk_image.GetOrigin())
            sitk_output.SetDirection(reference_sitk_image.GetDirection())
        sitk.WriteImage(sitk_output, output_path, useCompression=use_compression)
    
    elif file_ext.endswith('.tif') or file_ext.endswith('.tiff'):
        # Save as TIFF using SimpleITK
        print(f"Saving as TIFF to: {output_path}")
        sitk_output = sitk.GetImageFromArray(data)
        if reference_sitk_image is not None:
            sitk_output.SetSpacing(reference_sitk_image.GetSpacing())
            sitk_output.SetOrigin(reference_sitk_image.GetOrigin())
            sitk_output.SetDirection(reference_sitk_image.GetDirection())
        sitk.WriteImage(sitk_output, output_path, useCompression=use_compression)
    
    elif file_ext.endswith('.zarr'):
        # Save as zarr
        print(f"Saving as zarr to: {output_path}")
        zarr.array(
            data,
            dtype=data.dtype,
            store=output_path,
            chunks=(min(3, data.shape[0]), 512, 512, 512) if len(data.shape) == 4 else (512, 512, 512),
            overwrite=True,
        )
    
    else:
        raise ValueError(f"Unsupported output format: {output_path}")
    
    print(f"Saved to: {output_path}, shape: {data.shape}, dtype: {data.dtype}")


def process_watershed_and_eval(
    prediction_file: str,
    output_path: str,
    zarr_key: str = "",
    affinity_channel_idx: int = 0,
    skeleton_channel_idx: int = -1,
    binary_threshold: float = 0.5,
    skeleton_threshold: float = 0.0,
    watershed_min_size: int = 100,
    edt_downsample_factor: int = 1,
    use_fast_edt: bool = True,
    edt_parallel: int = 4,
    edt_anisotropy: tuple = None,
    output_format: str = "auto",
    gt_file: str = None,
    eval_initial: bool = False,
    mask_file: str = None,
    mask_h5_key: str = "",
    affinity_file: str = None,
    skeleton_file: str = None,
    save_intermediate: bool = True,
    use_compression: bool = True,
    crop_to_foreground: bool = True,
):
    """
    Load prediction data from zarr/h5 file, apply watershed segmentation, and evaluate.
    
    Args:
        prediction_file: Path to the prediction file (zarr/h5 or 4D image) containing all channels.
        output_path: Path to output directory.
        zarr_key: Key/path to the data in the zarr/h5 file. If empty, uses first key or root array.
        affinity_channel_idx: Index of the affinity channel (default: 0). Can be negative for reverse indexing.
        skeleton_channel_idx: Index of the skeleton channel (default: -1 for last channel). Can be negative for reverse indexing.
        binary_threshold: Threshold for affinity channel_0 to create foreground mask (default: 0.5).
        skeleton_threshold: Threshold for skeleton channel to create seeds (default: 0.0).
        watershed_min_size: Minimum size of segmented regions (default: 100).
        edt_downsample_factor: Downsample factor for distance transform computation (default: 1, no downsampling).
        use_fast_edt: Whether to use fast edt library if available (default: True).
        edt_parallel: Number of parallel threads for fast edt (default: 4).
        edt_anisotropy: Anisotropy tuple for edt (e.g., (1.0, 1.0, 1.0)). If None, auto-detect (default: None).
        output_format: Output format: "auto" (same as input), "zarr", "h5", "nii.gz", "tiff" (default: "auto").
        gt_file: Path to ground truth file for evaluation. If provided, will evaluate final_instance_seg (default: None).
        eval_initial: Whether to also evaluate initial instance segmentation (before affinity refinement). Requires gt_file (default: False).
        mask_file: Path to mask file. If provided, predictions outside mask will be set to 0 for evaluation (default: None).
        mask_h5_key: Key/path to the mask data in the h5 file (only used for h5 files). If empty, uses first key (default: "").
        affinity_file: Optional path to standalone affinity image file (for separate TIFF/NIfTI channel outputs).
        skeleton_file: Optional path to standalone skeleton image file (for separate TIFF/NIfTI channel outputs).
        save_intermediate: If False, only save final instance segmentation.
        use_compression: If False, disable output compression to reduce write time.
        crop_to_foreground: If True, run EDT/watershed on foreground ROI only for speed.
    """
    # Load prediction data
    load_source = prediction_file
    if affinity_file and skeleton_file:
        load_source = f"affinity={affinity_file}, skeleton={skeleton_file}"
    print(f"Loading prediction data from: {load_source}")
    affinity_channel, skeleton_channel, reference_sitk_image = load_prediction_data(
        prediction_file, 
        zarr_key=zarr_key,
        affinity_channel_idx=affinity_channel_idx,
        skeleton_channel_idx=skeleton_channel_idx,
        affinity_file=affinity_file,
        skeleton_file=skeleton_file,
    )
    
    print(f"Affinity channel shape: {affinity_channel.shape}, dtype: {affinity_channel.dtype}")
    print(f"Affinity channel range: [{np.min(affinity_channel):.4f}, {np.max(affinity_channel):.4f}]")
    print(f"Skeleton channel shape: {skeleton_channel.shape}, dtype: {skeleton_channel.dtype}")
    print(f"Skeleton channel range: [{np.min(skeleton_channel):.4f}, {np.max(skeleton_channel):.4f}]")
    
    # Determine output format
    if output_format == "auto":
        # Use the same format as input
        input_source = prediction_file or affinity_file or skeleton_file or ""
        input_ext = input_source.lower()
        if input_ext.endswith('.h5') or input_ext.endswith('.hdf5'):
            output_ext = '.h5'
        elif input_ext.endswith('.zarr') or (prediction_file is not None and os.path.isdir(prediction_file)):
            output_ext = '.zarr'
        elif input_ext.endswith('.tif') or input_ext.endswith('.tiff'):
            output_ext = '.tiff'
        else:
            output_ext = '.nii.gz'  # Default
    else:
        output_ext = '.' + output_format.replace('.', '')
    
    # Create output directory
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load mask file if provided
    mask_data = None
    if mask_file is not None:
        if not os.path.exists(mask_file):
            print(f"⚠️  Warning: Mask file not found: {mask_file}")
            print("    Continuing without mask.")
        else:
            print(f"\n=== Loading mask file ===")
            print(f"Mask file: {mask_file}")
            mask_data = load_mask_data(mask_file, h5_key=mask_h5_key)
            
            # Check if mask shape matches prediction shape
            if mask_data.shape != affinity_channel.shape:
                print(f"⚠️  Warning: Mask shape {mask_data.shape} does not match prediction shape {affinity_channel.shape}")
                print("    Attempting to resize mask...")
                # Try to resize mask to match prediction shape
                from scipy.ndimage import zoom
                zoom_factors = [pred_dim / mask_dim for pred_dim, mask_dim in zip(affinity_channel.shape, mask_data.shape)]
                mask_data = zoom(mask_data.astype(np.float32), zoom_factors, order=0) > 0.5
                mask_data = mask_data.astype(bool)
                print(f"    Resized mask shape: {mask_data.shape}")
            
            # use mask to filter affinity channel and skeleton channel
            affinity_channel = affinity_channel * mask_data
            skeleton_channel = skeleton_channel * mask_data
            print(f"Mask will be applied to restrict evaluation region.")
        
    
    # Apply watershed segmentation
    print("\n=== Applying two-stage watershed segmentation ===")
    foreground_mask, initial_instance_seg, final_instance_seg = apply_watershed_segmentation(
        affinity_channel,
        skeleton_channel,
        binary_thr=binary_threshold,
        skeleton_thr=skeleton_threshold,
        min_size=watershed_min_size,
        edt_downsample_factor=edt_downsample_factor,
        use_fast_edt=use_fast_edt,
        edt_parallel=edt_parallel,
        edt_anisotropy=edt_anisotropy,
        crop_to_foreground=crop_to_foreground,
    )
    
    # Apply mask to segmentation results if mask is provided
    if mask_data is not None:
        print("\n=== Applying mask to segmentation results ===")
        print(f"Before masking: {np.max(final_instance_seg)} regions, {np.sum(final_instance_seg > 0)} pixels")
        
        # Apply mask: set regions outside mask to 0
        # final_instance_seg = final_instance_seg.copy()
        # final_instance_seg[~mask_data] = 0
        
        # # Relabel to have consecutive labels after masking
        # final_instance_seg, num_final = ndimage.label(final_instance_seg > 0)
        
        # Also apply mask to other outputs for consistency
        initial_instance_seg = initial_instance_seg.copy()
        initial_instance_seg[~mask_data] = 0
        
        foreground_mask = foreground_mask.copy()
        foreground_mask[~mask_data] = False
        
    
    # Save results
    if save_intermediate:
        foreground_file = output_dir / f"foreground_mask{output_ext}"
        foreground_uint8 = (foreground_mask * 255).astype(np.uint8)
        save_image_data(
            foreground_uint8,
            str(foreground_file),
            reference_sitk_image,
            h5_key="foreground_mask",
            use_compression=use_compression,
        )

        initial_seg_file = output_dir / f"initial_instance_seg{output_ext}"
        initial_seg_uint32 = initial_instance_seg.astype(np.uint32)
        save_image_data(
            initial_seg_uint32,
            str(initial_seg_file),
            reference_sitk_image,
            h5_key="initial_seg",
            use_compression=use_compression,
        )
    else:
        initial_seg_file = output_dir / f"initial_instance_seg{output_ext}"
    
    # Save final instance segmentation (refined with affinity)
    final_seg_file = output_dir / f"final_instance_seg{output_ext}"
    final_seg_uint32 = final_instance_seg.astype(np.uint32)
    save_image_data(
        final_seg_uint32,
        str(final_seg_file),
        reference_sitk_image,
        h5_key="final_seg",
        use_compression=use_compression,
    )
    
    print(f"\n=== Watershed results saved to: {output_dir} ===")
    if save_intermediate:
        print(f"  - Foreground mask: {foreground_file.name}")
        print(f"  - Initial instance seg (from skeleton): {initial_seg_file.name}")
    print(f"  - Final instance seg (refined): {final_seg_file.name}")
    
    # Evaluate final instance segmentation if ground truth is provided
    if gt_file is not None:
        if not EVALUATION_AVAILABLE:
            print("\n⚠️  Warning: Evaluation module not available. Skipping evaluation.")
            print("    Install required packages: pandas, connectomics")
        elif not os.path.exists(gt_file):
            print(f"\n⚠️  Warning: Ground truth file not found: {gt_file}")
            print("    Skipping evaluation.")
        else:
            print(f"\n=== Evaluating final instance segmentation ===")
            print(f"Ground truth file: {gt_file}")
            if mask_file:
                print(f"Mask file: {mask_file} (restricting evaluation to mask region)")
            
            try:
                # Evaluate using the evaluation module
                metrics = evaluate_single_file(
                    pred_file=str(final_seg_file),
                    gt_file=gt_file,
                    save_results=True,
                    mask_file=mask_file
                )
                
                # Print key metrics
                print(f"\n=== Evaluation Results ===")
                print(f"Instance Segmentation Metrics:")
                print(f"  Precision: {metrics.get('precision', 0):.4f}")
                print(f"  Recall: {metrics.get('recall', 0):.4f}")
                print(f"  F1 Score: {metrics.get('f1', 0):.4f}")
                print(f"  Accuracy: {metrics.get('accuracy', 0):.4f}")
                print(f"  Panoptic Quality: {metrics.get('panoptic_quality', 0):.4f}")
                print(f"\nBinary Segmentation Metrics:")
                print(f"  Binary Precision: {metrics.get('binary_precision', 0):.4f}")
                print(f"  Binary Recall: {metrics.get('binary_recall', 0):.4f}")
                print(f"  Binary F1: {metrics.get('binary_f1', 0):.4f}")
                print(f"\nInstance Counts:")
                print(f"  Ground Truth Instances: {metrics.get('n_true', 0)}")
                print(f"  Predicted Instances: {metrics.get('n_pred', 0)}")
                print(f"  True Positives: {metrics.get('tp', 0)}")
                print(f"  False Positives: {metrics.get('fp', 0)}")
                print(f"  False Negatives: {metrics.get('fn', 0)}")
                
                # Save evaluation summary to output directory
                eval_summary_file = output_dir / "evaluation_summary.txt"
                with open(eval_summary_file, 'w') as f:
                    f.write("Final Instance Segmentation Evaluation\n")
                    f.write("=" * 50 + "\n\n")
                    f.write(f"Prediction file: {final_seg_file.name}\n")
                    f.write(f"Ground truth file: {gt_file}\n")
                    if mask_file:
                        f.write(f"Mask file: {mask_file}\n")
                    f.write("\n")
                    
                    f.write("Instance Segmentation Metrics:\n")
                    f.write(f"  Precision: {metrics.get('precision', 0):.4f}\n")
                    f.write(f"  Recall: {metrics.get('recall', 0):.4f}\n")
                    f.write(f"  F1 Score: {metrics.get('f1', 0):.4f}\n")
                    f.write(f"  Accuracy: {metrics.get('accuracy', 0):.4f}\n")
                    f.write(f"  Panoptic Quality: {metrics.get('panoptic_quality', 0):.4f}\n\n")
                    
                    f.write("Binary Segmentation Metrics:\n")
                    f.write(f"  Binary Precision: {metrics.get('binary_precision', 0):.4f}\n")
                    f.write(f"  Binary Recall: {metrics.get('binary_recall', 0):.4f}\n")
                    f.write(f"  Binary F1: {metrics.get('binary_f1', 0):.4f}\n\n")
                    
                    f.write("Instance Counts:\n")
                    f.write(f"  Ground Truth Instances: {metrics.get('n_true', 0)}\n")
                    f.write(f"  Predicted Instances: {metrics.get('n_pred', 0)}\n")
                    f.write(f"  True Positives: {metrics.get('tp', 0)}\n")
                    f.write(f"  False Positives: {metrics.get('fp', 0)}\n")
                    f.write(f"  False Negatives: {metrics.get('fn', 0)}\n")
                
                print(f"\nEvaluation summary saved to: {eval_summary_file}")
                
            except Exception as e:
                print(f"\n⚠️  Error during evaluation: {e}")
                import traceback
                traceback.print_exc()
                print("    Continuing without evaluation.")
    
    # Evaluate initial instance segmentation if requested
    if gt_file is not None and eval_initial:
        if not EVALUATION_AVAILABLE:
            print("\n⚠️  Warning: Evaluation module not available. Skipping initial segmentation evaluation.")
            print("    Install required packages: pandas, connectomics")
        elif not os.path.exists(gt_file):
            print(f"\n⚠️  Warning: Ground truth file not found: {gt_file}")
            print("    Skipping initial segmentation evaluation.")
        else:
            print(f"\n=== Evaluating initial instance segmentation ===")
            print(f"Ground truth file: {gt_file}")
            if mask_file:
                print(f"Mask file: {mask_file} (restricting evaluation to mask region)")
            
            try:
                # Evaluate initial instance segmentation
                initial_metrics = evaluate_single_file(
                    pred_file=str(initial_seg_file),
                    gt_file=gt_file,
                    save_results=True,
                    mask_file=mask_file
                )
                
                # Print key metrics for initial segmentation
                print(f"\n=== Initial Instance Segmentation Evaluation Results ===")
                print(f"Instance Segmentation Metrics:")
                print(f"  Precision: {initial_metrics.get('precision', 0):.4f}")
                print(f"  Recall: {initial_metrics.get('recall', 0):.4f}")
                print(f"  F1 Score: {initial_metrics.get('f1', 0):.4f}")
                print(f"  Accuracy: {initial_metrics.get('accuracy', 0):.4f}")
                print(f"  Panoptic Quality: {initial_metrics.get('panoptic_quality', 0):.4f}")
                print(f"\nBinary Segmentation Metrics:")
                print(f"  Binary Precision: {initial_metrics.get('binary_precision', 0):.4f}")
                print(f"  Binary Recall: {initial_metrics.get('binary_recall', 0):.4f}")
                print(f"  Binary F1: {initial_metrics.get('binary_f1', 0):.4f}")
                print(f"\nInstance Counts:")
                print(f"  Ground Truth Instances: {initial_metrics.get('n_true', 0)}")
                print(f"  Predicted Instances: {initial_metrics.get('n_pred', 0)}")
                print(f"  True Positives: {initial_metrics.get('tp', 0)}")
                print(f"  False Positives: {initial_metrics.get('fp', 0)}")
                print(f"  False Negatives: {initial_metrics.get('fn', 0)}")
                
                # Save initial evaluation summary to output directory
                initial_eval_summary_file = output_dir / "initial_evaluation_summary.txt"
                with open(initial_eval_summary_file, 'w') as f:
                    f.write("Initial Instance Segmentation Evaluation\n")
                    f.write("=" * 50 + "\n\n")
                    f.write(f"Prediction file: {initial_seg_file.name}\n")
                    f.write(f"Ground truth file: {gt_file}\n")
                    if mask_file:
                        f.write(f"Mask file: {mask_file}\n")
                    f.write("\n")
                    
                    f.write("Instance Segmentation Metrics:\n")
                    f.write(f"  Precision: {initial_metrics.get('precision', 0):.4f}\n")
                    f.write(f"  Recall: {initial_metrics.get('recall', 0):.4f}\n")
                    f.write(f"  F1 Score: {initial_metrics.get('f1', 0):.4f}\n")
                    f.write(f"  Accuracy: {initial_metrics.get('accuracy', 0):.4f}\n")
                    f.write(f"  Panoptic Quality: {initial_metrics.get('panoptic_quality', 0):.4f}\n\n")
                    
                    f.write("Binary Segmentation Metrics:\n")
                    f.write(f"  Binary Precision: {initial_metrics.get('binary_precision', 0):.4f}\n")
                    f.write(f"  Binary Recall: {initial_metrics.get('binary_recall', 0):.4f}\n")
                    f.write(f"  Binary F1: {initial_metrics.get('binary_f1', 0):.4f}\n\n")
                    
                    f.write("Instance Counts:\n")
                    f.write(f"  Ground Truth Instances: {initial_metrics.get('n_true', 0)}\n")
                    f.write(f"  Predicted Instances: {initial_metrics.get('n_pred', 0)}\n")
                    f.write(f"  True Positives: {initial_metrics.get('tp', 0)}\n")
                    f.write(f"  False Positives: {initial_metrics.get('fp', 0)}\n")
                    f.write(f"  False Negatives: {initial_metrics.get('fn', 0)}\n")
                
                print(f"\nInitial evaluation summary saved to: {initial_eval_summary_file}")
                
            except Exception as e:
                print(f"\n⚠️  Error during initial segmentation evaluation: {e}")
                import traceback
                traceback.print_exc()
                print("    Continuing without initial segmentation evaluation.")
    
    print(f"\nAll outputs saved to: {output_dir}")
    print("Watershed and evaluation completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Apply watershed segmentation on prediction data (zarr/h5/tiff/nii) and evaluate",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic watershed segmentation
  python watershed_and_eval.py --prediction_file predictions.zarr --output_path output_dir/
  
  # Apply watershed with custom thresholds
  python watershed_and_eval.py --prediction_file predictions.zarr --output_path output_dir/ \\
      --binary_threshold 0.5 --skeleton_threshold 0.0 --watershed_min_size 100
  
  # Apply watershed and evaluate against ground truth
  python watershed_and_eval.py --prediction_file predictions.zarr --output_path output_dir/ \\
      --gt_file ground_truth.nii.gz
  
  # Apply watershed and evaluate both initial and final segmentation
  python watershed_and_eval.py --prediction_file predictions.zarr --output_path output_dir/ \\
      --gt_file ground_truth.nii.gz --eval_initial
  
  # Apply watershed with mask and evaluate (only evaluate regions within mask)
  python watershed_and_eval.py --prediction_file predictions.zarr --output_path output_dir/ \\
      --gt_file ground_truth.nii.gz --mask_file mask.tiff
  
  # Specify channel indices (affinity at index 0, skeleton at index 6)
  python watershed_and_eval.py --prediction_file predictions.zarr --output_path output_dir/ \\
      --affinity_channel_idx 0 --skeleton_channel_idx 6
  
  # Use fast edt library with 8 parallel threads (5-10x faster, requires: pip install edt)
  python watershed_and_eval.py --prediction_file predictions.zarr --output_path output_dir/ \\
      --edt_parallel 8
  
  # Speed up EDT with downsampling (e.g., 2x downsample = ~8x faster for 3D)
  python watershed_and_eval.py --prediction_file predictions.zarr --output_path output_dir/ \\
      --edt_downsample_factor 2
  
  # Combine fast edt + downsampling for maximum speed
  python watershed_and_eval.py --prediction_file predictions.zarr --output_path output_dir/ \\
      --edt_parallel 8 --edt_downsample_factor 2
  
  # Save in specific format
  python watershed_and_eval.py --prediction_file predictions.zarr --output_path output_dir/ \\
      --output_format nii.gz

  # Use separate affinity/skeleton TIFF files
  python watershed_and_eval.py --affinity_file affinity_0.tiff --skeleton_file skeleton.tiff --output_path output_dir/

  # Use a multi-channel TIFF and select channels
  python watershed_and_eval.py --prediction_file prediction_multichannel.tiff --output_path output_dir/ \\
      --affinity_channel_idx 0 --skeleton_channel_idx -1

  # Faster writes and less I/O (save final output only, disable compression)
  python watershed_and_eval.py --affinity_file affinity_0.tiff --skeleton_file skeleton.tiff --output_path output_dir/ \\
      --save_final_only --no_compression --edt_downsample_factor 2 --edt_parallel 16
        """
    )
    
    # Required arguments
    parser.add_argument("--prediction_file", type=str, required=False, 
                       help="Path to prediction file containing all channels (.zarr/.h5 or 4D .tiff/.nii)")
    parser.add_argument("--affinity_file", type=str, default=None,
                       help="Path to standalone affinity image file (e.g., affinity_0.tiff).")
    parser.add_argument("--skeleton_file", type=str, default=None,
                       help="Path to standalone skeleton image file (e.g., skeleton.tiff).")
    parser.add_argument("--output_path", type=str, required=True, 
                       help="Path to output directory")
    
    # File loading options
    parser.add_argument("--zarr_key", type=str, default="", 
                       help="Key/path to the data in the zarr/h5 file. If empty, uses first key or root array (default: '')")
    parser.add_argument("--affinity_channel_idx", type=int, default=0, 
                       help="Index of the affinity channel (default: 0). Can be negative for reverse indexing.")
    parser.add_argument("--skeleton_channel_idx", type=int, default=-1, 
                       help="Index of the skeleton channel (default: -1 for last channel). Can be negative for reverse indexing.")
    
    # Watershed segmentation options
    parser.add_argument("--binary_threshold", type=float, default=0.5, 
                       help="[Stage 2] Threshold for affinity channel_0 to create foreground mask (default: 0.5)")
    parser.add_argument("--skeleton_threshold", type=float, default=0.0, 
                       help="[Stage 1] Threshold for skeleton channel to create initial instances (default: 0.0)")
    parser.add_argument("--watershed_min_size", type=int, default=200, 
                       help="Minimum size of segmented regions to keep (default: 100)")
    parser.add_argument("--edt_downsample_factor", type=int, default=1, 
                       help="Downsample factor for distance transform computation to speed up processing (default: 1, no downsampling). "
                            "Higher values (e.g., 2, 4) reduce computation time but may slightly affect boundary precision.")
    parser.add_argument("--use_fast_edt", action="store_true", default=True,
                       help="Use fast edt library if available (5-10x faster than scipy, default: True)")
    parser.add_argument("--no_fast_edt", action="store_false", dest="use_fast_edt",
                       help="Disable fast edt library and use scipy instead")
    parser.add_argument("--edt_parallel", type=int, default=4,
                       help="Number of parallel threads for fast edt library (default: 4)")
    parser.add_argument("--edt_anisotropy", type=float, nargs='+', default=None,
                       help="Anisotropy values for edt (e.g., --edt_anisotropy 1.0 1.0 1.0 for 3D). If not specified, uses isotropic (default: None)")
    
    # Output options
    parser.add_argument("--output_format", type=str, default="auto", 
                       choices=["auto", "zarr", "h5", "nii.gz", "tiff"],
                       help="Output format: 'auto' (same as input), 'zarr', 'h5', 'nii.gz', 'tiff' (default: auto)")
    parser.add_argument("--save_intermediate", action="store_true", default=True,
                       help="Save foreground and initial-instance outputs in addition to final segmentation (default: True)")
    parser.add_argument("--save_final_only", action="store_false", dest="save_intermediate",
                       help="Only save final_instance_seg output to reduce I/O time")
    parser.add_argument("--use_compression", action="store_true", default=True,
                       help="Use compression when writing outputs (default: True)")
    parser.add_argument("--no_compression", action="store_false", dest="use_compression",
                       help="Disable compression for faster output writes")
    parser.add_argument("--crop_to_foreground", action="store_true", default=True,
                       help="Crop EDT/watershed computation to foreground ROI for speed (default: True)")
    parser.add_argument("--no_crop_to_foreground", action="store_false", dest="crop_to_foreground",
                       help="Disable foreground ROI cropping")
    
    # Evaluation options
    parser.add_argument("--gt_file", type=str, default=None, 
                       help="Path to ground truth file for evaluation. If provided, will evaluate final_instance_seg (default: None)")
    parser.add_argument("--eval_initial", action="store_true", default=False,
                       help="Also evaluate initial instance segmentation (before affinity refinement). Requires --gt_file (default: False)")
    parser.add_argument("--mask_file", type=str, default=None, 
                       help="Path to mask file. If provided, predictions outside mask will be set to 0 for evaluation (default: None)")
    parser.add_argument("--mask_h5_key", type=str, default="", 
                       help="Key/path to the mask data in the h5 file (only used for h5 files). If empty, uses first key (default: '')")
    
    args = parser.parse_args()
    
    # Validate paths
    if not args.prediction_file and not (args.affinity_file and args.skeleton_file):
        parser.error("Please provide --prediction_file, or provide both --affinity_file and --skeleton_file.")

    if args.prediction_file and not os.path.exists(args.prediction_file):
        raise FileNotFoundError(f"Prediction file not found: {args.prediction_file}")

    if args.affinity_file and not os.path.exists(args.affinity_file):
        raise FileNotFoundError(f"Affinity file not found: {args.affinity_file}")

    if args.skeleton_file and not os.path.exists(args.skeleton_file):
        raise FileNotFoundError(f"Skeleton file not found: {args.skeleton_file}")
    
    # Convert edt_anisotropy list to tuple if provided
    edt_anisotropy = tuple(args.edt_anisotropy) if args.edt_anisotropy is not None else None
    
    process_watershed_and_eval(
        prediction_file=args.prediction_file,
        output_path=args.output_path,
        zarr_key=args.zarr_key,
        affinity_channel_idx=args.affinity_channel_idx,
        skeleton_channel_idx=args.skeleton_channel_idx,
        binary_threshold=args.binary_threshold,
        skeleton_threshold=args.skeleton_threshold,
        watershed_min_size=args.watershed_min_size,
        edt_downsample_factor=args.edt_downsample_factor,
        use_fast_edt=args.use_fast_edt,
        edt_parallel=args.edt_parallel,
        edt_anisotropy=edt_anisotropy,
        output_format=args.output_format,
        gt_file=args.gt_file,
        eval_initial=args.eval_initial,
        mask_file=args.mask_file,
        mask_h5_key=args.mask_h5_key,
        affinity_file=args.affinity_file,
        skeleton_file=args.skeleton_file,
        save_intermediate=args.save_intermediate,
        use_compression=args.use_compression,
        crop_to_foreground=args.crop_to_foreground,
    )
