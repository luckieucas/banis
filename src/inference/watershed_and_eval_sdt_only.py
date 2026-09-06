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


def load_mask_data(file_path: str, h5_key: str = ""):
    """
    Load mask data from h5, nii.gz, tiff, or zarr files.
    """
    file_ext = file_path.lower()
    
    if file_ext.endswith('.h5') or file_ext.endswith('.hdf5'):
        # Read h5 data
        print(f"Loading mask from h5 file: {file_path}")
        with h5py.File(file_path, 'r') as f:
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
                mask_data = zarr_data[:]
            else:
                if h5_key and h5_key in zarr_data:
                    mask_data = zarr_data[h5_key][:]
                else:
                    available_keys = list(zarr_data.keys())
                    if len(available_keys) == 0:
                        raise ValueError("No keys found in zarr group")
                    key = available_keys[0]
                    mask_data = zarr_data[key][:]
                    print(f"Using first key: {key}")
            
            if hasattr(mask_data, 'compute'):
                mask_data = mask_data.compute()
        except Exception as e:
            raise ValueError(f"Failed to open mask zarr file at {file_path}: {e}")
    
    else:
        raise ValueError(f"Unsupported mask file format: {file_path}")
    
    mask_data = mask_data > 0
    print(f"Mask shape: {mask_data.shape}, dtype: {mask_data.dtype}")
    print(f"Mask valid pixels: {np.sum(mask_data)} / {mask_data.size} ({100 * np.sum(mask_data) / mask_data.size:.2f}%)")
    
    return mask_data


def load_sdt_channel(file_path: str, zarr_key: str = "", sdt_channel_idx: int = -1):
    """
    Load SDT (Signed Distance Transform) channel from zarr, h5, nii.gz, or tiff files.
    """
    file_ext = file_path.lower()
    sitk_image = None
    
    # Check if it's a zarr file
    is_zarr = False
    if file_ext.endswith('.zarr'):
        is_zarr = True
    elif os.path.isdir(file_path):
        if os.path.exists(os.path.join(file_path, 'zarr.json')) or \
           os.path.exists(os.path.join(file_path, '.zarray')):
            is_zarr = True
    
    if is_zarr:
        # ... (Zarr loading logic remains same) ...
        print(f"Loading zarr data from: {file_path}")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Zarr file/directory not found: {file_path}")
        
        try:
            zarr_data = zarr.open_array(file_path, mode='r')
        except Exception as e:
            raise ValueError(f"Failed to open zarr file at {file_path}: {e}")
        
        if zarr_key:
            if zarr_key in zarr_data:
                data = zarr_data[zarr_key][:]
                print(f"Using specified key: {zarr_key}")
            else:
                print(f"Available keys in zarr file: {list(zarr_data.keys())}")
                raise KeyError(f"Key '{zarr_key}' not found in zarr file")
        else:
            if hasattr(zarr_data, 'shape'):
                data = zarr_data[:]
                print(f"Loaded zarr array with shape: {data.shape}")
            else:
                possible_keys = ['data', 'prediction', 'pred', 'sdt']
                found = False
                for key in possible_keys:
                    if key in zarr_data:
                        data = zarr_data[key][:]
                        print(f"Found data at key: {key}, shape: {data.shape}")
                        found = True
                        break
                
                if not found:
                    available_keys = list(zarr_data.keys())
                    if len(available_keys) == 0:
                        raise ValueError("No keys found in zarr group")
                    key = available_keys[0]
                    data = zarr_data[key][:]
                    print(f"No key provided, using first key: {key}, shape: {data.shape}")
        
        print(f"Loaded data shape: {data.shape}, dtype: {data.dtype}")
        
        if len(data.shape) == 3:
            sdt_channel = data
            print(f"3D data detected, using as SDT channel directly, shape: {sdt_channel.shape}")
        elif len(data.shape) == 4:
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
            
            if channels_first:
                if sdt_channel_idx < 0:
                    sdt_channel_idx = num_channels + sdt_channel_idx
                sdt_channel = data[sdt_channel_idx]
            else:
                if sdt_channel_idx < 0:
                    sdt_channel_idx = num_channels + sdt_channel_idx
                sdt_channel = data[..., sdt_channel_idx]
            
            print(f"Extracted SDT channel {sdt_channel_idx}, shape: {sdt_channel.shape}")
        else:
            raise ValueError(f"Unsupported data shape: {data.shape}. Expected 3D or 4D array.")
        
        if hasattr(sdt_channel, 'compute'):
            sdt_channel = sdt_channel.compute()
        
        return sdt_channel, sitk_image
    
    elif file_ext.endswith('.h5') or file_ext.endswith('.hdf5'):
        # ... (H5 loading logic remains same) ...
        print(f"Loading h5 data from: {file_path}")
        with h5py.File(file_path, 'r') as f:
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
        
        print(f"Loaded data shape: {data.shape}, dtype: {data.dtype}")
        
        if len(data.shape) == 3:
            sdt_channel = data
            print(f"3D data detected, using as SDT channel directly, shape: {sdt_channel.shape}")
        elif len(data.shape) == 4:
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
            
            if channels_first:
                if sdt_channel_idx < 0:
                    sdt_channel_idx = num_channels + sdt_channel_idx
                sdt_channel = data[sdt_channel_idx]
            else:
                if sdt_channel_idx < 0:
                    sdt_channel_idx = num_channels + sdt_channel_idx
                sdt_channel = data[..., sdt_channel_idx]
            
            print(f"Extracted SDT channel {sdt_channel_idx}, shape: {sdt_channel.shape}")
        else:
            raise ValueError(f"Unsupported data shape: {data.shape}. Expected 3D or 4D array.")
        
        return sdt_channel, sitk_image
    
    # ------------------ Modified Logic for TIFF/NIfTI ------------------
    elif file_ext.endswith('.nii.gz') or file_ext.endswith('.nii') or file_ext.endswith('.tif') or file_ext.endswith('.tiff'):
        # Read image data using SimpleITK
        print(f"Loading image data from: {file_path}")
        sitk_image = sitk.ReadImage(file_path)
        data = sitk.GetArrayFromImage(sitk_image)
        print(f"Loaded image with shape: {data.shape}")
        
        # Explicitly handle 2D or 3D scalar inputs as direct SDT maps
        if len(data.shape) == 2:
            sdt_channel = data
            print(f"2D data detected, using directly as SDT channel, shape: {sdt_channel.shape}")
            
        elif len(data.shape) == 3:
            # Assumes standard (Z, Y, X) for 3D scalar images or single channel volumes
            sdt_channel = data
            print(f"3D data detected, using directly as SDT channel (Z, Y, X), shape: {sdt_channel.shape}")
            
        elif len(data.shape) == 4:
            # Assume channels are in the first dimension (SimpleITK usually reads as C last, but GetArray flips to C first or Z first)
            # Typically Sitk GetArrayFromImage returns (Z, Y, X, C) if vectors? No, usually (Z, Y, X) for scalar.
            # If 4D, usually (Time/Channel, Z, Y, X) in numpy.
            
            print(f"4D data detected (likely multi-channel), extracting channel {sdt_channel_idx}")
            num_channels = data.shape[0]
            if sdt_channel_idx < 0:
                sdt_channel_idx = num_channels + sdt_channel_idx
            sdt_channel = data[sdt_channel_idx]
            print(f"Extracted SDT channel {sdt_channel_idx}, shape: {sdt_channel.shape}")
            
        else:
            raise ValueError(f"Unsupported data shape: {data.shape}. Expected 2D, 3D or 4D array.")
        
        return sdt_channel, sitk_image
    # -------------------------------------------------------------------
    
    else:
        raise ValueError(f"Unsupported file format: {file_path}. Supported formats: .zarr, .h5, .hdf5, .nii, .nii.gz, .tif, .tiff")


def apply_watershed_segmentation_sdt_only(sdt_channel, skeleton_thr=0.0, min_size=100, 
                                          edt_downsample_factor=1, use_fast_edt=True, 
                                          edt_parallel=4, edt_anisotropy=None,
                                          foreground_thr=0.0):
    """
    Apply two-stage watershed segmentation using only SDT channel.
    """
    print(f"\n=== Stage 1: Initial instance segmentation from SDT (skeleton_thr={skeleton_thr}) ===")
    
    # Stage 1: Use SDT > skeleton_thr to get initial instance seeds
    skeleton_mask = sdt_channel > skeleton_thr
    print(f"SDT > {skeleton_thr} mask stats: {np.sum(skeleton_mask)} pixels")
    
    # Get initial instance segmentation using connected components
    # For 2D data, ensure we use appropriate connectivity (8 for 2D, 26 for 3D)
    connectivity = 8 if sdt_channel.ndim == 2 else 26
    print(f"Using connectivity: {connectivity}")
    
    initial_instance_seg = cc3d.connected_components(skeleton_mask, connectivity=connectivity)
    
    if len(np.unique(initial_instance_seg)) == 0:
        print("Warning: No seeds found! Returning empty segmentation.")
        return (np.zeros_like(skeleton_mask, dtype=np.uint8), 
                np.zeros_like(skeleton_mask, dtype=np.uint16), 
                np.zeros_like(skeleton_mask, dtype=np.uint16))
    
    print(f"Initial instance segmentation: {np.max(initial_instance_seg)} regions")
    
    # Stage 2: Use SDT > foreground_thr as foreground mask (binary mask)
    print(f"\n=== Stage 2: Create foreground mask from SDT > {foreground_thr} ===")
    
    foreground_mask = sdt_channel > foreground_thr
    print(f"Foreground mask (SDT > {foreground_thr}) stats: foreground pixels {np.sum(foreground_mask)}, background pixels {np.sum(~foreground_mask)}")
    
    foreground_mask = foreground_mask.astype(bool)
    foreground_mask = foreground_mask.astype(np.uint8)
    
    # Use initial instance segmentation as seeds
    refined_seeds = initial_instance_seg
    
    # Check how many instances remain
    unique_seeds = np.unique(refined_seeds)
    num_refined_seeds = len(unique_seeds) - 1  # Exclude background 0
    print(f"Refined seeds: {num_refined_seeds} regions (from {np.max(initial_instance_seg)} initial)")
    
    if num_refined_seeds == 0:
        print("Warning: No seeds remain! Using initial segmentation.")
        final_instance_seg = initial_instance_seg.copy()
        final_instance_seg[~foreground_mask.astype(bool)] = 0
    else:
        # Compute distance transform on foreground (with optional downsampling)
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
            original_shape = foreground_mask.shape
            print(f"EDT downsampling: factor={edt_downsample_factor}, original shape={original_shape}")
            
            # Downsample foreground mask using nearest neighbor interpolation
            downsample_zoom = 1.0 / edt_downsample_factor
            foreground_mask_ds = zoom(foreground_mask.astype(np.float32), downsample_zoom, order=0) > 0.5
            print(f"Downsampled foreground mask shape: {foreground_mask_ds.shape}")
            
            # Compute distance transform on downsampled mask
            if use_edt_lib:
                if edt_anisotropy is None:
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
            
            # Scale the distance values by the downsample factor
            distance_fg = distance_fg * edt_downsample_factor
            print(f"Upsampled distance transform shape: {distance_fg.shape}")
        else:
            # Compute EDT at full resolution
            if use_edt_lib:
                if edt_anisotropy is None:
                    edt_anisotropy = tuple([1.0] * len(foreground_mask.shape))
                distance_fg = edt.edt(
                    foreground_mask.astype(np.uint8),
                    anisotropy=edt_anisotropy,
                    black_border=True,
                    parallel=edt_parallel
                )
            else:
                distance_fg = ndimage.distance_transform_edt(foreground_mask)
        
        # Watershed using distance transform
        final_instance_seg = watershed(-distance_fg, refined_seeds, mask=foreground_mask)
        print(f"After watershed: {np.max(final_instance_seg)} regions")
    
    # Remove small regions
    unique_labels, counts = np.unique(final_instance_seg, return_counts=True)
    remove_labels = unique_labels[(unique_labels > 0) & (counts < min_size)]
    small_regions_removed = int(remove_labels.size)
    if small_regions_removed > 0:
        final_instance_seg[np.isin(final_instance_seg, remove_labels)] = 0
    
    if small_regions_removed > 0:
        print(f"Removed {small_regions_removed} small regions (< {min_size} pixels)")
    
    num_final = np.max(final_instance_seg)
    print(f"Final instance segmentation: {num_final} regions")
    
    return foreground_mask, initial_instance_seg, final_instance_seg


def save_image_data(data, output_path, reference_sitk_image=None, h5_key="data"):
    """
    Save image data to file in the same format as input.
    """
    file_ext = output_path.lower()
    
    if file_ext.endswith('.h5') or file_ext.endswith('.hdf5'):
        print(f"Saving as h5 to: {output_path}")
        with h5py.File(output_path, 'w') as f:
            f.create_dataset(h5_key, data=data, compression='gzip', compression_opts=6)
    
    elif file_ext.endswith('.nii.gz') or file_ext.endswith('.nii'):
        print(f"Saving as NIfTI to: {output_path}")
        sitk_output = sitk.GetImageFromArray(data)
        if reference_sitk_image is not None:
            # Only copy metadata if dimensions match (handling 2D input -> 3D ref edge case)
            if sitk_output.GetDimension() == reference_sitk_image.GetDimension():
                sitk_output.SetSpacing(reference_sitk_image.GetSpacing())
                sitk_output.SetOrigin(reference_sitk_image.GetOrigin())
                sitk_output.SetDirection(reference_sitk_image.GetDirection())
        sitk.WriteImage(sitk_output, output_path, useCompression=True)
    
    elif file_ext.endswith('.tif') or file_ext.endswith('.tiff'):
        print(f"Saving as TIFF to: {output_path}")
        sitk_output = sitk.GetImageFromArray(data)
        if reference_sitk_image is not None:
             if sitk_output.GetDimension() == reference_sitk_image.GetDimension():
                sitk_output.SetSpacing(reference_sitk_image.GetSpacing())
                sitk_output.SetOrigin(reference_sitk_image.GetOrigin())
                sitk_output.SetDirection(reference_sitk_image.GetDirection())
        sitk.WriteImage(sitk_output, output_path, useCompression=True)
    
    elif file_ext.endswith('.zarr'):
        print(f"Saving as zarr to: {output_path}")
        zarr.array(
            data,
            dtype=data.dtype,
            store=output_path,
            chunks=(min(3, data.shape[0]), 512, 512, 512) if len(data.shape) == 4 else (512, 512, 512) if len(data.shape) == 3 else True,
            overwrite=True,
        )
    
    else:
        raise ValueError(f"Unsupported output format: {output_path}")
    
    print(f"Saved to: {output_path}, shape: {data.shape}, dtype: {data.dtype}")


def process_watershed_and_eval_sdt_only(
    prediction_file: str,
    output_path: str,
    zarr_key: str = "",
    sdt_channel_idx: int = -1,
    skeleton_threshold: float = 0.0,
    foreground_threshold: float = 0.0,
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
):
    """
    Main processing function.
    """
    # Load SDT channel
    print(f"Loading SDT channel from: {prediction_file}")
    sdt_channel, reference_sitk_image = load_sdt_channel(
        prediction_file, 
        zarr_key=zarr_key,
        sdt_channel_idx=sdt_channel_idx
    )
    
    print(f"SDT channel shape: {sdt_channel.shape}, dtype: {sdt_channel.dtype}")
    print(f"SDT channel range: [{np.min(sdt_channel):.4f}, {np.max(sdt_channel):.4f}]")
    
    # Determine output format
    if output_format == "auto":
        input_ext = prediction_file.lower()
        if input_ext.endswith('.h5') or input_ext.endswith('.hdf5'):
            output_ext = '.h5'
        elif input_ext.endswith('.zarr') or os.path.isdir(prediction_file):
            output_ext = '.zarr'
        elif input_ext.endswith('.tif') or input_ext.endswith('.tiff'):
            output_ext = '.tiff'
        else:
            output_ext = '.nii.gz'
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
            if mask_data.shape != sdt_channel.shape:
                print(f"⚠️  Warning: Mask shape {mask_data.shape} does not match prediction shape {sdt_channel.shape}")
                print("    Attempting to resize mask...")
                from scipy.ndimage import zoom
                zoom_factors = [pred_dim / mask_dim for pred_dim, mask_dim in zip(sdt_channel.shape, mask_data.shape)]
                mask_data = zoom(mask_data.astype(np.float32), zoom_factors, order=0) > 0.5
                mask_data = mask_data.astype(bool)
                print(f"    Resized mask shape: {mask_data.shape}")
            
            # Use mask to filter SDT channel
            sdt_channel = sdt_channel * mask_data
            print(f"Mask will be applied to restrict evaluation region.")
    
    # Apply watershed segmentation using SDT only
    print("\n=== Applying watershed segmentation using SDT channel only ===")
    print(f"SDT > {foreground_threshold} will be used as binary foreground mask")
    print(f"SDT > {skeleton_threshold} will be used for initial instance seeds")
    
    foreground_mask, initial_instance_seg, final_instance_seg = apply_watershed_segmentation_sdt_only(
        sdt_channel,
        skeleton_thr=skeleton_threshold,
        min_size=watershed_min_size,
        edt_downsample_factor=edt_downsample_factor,
        use_fast_edt=use_fast_edt,
        edt_parallel=edt_parallel,
        edt_anisotropy=edt_anisotropy,
        foreground_thr=foreground_threshold,
    )
    
    # Apply mask to segmentation results if mask is provided
    if mask_data is not None:
        print("\n=== Applying mask to segmentation results ===")
        print(f"Before masking: {np.max(final_instance_seg)} regions, {np.sum(final_instance_seg > 0)} pixels")
        
        initial_instance_seg = initial_instance_seg.copy()
        initial_instance_seg[~mask_data] = 0
        
        foreground_mask = foreground_mask.copy()
        foreground_mask[~mask_data] = False
    
    # Save results
    foreground_file = output_dir / f"foreground_mask{output_ext}"
    foreground_uint8 = (foreground_mask * 255).astype(np.uint8)
    save_image_data(foreground_uint8, str(foreground_file), reference_sitk_image, h5_key="foreground_mask")
    
    initial_seg_file = output_dir / f"initial_instance_seg{output_ext}"
    initial_seg_uint16 = initial_instance_seg.astype(np.uint16)
    save_image_data(initial_seg_uint16, str(initial_seg_file), reference_sitk_image, h5_key="initial_seg")
    
    final_seg_file = output_dir / f"final_instance_seg{output_ext}"
    final_seg_uint16 = final_instance_seg.astype(np.uint16)
    save_image_data(final_seg_uint16, str(final_seg_file), reference_sitk_image, h5_key="final_seg")
    
    print(f"\n=== Watershed results saved to: {output_dir} ===")
    print(f"  - Foreground mask (SDT > {foreground_threshold}): {foreground_file.name}")
    print(f"  - Initial instance seg (SDT > {skeleton_threshold}): {initial_seg_file.name}")
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
                metrics = evaluate_single_file(
                    pred_file=str(final_seg_file),
                    gt_file=gt_file,
                    save_results=True,
                    mask_file=mask_file
                )
                
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
                
                eval_summary_file = output_dir / "evaluation_summary.txt"
                with open(eval_summary_file, 'w') as f:
                    f.write("Final Instance Segmentation Evaluation (SDT Only)\n")
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
        elif not os.path.exists(gt_file):
            print(f"\n⚠️  Warning: Ground truth file not found: {gt_file}")
        else:
            print(f"\n=== Evaluating initial instance segmentation ===")
            print(f"Ground truth file: {gt_file}")
            if mask_file:
                print(f"Mask file: {mask_file} (restricting evaluation to mask region)")
            
            try:
                initial_metrics = evaluate_single_file(
                    pred_file=str(initial_seg_file),
                    gt_file=gt_file,
                    save_results=True,
                    mask_file=mask_file
                )
                
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
                
                initial_eval_summary_file = output_dir / "initial_evaluation_summary.txt"
                with open(initial_eval_summary_file, 'w') as f:
                    f.write("Initial Instance Segmentation Evaluation (SDT Only)\n")
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
    
    print(f"\nAll outputs saved to: {output_dir}")
    print("Watershed and evaluation (SDT only) completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Apply watershed segmentation using SDT channel only (SDT > 0 as binary mask)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic watershed segmentation using SDT only (TIFF or Zarr/H5)
  python watershed_and_eval_sdt_only.py --prediction_file predictions.zarr --output_path output_dir/
  python watershed_and_eval_sdt_only.py --prediction_file my_sdt_prediction.tiff --output_path output_dir/
  
  # Apply watershed with custom seed threshold
  python watershed_and_eval_sdt_only.py --prediction_file predictions.zarr --output_path output_dir/ \
      --skeleton_threshold 0.5 --watershed_min_size 100
  
  # Apply watershed and evaluate against ground truth
  python watershed_and_eval_sdt_only.py --prediction_file predictions.zarr --output_path output_dir/ \
      --gt_file ground_truth.nii.gz
  
  # Apply watershed and evaluate both initial and final segmentation
  python watershed_and_eval_sdt_only.py --prediction_file predictions.zarr --output_path output_dir/ \
      --gt_file ground_truth.nii.gz --eval_initial
  
  # Apply watershed with mask and evaluate
  python watershed_and_eval_sdt_only.py --prediction_file predictions.zarr --output_path output_dir/ \
      --gt_file ground_truth.nii.gz --mask_file mask.tiff
  
  # Specify SDT channel index (e.g., last channel) - useful for multi-channel Zarr/Nifti
  python watershed_and_eval_sdt_only.py --prediction_file predictions.zarr --output_path output_dir/ \
      --sdt_channel_idx -1
  
  # Use fast edt library with 8 parallel threads
  python watershed_and_eval_sdt_only.py --prediction_file predictions.zarr --output_path output_dir/ \
      --edt_parallel 8
  
  # Speed up EDT with downsampling
  python watershed_and_eval_sdt_only.py --prediction_file predictions.zarr --output_path output_dir/ \
      --edt_downsample_factor 2
  
  # Save in specific format
  python watershed_and_eval_sdt_only.py --prediction_file predictions.zarr --output_path output_dir/ \
      --output_format nii.gz

Key Differences from watershed_and_eval.py:
  - Uses only SDT (Signed Distance Transform) channel
  - SDT > 0 serves as binary foreground mask (replacing affinity channel)
  - SDT > skeleton_threshold serves as initial instance seeds
  - No separate affinity channel input needed
        """
    )
    
    # Required arguments
    parser.add_argument("--prediction_file", type=str, required=True, 
                        help="Path to the prediction file (zarr, h5, or tiff/nii) containing SDT channel")
    parser.add_argument("--output_path", type=str, required=True, 
                        help="Path to output directory")
    
    # File loading options
    parser.add_argument("--zarr_key", type=str, default="", 
                        help="Key/path to the data in the zarr/h5 file. If empty, uses first key or root array (default: '')")
    parser.add_argument("--sdt_channel_idx", type=int, default=-1, 
                        help="Index of the SDT channel (default: -1 for last channel). Ignored if input is single-channel TIFF/NIfTI.")
    
    # Watershed segmentation options
    parser.add_argument("--skeleton_threshold", type=float, default=0.0, 
                        help="Threshold for SDT channel to create initial instance seeds (default: 0.0)")
    parser.add_argument("--foreground_threshold", type=float, default=0.0,
                        help="Threshold for SDT channel to create the binary foreground mask (default: 0.0)")
    parser.add_argument("--watershed_min_size", type=int, default=200, 
                        help="Minimum size of segmented regions to keep (default: 200)")
    parser.add_argument("--edt_downsample_factor", type=int, default=1, 
                        help="Downsample factor for distance transform computation (default: 1, no downsampling)")
    parser.add_argument("--use_fast_edt", action="store_true", default=True,
                        help="Use fast edt library if available (default: True)")
    parser.add_argument("--no_fast_edt", action="store_false", dest="use_fast_edt",
                        help="Disable fast edt library and use scipy instead")
    parser.add_argument("--edt_parallel", type=int, default=4,
                        help="Number of parallel threads for fast edt library (default: 4)")
    parser.add_argument("--edt_anisotropy", type=float, nargs='+', default=None,
                        help="Anisotropy values for edt (e.g., --edt_anisotropy 1.0 1.0 1.0 for 3D)")
    
    # Output options
    parser.add_argument("--output_format", type=str, default="auto", 
                        choices=["auto", "zarr", "h5", "nii.gz", "tiff"],
                        help="Output format: 'auto' (same as input), 'zarr', 'h5', 'nii.gz', 'tiff' (default: auto)")
    
    # Evaluation options
    parser.add_argument("--gt_file", type=str, default=None, 
                        help="Path to ground truth file for evaluation (default: None)")
    parser.add_argument("--eval_initial", action="store_true", default=False,
                        help="Also evaluate initial instance segmentation. Requires --gt_file (default: False)")
    parser.add_argument("--mask_file", type=str, default=None, 
                        help="Path to mask file. Predictions outside mask will be set to 0 (default: None)")
    parser.add_argument("--mask_h5_key", type=str, default="", 
                        help="Key/path to the mask data in h5 file (default: '')")
    
    args = parser.parse_args()
    
    # Validate paths
    if not os.path.exists(args.prediction_file):
        raise FileNotFoundError(f"Prediction file not found: {args.prediction_file}")
    
    # Convert edt_anisotropy list to tuple if provided
    edt_anisotropy = tuple(args.edt_anisotropy) if args.edt_anisotropy is not None else None
    
    process_watershed_and_eval_sdt_only(
        prediction_file=args.prediction_file,
        output_path=args.output_path,
        zarr_key=args.zarr_key,
        sdt_channel_idx=args.sdt_channel_idx,
        skeleton_threshold=args.skeleton_threshold,
        foreground_threshold=args.foreground_threshold,
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
    )
