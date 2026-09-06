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

try:
    from skimage.feature import peak_local_maxima
except ImportError:
    from skimage.feature import peak_local_max as peak_local_maxima

# Add parent directory to path to import BANIS and inference modules
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from BANIS import BANIS
from inference import patched_inference, patched_inference_batch

# Try to import evaluation module
try:
    evaluation_dir = os.path.join(project_root, 'src', 'evaluation')
    if evaluation_dir not in sys.path:
        sys.path.insert(0, evaluation_dir)
    from evaluate_res import evaluate_single_file
    EVALUATION_AVAILABLE = True
except ImportError:
    EVALUATION_AVAILABLE = False
    print("Warning: Evaluation module not available. Install required packages to enable evaluation.")


def load_image_data(file_path: str, h5_key: str = "", zarr_key: str = ""):
    """
    Load image data from h5, nii.gz, or tiff files.
    
    Args:
        file_path: Path to the input file.
        h5_key: Key/path to the data in the h5 file (only used for h5 files).
        zarr_key: Array key in a Zarr group (only used for Zarr groups).
    
    Returns:
        tuple: (numpy.ndarray, sitk.Image or None) - The loaded image data and SimpleITK image object (for metadata)
    """
    file_ext = file_path.lower()
    sitk_image = None
    
    if file_ext.endswith('.h5') or file_ext.endswith('.hdf5'):
        # Read h5 data
        print(f"Loading h5 data from: {file_path}")
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
            
            img_data = f[h5_key][:]
    
    elif file_ext.endswith('.zarr') or os.path.isdir(file_path):
        print(f"Loading Zarr data from: {file_path}")
        source = zarr.open(file_path, mode='r')
        if isinstance(source, zarr.Group):
            if not zarr_key:
                available_keys = list(source.array_keys())
                if len(available_keys) != 1:
                    raise ValueError(
                        f"Zarr group contains arrays {available_keys}; provide --zarr_input_key"
                    )
                zarr_key = available_keys[0]
                print(f"No key provided, using only array: {zarr_key}")
            if zarr_key not in source:
                raise KeyError(f"Zarr key '{zarr_key}' not found; available arrays: {list(source.array_keys())}")
            source = source[zarr_key]
        img_data = np.asarray(source)

    elif file_ext.endswith('.nii.gz') or file_ext.endswith('.nii') or file_ext.endswith('.tif') or file_ext.endswith('.tiff'):
        # Read image data using SimpleITK (supports NIfTI, TIFF, and many other medical image formats)
        print(f"Loading image data using SimpleITK from: {file_path}")
        sitk_image = sitk.ReadImage(file_path)
        img_data = sitk.GetArrayFromImage(sitk_image)
        print(f"Loaded image with spacing: {sitk_image.GetSpacing()}")
        print(f"Loaded image with origin: {sitk_image.GetOrigin()}")
        print(f"Loaded image with direction: {sitk_image.GetDirection()}")
    
    else:
        raise ValueError(f"Unsupported file format: {file_path}. Supported formats: .h5, .hdf5, .nii, .nii.gz, .tif, .tiff")
    
    return img_data, sitk_image


def apply_watershed_segmentation(affinity_channel, skeleton_data, binary_thr=0.5, skeleton_thr=0.0, min_size=100):
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
    
    # Choose appropriate structure element based on data dimension
    if len(skeleton_mask.shape) == 3:
        # 3D data uses ball structure element
        structure = ball(1)
    else:
        # 2D data uses disk structure element
        structure = disk(1)
    
    # Step 1.2: Morphological operations to clean skeleton mask
    skeleton_mask = ndimage.binary_fill_holes(skeleton_mask)
    skeleton_mask = ndimage.binary_opening(skeleton_mask, structure=structure)
    print(f"After morphological cleaning: skeleton pixels {np.sum(skeleton_mask)}")
    
    # Step 1.3: Use skeleton as seeds - label connected components
    skeleton_seeds, num_seeds = ndimage.label(skeleton_mask)
    print(f"Found {num_seeds} skeleton seed regions")
    
    if num_seeds == 0:
        print("Warning: No skeleton seeds found! Returning empty segmentation.")
        return (np.zeros_like(skeleton_mask, dtype=np.uint8), 
                np.zeros_like(skeleton_mask, dtype=np.uint16), 
                np.zeros_like(skeleton_mask, dtype=np.uint16), 
                np.zeros_like(skeleton_mask, dtype=np.uint16))
    
    # Step 1.4: Expand skeleton to get initial foreground for first watershed
    # Use dilation to expand skeleton regions
    dilated_skeleton = ndimage.binary_dilation(skeleton_mask, structure=structure, iterations=3)
    
    # Step 1.5: Compute distance transform on dilated skeleton
    distance_skeleton = ndimage.distance_transform_edt(dilated_skeleton)
    
    # Step 1.6: First watershed - use skeleton seeds on dilated skeleton mask
    initial_instance_seg = watershed(-distance_skeleton, skeleton_seeds, mask=dilated_skeleton)
    print(f"Initial instance segmentation: {np.max(initial_instance_seg)} regions")
    
    # Stage 2: Refine using affinity channel
    print(f"\n=== Stage 2: Refine with affinity channel (binary_thr={binary_thr}) ===")
    
    # Step 2.1: Create foreground mask from affinity channel_0
    foreground_mask = affinity_channel > binary_thr
    print(f"Foreground mask stats: foreground pixels {np.sum(foreground_mask)}, background pixels {np.sum(~foreground_mask)}")
    
    # Step 2.2: Morphological operations to clean foreground mask
    foreground_mask = ndimage.binary_fill_holes(foreground_mask)
    foreground_mask = ndimage.binary_opening(foreground_mask, structure=structure)
    print(f"After morphological cleaning: foreground pixels {np.sum(foreground_mask)}")
    
    # Step 2.3: Use initial instance segmentation as seeds
    # Only keep instances that overlap with foreground
    refined_seeds = initial_instance_seg.copy()
    refined_seeds[~foreground_mask] = 0  # Remove seeds outside foreground
    
    # Check how many instances remain
    unique_seeds = np.unique(refined_seeds)
    num_refined_seeds = len(unique_seeds) - 1  # Exclude background 0
    print(f"Refined seeds: {num_refined_seeds} regions (from {np.max(initial_instance_seg)} initial)")
    
    if num_refined_seeds == 0:
        print("Warning: No seeds remain after foreground filtering! Using initial segmentation.")
        final_instance_seg = initial_instance_seg.copy()
        final_instance_seg[~foreground_mask] = 0
    else:
        # Step 2.4: Compute distance transform on foreground
        distance_fg = ndimage.distance_transform_edt(foreground_mask)
        
        # Step 2.5: Second watershed - use initial instances as seeds on foreground
        final_instance_seg = watershed(-distance_fg, refined_seeds, mask=foreground_mask)
        print(f"After refinement watershed: {np.max(final_instance_seg)} regions")
    
    # Step 2.6: Remove small regions
    unique_labels, counts = np.unique(final_instance_seg, return_counts=True)
    small_regions_removed = 0
    for label, count in zip(unique_labels, counts):
        if count < min_size and label > 0:  # Keep background label 0
            final_instance_seg[final_instance_seg == label] = 0
            small_regions_removed += 1
    
    if small_regions_removed > 0:
        print(f"Removed {small_regions_removed} small regions (< {min_size} pixels)")
    
    # Step 2.7: Relabel to have consecutive labels
    final_instance_seg, num_final = ndimage.label(final_instance_seg > 0)
    
    print(f"Final instance segmentation: {num_final} regions")
    
    return foreground_mask, skeleton_seeds, initial_instance_seg, final_instance_seg


def save_image_data(data, output_path, reference_sitk_image=None, h5_key="data"):
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
            f.create_dataset(h5_key, data=data, compression='gzip', compression_opts=6)
    
    elif file_ext.endswith('.nii.gz') or file_ext.endswith('.nii'):
        # Save as NIfTI using SimpleITK
        print(f"Saving as NIfTI to: {output_path}")
        sitk_output = sitk.GetImageFromArray(data)
        if reference_sitk_image is not None:
            sitk_output.SetSpacing(reference_sitk_image.GetSpacing())
            sitk_output.SetOrigin(reference_sitk_image.GetOrigin())
            sitk_output.SetDirection(reference_sitk_image.GetDirection())
        sitk.WriteImage(sitk_output, output_path, useCompression=True)
    
    elif file_ext.endswith('.tif') or file_ext.endswith('.tiff'):
        # Save as TIFF using SimpleITK
        print(f"Saving as TIFF to: {output_path}")
        sitk_output = sitk.GetImageFromArray(data)
        if reference_sitk_image is not None:
            sitk_output.SetSpacing(reference_sitk_image.GetSpacing())
            sitk_output.SetOrigin(reference_sitk_image.GetOrigin())
            sitk_output.SetDirection(reference_sitk_image.GetDirection())
        sitk.WriteImage(sitk_output, output_path, useCompression=True)
    
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


def predict_data(
    checkpoint_path: str,
    input_path: str,
    output_path: str,
    h5_input_key: str = "",
    zarr_input_key: str = "",
    prediction_channels: int = 3,
    small_size: int = 128,
    divide: int = 255,
    use_batch: bool = True,
    batch_size: int = 4,
    padding_mode: str = "constant",
    padding_position: str = "end",
    padding_constant: float = 0,
    save_channels: str = "all",
    apply_watershed: bool = False,
    binary_threshold: float = 0.5,
    skeleton_threshold: float = 0.0,
    watershed_min_size: int = 100,
    output_format: str = "auto",
    gt_file: str = None,
):
    """
    Predict affinities from image data (h5, nii.gz, or tiff) using BANIS model.

    Args:
        checkpoint_path: Path to the model checkpoint.
        input_path: Path to the input file (supports HDF5, Zarr, NIfTI, and TIFF).
        output_path: Path to save the output predictions (format determined by output_format).
        h5_input_key: Key/path to the data in the h5 file (only used for h5 files). If empty, will use the first key (default: "").
        zarr_input_key: Array key in a Zarr group. Required when the group has multiple arrays.
        prediction_channels: Number of prediction channels (default: 3).
        small_size: Size of patches for inference (default: 128).
        divide: Divisor for the image normalization (default: 255).
        use_batch: Whether to use batched inference for faster processing (default: True).
        batch_size: Number of patches to process in parallel when use_batch=True (default: 4).
        padding_mode: np.pad mode when input is smaller than patch size (default: constant).
        padding_position: Where to put padding for short dimensions: end or center (default: end).
        padding_constant: Constant value for constant padding (default: 0).
        save_channels: Which channels to save: "all", "affinity" (first 3), "skeleton" (last), or comma-separated indices like "0,1,2" (default: "all").
        apply_watershed: Whether to apply watershed segmentation on skeleton channel (default: False).
        binary_threshold: Threshold for affinity channel_0 to create foreground mask (default: 0.5).
        skeleton_threshold: Threshold for skeleton channel to create seeds (default: 0.0).
        watershed_min_size: Minimum size of segmented regions (default: 100).
        output_format: Output format: "auto" (same as input), "zarr", "h5", "nii.gz", "tiff" (default: "auto").
        gt_file: Path to ground truth file for evaluation. If provided, will evaluate final_instance_seg (default: None).
    """
    print(f"Loading model from checkpoint: {checkpoint_path}")
    model = BANIS.load_from_checkpoint(checkpoint_path)
    model.eval()
    model.cuda()
    
    # Suppress torch.compile dynamo errors if they occur
    # This helps with compilation errors during inference
    try:
        import torch._dynamo
        torch._dynamo.config.suppress_errors = True
        print("Enabled torch._dynamo error suppression for inference")
    except:
        pass
    
    # Load image data using the appropriate reader
    img_data, reference_sitk_image = load_image_data(input_path, h5_input_key, zarr_input_key)
    print(f"Input data shape: {img_data.shape}")
    print(f"Input data dtype: {img_data.dtype}")
    
    # Handle different input shapes
    # patched_inference expects shape (x, y, z, channel)
    if len(img_data.shape) == 3:
        # 3D data without channel dimension - add channel dimension
        # Assume shape is (z, y, x) or (x, y, z), add channel as last dimension
        img_data = np.expand_dims(img_data, axis=-1)
        print(f"Added channel dimension, new shape: {img_data.shape}")
    elif len(img_data.shape) == 4:
        # 4D data - check if it's (channel, x, y, z) or (x, y, z, channel)
        if img_data.shape[0] < img_data.shape[-1]:
            # Likely (channel, x, y, z), transpose to (x, y, z, channel)
            img_data = np.moveaxis(img_data, 0, -1)
            print(f"Transposed data from (channel, x, y, z) to (x, y, z, channel): {img_data.shape}")
        # If already (x, y, z, channel), no change needed
    else:
        raise ValueError(f"Unsupported data shape: {img_data.shape}. Expected 3D or 4D array.")
    
    # Perform inference
    print("Starting inference...")
    if use_batch:
        print(f"Using batched inference with batch_size={batch_size}")
        aff_pred = patched_inference_batch(
            img_data,
            model=model,
            do_overlap=True,
            prediction_channels=prediction_channels,
            divide=divide,
            small_size=small_size,
            batch_size=batch_size,
            padding_mode=padding_mode,
            padding_position=padding_position,
            padding_constant=padding_constant,
        )
    else:
        print("Using sequential inference")
        aff_pred = patched_inference(
            img_data,
            model=model,
            do_overlap=True,
            prediction_channels=prediction_channels,
            divide=divide,
            small_size=small_size,
            padding_mode=padding_mode,
            padding_position=padding_position,
            padding_constant=padding_constant,
        )
    
    print(f"Prediction shape: {aff_pred.shape}")
    print(f"Prediction dtype: {aff_pred.dtype}")
    
    # Determine output format
    if output_format == "auto":
        # Use the same format as input
        input_ext = input_path.lower()
        if input_ext.endswith('.h5') or input_ext.endswith('.hdf5'):
            output_ext = '.h5'
        elif input_ext.endswith('.nii.gz'):
            output_ext = '.nii.gz'
        elif input_ext.endswith('.nii'):
            output_ext = '.nii'
        elif input_ext.endswith('.tif') or input_ext.endswith('.tiff'):
            output_ext = '.tiff'
        else:
            output_ext = '.zarr'
    else:
        output_ext = '.' + output_format.replace('.', '')
    
    # Create output directory
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine which channels to save
    num_pred_channels = aff_pred.shape[0]
    print(f"Total prediction channels: {num_pred_channels}")
    
    if save_channels == "all":
        channels_to_save = list(range(num_pred_channels))
        # Name channels meaningfully: first 3 are affinity, last is skeleton, others are numbered
        channel_names = []
        for i in range(num_pred_channels):
            if i < 3:
                channel_names.append(f"affinity_{i}")
            elif i == num_pred_channels - 1:
                channel_names.append("skeleton")
            else:
                channel_names.append(f"channel_{i}")
    elif save_channels == "affinity":
        channels_to_save = list(range(min(3, num_pred_channels)))
        channel_names = [f"affinity_{i}" for i in range(len(channels_to_save))]
    elif save_channels == "skeleton":
        channels_to_save = [num_pred_channels - 1]
        channel_names = ["skeleton"]
    else:
        # Parse comma-separated channel indices
        channels_to_save = [int(ch.strip()) for ch in save_channels.split(',')]
        channel_names = [f"channel_{i}" for i in channels_to_save]
    
    print(f"Saving channels: {channels_to_save}")
    
    # Save individual channels
    for ch_idx, ch_name in zip(channels_to_save, channel_names):
        if ch_idx >= num_pred_channels:
            print(f"Warning: Channel {ch_idx} not found in predictions (max: {num_pred_channels-1}), skipping")
            continue
        
        channel_data = aff_pred[ch_idx]
        output_file = output_dir / f"{ch_name}{output_ext}"
        
        # Convert to appropriate dtype
        if output_ext in ['.h5', '.hdf5', '.nii', '.nii.gz', '.tiff', '.tif']:
            channel_data = channel_data.astype(np.float32)
        else:
            channel_data = channel_data.astype(np.float16)
        
        save_image_data(channel_data, str(output_file), reference_sitk_image, h5_key=ch_name)
    
    # Apply watershed segmentation if requested and channels exist
    if apply_watershed:
        if num_pred_channels >= 4:
            print("\n=== Applying two-stage watershed segmentation ===")
            affinity_channel_0 = aff_pred[0]  # First channel for foreground
            skeleton_channel = aff_pred[-1]  # Last channel is skeleton
            
            foreground_mask, skeleton_seeds, initial_instance_seg, final_instance_seg = apply_watershed_segmentation(
                affinity_channel_0,
                skeleton_channel,
                binary_thr=binary_threshold,
                skeleton_thr=skeleton_threshold,
                min_size=watershed_min_size
            )
            
            # Save foreground mask (from affinity channel_0)
            foreground_file = output_dir / f"foreground_mask{output_ext}"
            foreground_uint8 = (foreground_mask * 255).astype(np.uint8)
            save_image_data(foreground_uint8, str(foreground_file), reference_sitk_image, h5_key="foreground_mask")
            
            # Save skeleton seeds (labeled connected components from skeleton)
            skeleton_seeds_file = output_dir / f"skeleton_seeds{output_ext}"
            skeleton_seeds_uint16 = skeleton_seeds.astype(np.uint16)
            save_image_data(skeleton_seeds_uint16, str(skeleton_seeds_file), reference_sitk_image, h5_key="skeleton_seeds")
            
            # Save initial instance segmentation (from skeleton watershed)
            initial_seg_file = output_dir / f"initial_instance_seg{output_ext}"
            initial_seg_uint16 = initial_instance_seg.astype(np.uint16)
            save_image_data(initial_seg_uint16, str(initial_seg_file), reference_sitk_image, h5_key="initial_seg")
            
            # Save final instance segmentation (refined with affinity)
            final_seg_file = output_dir / f"final_instance_seg{output_ext}"
            final_seg_uint16 = final_instance_seg.astype(np.uint16)
            save_image_data(final_seg_uint16, str(final_seg_file), reference_sitk_image, h5_key="final_seg")
            
            print(f"\n=== Watershed results saved to: {output_dir} ===")
            print(f"  - Foreground mask: {foreground_file.name}")
            print(f"  - Skeleton seeds: {skeleton_seeds_file.name}")
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
                    
                    try:
                        # Evaluate using the evaluation module
                        metrics = evaluate_single_file(
                            pred_file=str(final_seg_file),
                            gt_file=gt_file,
                            save_results=True,
                            mask_file=None
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
                            f.write(f"Ground truth file: {gt_file}\n\n")
                            
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
                        print("    Continuing without evaluation.")
        else:
            print("Warning: Watershed segmentation requested but not enough channels (need at least 4 channels)")
            print(f"  Current channels: {num_pred_channels}, need: affinity channels + skeleton channel")
    
    print(f"\nAll outputs saved to: {output_dir}")
    print("Inference completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict affinities from image data (h5, nii.gz, or tiff) using BANIS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic prediction with all channels
  python predict_h5.py --checkpoint_path model.ckpt --input_path data.nii.gz --output_path output_dir/
  
  # Save only affinity channels
  python predict_h5.py --checkpoint_path model.ckpt --input_path data.nii.gz --output_path output_dir/ --save_channels affinity
  
  # Apply two-stage watershed segmentation for instance segmentation
  # Stage 1: Use skeleton to get initial instances
  # Stage 2: Use affinity channel_0 as foreground, refine with initial instances as seeds
  python predict_h5.py --checkpoint_path model.ckpt --input_path data.nii.gz --output_path output_dir/ \\
      --apply_watershed --binary_threshold 0.5 --skeleton_threshold 0.0
  
  # Apply watershed and evaluate against ground truth
  python predict_h5.py --checkpoint_path model.ckpt --input_path data.nii.gz --output_path output_dir/ \\
      --apply_watershed --gt_file ground_truth.nii.gz
  
  # Save in specific format
  python predict_h5.py --checkpoint_path model.ckpt --input_path data.h5 --output_path output_dir/ --output_format tiff
        """
    )
    
    # Required arguments
    parser.add_argument("--checkpoint_path", type=str, required=True, 
                       help="Path to the model checkpoint")
    parser.add_argument("--input_path", type=str, required=True, 
                       help="Path to the input file (supports .h5, .hdf5, .nii, .nii.gz, .tif, .tiff)")
    parser.add_argument("--output_path", type=str, required=True, 
                       help="Path to output directory (individual channels will be saved here)")
    
    # Model parameters
    parser.add_argument("--h5_input_key", type=str, default="", 
                       help="Key/path to the data in the h5 file (only used for h5 files). If not provided, will use the first key in the file (default: '')")
    parser.add_argument("--zarr_input_key", type=str, default="",
                       help="Array key in a Zarr group; required when the group has multiple arrays")
    parser.add_argument("--prediction_channels", type=int, default=7, 
                       help="Number of prediction channels (default: 7)")
    parser.add_argument("--small_size", type=int, default=128, 
                       help="Size of patches for inference (default: 128)")
    parser.add_argument("--divide", type=int, default=255, 
                       help="Divisor for image normalization (default: 255)")
    
    # Batch processing
    parser.add_argument("--use_batch", action=argparse.BooleanOptionalAction, default=True, 
                       help="Use batched inference for faster processing (default: True)")
    parser.add_argument("--batch_size", type=int, default=4, 
                       help="Number of patches to process in parallel when using batched inference (default: 4)")
    parser.add_argument("--padding_mode", type=str, default="constant",
                       choices=["constant", "edge", "reflect", "symmetric"],
                       help="np.pad mode used when an input dimension is smaller than small_size (default: constant)")
    parser.add_argument("--padding_position", type=str, default="end",
                       choices=["end", "center"],
                       help="Where to place padding for short dimensions: end or center (default: end)")
    parser.add_argument("--padding_constant", type=float, default=0,
                       help="Constant padding value when --padding_mode constant (default: 0)")
    
    # Output options
    parser.add_argument("--save_channels", type=str, default="all", 
                       help="Which channels to save: 'all', 'affinity' (first 3), 'skeleton' (last), or comma-separated indices (default: all)")
    parser.add_argument("--output_format", type=str, default="auto", 
                       choices=["auto", "zarr", "h5", "nii.gz", "tiff"],
                       help="Output format: 'auto' (same as input), 'zarr', 'h5', 'nii.gz', 'tiff' (default: auto)")
    
    # Watershed segmentation options (two-stage method)
    parser.add_argument("--apply_watershed", action="store_true", 
                       help="Apply two-stage watershed: 1) skeleton->initial instances, 2) refine with affinity foreground")
    parser.add_argument("--binary_threshold", type=float, default=0.5, 
                       help="[Stage 2] Threshold for affinity channel_0 to create foreground mask (default: 0.5)")
    parser.add_argument("--skeleton_threshold", type=float, default=0.0, 
                       help="[Stage 1] Threshold for skeleton channel to create initial instances (default: 0.0)")
    parser.add_argument("--watershed_min_size", type=int, default=100, 
                       help="Minimum size of segmented regions to keep (default: 100)")
    
    # Evaluation options
    parser.add_argument("--gt_file", type=str, default=None, 
                       help="Path to ground truth file for evaluation. If provided, will evaluate final_instance_seg (default: None)")
    
    args = parser.parse_args()
    
    # Validate paths
    if not os.path.exists(args.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint_path}")
    if not os.path.exists(args.input_path):
        raise FileNotFoundError(f"Input file not found: {args.input_path}")
    
    predict_data(
        checkpoint_path=args.checkpoint_path,
        input_path=args.input_path,
        output_path=args.output_path,
        h5_input_key=args.h5_input_key,
        zarr_input_key=args.zarr_input_key,
        prediction_channels=args.prediction_channels,
        small_size=args.small_size,
        divide=args.divide,
        use_batch=args.use_batch,
        batch_size=args.batch_size,
        padding_mode=args.padding_mode,
        padding_position=args.padding_position,
        padding_constant=args.padding_constant,
        save_channels=args.save_channels,
        apply_watershed=args.apply_watershed,
        binary_threshold=args.binary_threshold,
        skeleton_threshold=args.skeleton_threshold,
        watershed_min_size=args.watershed_min_size,
        output_format=args.output_format,
        gt_file=args.gt_file,
    )
