# Baseline for Affinity-based Mitochondria Instance Segmentation

**An easily adaptable baseline for the Mitochondria Instance Segmentation, [predicting affinities](https://arxiv.org/abs/1706.00120) with [modern architectures](https://arxiv.org/abs/2303.09975) and simple connected components for post-processing**

## Prerequisites

Set up a conda/mamba environment:

```bash
# With environment.yaml
mamba env create -f environment.yaml
mamba activate mito_sdt

# Without yaml
mamba create -n mito_sdt -c conda-forge python=3.11 -y
mamba activate mito_sdt
pip install torch torchvision torchaudio numpy connected-components-3d numba pytorch-lightning zarr monai scipy cython tensorboard
pip install git+https://github.com/MIC-DKFZ/MedNeXt.git#egg=mednextv1
pip install git+https://github.com/funkelab/funlib.evaluate.git
pip install git+https://github.com/PytorchConnectomics/em_util.git@8cc736ffb2a59bf4a8232e0c15ee8a5527935a76#egg=em_util
```

Tested on a Slurm cluster with nodes equipped with 1 NVIDIA A40 GPU and 500 GB RAM (stay tuned for a less RAM-intensive version).

## Training

For the cerebellum p7 workflow, including data preparation, training, testing,
ITK-SNAP export, and post-processing, see
[`docs/cerebellum_banis_handoff.md`](docs/cerebellum_banis_handoff.md).

Run a single training session (BANIS-S(mall)):

```bash
python BANIS.py --seed 0 --batch_size 2 --n_steps 1000000 --data_setting betaSeg han24 Jurkat Cardiac Kidney Liver Sperm Macrophage --base_data_path /path/to/sample_data --save_path ./checkpoints/mito --devices=1 --sdt 
```
Results are logged to TensorBoard by default. To also log scalars, images, hyperparameters, and optional checkpoint artifacts to Weights & Biases, pass `--wandb`:

```bash
python BANIS.py --seed 0 --batch_size 2 --n_steps 1000000 --data_setting betaSeg --base_data_path /path/to/sample_data --save_path ./checkpoints/mito --devices=1 --sdt --wandb --wandb_project banis
```

Useful W&B options include `--wandb_mode offline`, `--wandb_entity <team>`, `--wandb_group <group>`, `--wandb_tags tag1 tag2`, `--wandb_run_id <id> --wandb_resume allow`, and `--wandb_log_model` for checkpoint artifacts. For GPUs with less than 48 GB memory, reduce `batch_size` (and adjust `n_steps` / `learning_rate`). For BANIS-L(arge) add `--model_id L --kernel_size 5`. Additional options are in `parse_args` of `BANIS.py`.

To run multiple jobs on Slurm, adjust `config.yaml` and `aff_train.sh`, then:

```bash
python submit_slurm.py "python BANIS.py --seed 0 --batch_size 2 --n_steps 1000000 --data_setting betaSeg han24 Jurkat Cardiac Kidney Liver Sperm Macrophage --base_data_path /path/to/sample_data --save_path ./checkpoints/mito --devices=1 --sdt" -t 120:00:00 -p long -n train_allmito_sdt --constraint vr80g
```
## Inference
 ```bash
# Without evaluation (default behavior)
python src/inference/pred_and_eval.py \
    --checkpoint_path model.ckpt \
    --input_path data.nii.gz \
    --output_path output_dir/ \
    --apply_watershed

# With evaluation
python src/inference/pred_and_eval.py \
    --checkpoint_path model.ckpt \
    --input_path data.nii.gz \
    --output_path output_dir/ \
    --apply_watershed \
    --binary_threshold 0.6 \
    --skeleton_threshold 0.4 \
    --gt_file ground_truth.nii.gz 
 ```

 # decode from sdt only

 ```bash
python src/inference/watershed_and_eval_sdt_only.py --prediction_file path/to/pred_aff_test_high_c2.zarr --gt_file path/to/ground_truth.nii.gz  --output_path path/to/output_dir --skeleton_threshold 0.5  --output_format tiff --edt_parallel 8 --edt_downsample_factor 2 --eval_initial
 ```
