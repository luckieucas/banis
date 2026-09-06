import argparse
import gc
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List
import random
import warnings

import numpy as np
import pytorch_lightning as pl
import torch
import torchvision
import zarr
from nnunet_mednext import create_mednext_v1
from pytorch_lightning import LightningModule, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, DeviceStatsMonitor, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from torch.nn.functional import binary_cross_entropy_with_logits, mse_loss, tanh
from pytorch_lightning.strategies import DDPStrategy
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
from types import SimpleNamespace
try:
    import wandb
except ImportError:
    wandb = None

from src.data.data import load_data
from src.inference.inference import scale_sigmoid, patched_inference, compute_connected_component_segmentation
from src.evaluation.metrics import compute_metrics


def get_conf(config_path=None):
    """
    Loads a YAML config file and returns it as a nested SimpleNamespace object.
    If config_path is None, it parses command-line arguments for a --config path.
    """
    if config_path is None:
        # Use a separate parser to not interfere with other argument parsers
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument('--config', default='config.yaml', help='Path to the config file')
        args, _ = parser.parse_known_args()
        config_path = args.config

    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    # Recursively convert dict to SimpleNamespace
    def dict_to_sns(d):
        if isinstance(d, dict):
            return SimpleNamespace(**{k: dict_to_sns(v) for k, v in d.items()})
        elif isinstance(d, list):
            return [dict_to_sns(i) for i in d]
        return d

    return dict_to_sns(config_dict)


class BANIS(LightningModule):
    """
    PyTorch Lightning module for BANIS: Baseline for Affinity-based Neuron Instance Segmentation
    """

    def __init__(self, **kwargs: Any):
        super().__init__()
        self.save_hyperparameters()
        print(f"hparams: \n{self.hparams}")

        self.model = create_mednext_v1(
            num_input_channels=self.hparams.num_input_channels,
            num_classes=6 + int(self.hparams.sdt),  # 3 short + 3 long range affinities + (1 if self.hparams.sdt)
            model_id=self.hparams.model_id,
            kernel_size=self.hparams.kernel_size,
        )
        self.model.outside_block_checkpointing = True  # Save GPU memory

        if self.hparams.compile:
            self.model = torch.compile(self.model)

        self.best_nerl_so_far = defaultdict(float)  # for train/val/test
        self.best_thr_so_far = defaultdict(float)

        self.plotted = False
    def on_save_checkpoint(self, checkpoint):
        checkpoint["best_thr_so_far"] = self.best_thr_so_far
        checkpoint["best_nerl_so_far"] = self.best_nerl_so_far

    def on_load_checkpoint(self, checkpoint):
        self.best_thr_so_far = checkpoint.get("best_thr_so_far", defaultdict(float))
        self.best_nerl_so_far = checkpoint.get("best_nerl_so_far", defaultdict(float))

    def on_fit_start(self):
        self._add_text("hparams", str(self.hparams))

    def _wandb_log_metric(self, name: str, value: torch.Tensor | float) -> None:
        """Best-effort direct W&B logging for critical metrics."""
        if wandb is None or wandb.run is None or self._has_wandb_logger():
            return
        # Avoid duplicate logs from non-primary ranks.
        if hasattr(self, "trainer") and self.trainer is not None:
            if hasattr(self.trainer, "is_global_zero") and not self.trainer.is_global_zero:
                return
        if isinstance(value, torch.Tensor):
            value = float(value.detach().cpu().item())
        wandb.log({name: value}, step=int(self.global_step))

    def _iter_logger_experiments(self):
        if not hasattr(self, "trainer") or self.trainer is None:
            loggers = [self.logger] if self.logger is not None else []
        else:
            loggers = getattr(self.trainer, "loggers", None) or ([self.logger] if self.logger is not None else [])
        for logger in loggers:
            experiment = getattr(logger, "experiment", None)
            if experiment is not None:
                yield logger, experiment

    def _has_wandb_logger(self) -> bool:
        if not hasattr(self, "trainer") or self.trainer is None:
            loggers = [self.logger] if self.logger is not None else []
        else:
            loggers = getattr(self.trainer, "loggers", None) or ([self.logger] if self.logger is not None else [])
        return any(isinstance(logger, WandbLogger) for logger in loggers)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        if self.hparams.scheduler:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.n_steps)
            return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
        return optimizer

    def on_train_epoch_start(self):
        self.plotted = False

    def on_validation_epoch_start(self):
        self.plotted = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, data: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._step(data, "train")

    def validation_step(self, data: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._step(data, "val")

    def _step(self, data: Dict[str, torch.Tensor], mode: str) -> torch.Tensor:
        self.log_input(data["img"])
        self.log_weight_stats()
        pred = self(data["img"])
        if not torch.isfinite(pred).all():
            raise RuntimeError(f"Non-finite model predictions at step={self.global_step}, mode={mode}")
        if self.hparams.sdt:
            aff_pred, sdt_pred = pred[:, :-1], pred[:, -1]
        else:
            aff_pred = pred
        target = data["aff"].half()
        aff_loss_mask = data["aff"] >= 0
        if aff_loss_mask.any():
            aff_loss = binary_cross_entropy_with_logits(aff_pred[aff_loss_mask], target[aff_loss_mask])
        else:
            # Keep training numerically stable on extremely sparse patches.
            aff_loss = aff_pred.sum() * 0.0
            warnings.warn(
                f"Empty affinity supervision mask at step={self.global_step}, mode={mode}; "
                "using zero affinity loss for this batch."
            )
        self.log(
            f"{mode}_aff_loss",
            aff_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=(mode == "train"),
            batch_size=data["img"].shape[0],
        )
        self._wandb_log_metric(f"{mode}_aff_loss", aff_loss)

        if self.hparams.sdt:
            sdt_target = data["sdt"].half()
            assert -1 <= sdt_target.min() and sdt_target.max() <= 1
            sdt_loss_mask = data["sdt_mask"]
            if sdt_loss_mask.any():
                sdt_loss = mse_loss(tanh(sdt_pred[sdt_loss_mask]), sdt_target[sdt_loss_mask])
            else:
                sdt_loss = sdt_pred.sum() * 0.0
                warnings.warn(
                    f"Empty SDT supervision mask at step={self.global_step}, mode={mode}; "
                    "using zero SDT loss for this batch."
                )
            self.log(
                f"{mode}_sdt_loss",
                sdt_loss,
                on_step=True,
                on_epoch=True,
                prog_bar=(mode == "train"),
                batch_size=data["img"].shape[0],
            )
            self._wandb_log_metric(f"{mode}_sdt_loss", sdt_loss)
            loss = aff_loss + self.hparams.sdt_loss_weight * sdt_loss
        else:
            loss = aff_loss
        self.log(
            f"{mode}_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=data["img"].shape[0],
        )
        self._wandb_log_metric(f"{mode}_loss", loss)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step={self.global_step}, mode={mode}")


        if not self.plotted:
            self._log_images(data, pred, mode)
            self.plotted = True
        return loss

    def _log_images(self, data: Dict[str, torch.Tensor], pred: torch.Tensor, mode: str):
        middle = data["img"].shape[2] // 2
        self._add_image(f"{mode}_img", data["img"][:, :3, middle])
        self._add_image(f"{mode}_aff", data["aff"][:, :3, middle])
        self._add_image(f"{mode}_aff_pred", scale_sigmoid(pred[:, :3, middle]))
        self._add_image(f"{mode}_lr_aff", data["aff"][:, 3:6, middle])
        self._add_image(f"{mode}_lr_aff_pred", scale_sigmoid(pred[:, 3:6, middle]))
        
        if self.hparams.sdt:
            self._add_image(f"{mode}_sdt", data["sdt"][:, middle].unsqueeze(1))
            self._add_image(f"{mode}_sdt_pred", tanh(pred[:, -1, middle]).unsqueeze(1))

        seg_middle = data["seg"][:, middle]
        # max(1, ...)+2 so that empty all nonlabelled segments are colored black
        colormap = torch.rand(max(1,seg_middle.max()) + 2, 3)
        colormap[0] = 0
        colormap[-1] = 0
        colored_seg = colormap[seg_middle.cpu()].permute(0, 3, 1, 2)
        self._add_image(f"{mode}_seg", colored_seg)

    def _add_image(self, tag: str, img: torch.Tensor) -> None:
        grid = torchvision.utils.make_grid(img, value_range=(0, 1))
        for _, experiment in self._iter_logger_experiments():
            if hasattr(experiment, "add_image"):
                experiment.add_image(tag, grid, global_step=self.global_step)
            elif wandb is not None and hasattr(experiment, "log"):
                wandb_grid = grid.detach().float().cpu()
                wandb_grid = torch.nan_to_num(wandb_grid, nan=0.0, posinf=1.0, neginf=0.0)
                min_value = float(wandb_grid.min())
                max_value = float(wandb_grid.max())
                if min_value < 0.0 or max_value > 1.0:
                    denom = max(max_value - min_value, 1e-8)
                    wandb_grid = (wandb_grid - min_value) / denom
                wandb_grid = wandb_grid.clamp(0.0, 1.0)
                experiment.log({tag: wandb.Image(wandb_grid)}, step=int(self.global_step))

    def _add_text(self, tag: str, text: str) -> None:
        for _, experiment in self._iter_logger_experiments():
            if hasattr(experiment, "add_text"):
                experiment.add_text(tag, text)
            elif hasattr(experiment, "log"):
                experiment.log({tag: text}, step=int(self.global_step))

    def on_validation_epoch_end(self):
        if self.hparams.validate_extern:
            if self.trainer.is_global_zero:
                def format_value(value):
                    if isinstance(value, bool):
                        return str(value).lower()  # Convert booleans to lowercase strings (true/false)
                    elif isinstance(value, list):
                        return ' '.join(map(str, value))  # Convert list to a space-separated string
                    elif value is None:
                        return ''  # Skip None values
                    else:
                        return str(value)  # Convert other types to string

                args_list = [f"--{key} {format_value(value)}" for key, value in self.hparams.items()]
                args = ' '.join(args_list)

                command = f"sbatch --job-name {self.hparams.exp_name}_val --output {self.hparams.save_dir}/slurm-validation-log.txt validation_watcher.sh {args}"
                os.system(command)
                print(f"running validation: {command}")

        else:
            # self.full_cube_inference("val")
            print("skipping full cube inference")

    def on_train_end(self):
        if not getattr(self.hparams, "final_full_cube_inference", True):
            print("Skipping final full cube inference.")
            return
        # assert self.best_nerl_so_far["val"] > 0, "No best NERL found in validation"
        self.eval()
        print(f"device {next(self.parameters()).device}")
        self.cuda()
        self.full_cube_inference("val")
        assert self.best_nerl_so_far["val"] > 0, "No best NERL found in validation"
        self.full_cube_inference("test")
        self.full_cube_inference("train")

    @torch.no_grad()
    def full_cube_inference(self, mode: str, evaluate_thresholds: bool = True, all_seeds: bool = False, prediction_channels = 3, global_step=None, data_setting: str = None):
        """Perform full cube inference. Expensive!

        Args:
            mode: Either "train", "val", or "test".
            evaluate_thresholds: Whether to evaluate thresholds.
            all_seeds: Whether to evaluate all seeds or just the first one.
            prediction_channels: Number of prediction channels.
            global_step: Global step for logging.
            data_setting: Specific dataset to run inference on. If None and model was trained on
                         multiple datasets, will run on all of them. If None and model was trained
                         on single dataset, will use that dataset.
        """
        assert mode in ["train", "val", "test"], f"Invalid mode: {mode}"
        print(f"Full cube inference for {mode}")

        # Handle both single dataset (string) and multi-dataset (list) cases
        if data_setting is not None:
            # Use specified dataset
            datasets_to_infer = [data_setting]
        elif isinstance(self.hparams.data_setting, list):
            # Multi-dataset case: infer on all datasets
            datasets_to_infer = self.hparams.data_setting
            print(f"Running inference on all {len(datasets_to_infer)} datasets: {datasets_to_infer}")
        else:
            # Single dataset case (backward compatibility)
            datasets_to_infer = [self.hparams.data_setting]
        
        # Run inference on each dataset
        for dataset_name in datasets_to_infer:
            print(f"\n{'='*60}")
            print(f"Running inference on dataset: {dataset_name}")
            print(f"{'='*60}")
            
            base_path_mode = os.path.join(self.hparams.base_data_path, dataset_name, mode)
            
            if not os.path.exists(base_path_mode):
                print(f"Warning: Path does not exist: {base_path_mode}, skipping...")
                continue
                
            seeds_path_mode = sorted([f for f in os.listdir(base_path_mode) if os.path.isdir(os.path.join(base_path_mode, f))])
            
            if len(seeds_path_mode) == 0:
                print(f"Warning: No seeds found in {base_path_mode}, skipping...")
                continue
                
            if not all_seeds:
                seeds_path_mode = seeds_path_mode[:1]
                
            for x in seeds_path_mode:
                seed_path = os.path.join(base_path_mode, x)
                print(f"Processing seed: {x} from {dataset_name}")

                img_data = zarr.open(os.path.join(seed_path, "data.zarr"), mode="r")["img"]

                aff_pred = patched_inference(img_data, model=self, do_overlap=True, prediction_channels=prediction_channels, divide=255,
                                             small_size=self.hparams.small_size)

                # Include dataset name in output file
                output_name = f"pred_aff_{mode}_{dataset_name}_{x}.zarr" if len(datasets_to_infer) > 1 else f"pred_aff_{mode}_{x}.zarr"
                aff_pred = zarr.array(aff_pred, dtype=np.float16, store=f"{self.hparams.save_dir}/{output_name}",
                                      chunks=(3, 512, 512, 512), overwrite=True)

                if evaluate_thresholds:
                    self._evaluate_thresholds(aff_pred, os.path.join(seed_path, "skeleton.pkl"), f"{mode}_{dataset_name}" if len(datasets_to_infer) > 1 else mode, global_step)

    def _evaluate_thresholds(self, aff_pred: zarr.Array, skel_path: str, mode: str, global_step=None):
        best_voi = best_voi_no_merge = 1e100
        best_nerl = best_nerl_no_merge = -1
        best_nerl_metrics = None
        thresholds = self.hparams.eval_ranges if mode != "test" else [self.best_thr_so_far["val"]]

        for thr in tqdm(thresholds):
            gc.collect()
            torch.cuda.empty_cache()
            print(f"threshold {thr}")

            pred_seg = compute_connected_component_segmentation(
                aff_pred[:3] > thr  # hard affinities
            )

            metrics = compute_metrics(pred_seg, skel_path)

            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    self.safe_add_scalar(f"{mode}_{k}_thr_{thr}", v, global_step)

            if metrics["n_non0_mergers"] == 0:
                best_nerl_no_merge = max(best_nerl_no_merge, metrics["nerl"])
                best_voi_no_merge = min(best_voi_no_merge, metrics["voi_sum"])

            if metrics["nerl"] > best_nerl:
                best_nerl = metrics["nerl"]
                best_nerl_metrics = metrics
                if self.best_nerl_so_far[mode] < best_nerl:
                    self.best_nerl_so_far[mode] = best_nerl
                    self.best_thr_so_far[mode] = thr
                    with open(f"{self.hparams.save_dir}/best_thr_{mode}.txt", "w") as f:
                        f.write(str(self.best_thr_so_far[mode]))
                    seg_pred = zarr.array(pred_seg, dtype=np.uint32,
                                          store=f"{self.hparams.save_dir}/pred_seg_{mode}.zarr",
                                          chunks=(512, 512, 512), overwrite=True)
            best_voi = min(best_voi, metrics["voi_sum"])

        self.safe_add_scalar(f"{mode}_best_nerl", best_nerl, global_step)
        self.safe_add_scalar(f"{mode}_best_voi", best_voi, global_step)
        self.safe_add_scalar(f"{mode}_best_nerl_no_merge", best_nerl_no_merge, global_step)
        self.safe_add_scalar(f"{mode}_best_voi_no_merge", best_voi_no_merge, global_step)

        for k, v in best_nerl_metrics.items():
            if isinstance(v, (int, float)):
                self.safe_add_scalar(f"{mode}_best_nerl_{k}", v, global_step)

    def safe_add_scalar(self, name: str, value: float, global_step=None) -> None:
        step = self.global_step if global_step is None else global_step
        try:  # s.t. full_cube_inference can be called outside of .fit() without error
            for _, experiment in self._iter_logger_experiments():
                if hasattr(experiment, "add_scalar"):
                    experiment.add_scalar(name, value, step)
                elif hasattr(experiment, "log"):
                    experiment.log({name: value}, step=int(step))
        except Exception as e:
            print(f"Error logging {name}: {e}")

    def log_input(self, input):
        self.log_dict({
                f"input/min": input.min(),
                f"input/max": input.max(),
                f"input/mean": input.mean(),
                f"input/std": input.std(),
        })

    def register_activation_hooks(self):
        for name, module in self.named_modules():
            def hook_fn(module, input, output, block_name=name):  # capture name in default arg
                if not self.training:  # don't log during validation
                    return
                self.log_dict({
                    f"activations/{block_name}_min": output.min(),
                    f"activations/{block_name}_max": output.max(),
                    f"activations/{block_name}_mean": output.mean(),
                    f"activations/{block_name}_std": output.std(),
                })
                if torch.isnan(output).any():
                    print(f"NaN in output of {block_name}")
            module.register_forward_hook(hook_fn)

    def setup(self, stage: str):
        if stage == 'fit':
            self.register_activation_hooks()

    def log_weight_stats(self):
        for name, param in self.named_parameters():
            self.log_dict({
                f"weights/{name}_min": param.data.min(),
                f"weights/{name}_max": param.data.max(),
                f"weights/{name}_mean": param.data.mean(),
                f"weights/{name}_std": param.data.std(),
            })

    def configure_gradient_clipping(self, optimizer, gradient_clip_val, gradient_clip_algorithm):
        total_norm_before = torch.norm(torch.stack([p.grad.norm(2) for p in self.parameters() if p.grad is not None]))
        self.log("gradients/total_norm", total_norm_before.item())
        max_grad_before = max([p.grad.abs().max().item() for p in self.parameters() if p.grad is not None])
        self.log("gradients/max_grad", max_grad_before)

        self.clip_gradients(optimizer, gradient_clip_val=gradient_clip_val, gradient_clip_algorithm=gradient_clip_algorithm)

        total_norm_after = torch.norm(torch.stack([p.grad.norm(2) for p in self.parameters() if p.grad is not None]))
        self.log("gradients/total_norm_clipped", total_norm_after.item(), on_step=True)
        max_grad_after = max([p.grad.abs().max().item() for p in self.parameters() if p.grad is not None])
        self.log("gradients/max_grad_clipped", max_grad_after)


def main():
    args = parse_args()
    
    # Load config and get resolution for the data_setting(s)
    conf = get_conf("./config.yaml")
    
    # Support multiple data settings
    data_settings = args.data_setting if isinstance(args.data_setting, list) else [args.data_setting]
    
    # Get resolution for each data setting
    resolutions = []
    for ds in data_settings:
        resolution = None
        
        # Check mito settings first
        for setting in conf.mito.settings:
            if setting.name == ds:
                resolution = tuple(setting.resolution)
                break
        
        # If not found in mito, check rib settings
        if resolution is None and hasattr(conf, 'rib'):
            for setting in conf.rib.settings:
                if setting.name == ds:
                    resolution = tuple(setting.resolution)
                    break
        
        if resolution is None:
            raise ValueError(f"Resolution not found for data_setting '{ds}' in config")
        
        resolutions.append(resolution)
    
    # Store both data_settings and resolutions
    args.data_settings = data_settings
    args.resolutions = resolutions
    
    # For backward compatibility and logging, keep the first one as primary
    args.resolution = resolutions[0]

    seed_everything(args.seed, workers=True)

    torch.set_float32_matmul_precision("medium")

    # Create experiment name with multiple datasets
    data_setting_str = "+".join(args.data_settings) if len(args.data_settings) > 1 else args.data_settings[0]
    
    exp_name = (
            datetime.now().strftime("%y-%m-%d_%H-%M-%S-%f")
            + f"ds_{data_setting_str}"
              f"_lrng{args.long_range}_s{args.seed}_b{args.batch_size}_m{args.model_id}_k{args.kernel_size}_"
              f"lr{args.learning_rate}_wd{args.weight_decay}_sch{args.scheduler}_syn_{args.synthetic}"
              f"_drop{args.drop_slice_prob}_shift{args.shift_slice_prob}_int{args.intensity_aug}_noise{args.noise_scale}"
              f"_affine{args.affine}_ns{args.n_steps}_ss{args.small_size}"
              f"_sdt{int(args.sdt)}_sdtw{args.sdt_loss_weight}"
    ) if not args.exp_name else args.exp_name

    save_dir = os.path.join(args.save_path, exp_name)
    os.makedirs(save_dir, exist_ok=True)
    print(f"save dir: {save_dir}")
    
    # Check if resuming from checkpoint
    if args.resume_from_last_checkpoint:
        checkpoint_dir = os.path.join(save_dir, "default", "checkpoints")
        last_ckpt = os.path.join(checkpoint_dir, "last.ckpt")
        if not os.path.exists(last_ckpt):
            raise FileNotFoundError(
                f"Cannot resume: checkpoint not found at {last_ckpt}\n"
                f"Make sure --exp_name matches an existing experiment directory."
            )
        print(f"Resuming from checkpoint: {last_ckpt}")
    
    tb_logger = TensorBoardLogger(
        save_dir=args.save_path,
        name=exp_name,
        version="default",
    )
    tb_logger.experiment.add_text("save dir", save_dir)

    loggers = [tb_logger]
    wandb_logger = None
    if args.wandb and args.wandb_mode != "disabled":
        if wandb is None:
            raise ModuleNotFoundError(
                "wandb is not installed, but --wandb is enabled. "
                "Install wandb or pass --no-wandb."
            )
        wandb_logger_kwargs = {
            "project": args.wandb_project,
            "name": args.wandb_run_name if args.wandb_run_name else exp_name,
            "save_dir": save_dir,
            "offline": args.wandb_mode == "offline",
            "log_model": args.wandb_log_model,
        }
        if args.wandb_entity:
            wandb_logger_kwargs["entity"] = args.wandb_entity
        if args.wandb_group:
            wandb_logger_kwargs["group"] = args.wandb_group
        if args.wandb_tags:
            wandb_logger_kwargs["tags"] = args.wandb_tags
        if args.wandb_run_id:
            wandb_logger_kwargs["id"] = args.wandb_run_id
        if args.wandb_resume:
            wandb_logger_kwargs["resume"] = args.wandb_resume
        if args.wandb_sync_tensorboard:
            wandb_logger_kwargs["sync_tensorboard"] = True

        wandb_logger = WandbLogger(**wandb_logger_kwargs)
        wandb_logger.log_hyperparams({**vars(args), "save_dir": save_dir})
        loggers.append(wandb_logger)
        print(
            f"W&B enabled (project={args.wandb_project}, mode={args.wandb_mode}, "
            f"sync_tensorboard={args.wandb_sync_tensorboard})"
        )
    else:
        print("W&B disabled.")

    model_checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        save_last=True,
        mode="min",
        save_top_k=100,
        verbose=True,
        save_on_train_epoch_end=False  # automatically runs at the end of the validation
    )
    trainer = pl.Trainer(
        callbacks=[
            DeviceStatsMonitor(),
            model_checkpoint_callback,
            LearningRateMonitor(
                logging_interval='step'
            ),
        ],
        logger=loggers,
        max_steps=args.n_steps,
        accelerator="gpu",
        devices=args.devices,
        strategy=DDPStrategy(find_unused_parameters=False) if args.distributed else "auto",
        num_nodes=int(os.environ["SLURM_NNODES"]) if args.distributed else 1,
        log_every_n_steps=args.log_every_n_steps,
        limit_val_batches=100,
        precision="16-mixed",
        profiler="simple",
        default_root_dir=save_dir,
        val_check_interval=args.val_check_interval,  # validation full cube inference expensive so less frequent
        check_val_every_n_epoch=None,
        fast_dev_run=args.n_debug_steps,
        gradient_clip_val=1.0,
    )
    print(f"Checkpoints will be saved in: {trainer.default_root_dir}/checkpoints")

    train_data, val_data, n_channels = load_data(args)
    args.save_dir = save_dir
    args.num_input_channels = n_channels

    if os.path.exists(args.model_from_checkpoint):
        print(f"Loading model from checkpoint: {args.model_from_checkpoint}")
        model = BANIS(**vars(args))
        checkpoint = torch.load(args.model_from_checkpoint, map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"])
        model.hparams.update(vars(args))
    else:
        model = BANIS(**vars(args))

    try:
        trainer.fit(
            model=model,
            train_dataloaders=DataLoader(train_data, batch_size=args.batch_size, num_workers=args.workers, shuffle=True, drop_last=True),
            val_dataloaders=DataLoader(val_data, batch_size=args.batch_size, num_workers=args.workers),
            ckpt_path="last" if args.resume_from_last_checkpoint else None
        )
    finally:
        if wandb_logger is not None and wandb is not None and wandb.run is not None:
            wandb.finish()

    print("Training complete")


def parse_args():
    parser = argparse.ArgumentParser()

    # General arguments
    parser.add_argument("--exp_name", type=str, default="", help="Experiment name (if empty, will be filled automatically).")
    parser.add_argument("--save_path", type=str, default="/cajal/scratch/projects/misc/riegerfr/aff_nis/", help="Path to save the model and logs.")

    # Training arguments
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training.")
    parser.add_argument("--n_steps", type=int, default=20_000, help="Number of training steps.")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate for the optimizer.")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay for the optimizer.")
    parser.add_argument("--workers", type=int, default=8, help="Number of workers for data loading.")
    parser.add_argument("--scheduler", action=argparse.BooleanOptionalAction, default=True, help="Whether to use a learning rate scheduler.")
    parser.add_argument("--devices", type=int, default=-1, help="Number GPU devices to use (-1: all).")
    parser.add_argument("--n_debug_steps", type=int, default=0, help="Number of debug steps.")
    parser.add_argument("--log_every_n_steps", type=int, default=100, help="Log every n steps.")
    parser.add_argument("--val_check_interval", type=int, default=5000, help="Validation check interval.")
    parser.add_argument("--resume_from_last_checkpoint", action=argparse.BooleanOptionalAction, default=False, help="Resume training from the last checkpoint.")
    parser.add_argument("--model_from_checkpoint", type=str, default="", help="Load model from defined checkpoint.")
    parser.add_argument("--validate_extern", action=argparse.BooleanOptionalAction, default=False, help="Long training with a separate validation process.")
    parser.add_argument("--final_full_cube_inference", action=argparse.BooleanOptionalAction, default=True, help="Run full-cube inference/evaluation when training ends.")
    parser.add_argument("--distributed", action=argparse.BooleanOptionalAction, default=False, help="Use distributed training.")
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=False, help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb_project", type=str, default="banis", help="W&B project name.")
    parser.add_argument("--wandb_entity", type=str, default="", help="W&B entity/team (optional).")
    parser.add_argument(
        "--wandb_mode",
        type=str,
        default="online",
        choices=["online", "offline", "disabled"],
        help="W&B sync mode. Default is online.",
    )
    parser.add_argument(
        "--wandb_sync_tensorboard",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also ask W&B to sync TensorBoard event files. Usually unnecessary when W&B logger is enabled.",
    )
    parser.add_argument("--wandb_run_name", type=str, default="", help="Custom W&B run name (defaults to exp_name).")
    parser.add_argument("--wandb_group", type=str, default="", help="W&B group name (optional).")
    parser.add_argument("--wandb_tags", nargs="*", default=[], help="W&B tags.")
    parser.add_argument("--wandb_log_model", action=argparse.BooleanOptionalAction, default=False, help="Log checkpoints as W&B artifacts.")
    parser.add_argument("--wandb_run_id", type=str, default="", help="Existing W&B run id for resume/continuation.")
    parser.add_argument("--wandb_resume", type=str, default="", choices=["", "allow", "must", "never", "auto"], help="W&B resume policy when --wandb_run_id is set.")

    # Data arguments
    parser.add_argument("--base_data_path", type=str, default="/cajal/nvmescratch/projects/NISB/", help="Base path for the dataset.")
    parser.add_argument("--data_setting", type=str, nargs="+", default=["base"], help="Data setting identifier(s). Can specify multiple datasets to train together.")
    parser.add_argument("--real_data_path", type=str, default="/cajal/scratch/projects/misc/mdraw/data/funke/zebrafinch/training/", help="Path to the real dataset. See https://colab.research.google.com/github/funkelab/lsd/blob/master/lsd/tutorial/notebooks/lsd_data_download.ipynb ")
    parser.add_argument("--synthetic", type=float, default=1.0, help="Ratio of synthetic data to real data.")
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True, help="Use augmentations")
    parser.add_argument("--drop_slice_prob", type=float, default=0.05, help="Probability of dropping a slice during augmentation.")
    parser.add_argument("--shift_slice_prob", type=float, default=0.05, help="Probability of shifting a slice during augmentation.")
    parser.add_argument("--intensity_aug", action=argparse.BooleanOptionalAction, default=True, help="Whether to apply intensity augmentation.")
    parser.add_argument("--noise_scale", type=float, default=0.5, help="Scale of the noise to be added during augmentation.")
    parser.add_argument("--affine", type=float, default=0.5, help="Affine transformation probability.")
    parser.add_argument("--affine_scale", type=float, default=0.2, help="Scale for affine augmentation.")
    parser.add_argument("--affine_shear", type=float, default=0.5, help="Shear for affine augmentation.")
    parser.add_argument("--shift_magnitude", type=int, default=10, help="Shift augmentation magnitude (voxels).")
    parser.add_argument("--mul_int", type=float, default=0.1, help="Multiplicative augmentation intensity.")
    parser.add_argument("--add_int", type=float, default=0.1, help="Additive augmentation intensity.")
    parser.add_argument(
        "--ensure_foreground",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resample patches until they contain at least one foreground voxel (seg > 0).",
    )
    parser.add_argument(
        "--max_sampling_attempts",
        type=int,
        default=100,
        help="Maximum patch resampling attempts when foreground is required.",
    )

    # Model arguments
    parser.add_argument("--long_range", type=int, default=10, help="Long range affinities (voxels).")
    parser.add_argument("--model_id", type=str, default="S", help="Identifier for the mednext model architecture.")
    parser.add_argument("--kernel_size", type=int, default=3, help="Kernel size for the convolutional layers.")
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True, help="Whether to compile the model using torch.compile.")
    parser.add_argument("--eval_ranges", type=float, nargs="+", default=torch.sigmoid(torch.tensor(list(range(-1, 12))).double() * 0.2).numpy().round(4).tolist(), help="List of evaluation thresholds.")
    parser.add_argument("--small_size", type=int, default=128, help="Size of the patches.")

    # make it so that adding it makes it true
    parser.add_argument("--sdt", action="store_true", help="Whether to train to predict SDT.")
    parser.add_argument("--sdt_loss_weight", type=float, default=1.0, help="Weight of the SDT loss.")

    return parser.parse_args()


if __name__ == "__main__":
    main()
