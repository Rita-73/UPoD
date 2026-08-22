"""
Common utilities for trainers.
Provides shared functionality to reduce code duplication across trainers.
"""
import os
import random
import sys
from datetime import timedelta
from typing import Optional, Dict, Any, List

import logging

import numpy as np
import torch
import wandb
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed Python / NumPy / PyTorch RNGs for reproducible training.

    Data-layer seeds (e.g. `random.seed(42)` inside dataset __init__) run before
    this call and stay deterministic across seeds — that protects train/test
    splits. Anything after this call (model init, dropout, shuffle, negative
    sampling at __getitem__ time) varies with `seed`.

    Args:
        seed: Random seed.
        deterministic: If True, enable cudnn deterministic mode. Slower; only
            use when bit-exact reproducibility matters (rarely needed for the
            seed-variance matrix we run for benchmarks).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def gpu_supports_bf16() -> bool:
    """Return True only when bf16 GEMM is expected to work (Ampere+, SM >= 8.0).

    torch.cuda.is_bf16_supported() can be True on older Pascal GPUs while CUBLAS
    still rejects CUDA_R_16BF (TITAN X on this cluster).
    """
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    if major < 8:
        return False
    return torch.cuda.is_bf16_supported()


def setup_accelerator(
    split_batches: bool = True,
    gradient_accumulate_every: int = 1,
    amp: bool = False,
    mixed_precision_type: str = "fp16",
) -> Accelerator:
    """
    Setup and return an Accelerator instance.

    Args:
        split_batches: Whether to split batches across devices
        gradient_accumulate_every: Gradient accumulation steps
        amp: Whether to use automatic mixed precision
        mixed_precision_type: Type of mixed precision ("fp16" or "bf16")

    Returns:
        Configured Accelerator instance
    """
    resolved_amp = amp
    resolved_mp = mixed_precision_type
    if amp and mixed_precision_type == "bf16" and not gpu_supports_bf16():
        cap = torch.cuda.get_device_capability() if torch.cuda.is_available() else None
        logging.getLogger(__name__).warning(
            "bf16 is not supported on this GPU (capability=%s); falling back to fp32 training.",
            cap,
        )
        resolved_amp = False
        resolved_mp = "no"

    # NCCL collective timeout: the default 600s is too short for 1.7B SFT. A slow
    # per-epoch checkpoint save under host IO contention (e.g. two 4-GPU jobs sharing
    # one box) can starve a single rank past 600s, which trips the NCCL watchdog and
    # kills the whole process group — this is exactly what crashed the toys run
    # (see memory: onerec-sft-nccl-crash). 30 min gives ample slack with no downside.
    nccl_kwargs = InitProcessGroupKwargs(timeout=timedelta(minutes=30))
    return Accelerator(
        split_batches=split_batches,
        gradient_accumulation_steps=gradient_accumulate_every,
        mixed_precision=resolved_mp if resolved_amp else "no",
        kwargs_handlers=[nccl_kwargs],
    )


def setup_wandb(
    project: str,
    run_name: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    step_metrics: Optional[Dict[str, str]] = None,
) -> None:
    """
    Setup Weights & Biases logging.

    Args:
        project: WandB project name
        run_name: Run name (auto-generated if None)
        config: Configuration dict to log
        step_metrics: Dict mapping metric prefix to step metric name
                     e.g., {"train/*": "global_step", "eval/*": "epoch"}
    """
    wandb.login()
    wandb.init(
        project=project,
        name=run_name,
        config=config,
    )

    # Define step metrics for better visualization
    if step_metrics:
        for metric_pattern, step_metric in step_metrics.items():
            wandb.define_metric(metric_pattern, step_metric=step_metric)


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any] = None,
    epoch: Optional[int] = None,
    step: Optional[int] = None,
    extra: Optional[Dict[str, Any]] = None,
    logger=None,
) -> None:
    """
    Save a training checkpoint.

    Args:
        path: Path to save checkpoint
        model: Model to save
        optimizer: Optimizer to save
        scheduler: Optional learning rate scheduler
        epoch: Current epoch number
        step: Current step number
        extra: Extra data to save in checkpoint
        logger: Optional logger for info message
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }

    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()
    if epoch is not None:
        state["epoch"] = epoch
    if step is not None:
        state["step"] = step
    if extra is not None:
        state.update(extra)

    torch.save(state, path)

    if logger:
        logger.info(f"Saved checkpoint to {path}")


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: torch.device = torch.device("cpu"),
    logger=None,
) -> Dict[str, Any]:
    """
    Load a training checkpoint.

    Args:
        path: Path to checkpoint
        model: Model to load weights into
        optimizer: Optional optimizer to load state into
        scheduler: Optional scheduler to load state into
        device: Device to load checkpoint to
        logger: Optional logger for info message

    Returns:
        Dict containing checkpoint data (epoch, step, etc.)
    """
    if logger:
        logger.info(f"Loading checkpoint from {path}")

    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model"])

    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])

    return checkpoint
