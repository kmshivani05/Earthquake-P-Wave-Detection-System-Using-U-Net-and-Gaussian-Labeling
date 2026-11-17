"""
Main training execution script and pipeline coordinator.
"""

import logging
from pathlib import Path
import random
from typing import Optional
import numpy as np
import torch
from torch.utils.data import DataLoader

from earthquake_detection.config import ExperimentConfig, load_config
from earthquake_detection.data.dataset import create_dataset_splits
from earthquake_detection.models import build_model
from earthquake_detection.training.losses import build_loss
from earthquake_detection.training.trainer import Trainer, TrainingHistory

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Set seeds for reproducibility across random, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_training(config: ExperimentConfig) -> TrainingHistory:
    """Execute complete training workflow defined by configuration."""
    set_seed(config.project.seed)
    device = config.get_torch_device()
    logger.info(f"Target execution device: {device}")

    # Build dataset splits
    logger.info(f"Loading dataset from: {config.data.path}")
    train_ds, val_ds, test_ds = create_dataset_splits(
        hdf5_path=config.data.path,
        train_ratio=config.data.train_split,
        val_ratio=config.data.val_split,
        test_ratio=config.data.test_split,
        num_records=config.data.num_records,
        seed=config.project.seed,
        traces_key=config.data.traces_key,
        p_arrival_key=config.data.p_arrival_key,
        s_arrival_key=config.data.s_arrival_key,
        metadata_key=config.data.metadata_key,
        sampling_rate=config.data.sampling_rate or 100.0,
        gaussian_sigma=config.gaussian.sigma,
        apply_filter=config.preprocessing.filtering,
        lowcut=config.preprocessing.lowcut,
        highcut=config.preprocessing.highcut,
        filter_order=config.preprocessing.filter_order,
        zero_phase=config.preprocessing.zero_phase,
        normalization=config.preprocessing.normalization,
        compute_displacement=config.preprocessing.displacement.enabled,
    )

    logger.info(
        f"Dataset split: {len(train_ds)} train, {len(val_ds)} val, {len(test_ds)} test records"
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # Instantiate model
    model = build_model(config.model)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model architecture: {config.model.architecture} (Trainable params: {total_params:,})")

    # Loss criterion
    criterion = build_loss(
        loss_name=config.training.loss,
        deep_supervision=config.model.deep_supervision,
    )

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    # Scheduler
    scheduler = None
    if config.training.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.training.lr_step_size,
            gamma=config.training.lr_gamma,
        )

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        scheduler=scheduler,
        device=device,
        epochs=config.training.epochs,
        early_stopping_patience=config.training.early_stopping_patience,
        min_delta=config.training.min_delta,
        checkpoint_dir=config.training.checkpoint_dir,
        save_best_only=config.training.save_best_only,
    )

    history = trainer.fit()
    return history
