"""
CLI script: train 1D U-Net++ seismic phase picker.

Usage:
    python scripts/train_model.py --config configs/baseline.yaml
    python scripts/train_model.py --config configs/enhanced.yaml --epochs 10 --batch-size 8
"""

import argparse
import logging
from pathlib import Path
import sys

# Ensure src/ is importable
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

from earthquake_detection.config import load_config
from earthquake_detection.training.train import run_training
from earthquake_detection.visualization.plots import plot_training_history

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train seismic phase picking model.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/baseline.yaml",
        help="Path to YAML training configuration file.",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size.")
    parser.add_argument("--learning-rate", type=float, default=None, help="Override learning rate.")
    parser.add_argument("--records", type=int, default=None, help="Override number of dataset records to use.")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # CLI Overrides
    if args.epochs is not None:
        cfg.training.epochs = args.epochs
    if args.batch_size is not None:
        cfg.training.batch_size = args.batch_size
    if args.learning_rate is not None:
        cfg.training.learning_rate = args.learning_rate
    if args.records is not None:
        cfg.data.num_records = args.records

    logger.info(f"Loaded config from {args.config} for model {cfg.model.architecture}")
    history = run_training(cfg)

    # Plot loss history
    res_dir = Path(cfg.evaluation.results_dir)
    res_dir.mkdir(parents=True, exist_ok=True)
    loss_plot_path = res_dir / "loss_curves.png"
    plot_training_history(history.train_loss, history.val_loss, save_path=loss_plot_path)
    logger.info(f"Training completed. Loss curve plot saved to: {loss_plot_path}")


if __name__ == "__main__":
    main()
