"""
CLI script: evaluate trained model checkpoint on test dataset split.

Usage:
    python scripts/evaluate_model.py --checkpoint checkpoints/best_model.pt
    python scripts/evaluate_model.py --checkpoint checkpoints/best_model.pt --config configs/baseline.yaml
"""

import argparse
import json
import logging
from pathlib import Path
import sys
import numpy as np

# Ensure src/ is importable
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

from earthquake_detection.config import load_config
from earthquake_detection.data.dataset import create_dataset_splits
from earthquake_detection.evaluation.metrics import calculate_metrics
from earthquake_detection.inference.predict import SeismicPredictor
from earthquake_detection.visualization.plots import plot_error_distribution

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Evaluate model checkpoint on test split.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt).")
    parser.add_argument("--config", type=str, default="configs/baseline.yaml", help="Path to config YAML.")
    parser.add_argument("--num-records", type=int, default=100, help="Number of test records to evaluate.")
    parser.add_argument("--tolerance", type=int, default=None, help="Timing tolerance in samples.")
    parser.add_argument("--output-dir", type=str, default="results/evaluation", help="Directory for output reports.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    tol = args.tolerance or cfg.evaluation.timing_tolerance_samples
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Initialize predictor
    predictor = SeismicPredictor(checkpoint_path=args.checkpoint, config=cfg)

    # Load test split
    _, _, test_ds = create_dataset_splits(
        hdf5_path=cfg.data.path,
        train_ratio=cfg.data.train_split,
        val_ratio=cfg.data.val_split,
        test_ratio=cfg.data.test_split,
        num_records=cfg.data.num_records,
        seed=cfg.project.seed,
        sampling_rate=cfg.data.sampling_rate or 100.0,
        gaussian_sigma=cfg.gaussian.sigma,
        apply_filter=False,  # Predictor handles its own preprocessing
        normalization="none",
    )

    eval_count = min(len(test_ds), args.num_records)
    logger.info(f"Evaluating {eval_count} test records from {cfg.data.path}...")

    ground_truth_arrivals = []
    predicted_arrivals = []
    sample_errors = []

    for i in range(eval_count):
        item = test_ds[i]
        raw_waveform = item["waveform"].numpy()
        true_p = item["p_arrival"]

        det_result, _ = predictor.predict(raw_waveform)

        ground_truth_arrivals.append(true_p if true_p >= 0 else None)
        predicted_arrivals.append(det_result.arrival_index)

        if true_p >= 0 and det_result.arrival_index is not None:
            sample_errors.append(float(det_result.arrival_index - true_p))

    metrics = calculate_metrics(
        ground_truth_arrivals=ground_truth_arrivals,
        predicted_arrivals=predicted_arrivals,
        timing_tolerance_samples=tol,
        sampling_rate_hz=cfg.data.sampling_rate,
    )

    print("\n" + metrics.summary() + "\n")

    # Save JSON report
    report_path = out_dir / "evaluation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(metrics.to_json())
    logger.info(f"Saved evaluation metrics to: {report_path}")

    # Plot error distribution
    if len(sample_errors) > 0:
        err_plot = out_dir / "error_distribution.png"
        plot_error_distribution(
            errors_samples=sample_errors,
            tolerance_samples=tol,
            sampling_rate=cfg.data.sampling_rate,
            save_path=err_plot,
        )
        logger.info(f"Saved error distribution plot to: {err_plot}")


if __name__ == "__main__":
    main()
