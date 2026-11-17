"""
CLI script: run noise robustness benchmark across SNRs and noise archetypes.

Usage:
    python scripts/run_noise_experiment.py --checkpoint checkpoints/best_model.pt --num-records 20
"""

import argparse
import logging
from pathlib import Path
import sys

# Ensure src/ is importable
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

from earthquake_detection.config import load_config
from earthquake_detection.data.dataset import create_dataset_splits
from earthquake_detection.experiments.noise_robustness import run_noise_benchmark
from earthquake_detection.inference.predict import SeismicPredictor
from earthquake_detection.visualization.plots import plot_noise_robustness_curve

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Evaluate model robustness across noise levels.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt", help="Path to checkpoint.")
    parser.add_argument("--config", type=str, default="configs/baseline.yaml", help="Path to config file.")
    parser.add_argument("--num-records", type=int, default=20, help="Number of records to benchmark.")
    parser.add_argument("--save-plot", type=str, default="results/noise_robustness.png", help="Path to save curve.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ckpt = Path(args.checkpoint)
    if not ckpt.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}. Please train a model first.")

    predictor = SeismicPredictor(checkpoint_path=ckpt, config=cfg)

    # Load test traces
    _, _, test_ds = create_dataset_splits(
        hdf5_path=cfg.data.path,
        train_ratio=cfg.data.train_split,
        val_ratio=cfg.data.val_split,
        test_ratio=cfg.data.test_split,
        num_records=cfg.data.num_records,
        seed=cfg.project.seed,
        apply_filter=False,
        normalization="none",
    )

    n_eval = min(len(test_ds), args.num_records)
    logger.info(f"Preparing {n_eval} test waveforms for noise benchmark...")

    test_waveforms = []
    ground_truth_arrivals = []

    for i in range(n_eval):
        item = test_ds[i]
        test_waveforms.append(item["waveform"].numpy())
        ground_truth_arrivals.append(item["p_arrival"])

    # Run benchmark across clean, 20dB, 10dB, 0dB, -5dB
    results = run_noise_benchmark(
        predictor=predictor,
        test_waveforms=test_waveforms,
        ground_truth_arrivals=ground_truth_arrivals,
        snr_levels_db=[None, 20.0, 10.0, 0.0, -5.0],
        noise_types=["gaussian", "pink"],
        tolerance_samples=cfg.evaluation.timing_tolerance_samples,
        sampling_rate=cfg.data.sampling_rate,
    )

    print("\n" + "=" * 60)
    print("           NOISE ROBUSTNESS BENCHMARK RESULTS           ")
    print("=" * 60)
    for n_type, snr_data in results.items():
        print(f"\n--- Noise Archetype: {n_type.upper()} ---")
        print(f"{'SNR':<10} | {'F1-Score':<10} | {'Precision':<10} | {'Recall':<10} | {'MAE (ms)':<10}")
        print("-" * 55)
        for snr_lbl, metrics in snr_data.items():
            mae_str = f"{metrics.mae_ms:.1f}" if metrics.mae_ms is not None else "N/A"
            print(f"{snr_lbl:<10} | {metrics.f1_score:<10.4f} | {metrics.precision:<10.4f} | {metrics.recall:<10.4f} | {mae_str:<10}")
    print("=" * 60 + "\n")

    # Plot robustness curve
    out_plot = Path(args.save_plot)
    out_plot.parent.mkdir(parents=True, exist_ok=True)
    plot_noise_robustness_curve(results, metric_key="f1_score", save_path=out_plot)
    logger.info(f"Saved noise degradation curve to: {out_plot}")


if __name__ == "__main__":
    main()
