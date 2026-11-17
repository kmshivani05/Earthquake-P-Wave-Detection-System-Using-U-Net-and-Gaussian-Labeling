"""
CLI script: run inference on a seismic waveform using a trained checkpoint.

Usage:
    python scripts/predict.py --checkpoint checkpoints/best_model.pt --record-index 0
    python scripts/predict.py --checkpoint checkpoints/best_model.pt --config configs/baseline.yaml --record-index 15
"""

import argparse
import logging
from pathlib import Path
import sys
import h5py
import numpy as np

# Ensure src/ is importable
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

from earthquake_detection.config import load_config
from earthquake_detection.data.gaussian_labels import generate_gaussian_target
from earthquake_detection.inference.predict import SeismicPredictor
from earthquake_detection.visualization.plots import plot_prediction_alignment

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Predict P-wave arrival using trained model.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained model checkpoint (.pt).")
    parser.add_argument("--config", type=str, default="configs/baseline.yaml", help="Path to config YAML.")
    parser.add_argument("--record-index", type=int, default=0, help="Record index in HDF5 archive.")
    parser.add_argument("--threshold", type=float, default=None, help="Detection confidence threshold.")
    parser.add_argument("--save-plot", type=str, default="results/prediction_sample.png", help="Path to save plot.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    predictor = SeismicPredictor(checkpoint_path=args.checkpoint, config=cfg)

    # Load test waveform
    h5_path = Path(cfg.data.path)
    with h5py.File(h5_path, "r") as f:
        trace = np.array(f[cfg.data.traces_key][args.record_index], dtype=np.float32)
        true_p = int(f[cfg.data.p_arrival_key][args.record_index]) if cfg.data.p_arrival_key in f else None

    # Predict
    result, scores = predictor.predict(trace, threshold=args.threshold)

    print("\n" + "=" * 55)
    print(f"       INFERENCE RESULT (RECORD #{args.record_index:04d})       ")
    print("=" * 55)
    print(f"Ground Truth P-Arrival : {true_p} samples")
    print(f"Predicted P-Arrival    : {result.arrival_index} samples")
    print(f"Peak Confidence Score  : {result.confidence:.4f}")
    print(f"Detection Valid        : {result.has_detection}")
    if true_p is not None and result.arrival_index is not None:
        err = result.arrival_index - true_p
        print(f"Timing Error (samples) : {err:+d} samples")
        if cfg.data.sampling_rate is not None:
            err_ms = (err / cfg.data.sampling_rate) * 1000.0
            print(f"Timing Error (ms)      : {err_ms:+.1f} ms")
    print("=" * 55)

    # Visualization
    target = None
    if true_p is not None:
        target = generate_gaussian_target(
            waveform_length=len(scores),
            arrival_index=true_p,
            sigma=cfg.gaussian.sigma,
        )

    out_path = Path(args.save_plot)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plot_prediction_alignment(
        waveform=trace,
        target=target,
        predicted_scores=scores,
        actual_p_arrival=true_p,
        predicted_p_arrival=result.arrival_index,
        sampling_rate=cfg.data.sampling_rate,
        save_path=out_path,
    )
    print(f"\nPrediction alignment plot saved to: {out_path}\n")


if __name__ == "__main__":
    main()
