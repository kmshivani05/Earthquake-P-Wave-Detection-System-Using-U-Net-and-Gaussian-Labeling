"""
CLI script: visualize a seismic waveform record with P and S arrival markers.

Usage:
    python scripts/visualize_waveform.py --record-index 0
    python scripts/visualize_waveform.py --config configs/config.yaml --record-index 42
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
from earthquake_detection.data.gaussian_labels import generate_gaussian_target, plot_gaussian_label_alignment
from earthquake_detection.visualization.plots import plot_three_channel_waveform

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Visualize seismic record and phase picks.")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file.")
    parser.add_argument("--record-index", type=int, default=0, help="Record index in HDF5 file.")
    parser.add_argument("--save-dir", type=str, default="results/visualizations", help="Output directory for plots.")
    args = parser.parse_args()

    config_path = Path(args.config)
    if config_path.is_file():
        cfg = load_config(config_path)
        ds_path = cfg.data.path
        sampling_rate = cfg.data.sampling_rate
        sigma = cfg.gaussian.sigma
        p_key = cfg.data.p_arrival_key
        s_key = cfg.data.s_arrival_key
        traces_key = cfg.data.traces_key
    else:
        ds_path = "STEAD/test.hdf5"
        sampling_rate = 100.0
        sigma = 20.0
        p_key = "p_arrival"
        s_key = "s_arrival"
        traces_key = "traces"

    h5_file = Path(ds_path)
    if not h5_file.is_file():
        raise FileNotFoundError(f"Dataset file not found: {h5_file}")

    with h5py.File(h5_file, "r") as f:
        total = len(f[traces_key])
        if args.record_index < 0 or args.record_index >= total:
            raise IndexError(f"Record index {args.record_index} out of bounds [0, {total-1}]")

        trace = np.array(f[traces_key][args.record_index], dtype=np.float32)
        p_arrival = int(f[p_key][args.record_index]) if p_key in f else None
        s_arrival = int(f[s_key][args.record_index]) if s_key in f else None

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 1. Plot 3-channel waveform
    out_3c = save_dir / f"waveform_record_{args.record_index:04d}_3c.png"
    plot_three_channel_waveform(
        waveform=trace,
        p_arrival=p_arrival,
        s_arrival=s_arrival,
        sampling_rate=sampling_rate,
        title=f"Record #{args.record_index:04d} - 3-Channel Waveform with Phase Annotations",
        save_path=out_3c,
    )
    logger.info(f"Saved 3-channel plot to: {out_3c}")

    # 2. Plot Gaussian target alignment
    z_channel = trace[:, 0] if trace.shape[1] == 3 else trace[0]
    target = generate_gaussian_target(
        waveform_length=len(z_channel),
        arrival_index=p_arrival,
        sigma=sigma,
    )
    out_gauss = save_dir / f"gaussian_target_record_{args.record_index:04d}.png"
    plot_gaussian_label_alignment(
        waveform=z_channel,
        target=target,
        arrival_index=p_arrival or 0,
        title=f"Record #{args.record_index:04d} - P-Wave Onset & Gaussian Target (sigma={sigma})",
        save_path=str(out_gauss),
    )
    logger.info(f"Saved Gaussian alignment plot to: {out_gauss}")


if __name__ == "__main__":
    main()
