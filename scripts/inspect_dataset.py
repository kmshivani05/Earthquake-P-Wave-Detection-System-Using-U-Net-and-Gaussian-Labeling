"""
CLI script: inspect and audit HDF5 seismic dataset.

Usage:
    python scripts/inspect_dataset.py --config configs/config.yaml
    python scripts/inspect_dataset.py --dataset-path STEAD/test.hdf5 --output results/audit_report.json
"""

import argparse
import logging
from pathlib import Path
import sys

# Ensure src/ is importable
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

from earthquake_detection.config import load_config
from earthquake_detection.data.dataset_audit import audit_hdf5_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Inspect and audit HDF5 seismic dataset.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Optional override for HDF5 dataset path.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/dataset_audit_report.json",
        help="Path to save the JSON audit report.",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Optional cap on records scanned for fast inspection.",
    )

    args = parser.parse_args()

    # Load configuration if config file exists
    config_file = Path(args.config)
    if config_file.is_file():
        cfg = load_config(config_file)
        ds_path = args.dataset_path or cfg.data.path
        traces_k = cfg.data.traces_key
        p_k = cfg.data.p_arrival_key
        s_k = cfg.data.s_arrival_key
        meta_k = cfg.data.metadata_key
    else:
        ds_path = args.dataset_path or "STEAD/test.hdf5"
        traces_k = "traces"
        p_k = "p_arrival"
        s_k = "s_arrival"
        meta_k = "metadata"

    report = audit_hdf5_dataset(
        file_path=ds_path,
        traces_key=traces_k,
        p_arrival_key=p_k,
        s_arrival_key=s_k,
        metadata_key=meta_k,
        max_scan_records=args.max_records,
    )

    print("\n" + "=" * 60)
    print("                 DATASET AUDIT SUMMARY                ")
    print("=" * 60)
    print(f"File Path           : {report.file_path}")
    print(f"File Size           : {report.file_size_bytes / 1e6:.2f} MB")
    print(f"Keys Available      : {report.keys}")
    print(f"Number of Records   : {report.num_records:,}")
    print(f"Waveform Dimensions : {report.waveform_shape} ({report.waveform_dtype})")
    print(f"P-Arrival Range     : [{report.p_arrival_stats.get('min')}, {report.p_arrival_stats.get('max')}]")
    print(f"P-Arrival Median    : {report.p_arrival_stats.get('median')}")
    print(f"S-Arrival Range     : [{report.s_arrival_stats.get('min')}, {report.s_arrival_stats.get('max')}]")
    print(f"S-Arrival Median    : {report.s_arrival_stats.get('median')}")
    print(f"Signal Min / Max    : {report.waveform_signal_stats['min_amplitude']:.2f} / {report.waveform_signal_stats['max_amplitude']:.2f}")
    print(f"Signal Mean / Std   : {report.waveform_signal_stats['mean_amplitude']:.4f} / {report.waveform_signal_stats['std_amplitude']:.4f}")
    print(f"NaN / Inf Records   : {report.anomaly_counts['records_with_nan']} / {report.anomaly_counts['records_with_inf']}")
    print("=" * 60)

    report.save(args.output)
    print(f"\nDetailed report saved to: {args.output}\n")


if __name__ == "__main__":
    main()
