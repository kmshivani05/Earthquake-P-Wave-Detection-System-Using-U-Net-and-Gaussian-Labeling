"""
Seismic dataset inspection and audit module.

Safely examines HDF5 dataset structure, data types, dimensions, metadata,
arrival label statistics, anomaly rates, and waveform signal statistics
without loading multi-gigabyte tensors into memory.
"""

from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import h5py
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class DatasetAuditReport:
    """Structured report summarizing dataset inspection results."""
    file_path: str
    file_size_bytes: int
    attributes: Dict[str, Any]
    keys: List[str]
    datasets: Dict[str, Dict[str, Any]]
    num_records: int
    waveform_shape: List[int]
    waveform_dtype: str
    p_arrival_stats: Dict[str, Any]
    s_arrival_stats: Dict[str, Any]
    anomaly_counts: Dict[str, int]
    waveform_signal_stats: Dict[str, float]
    metadata_samples: List[str]

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(asdict(self), indent=indent, default=str)

    def save(self, output_path: Union[str, Path]) -> None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            f.write(self.to_json())
        logger.info(f"Audit report saved to: {out}")


def audit_hdf5_dataset(
    file_path: Union[str, Path],
    traces_key: str = "traces",
    p_arrival_key: str = "p_arrival",
    s_arrival_key: str = "s_arrival",
    metadata_key: str = "metadata",
    batch_size: int = 500,
    max_scan_records: Optional[int] = None,
) -> DatasetAuditReport:
    """Inspect and audit an HDF5 seismic dataset in streaming batches.

    Args:
        file_path: Path to the .hdf5 file.
        traces_key: HDF5 key for waveform array.
        p_arrival_key: HDF5 key for P-wave arrival indices.
        s_arrival_key: HDF5 key for S-wave arrival indices.
        metadata_key: HDF5 key for record metadata.
        batch_size: Batch size for streaming anomaly detection.
        max_scan_records: Optional cap on records to scan for faster auditing.

    Returns:
        DatasetAuditReport containing full inspection findings.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"HDF5 file does not exist: {path}")

    file_size = path.stat().st_size
    logger.info(f"Auditing HDF5 dataset: {path} ({file_size / 1e6:.2f} MB)")

    with h5py.File(path, "r") as h5:
        attrs = {k: str(v) for k, v in h5.attrs.items()}
        keys = list(h5.keys())

        datasets_info = {}
        for k in keys:
            ds = h5[k]
            datasets_info[k] = {
                "shape": list(ds.shape),
                "dtype": str(ds.dtype),
                "size": ds.size,
            }

        if traces_key not in h5:
            raise KeyError(f"Expected traces dataset '{traces_key}' not found in HDF5 keys: {keys}")

        traces_ds = h5[traces_key]
        num_records = traces_ds.shape[0]
        waveform_shape = list(traces_ds.shape[1:])
        waveform_dtype = str(traces_ds.dtype)
        T = waveform_shape[0]

        scan_count = num_records if max_scan_records is None else min(num_records, max_scan_records)

        # Audit P-arrival labels
        p_stats: Dict[str, Any] = {}
        if p_arrival_key in h5:
            p_arr = h5[p_arrival_key][:scan_count]
            valid_p = p_arr[(p_arr >= 0) & (p_arr < T)]
            p_stats = {
                "total_inspected": int(scan_count),
                "valid_count": int(len(valid_p)),
                "missing_or_negative_count": int(np.sum(p_arr < 0)),
                "out_of_bounds_count": int(np.sum(p_arr >= T)),
                "min": int(np.min(valid_p)) if len(valid_p) > 0 else None,
                "max": int(np.max(valid_p)) if len(valid_p) > 0 else None,
                "median": float(np.median(valid_p)) if len(valid_p) > 0 else None,
                "mean": float(np.mean(valid_p)) if len(valid_p) > 0 else None,
                "std": float(np.std(valid_p)) if len(valid_p) > 0 else None,
            }

        # Audit S-arrival labels
        s_stats: Dict[str, Any] = {}
        if s_arrival_key in h5:
            s_arr = h5[s_arrival_key][:scan_count]
            valid_s = s_arr[(s_arr >= 0) & (s_arr < T)]
            s_stats = {
                "total_inspected": int(scan_count),
                "valid_count": int(len(valid_s)),
                "missing_or_negative_count": int(np.sum(s_arr < 0)),
                "out_of_bounds_count": int(np.sum(s_arr >= T)),
                "min": int(np.min(valid_s)) if len(valid_s) > 0 else None,
                "max": int(np.max(valid_s)) if len(valid_s) > 0 else None,
                "median": float(np.median(valid_s)) if len(valid_s) > 0 else None,
                "mean": float(np.mean(valid_s)) if len(valid_s) > 0 else None,
                "std": float(np.std(valid_s)) if len(valid_s) > 0 else None,
            }

        # Audit metadata
        metadata_samples = []
        if metadata_key in h5:
            meta_ds = h5[metadata_key]
            sample_count = min(10, len(meta_ds))
            for i in range(sample_count):
                val = meta_ds[i]
                if isinstance(val, (bytes, bytearray)):
                    val = val.decode("utf-8", errors="replace")
                metadata_samples.append(str(val))

        # Streaming scan for NaNs, Infs, and signal statistics
        nan_records = 0
        inf_records = 0
        global_min = float("inf")
        global_max = float("-inf")
        sum_vals = 0.0
        sum_sq = 0.0
        total_samples_scanned = 0

        for start_idx in range(0, scan_count, batch_size):
            end_idx = min(start_idx + batch_size, scan_count)
            batch = traces_ds[start_idx:end_idx]

            # Check NaNs and Infs per record
            has_nan = np.isnan(batch).reshape(batch.shape[0], -1).any(axis=1)
            has_inf = np.isinf(batch).reshape(batch.shape[0], -1).any(axis=1)
            nan_records += int(np.sum(has_nan))
            inf_records += int(np.sum(has_inf))

            # Finite statistics
            finite_mask = np.isfinite(batch)
            if np.any(finite_mask):
                finite_vals = batch[finite_mask]
                batch_min = float(np.min(finite_vals))
                batch_max = float(np.max(finite_vals))
                if batch_min < global_min:
                    global_min = batch_min
                if batch_max > global_max:
                    global_max = batch_max

                sum_vals += float(np.sum(finite_vals, dtype=np.float64))
                sum_sq += float(np.sum(finite_vals.astype(np.float64) ** 2))
                total_samples_scanned += int(len(finite_vals))

        mean_val = sum_vals / total_samples_scanned if total_samples_scanned > 0 else 0.0
        var_val = (sum_sq / total_samples_scanned) - (mean_val ** 2) if total_samples_scanned > 0 else 0.0
        std_val = float(np.sqrt(max(0.0, var_val)))

        anomaly_counts = {
            "records_with_nan": nan_records,
            "records_with_inf": inf_records,
            "invalid_p_arrival": p_stats.get("out_of_bounds_count", 0) + p_stats.get("missing_or_negative_count", 0),
            "invalid_s_arrival": s_stats.get("out_of_bounds_count", 0) + s_stats.get("missing_or_negative_count", 0),
        }

        signal_stats = {
            "min_amplitude": global_min if global_min != float("inf") else 0.0,
            "max_amplitude": global_max if global_max != float("-inf") else 0.0,
            "mean_amplitude": float(mean_val),
            "std_amplitude": float(std_val),
        }

    return DatasetAuditReport(
        file_path=str(path.resolve()),
        file_size_bytes=file_size,
        attributes=attrs,
        keys=keys,
        datasets=datasets_info,
        num_records=num_records,
        waveform_shape=waveform_shape,
        waveform_dtype=waveform_dtype,
        p_arrival_stats=p_stats,
        s_arrival_stats=s_stats,
        anomaly_counts=anomaly_counts,
        waveform_signal_stats=signal_stats,
        metadata_samples=metadata_samples,
    )
