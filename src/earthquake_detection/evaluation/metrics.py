"""
Evaluation metrics for seismic phase arrival detection.

Calculates timing errors (MAE, Median AE, RMSE), detection tolerances,
and phase-picking classification metrics (Precision, Recall, F1-Score).
"""

from dataclasses import asdict, dataclass
import json
from typing import Dict, List, Optional, Union
import numpy as np


@dataclass
class EvaluationMetrics:
    """Comprehensive evaluation metrics report."""
    total_samples_evaluated: int
    true_events_count: int
    total_detections_count: int
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1_score: float
    detection_rate: float
    timing_tolerance_samples: int
    percentage_within_tolerance: float
    # Error statistics computed over True Positives
    mae_samples: Optional[float]
    median_ae_samples: Optional[float]
    rmse_samples: Optional[float]
    std_error_samples: Optional[float]
    # Millisecond equivalents (populated only when sampling_rate is verified)
    sampling_rate_hz: Optional[float]
    timing_tolerance_ms: Optional[float]
    mae_ms: Optional[float]
    median_ae_ms: Optional[float]
    rmse_ms: Optional[float]

    def to_dict(self) -> Dict[str, Union[float, int, None]]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def summary(self) -> str:
        lines = [
            "=" * 50,
            "         SEISMIC PHASE PICKING EVALUATION        ",
            "=" * 50,
            f"Evaluated Records     : {self.total_samples_evaluated}",
            f"True Events           : {self.true_events_count}",
            f"Model Detections      : {self.total_detections_count}",
            "-" * 50,
            f"True Positives (TP)   : {self.true_positives}",
            f"False Positives (FP)  : {self.false_positives}",
            f"False Negatives (FN)  : {self.false_negatives}",
            f"Precision             : {self.precision:.4f}",
            f"Recall                : {self.recall:.4f}",
            f"F1-Score              : {self.f1_score:.4f}",
            f"Detection Rate        : {self.detection_rate:.4f}",
            "-" * 50,
            f"Tolerance (Samples)   : +/-{self.timing_tolerance_samples}",
            f"Within Tolerance      : {self.percentage_within_tolerance:.2f}%",
        ]
        if self.mae_samples is not None:
            lines.extend([
                f"MAE (Samples)         : {self.mae_samples:.2f}",
                f"Median AE (Samples)   : {self.median_ae_samples:.2f}",
                f"RMSE (Samples)        : {self.rmse_samples:.2f}",
            ])
        if self.sampling_rate_hz is not None and self.mae_ms is not None:
            lines.extend([
                "-" * 50,
                f"Sampling Rate         : {self.sampling_rate_hz} Hz",
                f"Tolerance (ms)        : +/-{self.timing_tolerance_ms:.1f} ms",
                f"MAE (ms)              : {self.mae_ms:.2f} ms",
                f"Median AE (ms)        : {self.median_ae_ms:.2f} ms",
                f"RMSE (ms)             : {self.rmse_ms:.2f} ms",
            ])
        lines.append("=" * 50)
        return "\n".join(lines)


def calculate_metrics(
    ground_truth_arrivals: List[Optional[int]],
    predicted_arrivals: List[Optional[int]],
    timing_tolerance_samples: int = 20,
    sampling_rate_hz: Optional[float] = 100.0,
) -> EvaluationMetrics:
    """Compute phase picking accuracy, timing errors, and detection statistics.

    Args:
        ground_truth_arrivals: List of true arrival indices (None if no true event).
        predicted_arrivals: List of predicted arrival indices (None if no detection).
        timing_tolerance_samples: Maximum absolute difference (in samples) for a true positive.
        sampling_rate_hz: Sampling frequency in Hz to compute ms metrics.

    Returns:
        EvaluationMetrics dataclass instance.
    """
    if len(ground_truth_arrivals) != len(predicted_arrivals):
        raise ValueError(
            f"Length mismatch: {len(ground_truth_arrivals)} ground truth vs "
            f"{len(predicted_arrivals)} predictions"
        )

    N = len(ground_truth_arrivals)
    if N == 0:
        raise ValueError("Cannot compute metrics on empty arrival lists.")

    tp = 0
    fp = 0
    fn = 0
    true_events = 0
    total_detections = 0
    sample_errors: List[float] = []

    for gt, pred in zip(ground_truth_arrivals, predicted_arrivals):
        has_gt = gt is not None and gt >= 0
        has_pred = pred is not None and pred >= 0

        if has_gt:
            true_events += 1
        if has_pred:
            total_detections += 1

        if has_gt and has_pred:
            err = float(pred - gt)
            abs_err = abs(err)
            if abs_err <= timing_tolerance_samples:
                tp += 1
                sample_errors.append(abs_err)
            else:
                # Detection was made but outside tolerance
                fp += 1
                fn += 1
        elif has_gt and not has_pred:
            # Missed earthquake arrival
            fn += 1
        elif not has_gt and has_pred:
            # False alarm on noise
            fp += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    detection_rate = total_detections / N if N > 0 else 0.0
    pct_within = (tp / true_events * 100.0) if true_events > 0 else 0.0

    mae_s = float(np.mean(sample_errors)) if len(sample_errors) > 0 else None
    med_s = float(np.median(sample_errors)) if len(sample_errors) > 0 else None
    rmse_s = float(np.sqrt(np.mean(np.array(sample_errors) ** 2))) if len(sample_errors) > 0 else None
    std_s = float(np.std(sample_errors)) if len(sample_errors) > 0 else None

    # Millisecond conversions
    mae_ms = None
    med_ae_ms = None
    rmse_ms = None
    tol_ms = None

    if sampling_rate_hz is not None and sampling_rate_hz > 0:
        ms_factor = 1000.0 / sampling_rate_hz
        tol_ms = timing_tolerance_samples * ms_factor
        if mae_s is not None:
            mae_ms = mae_s * ms_factor
        if med_s is not None:
            med_ae_ms = med_s * ms_factor
        if rmse_s is not None:
            rmse_ms = rmse_s * ms_factor
    else:
        med_ae_ms = None

    return EvaluationMetrics(
        total_samples_evaluated=N,
        true_events_count=true_events,
        total_detections_count=total_detections,
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        precision=precision,
        recall=recall,
        f1_score=f1,
        detection_rate=detection_rate,
        timing_tolerance_samples=timing_tolerance_samples,
        percentage_within_tolerance=pct_within,
        mae_samples=mae_s,
        median_ae_samples=med_s,
        rmse_samples=rmse_s,
        std_error_samples=std_s,
        sampling_rate_hz=sampling_rate_hz,
        timing_tolerance_ms=tol_ms,
        mae_ms=mae_ms,
        median_ae_ms=med_ae_ms if sampling_rate_hz is not None else None,
        rmse_ms=rmse_ms,
    )
