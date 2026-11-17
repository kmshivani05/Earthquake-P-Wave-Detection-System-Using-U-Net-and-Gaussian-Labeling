"""
Phase arrival decoding and temporal peak extraction.

Converts continuous model probability / score outputs into discrete P-wave arrival
sample indices with calibrated confidence filtering and prominence thresholding.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import numpy as np
from scipy import signal
import torch


@dataclass
class DetectionResult:
    """Detailed arrival detection outcome."""
    arrival_index: Optional[int]    # Sample index of picked arrival, or None if no detection
    confidence: float               # Peak score [0, 1]
    has_detection: bool             # True if confidence >= threshold
    all_peaks: List[int]            # All candidate peak indices detected
    all_scores: List[float]         # Peak scores corresponding to all_peaks


def decode_p_arrival(
    scores: Union[np.ndarray, torch.Tensor],
    threshold: float = 0.5,
    prominence: float = 0.2,
    min_distance: int = 50,
) -> DetectionResult:
    """Decode discrete P-wave arrival from continuous temporal model output.

    Args:
        scores: 1D array or tensor of temporal scores of length T.
        threshold: Minimum peak amplitude to qualify as an earthquake detection.
        prominence: Minimum topographic prominence required for peak detection.
        min_distance: Minimum sample separation between neighboring peaks.

    Returns:
        DetectionResult object containing arrival index, confidence, and candidates.
    """
    if isinstance(scores, torch.Tensor):
        arr = scores.squeeze().detach().cpu().numpy()
    else:
        arr = np.squeeze(scores)

    if arr.ndim != 1:
        raise ValueError(f"Expected 1D temporal scores, got shape {arr.shape}")

    T = len(arr)
    if T == 0:
        return DetectionResult(None, 0.0, False, [], [])

    # Find prominent peaks
    peaks, properties = signal.find_peaks(
        arr,
        height=threshold,
        prominence=prominence,
        distance=min_distance,
    )

    if len(peaks) == 0:
        # Fallback: check global argmax
        max_idx = int(np.argmax(arr))
        max_val = float(arr[max_idx])
        if max_val >= threshold:
            return DetectionResult(
                arrival_index=max_idx,
                confidence=max_val,
                has_detection=True,
                all_peaks=[max_idx],
                all_scores=[max_val],
            )
        return DetectionResult(
            arrival_index=None,
            confidence=max_val,
            has_detection=False,
            all_peaks=[],
            all_scores=[],
        )

    peak_heights = properties["peak_heights"]
    best_peak_pos = int(np.argmax(peak_heights))
    best_arrival = int(peaks[best_peak_pos])
    best_conf = float(peak_heights[best_peak_pos])

    return DetectionResult(
        arrival_index=best_arrival,
        confidence=best_conf,
        has_detection=True,
        all_peaks=[int(p) for p in peaks],
        all_scores=[float(h) for h in peak_heights],
    )


def reconcile_windowed_predictions(
    window_scores: List[np.ndarray],
    global_starts: List[int],
    total_length: int,
    aggregation: str = "max",
) -> np.ndarray:
    """Reconcile overlapping window predictions into a single global score trace.

    Args:
        window_scores: List of 1D score arrays per window.
        global_starts: Global start sample index for each window.
        total_length: Total length T of the complete recording.
        aggregation: 'max' for maximum overlapping score, 'mean' for running average.

    Returns:
        1D numpy array of shape (total_length,) with reconciled temporal scores.
    """
    if aggregation == "max":
        global_trace = np.zeros(total_length, dtype=np.float32)
        for scores, start in zip(window_scores, global_starts):
            end = min(start + len(scores), total_length)
            sub_len = end - start
            global_trace[start:end] = np.maximum(global_trace[start:end], scores[:sub_len])
        return global_trace

    elif aggregation == "mean":
        sum_trace = np.zeros(total_length, dtype=np.float64)
        count_trace = np.zeros(total_length, dtype=np.float64)
        for scores, start in zip(window_scores, global_starts):
            end = min(start + len(scores), total_length)
            sub_len = end - start
            sum_trace[start:end] += scores[:sub_len]
            count_trace[start:end] += 1.0

        count_trace = np.where(count_trace == 0, 1.0, count_trace)
        return (sum_trace / count_trace).astype(np.float32)

    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")
