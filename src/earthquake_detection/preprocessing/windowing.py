"""
Sliding-window segmentation and arrival index adjustment.

Divides continuous seismic traces into overlapping or contiguous windows,
recalculating local phase arrival indices and assigning null labels when
an arrival falls outside the window.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np


@dataclass
class WindowedSegment:
    """A single segmented window with its global metadata."""
    data: np.ndarray             # Windowed waveform shaped (C, window_length)
    global_start: int            # Start sample index in original trace
    global_end: int              # End sample index in original trace
    local_p_arrival: Optional[int] # Arrival sample index relative to window start, or None


def segment_waveform(
    waveform: np.ndarray,
    window_length: int,
    step_size: int,
    global_p_arrival: Optional[int] = None,
    channel_axis: int = 0,
) -> List[WindowedSegment]:
    """Segment waveform into sliding windows and adjust arrival indices.

    Args:
        waveform: Array of shape (C, T) or (T, C).
        window_length: Length of each window in samples.
        step_size: Stride/step between consecutive windows in samples.
        global_p_arrival: Global arrival sample index in original waveform.
        channel_axis: 0 for (C, T), 1 for (T, C).

    Returns:
        List of WindowedSegment objects.
    """
    if channel_axis == 1:
        wf = np.swapaxes(waveform, 0, 1)
    else:
        wf = waveform

    C, T = wf.shape
    if window_length > T:
        raise ValueError(f"window_length ({window_length}) cannot exceed waveform length ({T})")
    if step_size <= 0:
        raise ValueError(f"step_size must be positive, got {step_size}")

    segments: List[WindowedSegment] = []
    start = 0

    while start + window_length <= T:
        end = start + window_length
        sub_trace = wf[:, start:end].copy()

        local_arrival: Optional[int] = None
        if global_p_arrival is not None and global_p_arrival >= 0:
            if start <= global_p_arrival < end:
                local_arrival = int(global_p_arrival - start)

        segments.append(
            WindowedSegment(
                data=sub_trace,
                global_start=start,
                global_end=end,
                local_p_arrival=local_arrival,
            )
        )
        start += step_size

    return segments
