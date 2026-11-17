"""
Seismic P-wave inference and arrival time prediction.

Loads trained checkpoints, processes raw 3-channel seismograms, and predicts
discrete arrival times and continuous detection confidence curves.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn

from earthquake_detection.config import ExperimentConfig
from earthquake_detection.inference.decoder import DetectionResult, decode_p_arrival
from earthquake_detection.models import build_model
from earthquake_detection.preprocessing.filtering import apply_bandpass_filter
from earthquake_detection.preprocessing.normalization import normalize_waveform
from earthquake_detection.preprocessing.displacement import compute_displacement_and_magnitude


class SeismicPredictor:
    """Inference engine for 1D U-Net++ seismic phase picking."""

    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        config: ExperimentConfig,
        device: Optional[Union[str, torch.device]] = None,
    ):
        """Initialize predictor.

        Args:
            checkpoint_path: Path to saved model checkpoint (.pt).
            config: Experiment configuration matching model architecture and preprocessing.
            device: Target torch device.
        """
        self.config = config
        if device is None:
            self.device = config.get_torch_device()
        else:
            self.device = torch.device(device) if isinstance(device, str) else device

        # Inspect checkpoint and auto-adapt model configuration if needed
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)

        # Auto-align architectural flags from checkpoint state dictionary
        has_dilated = any("dilated_block" in k for k in state_dict.keys())
        has_attention = any("attention_block" in k for k in state_dict.keys())
        has_deep_sup = any("heads." in k for k in state_dict.keys())

        config.model.dilated_convolutions = has_dilated
        config.model.attention = has_attention
        config.model.deep_supervision = has_deep_sup

        # Build model and load weights
        self.model = build_model(config.model)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def preprocess_trace(self, waveform: np.ndarray) -> torch.Tensor:
        """Apply configured bandpass filter and normalization to raw 3-channel trace.

        Args:
            waveform: Raw array of shape (T, 3) or (3, T).

        Returns:
            Preprocessed tensor of shape (1, C, T) ready for model input.
        """
        arr = np.array(waveform, dtype=np.float32)

        # Transpose from (T, 3) to (3, T) if needed
        if arr.ndim == 2 and arr.shape[0] > arr.shape[1]:
            trace = arr.T.copy()
        else:
            trace = arr.copy()

        # 1. Bandpass Filtering
        if self.config.preprocessing.filtering and self.config.data.sampling_rate is not None:
            trace = apply_bandpass_filter(
                data=trace,
                lowcut=self.config.preprocessing.lowcut,
                highcut=self.config.preprocessing.highcut,
                sampling_rate=self.config.data.sampling_rate,
                order=self.config.preprocessing.filter_order,
                zero_phase=self.config.preprocessing.zero_phase,
                axis=-1,
            )

        # 2. Normalization
        trace = normalize_waveform(
            data=trace,
            method=self.config.preprocessing.normalization,
            channel_axis=0,
        )

        # 3. Optional Displacement Magnitude
        if (
            self.config.preprocessing.displacement.enabled
            and self.config.data.sampling_rate is not None
        ):
            _, _, disp_mag = compute_displacement_and_magnitude(
                trace,
                sampling_rate=self.config.data.sampling_rate,
                channel_axis=0,
            )
            disp_mag = normalize_waveform(disp_mag[np.newaxis, :], method="robust", channel_axis=0)
            trace = np.concatenate([trace, disp_mag], axis=0)

        tensor = torch.from_numpy(trace).unsqueeze(0).float()
        return tensor

    @torch.no_grad()
    def predict(
        self,
        waveform: np.ndarray,
        threshold: Optional[float] = None,
        prominence: Optional[float] = None,
    ) -> Tuple[DetectionResult, np.ndarray]:
        """Predict P-wave arrival for a single seismic record.

        Args:
            waveform: Raw 3-channel array of shape (T, 3) or (3, T).
            threshold: Confidence threshold (defaults to config value).
            prominence: Peak prominence threshold.

        Returns:
            (detection_result, continuous_scores_1d)
        """
        thresh = threshold if threshold is not None else self.config.inference.threshold
        prom = prominence if prominence is not None else self.config.inference.prominence
        min_dist = self.config.inference.min_distance

        inp = self.preprocess_trace(waveform).to(self.device)
        pred = self.model(inp)

        if isinstance(pred, list):
            pred = pred[-1]

        scores = pred.squeeze().cpu().numpy()
        scores = np.clip(scores, 0.0, 1.0)

        result = decode_p_arrival(
            scores=scores,
            threshold=thresh,
            prominence=prom,
            min_distance=min_dist,
        )

        return result, scores
