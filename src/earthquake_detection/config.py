"""Configuration loading, validation, and schema definitions."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import yaml
import torch


@dataclass
class ProjectConfig:
    name: str = "earthquake-p-wave-detection"
    description: str = "P-Wave detection using 1D U-Net++"
    seed: int = 42
    device: str = "auto"


@dataclass
class DataConfig:
    path: str = "STEAD/test.hdf5"
    num_records: Optional[int] = None
    train_split: float = 0.70
    val_split: float = 0.15
    test_split: float = 0.15
    sampling_rate: Optional[float] = 100.0
    num_channels: int = 3
    waveform_length: int = 6000
    p_arrival_key: str = "p_arrival"
    s_arrival_key: str = "s_arrival"
    traces_key: str = "traces"
    metadata_key: str = "metadata"


@dataclass
class DisplacementConfig:
    enabled: bool = False
    detrend: bool = True
    highpass_cutoff: float = 0.1


@dataclass
class PreprocessingConfig:
    filtering: bool = True
    filter_type: str = "butterworth"
    filter_order: int = 4
    lowcut: float = 0.5
    highcut: float = 10.0
    zero_phase: bool = True
    normalization: str = "per_channel"
    displacement: DisplacementConfig = field(default_factory=DisplacementConfig)


@dataclass
class WindowingConfig:
    enabled: bool = False
    window_length: int = 1000
    step_size: int = 500


@dataclass
class GaussianConfig:
    sigma: float = 20.0
    normalize_peak: bool = True


@dataclass
class ModelConfig:
    architecture: str = "unetplusplus_1d"
    in_channels: int = 3
    num_classes: int = 1
    base_channels: int = 32
    depth: int = 4
    dilated_convolutions: bool = False
    dilation_rates: List[int] = field(default_factory=lambda: [1, 2, 4, 8])
    attention: bool = False
    attention_heads: int = 4
    deep_supervision: bool = False
    dropout: float = 0.1


@dataclass
class TrainingConfig:
    batch_size: int = 16
    epochs: int = 50
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    loss: str = "mse"
    scheduler: str = "step"
    lr_step_size: int = 10
    lr_gamma: float = 0.9
    early_stopping_patience: int = 10
    min_delta: float = 0.0001
    checkpoint_dir: str = "checkpoints"
    save_best_only: bool = True
    num_workers: int = 0


@dataclass
class InferenceConfig:
    threshold: float = 0.5
    min_distance: int = 50
    prominence: float = 0.2


@dataclass
class EvaluationConfig:
    timing_tolerance_samples: int = 20
    timing_tolerance_ms: Optional[float] = 200.0
    metrics: List[str] = field(default_factory=lambda: [
        "mae", "median_ae", "rmse", "precision", "recall", "f1", "detection_rate"
    ])
    results_dir: str = "results"


@dataclass
class ExperimentConfig:
    project: ProjectConfig = field(default_factory=ProjectConfig)
    data: DataConfig = field(default_factory=DataConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    windowing: WindowingConfig = field(default_factory=WindowingConfig)
    gaussian: GaussianConfig = field(default_factory=GaussianConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def get_torch_device(self) -> torch.device:
        """Resolve device string to torch.device object."""
        if self.project.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.project.device)


def load_config(config_path: Union[str, Path]) -> ExperimentConfig:
    """Load configuration from a YAML file, merging with default dataclass structures."""
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data: Dict[str, Any] = yaml.safe_load(f) or {}

    proj = ProjectConfig(**data.get("project", {}))
    dt = DataConfig(**data.get("data", {}))

    prep_data = data.get("preprocessing", {})
    disp_data = prep_data.pop("displacement", {}) if "displacement" in prep_data else {}
    disp = DisplacementConfig(**disp_data)
    prep = PreprocessingConfig(displacement=disp, **prep_data)

    win = WindowingConfig(**data.get("windowing", {}))
    gauss = GaussianConfig(**data.get("gaussian", {}))
    mdl = ModelConfig(**data.get("model", {}))
    trn = TrainingConfig(**data.get("training", {}))
    inf = InferenceConfig(**data.get("inference", {}))
    evl = EvaluationConfig(**data.get("evaluation", {}))

    return ExperimentConfig(
        project=proj,
        data=dt,
        preprocessing=prep,
        windowing=win,
        gaussian=gauss,
        model=mdl,
        training=trn,
        inference=inf,
        evaluation=evl,
    )
