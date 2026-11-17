"""
Integration smoke test: verify end-to-end training and inference execution.
"""

from pathlib import Path
import tempfile
import pytest
import torch

from earthquake_detection.config import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    PreprocessingConfig,
    ProjectConfig,
    TrainingConfig,
)
from earthquake_detection.inference.predict import SeismicPredictor
from earthquake_detection.training.train import run_training


def test_full_training_and_inference_pipeline(synthetic_hdf5_path):
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        ckpt_dir = tmp_path / "checkpoints"

        cfg = ExperimentConfig(
            project=ProjectConfig(device="cpu", seed=42),
            data=DataConfig(
                path=str(synthetic_hdf5_path),
                num_records=16,
                train_split=0.6,
                val_split=0.2,
                test_split=0.2,
                sampling_rate=100.0,
                waveform_length=1000,
            ),
            preprocessing=PreprocessingConfig(
                filtering=False,
                normalization="per_channel",
            ),
            model=ModelConfig(
                architecture="unetplusplus_1d",
                in_channels=3,
                base_channels=8,
                depth=3,
                dilated_convolutions=True,
                dilation_rates=[1, 2],
                attention=True,
                attention_heads=2,
            ),
            training=TrainingConfig(
                batch_size=4,
                epochs=2,
                learning_rate=0.005,
                checkpoint_dir=str(ckpt_dir),
                num_workers=0,
            ),
        )

        history = run_training(cfg)
        assert len(history.train_loss) == 2
        assert len(history.val_loss) == 2

        # Verify checkpoints created
        best_pt = ckpt_dir / "best_model.pt"
        last_pt = ckpt_dir / "last_model.pt"
        hist_json = ckpt_dir / "training_history.json"

        assert best_pt.is_file()
        assert last_pt.is_file()
        assert hist_json.is_file()

        # Test inference using saved best model
        predictor = SeismicPredictor(checkpoint_path=best_pt, config=cfg, device="cpu")
        sample_trace = torch.randn(3, 1000).numpy()
        result, scores = predictor.predict(sample_trace)

        assert scores.shape == (1000,)
        assert isinstance(result.has_detection, bool)
