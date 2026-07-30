"""Typed configuration loaded from a YAML file (see configs/stage1.yaml)."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    # Root where consolidated, canonical-class images are written.
    processed_dir: str = "data/processed"
    # Where kaggle downloads are unpacked before ingest.
    downloads_dir: str = "data/downloads"
    # Split manifest (CSV) written by edgewaste.data.splits.
    manifest: str = "data/splits.csv"
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    seed: int = 42


@dataclass
class ModelConfig:
    # timm backbone names. Defaults chosen to fit a 4GB GPU / run on CPU.
    convnext: str = "convnext_tiny"
    vit: str = "vit_small_patch16_224"
    image_size: int = 224
    fusion_dim: int = 512  # projected dim per backbone before attention fusion
    dropout: float = 0.2
    pretrained: bool = True


@dataclass
class TrainConfig:
    epochs: int = 15
    batch_size: int = 32
    lr: float = 3.0e-4
    weight_decay: float = 0.05
    warmup_epochs: int = 1
    label_smoothing: float = 0.1
    num_workers: int = 4
    amp: bool = True  # mixed precision (ignored on CPU)
    use_class_weights: bool = True  # handle the known class imbalance
    grad_clip: float = 1.0
    output_dir: str = "runs/stage1"
    early_stop_patience: int = 5


@dataclass
class DetectConfig:
    # Where the TACO (YOLO-format) detection dataset is unpacked + prepared.
    raw_dir: str = "data/detect/taco_raw"
    prepared_dir: str = "data/detect/taco"
    # Kaggle slug for the pre-converted-to-YOLO-format TACO mirror.
    kaggle_locator: str = "vencerlanz09/taco-dataset-yolo-format"
    # data.yaml written for ultralytics; single class since the detector's job
    # is localization only ("is there an item, where") — material typing is
    # the Stage-1 classifier's job on the resulting crop.
    data_yaml: str = "data/detect/taco/data.yaml"
    class_name: str = "waste_item"
    model: str = "yolo26n.pt"
    image_size: int = 640
    epochs: int = 60
    batch_size: int = 16
    output_dir: str = "runs/detect"
    conf_threshold: float = 0.35  # inference-time detection confidence gate


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    detect: DetectConfig = field(default_factory=DetectConfig)

    @staticmethod
    def load(path: str | Path | None) -> "Config":
        cfg = Config()
        if path is None:
            return cfg
        raw: dict[str, Any] = yaml.safe_load(Path(path).read_text()) or {}
        if "data" in raw:
            cfg.data = DataConfig(**{**asdict(cfg.data), **raw["data"]})
        if "model" in raw:
            cfg.model = ModelConfig(**{**asdict(cfg.model), **raw["model"]})
        if "train" in raw:
            cfg.train = TrainConfig(**{**asdict(cfg.train), **raw["train"]})
        if "detect" in raw:
            cfg.detect = DetectConfig(**{**asdict(cfg.detect), **raw["detect"]})
        return cfg

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
