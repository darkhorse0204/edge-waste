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
    # Inference-time detection gate. Raised from 0.35 to 0.45: the live camera
    # launchers pass configs/stage1.yaml, which has no `detect:` section, so
    # this default — not detect.yaml's value — is what the demo actually used.
    # Weak boxes are the ones with sloppy edges, and a sloppy box means a
    # sloppy crop and an unstable prediction.
    conf_threshold: float = 0.45


@dataclass
class InferenceConfig:
    """Live-demo behaviour. Affects nothing about training or the checkpoints —
    every field here is applied after the models have already spoken."""

    # --- confidence gate -------------------------------------------------
    # Smoothed probability below this displays `unknown_label` instead of a
    # class. Raise toward 0.7 for a stricter demo, lower to 0.5 if too much
    # reads as unknown.
    cls_conf_threshold: float = 0.60
    unknown_label: str = "unknown"

    # --- crop --------------------------------------------------------------
    # Fraction added to each side of the detector box before cropping.
    pad_frac: float = 0.15
    # Drop boxes sitting >80% inside a bigger box (bottle-cap-inside-bottle).
    containment_thresh: float = 0.80

    # --- temporal smoothing ------------------------------------------------
    # "mean" = average softmax over the window, "vote" = majority argmax,
    # "none" = per-frame prediction (the old behaviour).
    smoothing: str = "mean"
    history: int = 10  # frames remembered per tracked object
    switch_margin: float = 0.05  # hysteresis before the label may change
    iou_match: float = 0.30  # box overlap that counts as "same object"
    max_age: int = 5  # frames a track survives a detector dropout

    # --- display -----------------------------------------------------------
    colour_high: float = 0.80  # >= this -> green
    colour_mid: float = 0.60  # >= this -> yellow, below -> red
    show_hud: bool = True


@dataclass
class ExplainConfig:
    """Live Grad-CAM second window. Inference-side only."""

    # Start with the window open. Toggle at runtime with 'g'.
    enabled: bool = False
    # Frames between recomputes. Measured cost is ~88 ms per Grad-CAM against
    # ~30 ms for a classification, so every frame would roughly halve the frame
    # rate. At 10 the average cost is ~9 ms/frame — visually free. Lower it for
    # a more responsive heatmap, raise it if the demo machine is slower.
    interval: int = 10
    # Heatmap share of the blend: 0 = original crop, 1 = pure heatmap.
    alpha: float = 0.5
    # Side length of each of the two tiles in the panel, in pixels.
    tile: int = 300


@dataclass
class RecognizeConfig:
    """COCO object-identity stage (see recognize.py). Pretrained, never trained
    here — it only supplies a prior over the material classifier's output."""

    enabled: bool = True
    # COCO-pretrained weights. yolo26n.pt is the same base checkpoint the TACO
    # detector was fine-tuned from, so no new architecture enters the project.
    model: str = "yolo26n.pt"
    # 0.50, not COCO's usual 0.25: a *wrong* identity can drag a correct
    # material prediction down (a glass jar mislabelled "book" at 48% pulls
    # toward paper). Testing on real data showed 0.50 drops the false IDs
    # while keeping the true ones, which sat at 0.60-0.90.
    conf_threshold: float = 0.50
    # IoU at which a COCO box is considered the same object as a waste box.
    iou_match: float = 0.45
    # Add confidently-recognised waste objects the TACO detector missed. This
    # is what lets a clean banana or pair of scissors be found at all — TACO is
    # trained on street litter and often doesn't localise tidy objects.
    add_unmatched: bool = True
    # Mass placed on the implied class for a "certain" mapping, scaled by the
    # recogniser's own confidence. <1.0 so a shaky identity stays recoverable.
    certainty: float = 0.95


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    detect: DetectConfig = field(default_factory=DetectConfig)
    infer: InferenceConfig = field(default_factory=InferenceConfig)
    recognize: RecognizeConfig = field(default_factory=RecognizeConfig)
    explain: ExplainConfig = field(default_factory=ExplainConfig)

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
        if "infer" in raw:
            cfg.infer = InferenceConfig(**{**asdict(cfg.infer), **raw["infer"]})
        if "recognize" in raw:
            cfg.recognize = RecognizeConfig(
                **{**asdict(cfg.recognize), **raw["recognize"]})
        if "explain" in raw:
            cfg.explain = ExplainConfig(**{**asdict(cfg.explain), **raw["explain"]})
        return cfg

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
