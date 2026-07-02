"""Evaluation against a pooled validation set spanning every node, using
Ultralytics' own DetectionValidator (via the standard YOLO.val() entrypoint)
for trusted, standard mAP numbers.
"""

from __future__ import annotations

from pathlib import Path

from ultralytics import YOLO

from .config import FedYoloConfig
from .model import build_model
from .pooling import materialize_pooled_dataset


def evaluate_state_dict(cfg: FedYoloConfig, state_dict: dict, name: str = "model"):
    """Build a model with the given weights and report mAP on the pooled val set."""
    pooled_yaml = materialize_pooled_dataset(cfg, split="val", out_dir=Path(cfg.output_dir) / "pooled_val")

    model = build_model(cfg.model.arch, cfg.nc, cfg.model.imgsz)
    model.load_state_dict(state_dict)
    model.names = {i: n for i, n in enumerate(cfg.global_classes)}

    yolo = YOLO(cfg.model.arch)  # builds a throwaway default model, immediately replaced below
    yolo.model = model

    metrics = yolo.val(
        data=str(pooled_yaml),
        imgsz=cfg.model.imgsz,
        batch=cfg.federation.batch_size,
        device=cfg.federation.device,
        plots=False,
        verbose=True,
        project=str(Path(cfg.output_dir) / "eval"),
        name=name,
    )
    return metrics
