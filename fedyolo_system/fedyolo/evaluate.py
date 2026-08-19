"""Evaluation against a pooled validation set spanning every node, using
Ultralytics' own DetectionValidator (via the standard YOLO.val() entrypoint)
for trusted, standard mAP numbers.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from ultralytics import YOLO

from .config import FedYoloConfig
from .model import build_model
from .pooling import materialize_pooled_dataset


def per_class_map50(metrics) -> dict[str, float]:
    """Per-class AP50 (IoU=0.5), full nc-length with a fallback for classes
    Ultralytics didn't evaluate (no validation instances that round).

    Ultralytics' own `metrics.box.maps` gives this shape for AP50-95, but
    doesn't expose an AP50 equivalent -- `metrics.box.ap50` is only indexed
    by `ap_class_index` (classes that had >=1 val instance), so a class
    outside that dies silently when zipped positionally against every
    class name. Rebuilt here the same way Ultralytics' own `Metric.maps`
    property builds its AP50-95 version: start every class at the OVERALL
    mAP50 (not 0 -- an untested class isn't known to be zero, so filling
    with the mean is Ultralytics' own convention, not a fabricated number),
    then overwrite the classes that were actually measured.
    """
    box = metrics.box
    names = list(metrics.names.values())
    out = np.full(len(names), box.map50, dtype=float)
    for i, c in enumerate(box.ap_class_index):
        out[c] = box.ap50[i]
    return {name: float(v) for name, v in zip(names, out)}


def evaluate_state_dict(cfg: FedYoloConfig, state_dict: dict,
                        name: str = "model", verbose: bool = False):
    """Run validation and return an Ultralytics metrics object.

    verbose=False suppresses all Ultralytics stdout (table, per-class rows,
    speed summary) so the caller can display only what it wants via Rich.
    """
    pooled_yaml = materialize_pooled_dataset(
        cfg, split="val", out_dir=Path(cfg.output_dir) / "pooled_val",
        fraction=cfg.federation.balanced_fraction, seed=cfg.seed,
    )
    model = build_model(cfg.model.arch, cfg.nc, cfg.model.imgsz, pretrained=cfg.model.pretrained)
    model.load_state_dict(state_dict)
    model.names = {i: n for i, n in enumerate(cfg.global_classes)}
    yolo = YOLO(cfg.model.arch)
    yolo.model = model
    metrics = yolo.val(
        data=str(pooled_yaml),
        imgsz=cfg.model.imgsz,
        batch=cfg.federation.batch_size,
        device=cfg.federation.device,
        plots=False,
        verbose=verbose,
        project=str(Path(cfg.output_dir) / "eval"),
        name=name,
    )
    return metrics
