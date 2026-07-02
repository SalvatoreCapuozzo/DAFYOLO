"""Configuration schema and loader for a federated object-detection run.

Everything about a federation (global class list, model architecture, per-node
data locations and owned classes, federated hyperparameters) lives in one YAML
file. See configs/example_federation.yaml for a worked example.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class ModelConfig:
    arch: str = "yolov8n.yaml"  # ultralytics architecture yaml -> random (blank) init
    imgsz: int = 640


@dataclass
class PseudoLabelConfig:
    enabled: bool = False
    start_round: int = 5       # don't pseudo-label until the global model is half-decent
    conf_thresh: float = 0.5   # only trust confident pseudo boxes
    weight: float = 0.5        # down-weight pseudo-labeled loss vs real ground truth


@dataclass
class FederationConfig:
    rounds: int = 20
    local_epochs: int = 2
    batch_size: int = 8
    lr0: float = 0.001
    warmup_steps: int = 20     # linear LR warmup at the start of every local round (weights were just reloaded)
    grad_clip: float = 10.0    # gradient-norm clipping; random-init detectors are prone to early spikes
    workers: int = 0
    device: str = "cpu"
    pseudo_label: PseudoLabelConfig = field(default_factory=PseudoLabelConfig)


@dataclass
class NodeConfig:
    name: str
    data_yaml: str
    owned_classes: list[str]   # subset of global_classes, IN THE ORDER local label ids 0..k-1 map to them

    def class_id_map(self, global_classes: list[str]) -> dict[int, int]:
        """local label id -> global label id."""
        return {local_id: global_classes.index(name) for local_id, name in enumerate(self.owned_classes)}

    def owned_global_ids(self, global_classes: list[str]) -> list[int]:
        return [global_classes.index(name) for name in self.owned_classes]


@dataclass
class FedYoloConfig:
    global_classes: list[str]
    model: ModelConfig
    federation: FederationConfig
    nodes: list[NodeConfig]
    output_dir: str = "runs/federation"
    seed: int = 0

    @property
    def nc(self) -> int:
        return len(self.global_classes)


def load_config(path: str | Path) -> FedYoloConfig:
    raw = yaml.safe_load(Path(path).read_text())

    model = ModelConfig(**raw.get("model", {}))

    fed_raw = dict(raw.get("federation", {}))
    pl_raw = fed_raw.pop("pseudo_label", {})
    federation = FederationConfig(pseudo_label=PseudoLabelConfig(**pl_raw), **fed_raw)

    nodes = [NodeConfig(**n) for n in raw["nodes"]]

    cfg = FedYoloConfig(
        global_classes=list(raw["global_classes"]),
        model=model,
        federation=federation,
        nodes=nodes,
        output_dir=raw.get("output_dir", "runs/federation"),
        seed=raw.get("seed", 0),
    )
    _validate(cfg)
    return cfg


def _validate(cfg: FedYoloConfig) -> None:
    for node in cfg.nodes:
        for cls in node.owned_classes:
            if cls not in cfg.global_classes:
                raise ValueError(
                    f"Node '{node.name}' owns class '{cls}' which is not in global_classes={cfg.global_classes}"
                )
        if len(set(node.owned_classes)) != len(node.owned_classes):
            raise ValueError(f"Node '{node.name}' lists duplicate owned_classes: {node.owned_classes}")

    covered = set()
    for node in cfg.nodes:
        covered.update(node.owned_classes)
    missing = set(cfg.global_classes) - covered
    if missing:
        raise ValueError(
            f"No node owns these global classes, they can never be learned: {sorted(missing)}"
        )
