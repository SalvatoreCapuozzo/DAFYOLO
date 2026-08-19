"""Configuration schema and loader for a federated object-detection run.

Everything about a federation (global class list, model architecture, per-node
data locations and owned classes, federated hyperparameters) lives in one YAML
file. See configs/example_federation.yaml for a worked example.

federation.mode controls which server is used:
  "sync"  — classic FedAvg barrier: all nodes finish round N before
            the server aggregates and starts round N+1 (FedServer).
  "async" — each node independently cycles pull→train→push; the server
            aggregates immediately on every push with a staleness discount
            (AsyncFedServer). No barrier, no rounds — the global model
            version increments by 1 after each single-node submission.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class ModelConfig:
    arch: str = "yolov8n.yaml"  # ultralytics architecture yaml
    imgsz: int = 640
    pretrained: bool = True     # init SHARED params (backbone/neck/box-head/DFL)
                                # from arch's official COCO checkpoint (e.g.
                                # yolov8m.yaml -> yolov8m.pt). The per-class
                                # head always stays random (nc != COCO's 80).
                                # Set False to reproduce the original from-
                                # scratch (fully blank) behavior.


@dataclass
class PseudoLabelConfig:
    enabled: bool = False
    # sync: interpreted as round index; async: interpreted as per-node cycle index
    start_round: int = 5
    conf_thresh: float = 0.5
    weight: float = 0.5


@dataclass
class FederationConfig:
    # ── selector ──────────────────────────────────────────────────────────────
    mode: str = "sync"              # "sync" | "async"

    # ── shared by both modes ──────────────────────────────────────────────────
    local_epochs: int = 5           # epochs per round (sync) or per cycle (async)
    batch_size: int = 8
    lr0: float = 0.001
    warmup_steps: int = 20          # per-round/cycle LR warmup steps
    grad_clip: float = 10.0
    workers: int = 0
    device: str = "auto"            # "auto" resolves at load time to the best
                                    # available backend on THIS machine: CUDA
                                    # if present, else Apple MPS, else CPU. Set
                                    # an explicit "cuda"/"cuda:0"/"mps"/"cpu"
                                    # to pin it instead (e.g. to force CPU on a
                                    # CUDA box, or to target a specific GPU).
    pseudo_label: PseudoLabelConfig = field(default_factory=PseudoLabelConfig)
    data_fraction: float = 1.0      # use only the first N% of each node's images (both
                                    # splits); for fast smoke-tests on a large real dataset
                                    # without touching the config's node/class layout.
                                    # 1.0 = full dataset (default, no behaviour change).
                                    # This is Ultralytics' own naive prefix slice -- NOT
                                    # class-aware. Prefer balanced_fraction below on any
                                    # dataset with imbalanced classes (a plain fraction can
                                    # silently drop a rare class to zero images).
    balanced_fraction: float | None = None
                                    # Class-stratified alternative to data_fraction, for fast
                                    # local iteration that still checks every class: every
                                    # class keeps max(1, round(count * balanced_fraction)) of
                                    # its images, in BOTH train and the pooled val set used
                                    # for evaluation (data_fraction only ever touched train).
                                    # None (default) = disabled, data_fraction applies as
                                    # normal. Takes priority over data_fraction when set.
                                    # e.g. 0.10 for a balanced ~10% subset.

    # ── sync only (ignored in async mode) ─────────────────────────────────────
    rounds: int = 20                # federated rounds; total epochs/node = rounds × local_epochs

    # ── async only (ignored in sync mode) ─────────────────────────────────────
    async_node_cycles: int = 10     # pull→train→push cycles per node
                                    # total submissions = n_nodes × async_node_cycles
                                    # total epochs/node = async_node_cycles × local_epochs
    staleness_alpha: float = 0.5    # staleness discount: w = 1 / (1 + alpha × staleness)
                                    # staleness = global_version_now − version_when_node_pulled
                                    # higher alpha = stronger penalty for stale updates
    max_concurrent_nodes: int = 1   # how many nodes may train simultaneously — both modes.
                                    # sync:  caps the round's ProcessPoolExecutor workers
                                    #        (nodes within a round are independent; the
                                    #        barrier still waits for all of them either way).
                                    # async: 1 = sequential execution with async semantics
                                    #        (safe default); >1 = true parallelism.
                                    # >1 requires enough RAM/VRAM for N models simultaneously
                                    # (each yolov8l at imgsz=960 ≈ 1–2 GB).

    # ── live evaluation (both modes) ──────────────────────────────────────────
    eval_interval: int = 1          # sync:  evaluate every N rounds
                                    # async: evaluate every N submissions
                                    #        (default 1 = every submission; set to n_nodes for
                                    #         one eval per "virtual round" worth of submissions)


@dataclass
class NodeConfig:
    name: str
    data_yaml: str
    owned_classes: list[str]   # subset of global_classes, IN THE ORDER local label ids 0..k-1 map to them

    def class_id_map(self, global_classes: list[str]) -> dict[int, int]:
        """local label id -> global label id."""
        return {local_id: global_classes.index(name) 
        	for local_id, name in enumerate(self.owned_classes)}

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


def _resolve_device(device: str) -> str:
    """Turn "auto" into a concrete torch device string for this machine.
    Any other value (including an already-concrete one) passes through
    unchanged, so pinning a device explicitly always still works."""
    if device != "auto":
        return device
    import torch
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_config(path: str | Path) -> FedYoloConfig:
    raw = yaml.safe_load(Path(path).read_text())

    model = ModelConfig(**raw.get("model", {}))

    fed_raw = dict(raw.get("federation", {}))
    pl_raw = fed_raw.pop("pseudo_label", {})
    federation = FederationConfig(pseudo_label=PseudoLabelConfig(**pl_raw), **fed_raw)
    federation.device = _resolve_device(federation.device)

    if federation.mode not in ("sync", "async"):
        raise ValueError(f"federation.mode must be 'sync' or 'async', got '{federation.mode}'")

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
                    f"Node '{node.name}' owns class '{cls}' which is not in "
                    f"global_classes={cfg.global_classes}"
                )
        if len(set(node.owned_classes)) != len(node.owned_classes):
            raise ValueError(
                f"Node '{node.name}' has duplicate owned_classes: {node.owned_classes}"
            )

    covered = {cls for node in cfg.nodes for cls in node.owned_classes}
    missing = set(cfg.global_classes) - covered
    if missing:
        raise ValueError(
            f"No node owns these global classes, they can never be learned: {sorted(missing)}"
        )
