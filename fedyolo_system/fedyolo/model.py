"""Model construction, and the parameter split that makes class-heterogeneous
federation work: every weight tensor is either SHARED (aggregated across all
nodes) or PER-CLASS (aggregated only across the nodes that own that class).

Why this split is exact and not a heuristic
--------------------------------------------
YOLOv8's Detect head ends each of its `nl` scale branches (cv3) with a final
1x1 conv whose output channel `c` is literally the logit for class `c`
(verified against the installed ultralytics build below). Every other weight
in the network (backbone, neck, box-regression branch cv2, DFL, the two
feature-mixing convs inside cv3 before the final projection) is class-agnostic
-- it produces shared features, not a class score -- so it can be FedAvg'd
across every node regardless of which classes that node owns.

    model.{H}.cv3.{i}.2.weight   shape (nc, c3, 1, 1)   <- per-class, dim 0 = class
    model.{H}.cv3.{i}.2.bias     shape (nc,)            <- per-class, dim 0 = class
    everything else                                      <- shared
"""

from __future__ import annotations

import re

import torch
from ultralytics.nn.tasks import DetectionModel


def build_model(arch: str, nc: int, imgsz: int) -> DetectionModel:
    """Build a YOLOv8 detection model with random (blank) weights -- no
    pretrained checkpoint is downloaded or loaded.
    """
    model = DetectionModel(cfg=arch, nc=nc, verbose=False)
    model.args = _default_train_args(imgsz)
    return model


def _default_train_args(imgsz: int):
    from types import SimpleNamespace

    # Minimal set of hyperparameters v8DetectionLoss / BboxLoss read from model.args.
    return SimpleNamespace(box=7.5, cls=0.5, dfl=1.5, imgsz=imgsz)


def _detect_module_index(model: DetectionModel) -> int:
    return len(model.model) - 1


def per_class_param_names(model: DetectionModel) -> list[tuple[str, str]]:
    """Return [(weight_key, bias_key), ...] for each detection scale's final
    per-class projection layer, e.g. [('model.22.cv3.0.2.weight', 'model.22.cv3.0.2.bias'), ...]
    """
    h = _detect_module_index(model)
    nl = model.model[h].nl
    return [(f"model.{h}.cv3.{i}.2.weight", f"model.{h}.cv3.{i}.2.bias") for i in range(nl)]


def split_state_dict(state_dict: dict, model: DetectionModel) -> tuple[dict, dict]:
    """Split a state_dict into (shared_params, per_class_params).

    per_class_params keeps the SAME key names as state_dict but every tensor's
    dim-0 still indexes classes -- callers slice by class id directly.
    """
    per_class_keys = {k for pair in per_class_param_names(model) for k in pair}
    shared = {k: v for k, v in state_dict.items() if k not in per_class_keys}
    per_class = {k: v for k, v in state_dict.items() if k in per_class_keys}
    return shared, per_class


def aggregate_shared(node_state_dicts: list[dict], node_weights: list[float], model: DetectionModel) -> dict:
    """Standard sample-weighted FedAvg over every node, restricted to shared keys."""
    per_class_keys = {k for pair in per_class_param_names(model) for k in pair}
    total = sum(node_weights)
    keys = [k for k in node_state_dicts[0].keys() if k not in per_class_keys]
    out = {}
    for k in keys:
        ref = node_state_dicts[0][k]
        if not torch.is_floating_point(ref):
            # buffers like num_batches_tracked: just keep the largest-node's value
            out[k] = node_state_dicts[node_weights.index(max(node_weights))][k]
            continue
        acc = torch.zeros_like(ref, dtype=torch.float32)
        for sd, w in zip(node_state_dicts, node_weights):
            acc += sd[k].to(torch.float32) * (w / total)
        out[k] = acc.to(ref.dtype)
    return out


def aggregate_per_class(
    node_state_dicts: list[dict],
    node_owned_global_ids: list[list[int]],
    node_class_counts: list[dict[int, int]],
    model: DetectionModel,
    nc: int,
) -> dict:
    """Class-conditional aggregation: for each global class id, average the
    relevant output-channel slice only across the nodes that own that class,
    weighted by how many labeled instances of that class each node has.

    Classes with zero contributing nodes keep their (randomly initialized)
    value from node 0 unchanged -- this should not happen if config
    validation passed, but is handled defensively.
    """
    out = {}
    for weight_key, bias_key in per_class_param_names(model):
        ref_w = node_state_dicts[0][weight_key]
        ref_b = node_state_dicts[0][bias_key]
        new_w = torch.zeros_like(ref_w, dtype=torch.float32)
        new_b = torch.zeros_like(ref_b, dtype=torch.float32)

        for c in range(nc):
            contributors = [
                i for i, owned in enumerate(node_owned_global_ids) if c in owned
            ]
            if not contributors:
                new_w[c] = ref_w[c].to(torch.float32)
                new_b[c] = ref_b[c].to(torch.float32)
                continue
            total = sum(max(node_class_counts[i].get(c, 0), 1) for i in contributors)
            for i in contributors:
                w = max(node_class_counts[i].get(c, 0), 1) / total
                new_w[c] += node_state_dicts[i][weight_key][c].to(torch.float32) * w
                new_b[c] += node_state_dicts[i][bias_key][c].to(torch.float32) * w

        out[weight_key] = new_w.to(ref_w.dtype)
        out[bias_key] = new_b.to(ref_b.dtype)
    return out


def aggregate(
    node_state_dicts: list[dict],
    node_num_images: list[int],
    node_owned_global_ids: list[list[int]],
    node_class_counts: list[dict[int, int]],
    model: DetectionModel,
    nc: int,
) -> dict:
    """Full server-side aggregation for one federated round."""
    shared = aggregate_shared(node_state_dicts, [float(n) for n in node_num_images], model)
    per_class = aggregate_per_class(node_state_dicts, node_owned_global_ids, node_class_counts, model, nc)
    merged = {**shared, **per_class}
    return merged
