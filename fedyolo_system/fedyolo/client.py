"""One node's local training for one federated round.

`run_node_round` is the function each worker process executes. It:
  1. builds a model with the GLOBAL class count and loads the current global weights
  2. sets a binary class mask so the classification loss is zero for classes
     this node doesn't own (the false-negative-suppression fix)
  3. optionally adds a self-distillation term against the previous round's
     global model for not-owned classes (pseudo-labeling)
  4. trains locally for `local_epochs`
  5. returns the updated state_dict (on CPU, ready to pickle back to the server)
     plus the metadata the server needs for weighted aggregation
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from ultralytics.utils.loss import v8DetectionLoss

from .config import FedYoloConfig, NodeConfig
from .data import build_node_dataloader, build_node_dataset, per_class_instance_counts
from .model import build_model
from .optim_utils import set_warmup_lr
from .pseudo_label import distillation_loss


@dataclass
class NodeRoundResult:
    name: str
    state_dict: dict
    num_images: int
    class_counts: dict
    owned_global_ids: list


def _class_mask(nc: int, owned_ids: list[int], device) -> torch.Tensor:
    mask = torch.zeros(nc, device=device)
    mask[owned_ids] = 1.0
    return mask


def run_node_round(
    node: NodeConfig,
    cfg: FedYoloConfig,
    global_state_dict: dict,
    round_idx: int,
    teacher_state_dict: dict | None,
) -> NodeRoundResult:
    torch.manual_seed(cfg.seed + round_idx)
    device = torch.device(cfg.federation.device)

    owned_ids = node.owned_global_ids(cfg.global_classes)
    unowned_ids = [c for c in range(cfg.nc) if c not in owned_ids]

    model = build_model(cfg.model.arch, cfg.nc, cfg.model.imgsz).to(device)
    model.load_state_dict(global_state_dict)
    model.class_weights = _class_mask(cfg.nc, owned_ids, device)  # zeroes loss for unowned classes
    model.train()

    criterion = v8DetectionLoss(model)

    use_distill = (
        cfg.federation.pseudo_label.enabled
        and round_idx >= cfg.federation.pseudo_label.start_round
        and teacher_state_dict is not None
        and len(unowned_ids) > 0
    )
    teacher = None
    if use_distill:
        teacher = build_model(cfg.model.arch, cfg.nc, cfg.model.imgsz).to(device)
        teacher.load_state_dict(teacher_state_dict)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)

    dataset = build_node_dataset(node, cfg, split="train")
    dataloader = build_node_dataloader(node, cfg, split="train")
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.federation.lr0, momentum=0.9, weight_decay=5e-4)

    step = 0
    for epoch in range(cfg.federation.local_epochs):
        for batch in dataloader:
            set_warmup_lr(optimizer, step, cfg.federation.warmup_steps, cfg.federation.lr0)
            step += 1

            batch["img"] = batch["img"].to(device).float() / 255.0
            optimizer.zero_grad()

            preds = model.forward(batch["img"])
            parsed = criterion.parse_output(preds)
            loss, _ = criterion.loss(parsed, batch)
            loss = loss.sum()

            if use_distill:
                with torch.no_grad():
                    t_preds = teacher.forward(batch["img"])
                    t_parsed = criterion.parse_output(t_preds)
                student_scores = parsed["scores"].permute(0, 2, 1).contiguous()
                teacher_scores = t_parsed["scores"].permute(0, 2, 1).contiguous()
                d_loss = distillation_loss(
                    student_scores, teacher_scores, unowned_ids, cfg.federation.pseudo_label.conf_thresh
                )
                loss = loss + cfg.federation.pseudo_label.weight * d_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.federation.grad_clip)
            optimizer.step()

    state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    class_counts = per_class_instance_counts(dataset, cfg.nc)

    return NodeRoundResult(
        name=node.name,
        state_dict=state_dict,
        num_images=len(dataset),
        class_counts=class_counts,
        owned_global_ids=owned_ids,
    )
