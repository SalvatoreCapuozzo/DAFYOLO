"""Centralized baseline: a single model trained directly (no federated
aggregation at all) by cycling through every node's dataloader and stepping
the SAME set of weights every time, switching the class mask to match
whichever node's batch is currently being trained on.

This is deliberately NOT "ignore node boundaries and assume every image has
every class labeled" -- that data usually doesn't exist (a node's images
were never annotated for classes it doesn't own). Instead this baseline
gets exactly the same information the federation gets (same images, same
partial per-node labels, same total number of local epochs over each node's
data) but updates one set of weights directly instead of aggregating
independently-trained copies. That isolates the cost of federation itself
from the cost of partial labels, which both approaches face equally.

If you DO have a fully cross-annotated dataset (every image labeled for
every class), point a single extra "oracle" node at it and evaluate that
checkpoint instead -- that measures the federation+masking cost combined
with the partial-label cost, i.e. the absolute ceiling.
"""

from __future__ import annotations

import logging

import torch

from .client import _class_mask
from .config import FedYoloConfig
from .data import build_node_dataloader
from .model import build_model
from .optim_utils import set_warmup_lr

log = logging.getLogger("fedyolo.centralized")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def train_centralized_baseline(cfg: FedYoloConfig) -> dict:
    device = torch.device(cfg.federation.device)
    torch.manual_seed(cfg.seed)

    model = build_model(cfg.model.arch, cfg.nc, cfg.model.imgsz, pretrained=cfg.model.pretrained).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.federation.lr0, momentum=0.9, weight_decay=5e-4)

    node_loaders = [
        (node, node.owned_global_ids(cfg.global_classes), build_node_dataloader(node, cfg, split="train"))
        for node in cfg.nodes
    ]

    # Match total epochs to whichever mode is active
    if cfg.federation.mode == "async":
        total_epochs = cfg.federation.async_node_cycles * cfg.federation.local_epochs
    else:
        total_epochs = cfg.federation.rounds * cfg.federation.local_epochs
        
    step = 0
    for epoch in range(total_epochs):
        log.info(f"=== centralized epoch {epoch + 1}/{total_epochs} ===")
        for node, owned_ids, loader in node_loaders:
            model.class_weights = _class_mask(cfg.nc, owned_ids, device)
            model.criterion = None  # force rebuild so it picks up the new mask
            model.train()
            for batch in loader:
                set_warmup_lr(optimizer, step, cfg.federation.warmup_steps, cfg.federation.lr0)
                step += 1
                batch["img"] = batch["img"].to(device).float() / 255.0
                optimizer.zero_grad()
                loss, _ = model.loss(batch)
                loss.sum().backward()
                torch.nn.utils.clip_grad_norm_(
                	model.parameters(), max_norm=cfg.federation.grad_clip
                )
                optimizer.step()

    return {k: v.detach().cpu() for k, v in model.state_dict().items()}
