"""Server-side orchestration.

Each round: broadcast the current global state_dict to every node, train each
node in its own OS process (real isolation -- a node's process never touches
another node's data or in-memory state, same as it would across machines),
collect updated weights + per-class counts, then aggregate.
"""

from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch
import torch.multiprocessing as mp

# Each round ships a ~350-tensor state_dict to every node process and back.
# Torch's default "file_descriptor" CPU tensor sharing strategy opens one fd
# per tensor for this kind of IPC and can exhaust the process's fd limit
# over many rounds; "file_system" avoids that at a small disk-temp-file cost.
mp.set_sharing_strategy("file_system")

from .client import NodeRoundResult, run_node_round
from .config import FedYoloConfig
from .model import aggregate, build_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("fedyolo.server")


def _node_worker(args):
    node, cfg, global_sd, round_idx, teacher_sd = args
    return run_node_round(node, cfg, global_sd, round_idx, teacher_sd)


class FedServer:
    def __init__(self, cfg: FedYoloConfig):
        self.cfg = cfg
        self.model = build_model(cfg.model.arch, cfg.nc, cfg.model.imgsz)
        self.global_state_dict = {k: v.clone() for k, v in self.model.state_dict().items()}
        self.out_dir = Path(cfg.output_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        torch.manual_seed(cfg.seed)

    def run(self) -> dict:
        previous_round_sd = None
        max_workers = max(1, len(self.cfg.nodes))
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            for round_idx in range(self.cfg.federation.rounds):
                log.info(f"=== round {round_idx + 1}/{self.cfg.federation.rounds} ===")

                tasks = [
                    (node, self.cfg, self.global_state_dict, round_idx, previous_round_sd)
                    for node in self.cfg.nodes
                ]

                results: list[NodeRoundResult] = list(pool.map(_node_worker, tasks))

                for r in results:
                    owned = [self.cfg.global_classes[c] for c in r.owned_global_ids]
                    log.info(f"  node={r.name:10s} images={r.num_images:4d} owns={owned}")

                previous_round_sd = {k: v.clone() for k, v in self.global_state_dict.items()}

                self.global_state_dict = aggregate(
                    node_state_dicts=[r.state_dict for r in results],
                    node_num_images=[r.num_images for r in results],
                    node_owned_global_ids=[r.owned_global_ids for r in results],
                    node_class_counts=[r.class_counts for r in results],
                    model=self.model,
                    nc=self.cfg.nc,
                )

                ckpt_path = self.out_dir / f"global_round{round_idx + 1:03d}.pt"
                torch.save(
                    {"state_dict": self.global_state_dict, "global_classes": self.cfg.global_classes},
                    ckpt_path,
                )

        final_path = self.out_dir / "global_final.pt"
        torch.save(
            {"state_dict": self.global_state_dict, "global_classes": self.cfg.global_classes},
            final_path,
        )
        log.info(f"federation complete -> {final_path}")
        return self.global_state_dict
