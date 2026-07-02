"""CLI entry point.

    python -m fedyolo.simulate --config configs/example_federation.yaml
    python -m fedyolo.simulate --config configs/example_federation.yaml --also-centralized
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch

from .config import load_config
from .evaluate import evaluate_state_dict
from .server import FedServer

log = logging.getLogger("fedyolo.simulate")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def main():
    parser = argparse.ArgumentParser(description="Federated object detection with heterogeneous class sets")
    parser.add_argument("--config", required=True, help="path to federation YAML config")
    parser.add_argument(
        "--also-centralized",
        action="store_true",
        help="also train the non-federated centralized baseline for comparison (slower)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    log.info(f"global classes: {cfg.global_classes}")
    for node in cfg.nodes:
        log.info(f"  node={node.name} owns={node.owned_classes}")

    server = FedServer(cfg)
    federated_sd = server.run()

    log.info("evaluating federated global model on pooled validation set...")
    fed_metrics = evaluate_state_dict(cfg, federated_sd, name="federated")
    summary = {"federated": _summarize(fed_metrics)}

    if args.also_centralized:
        from .centralized import train_centralized_baseline

        log.info("training non-federated centralized baseline (same total local epochs)...")
        central_sd = train_centralized_baseline(cfg)
        torch.save(
            {"state_dict": central_sd, "global_classes": cfg.global_classes},
            Path(cfg.output_dir) / "centralized_baseline.pt",
        )
        log.info("evaluating centralized baseline on the same pooled validation set...")
        central_metrics = evaluate_state_dict(cfg, central_sd, name="centralized")
        summary["centralized"] = _summarize(central_metrics)

    summary_path = Path(cfg.output_dir) / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    log.info(f"wrote comparison summary -> {summary_path}")
    log.info(json.dumps(summary, indent=2))


def _summarize(metrics) -> dict:
    return {
        "map50-95": float(metrics.box.map),
        "map50": float(metrics.box.map50),
        "per_class_map50-95": {
            name: float(ap) for name, ap in zip(metrics.names.values(), metrics.box.maps)
        },
    }


if __name__ == "__main__":
    main()
