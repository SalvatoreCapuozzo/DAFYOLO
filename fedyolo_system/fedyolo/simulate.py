"""CLI entry point.

    python -m fedyolo.simulate --config configs/example_federation.yaml
    python -m fedyolo.simulate --config configs/example_federation.yaml --also-centralized
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import torch

# Import rich for terminal UI
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel

from .config import load_config
from .evaluate import evaluate_state_dict
from .server import AsyncFedServer, FedServer

# 1. Upgrade logging to use RichHandler for beautiful terminal logs
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, markup=True)]
)

log = logging.getLogger("fedyolo.simulate")
console = Console()

def main():
    parser = argparse.ArgumentParser(description="Federated object detection with heterogeneous class sets")
    parser.add_argument("--config", required=True, help="path to federation YAML config")
    parser.add_argument(
        "--also-centralized",
        action="store_true",
        help="also train the non-federated centralized baseline for comparison (slower)",
    )
    parser.add_argument(
        "--session",
        default=None,
        help="name this run's output subfolder (default: auto timestamp) so repeated "
             "runs of the same config never overwrite each other's checkpoints",
    )
    parser.add_argument(
        "--no-session",
        action="store_true",
        help="write directly into config.output_dir instead of a session subfolder "
             "(old behaviour — reruns will overwrite previous checkpoints there)",
    )
    parser.add_argument(
        "--auto-cleanup-oom",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="before starting: kill any lingering fedyolo process from a previous "
             "crashed run and clear stale GPU/shm state. If THIS run crashes: do the "
             "same cleanup before exiting, so the next launch doesn't inherit a dirty "
             "state. On by default; use --no-auto-cleanup-oom to disable.",
    )
    args = parser.parse_args()

    if args.auto_cleanup_oom:
        from .gpu_cleanup import preflight_cleanup
        preflight_cleanup()

    cfg = load_config(args.config)

    # ── session directory ────────────────────────────────────────────────────
    # config.output_dir is treated as a FAMILY folder; each invocation writes
    # into its own session subfolder so re-running the same config (or several
    # concurrent runs of it) never clobbers a previous run's checkpoints.
    if not args.no_session:
        base_dir = Path(cfg.output_dir)
        session_name = args.session or datetime.now().strftime("%Y%m%d_%H%M%S")
        cfg.output_dir = str(base_dir / session_name)
        base_dir.mkdir(parents=True, exist_ok=True)
        latest_link = base_dir / "latest"
        try:
            if latest_link.is_symlink() or latest_link.exists():
                latest_link.unlink()
            latest_link.symlink_to(session_name, target_is_directory=True)
        except (OSError, NotImplementedError):
            pass  # symlinks unsupported (e.g. some Windows setups) — non-fatal
    log.info(f"[bold cyan]Session output directory:[/bold cyan] {cfg.output_dir}")

    # ── startup panel ────────────────────────────────────────────────────────
    mode = cfg.federation.mode
    if mode == "async":
        mode_detail = (
            f"[bold magenta]ASYNC[/bold magenta] | "
            f"cycles/node={cfg.federation.async_node_cycles} | "
            f"staleness_alpha={cfg.federation.staleness_alpha} | "
            f"total submissions={len(cfg.nodes) * cfg.federation.async_node_cycles}"
        )
    else:
        mode_detail = (
            f"[bold blue]SYNC[/bold blue] | "
            f"rounds={cfg.federation.rounds} | "
            f"local_epochs={cfg.federation.local_epochs}"
        )

    console.print(Panel(
        f"[bold cyan]Mode:[/bold cyan] {mode_detail}\n"
        f"[bold cyan]Global classes ({cfg.nc}):[/bold cyan] {cfg.global_classes}\n"
        f"[bold cyan]Nodes:[/bold cyan] {[n.name for n in cfg.nodes]}",
        title="[bold white]DAFYOLO — Federation Setup[/bold white]",
    ))
    for node in cfg.nodes:
        log.info(f"[bold blue]Node {node.name}[/bold blue] owns: {node.owned_classes}")

    # ── server selection ─────────────────────────────────────────────────────
    if mode == "async":
        server = AsyncFedServer(cfg)
        log.info("[bold green]🚀 Starting ASYNC Federated Training...[/bold green]")
    else:
        server = FedServer(cfg)
        log.info("[bold green]🚀 Starting SYNC Federated Training...[/bold green]")

    try:
        federated_sd = server.run()
    except Exception:
        if args.auto_cleanup_oom:
            from .gpu_cleanup import cleanup_after_crash
            cleanup_after_crash()
        raise
    log.info("[bold green]✔ Federated training complete![/bold green]")

    # The server already evaluated after the last round/submission and printed
    # the live table. Reuse that result for the JSON summary instead of
    # re-running a full validation pass.
    if server._map_history:
        final_fed = server._map_history[-1]
        summary = {"federated": {
            "map50":    final_fed["map50"],
            "map50-95": final_fed["map5095"],
            "per_class_map50-95": final_fed["per_class"],
        }}
        log.info(
            f"[bold green]Final federated mAP50=[/bold green][bold magenta]{final_fed['map50']:.4f}[/bold magenta]  "
            f"[bold green]mAP50-95=[/bold green][bold magenta]{final_fed['map5095']:.4f}[/bold magenta]"
        )
    else:
        # Fallback: eval_interval was set so high that no mid-training eval ran
        with console.status("[bold yellow]Evaluating federated model...", spinner="bouncingBar"):
            fed_metrics = evaluate_state_dict(cfg, federated_sd, name="federated", verbose=True)
        summary = {"federated": _summarize(fed_metrics)}
    log.info("[bold green]✔ Evaluation complete![/bold green]")

    if args.also_centralized:
        from .centralized import train_centralized_baseline

        with console.status("[bold magenta]Training centralized baseline (same total local epochs)...", spinner="aesthetic"):
            try:
                central_sd = train_centralized_baseline(cfg)
            except Exception:
                if args.auto_cleanup_oom:
                    from .gpu_cleanup import cleanup_after_crash
                    cleanup_after_crash()
                raise
            torch.save(
                {"state_dict": central_sd, "global_classes": cfg.global_classes},
                Path(cfg.output_dir) / "centralized_baseline.pt",
            )
        log.info("[bold green]✔ Centralized baseline training complete![/bold green]")
        
        with console.status("[bold yellow]Evaluating centralized baseline...", spinner="bouncingBar"):
            central_metrics = evaluate_state_dict(cfg, central_sd, name="centralized")
            summary["centralized"] = _summarize(central_metrics)
        log.info("[bold green]✔ Centralized evaluation complete![/bold green]")

    summary_path = Path(cfg.output_dir) / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    
    log.info(f"Wrote comparison summary -> [bold underline]{summary_path}[/bold underline]")
    
    # 4. Print the final JSON beautifully
    console.print_json(data=summary)


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
