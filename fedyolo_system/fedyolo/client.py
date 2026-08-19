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

import json
import threading
from dataclasses import dataclass
from pathlib import Path

import torch
from ultralytics.utils.loss import v8DetectionLoss

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from .config import FedYoloConfig, NodeConfig
from .data import build_node_dataloader, build_node_dataset, per_class_instance_counts
from .model import build_model
from .optim_utils import set_warmup_lr
from .pseudo_label import distillation_loss

# Guards concurrent appends to live_training_logs.txt in async mode
_log_file_lock = threading.Lock()


@dataclass
class NodeRoundResult:
    name: str
    state_dict: dict
    num_images: int
    class_counts: dict
    owned_global_ids: list
    optimizer_state: dict


def _optimizer_state_to_cpu(state_dict: dict) -> dict:
    """Move an SGD optimizer state_dict's per-parameter buffers (momentum_buffer)
    to CPU so it can be pickled back to the server process (sync mode) or held
    across cycles (async mode) without pinning GPU memory between calls."""
    state_dict["state"] = {
        k: {kk: (vv.detach().cpu() if torch.is_tensor(vv) else vv) for kk, vv in v.items()}
        for k, v in state_dict["state"].items()
    }
    return state_dict


def _class_mask(nc: int, owned_ids: list[int], device) -> torch.Tensor:
    mask = torch.zeros(nc, device=device)
    mask[owned_ids] = 1.0
    return mask
    
def _read_latest_map(output_dir: str) -> str:
    """Read the latest mAP from live_map.json written by the server after each
    evaluation. Returns an empty string when no evaluation has run yet (first
    round/cycle) so the progress bar degrades gracefully."""
    try:
        data = json.loads(Path(output_dir).joinpath("live_map.json").read_text())
        return f" [dim]| mAP50: {data['map50']:.4f}[/dim]"
    except Exception:
        return ""


def run_node_round(
    node: NodeConfig,
    cfg: FedYoloConfig,
    global_state_dict: dict,
    round_idx: int,
    teacher_state_dict: dict | None,
    optimizer_state: dict | None = None,
) -> NodeRoundResult:
    torch.manual_seed(cfg.seed + round_idx)
    device = torch.device(cfg.federation.device)

    owned_ids = node.owned_global_ids(cfg.global_classes)
    unowned_ids = [c for c in range(cfg.nc) if c not in owned_ids]

    model = build_model(cfg.model.arch, cfg.nc, cfg.model.imgsz, pretrained=cfg.model.pretrained).to(device)
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
        teacher = build_model(cfg.model.arch, cfg.nc, cfg.model.imgsz, pretrained=cfg.model.pretrained).to(device)
        teacher.load_state_dict(teacher_state_dict)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)

    dataset = build_node_dataset(node, cfg, split="train")
    dataloader = build_node_dataloader(node, cfg, split="train")
    optimizer = torch.optim.SGD(
    	model.parameters(), lr=cfg.federation.lr0, momentum=0.9, weight_decay=5e-4
    )
    if optimizer_state is not None:
        # Carry this node's momentum buffers forward from its previous cycle/round
        # instead of restarting cold every time. The model WEIGHTS still get
        # reloaded from the (possibly aggregated-elsewhere) global state above --
        # only the gradient-momentum statistics persist. LR is unaffected: the
        # per-step warmup below overwrites param_groups['lr'] before every
        # optimizer.step() regardless of what's in the loaded state.
        optimizer.load_state_dict(optimizer_state)

    step = 0
    log_path = Path(cfg.output_dir) / "live_training_logs.txt"
    # width=200: force_terminal alone doesn't stop Rich truncating columns to
    # "..." -- with no real tty to measure, it falls back to a narrow default
    # (~80 cols) that a description+bar+counts+timers row doesn't fit in.
    custom_console = Console(force_terminal=True, width=200)
    
    # Total steps for the whole round = epochs × batches/epoch
    # This lets us use a SINGLE progress task instead of one per epoch,
    # which is what was causing the flickering.
    total_steps = cfg.federation.local_epochs * len(dataloader)

    # Read the latest mAP once before training starts (server writes this file
    # after each evaluation; we refresh it once per epoch below).
    map_str = _read_latest_map(cfg.output_dir)
    
    # Create a custom Rich progress bar with Elapsed and Remaining time separated cleanly
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("[green]⏱️ Passed:[/green]"),
        TimeElapsedColumn(),
        TextColumn("[yellow]⏳ Left:[/yellow]"),
        TimeRemainingColumn(),
        transient=False,
        console=custom_console,
    ) as progress:
        
        task = progress.add_task(
            f"[cyan]{node.name} Ep 1/{cfg.federation.local_epochs}[/cyan]",
            total=total_steps,
        )
        
        for epoch in range(cfg.federation.local_epochs):
            # Variables to track the average loss for this specific epoch
            epoch_loss = 0.0
            num_batches = 0

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

                # Accumulate the loss for our text file logging
                epoch_loss += loss.item()
                num_batches += 1
                
                # Update the SINGLE task — description changes every batch,
                # the bar advances by 1 step; no new task is created.
                progress.update(
                    task,
                    advance=1,
                    description=(
                        f"[cyan]{node.name} "
                        f"Ep {epoch + 1}/{cfg.federation.local_epochs}[/cyan] "
                        f"[white]Loss: {loss.item():.4f}[/white]"
                        f"{map_str}"
                    ),
                )

            # --- END OF EPOCH LOGGING ---
            avg_epoch_loss = epoch_loss / max(1, num_batches)
            
            # Refresh mAP string once per epoch so the next epoch's batches
            # show the latest evaluation result without hammering the filesystem.
            map_str = _read_latest_map(cfg.output_dir)
            
            # Append to the shared log file; lock guards concurrent writes in async mode.
            with _log_file_lock:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(
                        f"Round/Cycle: {round_idx + 1:02d} | Node: {node.name:10s} | "
                        f"Epoch: {epoch + 1:02d}/{cfg.federation.local_epochs:02d} | "
                        f"Avg Loss: {avg_epoch_loss:.4f}\n"
                    )

    custom_console.print(f"[bold green]✔ {node.name}[/bold green] finished local training!")

    state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    class_counts = per_class_instance_counts(dataset, cfg.nc)
    returned_optimizer_state = _optimizer_state_to_cpu(optimizer.state_dict())

    return NodeRoundResult(
        name=node.name,
        state_dict=state_dict,
        num_images=len(dataset),
        class_counts=class_counts,
        owned_global_ids=owned_ids,
        optimizer_state=returned_optimizer_state,
    )
