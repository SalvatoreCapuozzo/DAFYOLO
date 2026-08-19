"""Splits ONE global YOLO-format dataset (images/ + labels/ + classes_list.txt)
into several per-node datasets usable by fedyolo, simulating a federation
where different "sites" specialize in different classes.

What it does
------------
1. Parses classes_list.txt ("N: name" lines), drops the classes you tell it
   to drop entirely (e.g. bubble/dirt artifact classes -- removed from every
   label file and never assigned to any node).
2. Auto-detects whether your label .txt files use 0-based or 1-based class
   ids (common with some annotation tools) by scanning what's actually on
   disk, so you don't have to guess.
3. Greedily assigns the remaining classes to N nodes, balancing total
   annotated instance count per node (a class with very few boxes won't be
   stranded alone on a node with the most popular class also dumped there).
4. For every image, assigns it to EVERY node that owns at least one class
   present in that image -- so an image with both a Trichuris and an
   Ascaris box ends up in both the Trichuris-owning node's dataset and the
   Ascaris-owning node's dataset, but each copy's label file only keeps that
   node's own classes (this is exactly the partial-annotation scenario
   fedyolo's loss masking is built for). Images whose only annotations are
   dropped classes are excluded entirely (no owned class present anywhere).
5. Splits each node's images into train/val, writes the YOLO directory
   layout + data.yaml fedyolo expects, and writes a ready-to-use federation
   config YAML.

Usage
-----
    python tools/split_dataset_into_nodes.py \\
        --dataset /path/to/dataset_folder \\
        --out data/from_global \\
        --n-nodes 4 \\
        --drop-classes bubble,dirt \\
        --config-out configs/from_global_federation.yaml

Run with --dry-run first to see the proposed class-to-node split and
per-node image/instance counts before anything is written to disk.
"""

from __future__ import annotations

import argparse
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path

import yaml

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def parse_classes_list(path: Path) -> list[str]:
    """Parses 'N: name' lines (ignoring any header/blank lines), in file order."""
    entries = []
    for line in Path(path).read_text().splitlines():
        m = re.match(r"^\s*(\d+)\s*:\s*(.+?)\s*$", line)
        if m:
            entries.append((int(m.group(1)), m.group(2).strip()))
    entries.sort(key=lambda x: x[0])
    return [name for _, name in entries]


def detect_label_index_base(label_dir: Path, n_classes: int) -> int:
    """Returns 0 or 1: whether the .txt files' class ids are 0-based
    (standard YOLO, id 0..n_classes-1) or 1-based (id 1..n_classes,
    sometimes produced by annotation tools that export the human-facing
    'select the class' number directly).
    """
    used = set()
    for f in label_dir.glob("*.txt"):
        for line in f.read_text().splitlines():
            if line.strip():
                used.add(int(line.split()[0]))
    if not used:
        raise ValueError(f"No labels found in {label_dir}")
    lo, hi = min(used), max(used)
    if lo == 0:
        return 0   # conclusive: 1-based never produces id 0
    if hi == n_classes:
        print(f"[detect] ids {lo}..{hi}, max == class count ({n_classes}) → 1-based, shifting -1")
        return 1   # conclusive: 0-based never reaches n_classes
    print(f"[detect] ids {lo}..{hi} ambiguous for {n_classes} classes → defaulting 0-based. "
          f"Rerun with --label-index-base 1 if classes look wrong.")
    return 0


def load_dataset(dataset_dir: Path, drop_names: set[str], label_index_base: str):
    classes = parse_classes_list(dataset_dir / "classes_list.txt")
    label_dir = dataset_dir / "labels"
    img_dir = dataset_dir / "images"

    base = (int(label_index_base)
            if label_index_base != "auto"
            else detect_label_index_base(label_dir, len(classes)))

    kept_names = [c for c in classes if c.lower() not in drop_names]
    dropped = [c for c in classes if c.lower() in drop_names]
    if dropped:
        print(f"[load] dropping classes: {dropped}")
    old_to_new = {classes.index(n): i for i, n in enumerate(kept_names)}

    images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    image_boxes: dict[Path, list[tuple[int, str]]] = {}
    class_instance_count: dict[int, int] = defaultdict(int)
    n_dropped_boxes = n_excluded_images = 0

    for img_path in images:
        label_path = label_dir / (img_path.stem + ".txt")
        boxes = []
        if label_path.exists():
            for line in label_path.read_text().splitlines():
                if not line.strip():
                    continue
                parts = line.split()
                raw_cls = int(parts[0]) - base
                if raw_cls not in old_to_new:
                    n_dropped_boxes += 1
                    continue
                new_cls = old_to_new[raw_cls]
                boxes.append((new_cls, " ".join(parts[1:])))
                class_instance_count[new_cls] += 1
        if boxes:
            image_boxes[img_path] = boxes
        else:
            n_excluded_images += 1

    print(f"[load] {len(images)} images → {len(image_boxes)} kept, "
          f"{n_excluded_images} excluded, {n_dropped_boxes} boxes dropped")
    return kept_names, image_boxes, class_instance_count


def assign_classes_to_nodes(kept_names: list[str], class_instance_count: dict, n_nodes: int) -> list[list[int]]:
    """Greedy balance: sort classes by descending instance count, repeatedly
    drop each into whichever node currently has the smallest total.
    """
    if n_nodes > len(kept_names):
        raise ValueError(f"--n-nodes ({n_nodes}) can't exceed the number of kept classes ({len(kept_names)})")

    order = sorted(range(len(kept_names)), key=lambda c: -class_instance_count.get(c, 0))
    node_classes = [[] for _ in range(n_nodes)]
    node_totals = [0] * n_nodes
    for c in order:
        i = min(range(n_nodes), key=lambda n: node_totals[n])
        node_classes[i].append(c)
        node_totals[i] += class_instance_count.get(c, 0)
    for node in node_classes:
        node.sort()  # keep a stable, readable order
    return node_classes


def write_node(
    node_name: str,
    owned_compacted_ids: list[int],
    kept_names: list[str],
    image_boxes: dict,
    out_root: Path,
    val_frac: float,
    seed: int,
    dry_run: bool,
):
    owned_set = set(owned_compacted_ids)
    local_id = {gid: i for i, gid in enumerate(owned_compacted_ids)}
    owned_names = [kept_names[gid] for gid in owned_compacted_ids]

    node_images = [img for img, boxes in image_boxes.items()
                   if any(c in owned_set for c, _ in boxes)]
    rng = random.Random(seed)
    rng.shuffle(node_images)
    n_val = max(1, int(len(node_images) * val_frac)) if node_images else 0
    splits = {"val": node_images[:n_val], "train": node_images[n_val:]}
    total_inst = sum(1 for img in node_images
                     for c, _ in image_boxes[img] if c in owned_set)
    print(f"[node {node_name}] owns={owned_names} | "
          f"images={len(node_images)} (train={len(splits['train'])}, val={n_val}) | "
          f"instances={total_inst}")

    if dry_run:
        return owned_names

    node_dir = out_root / node_name
    for split, imgs in splits.items():
        img_out = node_dir / "images" / split
        lbl_out = node_dir / "labels" / split
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)
        for img in imgs:
            dst_img = img_out / img.name
            if not dst_img.exists():
                try:
                    dst_img.symlink_to(img.resolve())
                except (OSError, NotImplementedError):
                    shutil.copy(img, dst_img)
            lines = [f"{local_id[c]} {rest}"
                     for c, rest in image_boxes[img] if c in owned_set]
            (lbl_out / (img.stem + ".txt")).write_text("\n".join(lines))

    data_yaml = {
        "path": str(node_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {i: n for i, n in enumerate(owned_names)},
    }
    (node_dir / "data.yaml").write_text(yaml.safe_dump(data_yaml))
    return owned_names


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, help="path to dataset_folder (images/, labels/, classes_list.txt)")
    p.add_argument("--out", default="data/from_global", help="output root for per-node datasets")
    p.add_argument("--n-nodes", type=int, default=4)
    p.add_argument("--drop-classes", default="bubble,dirt", help="comma-separated class names to drop entirely")
    p.add_argument("--label-index-base", default="auto", choices=["auto", "0", "1"])
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--config-out", default=None, help="also write a ready-to-use federation config YAML here")
    p.add_argument("--dry-run", action="store_true", help="print the planned split, write nothing to disk")
    args = p.parse_args()

    dataset_dir = Path(args.dataset)
    out_root = Path(args.out)
    drop_names = {c.strip().lower() for c in args.drop_classes.split(",") if c.strip()}

    kept_names, image_boxes, class_instance_count = load_dataset(
        dataset_dir, drop_names, args.label_index_base
    )
    print(f"[classes] {len(kept_names)} kept: {kept_names}")
    print("[classes] instance counts:",
          {kept_names[c]: n for c, n in sorted(class_instance_count.items())})

    node_classes = assign_classes_to_nodes(kept_names, class_instance_count, args.n_nodes)
    node_names = [f"node_{chr(ord('A') + i)}" for i in range(args.n_nodes)]

    nodes_cfg = []
    for name, owned_ids in zip(node_names, node_classes):
        owned_names = write_node(
            name, owned_ids, kept_names, image_boxes,
            out_root, args.val_frac, args.seed, args.dry_run
        )
        nodes_cfg.append({
            "name": name,
            "data_yaml": str(out_root / name / "data.yaml"),
            "owned_classes": owned_names,
        })

    if args.config_out and not args.dry_run:
        # Write a fully-explicit config so every tunable field is visible
        # and no field relies on a dataclass default that could silently
        # produce wrong behaviour if the code is updated later.
        full_cfg = {
            "global_classes": kept_names,
            "model": {
                "arch": "yolov8n.yaml",   # yolov8n/s/m/l/x.yaml — blank init, no pretrain
                "imgsz": 960,             # >=256 recommended; 640 for real microscopy data
            },
            "federation": {
                # ── mode selector ──────────────────────────────────────────
                "mode": "async",          # "async" or "sync"

                # ── shared (both modes) ────────────────────────────────────
                "local_epochs": 5,        # epochs per round max_concurrent_nodes: int = 1   # how many nodes may train simultaneously in async mode
                                    # 1 = sequential execution with async semantics (safe default)
                                    # >1 = true parallelism; requires enough RAM for N models
                                    #      simultaneously (each yolov8l at imgsz=960 ≈ 1–2 GB)
                                    # (sync) / per cycle (async)
                "batch_size": 8,
                "lr0": 0.0008,
                "warmup_steps": 20,
                "grad_clip": 10.0,
                "workers": 0,
                "device": "cuda",          # "cuda" if GPU available

                # ── sync only (ignored when mode: async) ───────────────────
                "rounds": 5,

                # ── async only (ignored when mode: sync) ───────────────────
                "async_node_cycles": 15,  # total submissions = n_nodes * async_node_cycles
                "staleness_alpha": 0.5,   # w = 1 / (1 + alpha * staleness)
                "max_concurrent_nodes": 1,  # nodes training simultaneously
                            # 1 = safe default (sequential w/ async semantics)
                            # increase only if RAM allows N models at once

                # ── live mAP evaluation ────────────────────────────────────
                # sync:  evaluate every N rounds      (1 = every round)
                # async: evaluate every N submissions (n_nodes = once per virtual round)
                "eval_interval": len(nodes_cfg),

                # ── pseudo-labeling ────────────────────────────────────────
                "pseudo_label": {
                    "enabled": True,
                    "start_round": 5,     # round (sync) or cycle index (async)
                    "conf_thresh": 0.6,
                    "weight": 0.5,
                },
            },
            "nodes": nodes_cfg,
            "output_dir": "runs/from_global_federation",
            "seed": 0,
        }
        Path(args.config_out).write_text(yaml.safe_dump(full_cfg, sort_keys=False))
        print(f"[config] wrote {args.config_out}")
        print(f"         review model.arch, model.imgsz, federation.device, and")
        print(f"         federation.rounds / async_node_cycles before launching.")


if __name__ == "__main__":
    main()
