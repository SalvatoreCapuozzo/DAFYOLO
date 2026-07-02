"""Generates a tiny synthetic toy dataset standing in for parasite-egg
microscopy images, split across 3 nodes with deliberately different,
overlapping class sets -- so you can run the whole pipeline end to end
before pointing it at real data.

    python tools/make_synthetic_dataset.py --out data/synthetic --n-train 24 --n-val 8

Each "egg" class is a distinct colored ellipse shape so the detector has a
real (if trivial) visual signal to learn. Node ownership:
    node_A: ascaris, trichuris
    node_B: hookworm
    node_C: ascaris, hookworm
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from PIL import Image, ImageDraw

GLOBAL_CLASSES = ["ascaris", "trichuris", "hookworm"]
CLASS_COLOR = {"ascaris": (210, 180, 80), "trichuris": (90, 160, 210), "hookworm": (200, 90, 140)}
CLASS_SHAPE = {"ascaris": "ellipse_wide", "trichuris": "ellipse_thin", "hookworm": "ellipse_round"}

NODES = {
    "node_A": ["ascaris", "trichuris"],
    "node_B": ["hookworm"],
    "node_C": ["ascaris", "hookworm"],
}

IMG_SIZE = 256
BG_COLOR = (235, 235, 225)  # pale microscope-slide background


def _draw_egg(draw: ImageDraw.ImageDraw, cls_name: str, cx: float, cy: float):
    color = CLASS_COLOR[cls_name]
    shape = CLASS_SHAPE[cls_name]
    if shape == "ellipse_wide":
        w, h = random.randint(28, 36), random.randint(14, 18)
    elif shape == "ellipse_thin":
        w, h = random.randint(34, 42), random.randint(8, 11)
    else:
        w, h = random.randint(18, 24), random.randint(16, 22)
    box = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
    draw.ellipse(box, fill=color, outline=(40, 40, 40))
    return w, h


def _make_image(classes_present: list[str], n_objects: int):
    img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), BG_COLOR)
    draw = ImageDraw.Draw(img)
    labels = []  # (cls_idx_within_classes_present, cx, cy, w, h) normalized
    for _ in range(n_objects):
        cls_name = random.choice(classes_present)
        margin = 30
        cx = random.uniform(margin, IMG_SIZE - margin)
        cy = random.uniform(margin, IMG_SIZE - margin)
        w, h = _draw_egg(draw, cls_name, cx, cy)
        labels.append((cls_name, cx / IMG_SIZE, cy / IMG_SIZE, w / IMG_SIZE, h / IMG_SIZE))
    return img, labels


def _write_split(node_name: str, owned: list[str], out_dir: Path, split: str, n_images: int, seed: int):
    img_dir = out_dir / node_name / "images" / split
    lbl_dir = out_dir / node_name / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    random.seed(seed)
    for i in range(n_images):
        n_objects = rng.randint(1, 3)
        img, labels = _make_image(owned, n_objects)
        img.save(img_dir / f"{i:04d}.jpg", quality=90)
        lines = []
        for cls_name, cx, cy, w, h in labels:
            local_id = owned.index(cls_name)
            lines.append(f"{local_id} {cx:.5f} {cy:.5f} {w:.5f} {h:.5f}")
        (lbl_dir / f"{i:04d}.txt").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/synthetic")
    parser.add_argument("--n-train", type=int, default=24)
    parser.add_argument("--n-val", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(args.out)
    for idx, (node_name, owned) in enumerate(NODES.items()):
        _write_split(node_name, owned, out_dir, "train", args.n_train, seed=args.seed + idx * 2)
        _write_split(node_name, owned, out_dir, "val", args.n_val, seed=args.seed + idx * 2 + 1)

        data_yaml = out_dir / node_name / "data.yaml"
        names_block = "\n".join(f"  {i}: {c}" for i, c in enumerate(owned))
        data_yaml.write_text(
            f"path: {(out_dir / node_name).resolve()}\n"
            f"train: images/train\n"
            f"val: images/val\n"
            f"names:\n{names_block}\n"
        )
        print(f"wrote {node_name}: owns={owned} -> {data_yaml}")

    print(f"\nDone. Point configs/*.yaml nodes at: {out_dir.resolve()}/<node_name>/data.yaml")


if __name__ == "__main__":
    main()
