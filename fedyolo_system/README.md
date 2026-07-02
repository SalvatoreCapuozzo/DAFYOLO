# fedyolo

Federated learning for YOLOv8 object detectors where every node starts from
blank (randomly initialized) weights, all nodes share the same visual domain,
but different nodes can own **different, possibly non-overlapping subsets of
classes** (e.g. one node only ever sees Ascaris/Trichuris eggs, another only
Hookworm). The output of training is one global multi-class model.

## Why this needs more than plain FedAvg

If you naively pool gradients across nodes with different class sets, a node
that has never annotated "Hookworm" will, every batch, teach the model that
the Hookworm channel should always say "no object" -- even on images that
might contain one. Averaged into the global model, this actively erases what
other nodes have learned about that class. This framework fixes that with
three things working together:

1. **A class-decomposed head.** YOLOv8's classification branch ends each
   detection scale in a 1x1 conv whose output channel `c` is literally the
   logit for class `c` (`model.{H}.cv3.{i}.2.{weight,bias}`, verified against
   the installed Ultralytics build in `fedyolo/model.py`). Every other
   weight -- backbone, neck, box-regression branch, DFL, even the two
   feature-mixing convs inside the classification branch before that final
   projection -- is class-agnostic.
2. **Loss masking.** Each node zeroes the classification loss for classes it
   doesn't own (via Ultralytics' built-in `model.class_weights` hook), so it
   never penalizes a class it has no ground truth for.
3. **Class-conditional aggregation.** The server FedAvgs every shared weight
   across all nodes (sample-weighted), but for each per-class output channel
   it only averages across the nodes that actually own that class
   (instance-count-weighted). A class owned by one node is simply copied
   from that node until others learn it too.

There's an optional fourth piece, **cross-round pseudo-labeling**
(`fedyolo/pseudo_label.py`): once the global model is decent, each node
distills the previous round's global model's confident predictions for
classes it doesn't own into its own training, closing more of the gap.

See `fedyolo/model.py`, `fedyolo/client.py`, `fedyolo/server.py` for the
actual mechanics and `fedyolo/centralized.py` for the comparison baseline.

## Honest caveats

- **No literal guarantee.** How close the federated model gets to a
  pooled/centralized model depends on how disjoint the class sets are, how
  much per-class data each node has, total compute (rounds x local epochs),
  and how non-iid the images themselves are. This framework minimizes the
  *federation-specific* loss; it can't manufacture information nodes never
  collected.
- **What "centralized baseline" means here.** `fedyolo/centralized.py`
  trains one model directly via SGD over the union of all nodes' images,
  with the *same* per-node class masking the federation uses (since a node's
  images genuinely don't have ground truth for classes it doesn't own,
  pooling them doesn't fix that). This isolates the cost of federation
  *specifically* from the cost of partial labels, which both approaches face
  equally. If you have a separate, fully cross-annotated dataset (every
  image labeled for every class), evaluate against that too -- it's a
  stronger, different baseline (the true label-complete ceiling).
- **Blank-weight YOLO training is data/compute hungry, and image resolution
  matters more than you'd expect.** Validated during development: at 128px
  or 192px input, this tiny synthetic dataset's classifier head got stuck
  predicting "background" everywhere (sigmoid outputs stuck under 0.5%) even
  after 100+ epochs -- the feature maps were too coarse for small objects
  relative to image size. The exact same setup at 256px converged cleanly
  within the same epoch budget. If your real detector seems to be making no
  progress, check this before assuming a deeper bug: inspect raw classifier
  logits (see "Debugging a stuck classifier" below) before tuning learning
  rate further.
- **This is a single-machine simulation** (`ProcessPoolExecutor`, one OS
  process per node, real isolation -- a node's process never touches another
  node's data or memory). For actual cross-machine deployment, swap the
  `ProcessPoolExecutor` in `fedyolo/server.py` for a network transport (gRPC,
  HTTP, or a library like Flower) carrying the same state_dict payloads;
  `client.run_node_round` and `model.aggregate` don't need to change.

## Validated end-to-end

This was tested on a tiny synthetic 3-node, 3-class dataset (disjoint shapes
standing in for egg classes) through the full pipeline: federated training,
class-conditional aggregation, and evaluation on a pooled validation set.
Result after 4 rounds x 40 local epochs (256px, yolov8n from scratch, CPU):

```
            mAP50   mAP50-95
ascaris     0.957     0.765
trichuris   0.995     0.597   (1 val instance -- noisy, but learned)
hookworm    0.995     0.738
overall     0.982     0.700
```

All three classes were learned well despite no single node ever owning all
three -- node_A only ever saw ascaris/trichuris, node_B only hookworm,
node_C ascaris/hookworm. That's the class-decomposed aggregation working as
intended.

## Install

```bash
pip install -r requirements.txt
```

## Quickstart (synthetic data, validates the whole pipeline in a few minutes)

```bash
python tools/make_synthetic_dataset.py --out data/synthetic --n-per-node 24
python -m fedyolo.simulate --config configs/example_federation.yaml
# add --also-centralized to also train and report the non-federated baseline
```

Outputs land in `runs/example_federation/`: a checkpoint per round, the final
global model (`global_final.pt`), and `summary.json` with per-class and
overall mAP.

## Using your own data

Each node needs a standard Ultralytics YOLO dataset: an `images/{train,val}`
+ `labels/{train,val}` layout plus a `data.yaml`:

```yaml
path: data/your_node
train: images/train
val: images/val
names:
  0: ascaris
  1: trichuris
```

Label `.txt` files use **local** class indices (0..k-1) matching the order
of `names` above -- this must exactly match the `owned_classes` order you
give that node in the federation config:

```yaml
nodes:
  - name: node_A
    data_yaml: data/your_node/data.yaml
    owned_classes: [ascaris, trichuris]   # index 0 -> ascaris, index 1 -> trichuris
```

`global_classes` in the config is the full canonical class list across every
node; `fedyolo/config.py` validates that every global class is owned by at
least one node and that owned-class lists don't contain duplicates or
unknown names.

If your data is in COCO JSON format, convert it first:

```bash
python tools/coco_to_yolo.py --json path/to/annotations.json --images path/to/images --out data/your_node
```

## Splitting one global dataset into per-node datasets

If you have a single pooled YOLO-format dataset (one `images/` + `labels/` +
`classes_list.txt`, all classes mixed together) rather than data that's
already physically separated by site, `tools/split_dataset_into_nodes.py`
turns it into a federation testbed:

```bash
python tools/split_dataset_into_nodes.py \
    --dataset /path/to/dataset_folder \
    --out data/from_global \
    --n-nodes 4 \
    --drop-classes bubble,dirt \
    --config-out configs/from_global_federation.yaml
```

Run with `--dry-run` first to see the proposed class-to-node split and
per-node image/instance counts without writing anything.

What it does, and the choices it makes:

- **Class-to-node assignment is balanced automatically** -- classes are
  sorted by total instance count and greedily dropped onto whichever node
  currently has the smallest total, so no node ends up starved while
  another hoards every common class. `--n-nodes` controls how many.
- **An image can land in more than one node.** If a single image has both a
  Trichuris box (owned by node_A) and an Ascaris box (owned by node_B), it's
  included in *both* nodes' datasets, but each copy's label file only keeps
  that node's own classes -- exactly the partial-annotation scenario the
  framework's loss masking handles. Images whose only annotations are
  dropped classes are excluded entirely (no owned class present).
- **`--drop-classes`** (default `bubble,dirt`) removes named classes from
  every label file and the global class list entirely, before any node
  assignment happens -- use this for artifact/distractor classes that
  aren't real detection targets you want a node to ever own.
- **Label index base (0-based vs 1-based) is auto-detected** from what's
  actually in your label files (some annotation tools export 1-based ids
  matching a human-facing "select the class" number instead of standard
  YOLO 0-based ids). It's conclusive when class id `0` appears anywhere, or
  when the max id used equals the class count; otherwise it defaults to
  0-based with a warning. Override with `--label-index-base 0` or `1` if
  you know better than the auto-detection.
- **`--config-out`** writes a ready-to-use federation config pointing at the
  generated node datasets. It ships with placeholder `federation.rounds` /
  `local_epochs` / `imgsz` values copied from the validated example --
  review these against your actual dataset size before training (see the
  "Honest caveats" section above; what worked for a 12-image toy node will
  not be the right budget for your real data, and what's right for
  yours depends on dataset size, which you'll need to tune for).

This was validated end-to-end on a synthetic stand-in built from your actual
`classes_list.txt`: class dropping, the 0-based/1-based auto-detection (both
directions), the balanced split, the multi-node image assignment, and the
resulting node datasets loading correctly through `fedyolo.data` were all
checked before delivery.



| field | meaning |
|---|---|
| `global_classes` | canonical class list, shared index space across all nodes |
| `model.arch` | Ultralytics architecture yaml (e.g. `yolov8n.yaml`/`yolov8s.yaml`...) -- random init, no pretrained checkpoint downloaded |
| `model.imgsz` | input resolution; see the caveat above about small objects needing larger imgsz |
| `federation.rounds` / `local_epochs` | total local epochs per node = rounds x local_epochs |
| `federation.lr0` / `warmup_steps` / `grad_clip` | every local round reloads fresh global weights into a new optimizer, so a short per-round LR warmup + gradient clipping matters a lot for stability from random init |
| `federation.pseudo_label.*` | optional cross-round self-distillation for not-owned classes (see above) |
| `nodes[].owned_classes` | which global classes this node has ground truth for, in local-label-index order |

## Debugging a stuck classifier

If mAP stays at exactly 0 no matter how long you train, check raw logits
before tuning hyperparameters further:

```python
import torch
from fedyolo.config import load_config
from fedyolo.model import build_model

cfg = load_config("configs/your_config.yaml")
ckpt = torch.load("runs/your_run/global_final.pt")
model = build_model(cfg.model.arch, cfg.nc, cfg.model.imgsz)
model.load_state_dict(ckpt["state_dict"]); model.eval()

img = torch.rand(1, 3, cfg.model.imgsz, cfg.model.imgsz)  # or a real validation image
with torch.no_grad():
    _, raw = model.forward(img)
print(raw["scores"].sigmoid().amax(dim=(0, 2)))  # per-class max confidence anywhere in the image
```

If every class is stuck under ~1%, the classifier hasn't learned to fire at
all yet -- almost always more local epochs, a larger `imgsz` relative to
object size, or both, rather than a bug in the aggregation logic.

## Project layout

```
fedyolo/
  config.py        federation YAML schema + validation
  model.py          build_model(), the shared/per-class parameter split, aggregation
  data.py           per-node dataset loading with local -> global class id remapping
  client.py         one node's local training for one round (runs in its own process)
  server.py         round orchestration: dispatch to node processes, aggregate, checkpoint
  pseudo_label.py    optional cross-round self-distillation for not-owned classes
  centralized.py     non-federated baseline trainer for comparison
  pooling.py         materializes a pooled, class-remapped val set on disk for evaluation
  evaluate.py        runs Ultralytics' own validator on the pooled val set
  simulate.py         CLI entry point
tools/
  make_synthetic_dataset.py        tiny toy dataset generator for smoke-testing
  split_dataset_into_nodes.py      splits one pooled global dataset into a balanced federation
  coco_to_yolo.py                  COCO JSON -> per-node YOLO dataset converter
configs/
  example_federation.yaml
```

## How to launch
```
python tools/split_dataset_into_nodes.py \
    --dataset /path/to/dataset_folder \
    --out data/from_global \
    --n-nodes 4 \
    --drop-classes bubble,dirt \
    --config-out configs/from_global_federation.yaml
python -m fedyolo.simulate \
    --config configs/from_global_federation.yaml \
    --also-centralized
```