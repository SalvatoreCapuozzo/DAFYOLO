# DAFYOLO — Disjoint Asynchronous Federated YOLO

Federated object detection where nodes hold disjoint or overlapping class
subsets. Two implementations live in this repo:

## [`fedyolo_system/`](fedyolo_system/) — current

A class-decomposed rewrite: every per-class output channel aggregates only
across the nodes that own that class, so training doesn't need a merge
heuristic. Sync and async federation modes, a centralized baseline for
comparison, and a synthetic dataset generator for a from-scratch smoke test.
Start with [`fedyolo_system/README.md`](fedyolo_system/README.md).

## [`legacy/`](legacy/) — prior approach, kept for reference

Client/server scripts (`client_updated.py` + `server_updated.py`, plus
earlier `client_v2.py`/`server_v2.py` and `run_experiments.py`) implementing
a selectable-strategy merge approach (FedHead, Stitch, TIES, FedAvg,
FedProx, YOLO-Inc, DFKD...). `client_updated.py`/`server_updated.py` are the
most current pair in this set; `client_v2.py`/`server_v2.py` are kept only
for the experimental DFKD strategy, which isn't ported anywhere else.
Machine-specific paths (remote GPU box vs. this Mac) live in
[`legacy/server_paths.yaml`](legacy/server_paths.yaml).

## [`dafyolo_field_report.html`](dafyolo_field_report.html)

A standalone diagnostic report comparing both systems — logged results,
root-cause analysis, and a macOS compatibility audit. Open it directly in a
browser; it has no external dependencies besides a Google Fonts stylesheet.

## Install

Each implementation has its own `requirements.txt`:
```bash
pip install -r legacy/requirements.txt          # legacy client/server scripts
pip install -r fedyolo_system/requirements.txt   # fedyolo_system
```
