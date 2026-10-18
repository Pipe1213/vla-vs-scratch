#!/usr/bin/env python
"""Evaluate the frozen grid: every run x its 3 eval checkpoints x 50 episodes.

Uses scripts/eval_sweep.py (same fixed settings as the pilot: 256 px, init states,
seed 1000, batch 1) with the method's eval_args from configs/runs.yaml.
One CSV row per (run, checkpoint) in results/results.csv. Finished evaluations are
skipped, so it can be stopped and restarted at any time.

Usage (repo root, `lerobot` env, MUJOCO_GL=egl):
  python scripts/eval_grid.py --method act --dry-run      # list what would run
  python scripts/eval_grid.py --method act                # evaluate all trained ACT runs
  python scripts/eval_grid.py --method act --only act_obj_t1_n05
  python scripts/eval_grid.py --method smolvla --download # Phase 3: fetch checkpoints from the Hub first
  python scripts/eval_grid.py --method act --summary      # print per-run means from results.csv

Do not run while a training run is using the GPU (results stay valid, but both slow down).
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_grid import ROOT, load, runs_for  # noqa: E402

OUT = "results/results.csv"


def ckpt_path(run_name: str, ckpt: str) -> Path:
    return ROOT / "outputs/train" / run_name / "checkpoints" / ckpt / "pretrained_model"


def download_smolvla(run_name: str, ckpts: list[str]):
    """LeRobot appends a timestamp to the Hub repo name; take the newest matching repo."""
    from huggingface_hub import HfApi, snapshot_download

    api = HfApi()
    me = api.whoami()["name"]
    repos = sorted(m.id for m in api.list_models(author=me, search=run_name) if m.id.split("/")[1].startswith(run_name))
    if not repos:
        print(f"  [warn] no Hub repo found for {run_name}")
        return
    repo = repos[-1]
    patterns = [f"checkpoints/{c}/pretrained_model/*" for c in ckpts]
    print(f"  downloading {repo} ({', '.join(ckpts)})")
    snapshot_download(repo, allow_patterns=patterns, local_dir=ROOT / "outputs/train" / run_name)


def summary(method: str):
    path = ROOT / OUT
    if not path.exists():
        print("no results yet"); return
    agg = defaultdict(lambda: [0, 0, []])
    for r in csv.DictReader(open(path)):
        if r["name"].startswith(method + "_"):
            a = agg[r["name"]]
            a[0] += int(r["n_success"]); a[1] += int(r["n_episodes"]); a[2].append(f"{r['label']}:{r['n_success']}")
    print(f"{'run':<24} {'success':>11}  per checkpoint")
    for name in sorted(agg):
        k, n, per = agg[name]
        print(f"{name:<24} {k:>3}/{n:<3} {100 * k / n:5.1f}%  {'  '.join(per)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", choices=["act", "smolvla"], required=True)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--download", action="store_true", help="SmolVLA: download eval checkpoints from the Hub first")
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()

    if args.summary:
        return summary(args.method)

    cfg, splits = load()
    m = cfg["methods"][args.method]
    ckpts = m["eval_checkpoints"]
    episodes = cfg["evaluation"]["episodes_per_checkpoint"]
    runs = runs_for(args.method, cfg, splits)
    if args.only:
        runs = [r for r in runs if r["name"] in set(args.only)]

    for r in runs:
        if args.download and args.method == "smolvla" and not args.dry_run:
            download_smolvla(r["name"], ckpts)
        paths = [ckpt_path(r["name"], c) for c in ckpts]
        missing = [str(p) for p in paths if not (p / "model.safetensors").exists()]
        if missing:
            print(f"[skip] {r['name']}: not trained/downloaded yet ({len(missing)} checkpoint(s) missing)")
            continue
        cmd = [sys.executable, "scripts/eval_sweep.py", "--name", r["name"],
               "--task-id", str(cfg["tasks"][r["task"]]), "--episodes", str(episodes), "--out", OUT,
               *[str(p.relative_to(ROOT)) for p in paths], "--", *m["eval_args"]]
        print(f"\n=== {r['name']} (sim task {cfg['tasks'][r['task']]}, {len(r['episodes'])} demos) ===")
        if args.dry_run:
            print(" ".join(cmd)); continue
        subprocess.run(cmd, cwd=ROOT)

    if not args.dry_run:
        print(); summary(args.method)


if __name__ == "__main__":
    sys.exit(main())
