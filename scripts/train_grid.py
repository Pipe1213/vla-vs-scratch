#!/usr/bin/env python
"""Train the frozen experiment grid (configs/runs.yaml + configs/splits.json).

ACT runs train locally, one after another (Phase 2). SmolVLA runs are submitted to
HF Jobs (Phase 3). Everything comes from the frozen config: no settings live here.

Usage (repo root, `lerobot` env):
  python scripts/train_grid.py --method act --dry-run      # print the 12 commands, run nothing
  python scripts/train_grid.py --method act                # train all 12 ACT runs (skips finished ones)
  python scripts/train_grid.py --method act --only act_obj_t1_n05   # one run
  python scripts/train_grid.py --method smolvla --dry-run  # Phase 3

Resumable: a run is "done" when its last eval checkpoint exists (ACT) or its job was
submitted (SmolVLA, recorded in outputs/logs/submitted_jobs.txt). Re-running the
script after a crash/restart continues with the next unfinished run. A run that
crashed half-way is restarted from scratch (its output dir is moved to *_failed_<time>).

After each successful ACT run: keeps only the eval checkpoints' `pretrained_model`
(deletes other checkpoints and all `training_state` folders: ~0.6 GB -> ~0.2 GB per
checkpoint), and appends a row to results/compute_log.csv.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
SIZE_TAG = {"5": "n05", "10": "n10", "20": "n20", "all": "nall"}


def load():
    cfg = yaml.safe_load((ROOT / "configs/runs.yaml").read_text())
    splits = json.loads((ROOT / "configs/splits.json").read_text())
    assert splits["dataset"]["revision"] == cfg["dataset"]["revision"], "splits.json and runs.yaml disagree on dataset revision"
    return cfg, splits


def runs_for(method: str, cfg, splits):
    out = []
    for t_label in cfg["tasks"]:                      # t1, t2, t3
        for size in cfg["demo_sizes"]:                # 5, 10, 20, all
            name = f"{method}_obj_{t_label}_{SIZE_TAG[size]}"
            eps = splits["tasks"][t_label]["splits"][size]
            out.append({"name": name, "task": t_label, "size": size, "episodes": eps})
    return out


def train_cmd(method: str, run: dict, cfg) -> list[str]:
    m = cfg["methods"][method]
    common = [
        "lerobot-train",
        f"--dataset.repo_id={cfg['dataset']['repo_id']}",
        f"--dataset.revision={cfg['dataset']['revision']}",
        f"--dataset.episodes={json.dumps(run['episodes'])}",
        f"--seed={cfg['training_seed']}",
        "--log_freq=200",
        f"--output_dir=outputs/train/{run['name']}",
        f"--job_name={run['name']}",
        "--wandb.enable=true",
    ]
    extra = list(m["train_args"])
    if method == "smolvla":
        extra.append(f"--policy.repo_id=Pipe1213/{run['name']}")
    return common + extra


def ckpt_dir(run) -> Path:
    return ROOT / "outputs/train" / run["name"] / "checkpoints"


def is_done_act(run, cfg) -> bool:
    last = cfg["methods"]["act"]["eval_checkpoints"][-1]
    return (ckpt_dir(run) / last / "pretrained_model" / "model.safetensors").exists()


def prune_act(run, cfg):
    keep = set(cfg["methods"]["act"]["eval_checkpoints"])
    freed = 0
    for d in sorted(ckpt_dir(run).iterdir()):
        if d.is_symlink() or not d.is_dir():
            continue
        if d.name not in keep:
            freed += sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
            shutil.rmtree(d)
        else:
            ts = d / "training_state"
            if ts.exists():
                freed += sum(f.stat().st_size for f in ts.rglob("*") if f.is_file())
                shutil.rmtree(ts)
    last = ckpt_dir(run) / "last"
    if last.is_symlink() and not last.exists():   # 'last' pointed at a deleted folder
        last.unlink()
    print(f"  pruned checkpoints, freed {freed / 1e9:.1f} GB")


def log_compute(run, method, steps, hours, where, hardware, notes):
    path = ROOT / "results/compute_log.csv"
    with open(path, "a", newline="") as f:
        csv.writer(f).writerow([dt.date.today().isoformat(), run["name"], method, run["task"],
                                run["size"], steps, where, hardware, round(hours, 2), 0, notes])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", choices=["act", "smolvla"], required=True)
    ap.add_argument("--only", nargs="*", help="run names to train (default: all 12)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg, splits = load()
    runs = runs_for(args.method, cfg, splits)
    if args.only:
        runs = [r for r in runs if r["name"] in set(args.only)]
        assert runs, f"no run matches {args.only}"
    (ROOT / "outputs/logs").mkdir(parents=True, exist_ok=True)
    steps = next(a.split("=", 1)[1] for a in cfg["methods"][args.method]["train_args"] if a.startswith("--steps="))

    submitted_file = ROOT / "outputs/logs/submitted_jobs.txt"
    submitted = set(submitted_file.read_text().split()) if submitted_file.exists() else set()

    todo = []
    for r in runs:
        done = is_done_act(r, cfg) if args.method == "act" else r["name"] in submitted
        print(f"{'[done]' if done else '[todo]'} {r['name']:<22} task {r['task']} (sim {cfg['tasks'][r['task']]}), "
              f"{len(r['episodes']):>2} demos")
        if not done:
            todo.append(r)
    print(f"\n{len(todo)} run(s) to {'train' if args.method == 'act' else 'submit'}.")

    for i, r in enumerate(todo, 1):
        cmd = train_cmd(args.method, r, cfg)
        print(f"\n=== [{i}/{len(todo)}] {r['name']} ===")
        if args.dry_run:
            print(" \\\n  ".join(cmd))
            continue
        out_dir = ROOT / "outputs/train" / r["name"]
        if args.method == "act" and out_dir.exists():     # half-finished run from a crash
            moved = out_dir.with_name(out_dir.name + "_failed_" + dt.datetime.now().strftime("%m%d-%H%M"))
            out_dir.rename(moved)
            print(f"  previous incomplete output moved to {moved.name}")
        log_path = ROOT / "outputs/logs" / f"{r['name']}.log"
        t0 = time.time()
        with open(log_path, "w") as log:
            ret = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT).returncode
        hours = (time.time() - t0) / 3600
        if args.method == "act":
            if ret != 0 or not is_done_act(r, cfg):
                print(f"  FAILED (exit {ret}) after {hours * 60:.0f} min; see {log_path}. Continuing with the next run.")
                continue
            print(f"  finished in {hours * 60:.0f} min")
            prune_act(r, cfg)
            log_compute(r, "act", steps, hours, "laptop", "RTX 4050 6GB", "Phase 2 grid run")
        else:
            if ret != 0:
                print(f"  submission FAILED (exit {ret}); see {log_path}")
                continue
            with open(submitted_file, "a") as f:
                f.write(r["name"] + "\n")
            print(f"  submitted; job id / repo in {log_path}")

    if not args.dry_run:
        print("\nAll done. Check: python scripts/train_grid.py --method", args.method, "--dry-run  (should list every run as [done])")


if __name__ == "__main__":
    sys.exit(main())
