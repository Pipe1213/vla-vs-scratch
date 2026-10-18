#!/usr/bin/env python
"""Evaluate one or more checkpoints with the experiment's fixed evaluation settings.

Every run uses the same settings (fairness rules 3, 7, 8 of the plan):
  libero_object, one task, LIBERO's fixed initial states, hard reset (LeRobot default),
  256x256 rendering, batch size 1, seed 1000.
Results are appended to a CSV (one row per checkpoint, with per-episode outcomes) so
nothing has to be copied by hand. Finished evaluations are skipped on re-run.

Examples
  # ACT pilot: all saved checkpoints, 20 episodes each
  python scripts/eval_sweep.py --name act_t1_pilot --task-id 3 --episodes 20 \
      outputs/train/act_obj_t1_nall_pilot/checkpoints/*/pretrained_model

  # Same checkpoint, different inference setting (anything after -- goes to lerobot-eval)
  python scripts/eval_sweep.py --name act_t1_pilot_nas20 --task-id 3 --episodes 20 \
      outputs/train/act_obj_t1_nall_pilot/checkpoints/100000/pretrained_model \
      -- --policy.n_action_steps=20

  # SmolVLA (needs the camera rename used in training)
  python scripts/eval_sweep.py --name smolvla_t1_pilot --task-id 3 --episodes 20 \
      <dir1> <dir2> -- --policy.n_action_steps=10 \
      --rename_map='{"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}'

Run from the repo root inside the `lerobot` env with MUJOCO_GL=egl set.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import subprocess
import sys
import time
from pathlib import Path

FIXED = [
    "--env.type=libero",
    "--env.task=libero_object",
    "--env.init_states=true",
    "--env.observation_height=256",
    "--env.observation_width=256",
    "--eval.batch_size=1",
    "--seed=1000",
]
FIELDS = [
    "timestamp", "name", "label", "task_id", "n_episodes", "n_success", "pc_success",
    "ci95_lo", "ci95_hi", "s_per_episode", "wall_min", "extra_args", "checkpoint", "successes",
]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (100 * max(0.0, centre - half), 100 * min(1.0, centre + half))


def label_for(path: Path) -> str:
    """'.../checkpoints/020000/pretrained_model' -> '020000'; otherwise the folder name."""
    parts = path.resolve().parts
    if parts[-1] == "pretrained_model" and len(parts) >= 2:
        return parts[-2]
    return parts[-1]


def read_result(info_path: Path) -> dict:
    d = json.loads(info_path.read_text())
    o = d["overall"]
    succ = []
    for t in d.get("per_task", []):
        succ += [int(bool(x)) for x in t.get("metrics", {}).get("successes", [])]
    return {
        "n": int(o["n_episodes"]),
        "k": int(o.get("n_success", round(o["pc_success"] * o["n_episodes"] / 100))),
        "s_ep": float(o.get("eval_ep_s", float("nan"))),
        "succ": "".join(map(str, succ)),
    }


def main() -> None:
    argv = sys.argv[1:]
    extra = []
    if "--" in argv:
        i = argv.index("--")
        argv, extra = argv[:i], argv[i + 1 :]

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoints", nargs="+", help="pretrained_model directories (or Hub repo ids)")
    p.add_argument("--name", required=True, help="run group name, e.g. act_t1_pilot")
    p.add_argument("--task-id", type=int, required=True, help="simulator task id (--env.task_ids)")
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--out", default="results/pilot_sweep.csv")
    p.add_argument("--eval-root", default="outputs/eval")
    args = p.parse_args(argv)

    out_csv = Path(args.out)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    new_file = not out_csv.exists()
    rows = []

    ckpts = sorted(args.checkpoints, key=lambda c: label_for(Path(c)))
    print(f"{len(ckpts)} checkpoint(s), {args.episodes} episodes each, task {args.task_id}, extra={extra}")
    for c in ckpts:
        label = label_for(Path(c))
        odir = Path(args.eval_root) / args.name / f"{label}_ep{args.episodes}"
        info = odir / "eval_info.json"
        if info.exists():
            print(f"[skip] {label}: already evaluated ({info})")
            continue
        odir.mkdir(parents=True, exist_ok=True)
        cmd = ["lerobot-eval", f"--policy.path={c}", *FIXED,
               f"--env.task_ids=[{args.task_id}]", f"--eval.n_episodes={args.episodes}",
               f"--output_dir={odir}", *extra]
        print(f"\n[run] {label}  ->  {odir}")
        t0 = time.time()
        with open(odir / "eval.log", "w") as log:
            ret = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode
        wall = (time.time() - t0) / 60
        if ret != 0 or not info.exists():
            print(f"[FAIL] {label} (exit {ret}); see {odir / 'eval.log'}")
            continue
        r = read_result(info)
        lo, hi = wilson(r["k"], r["n"])
        row = {
            "timestamp": dt.datetime.now().isoformat(timespec="seconds"), "name": args.name,
            "label": label, "task_id": args.task_id, "n_episodes": r["n"], "n_success": r["k"],
            "pc_success": round(100 * r["k"] / r["n"], 1), "ci95_lo": round(lo, 1), "ci95_hi": round(hi, 1),
            "s_per_episode": round(r["s_ep"], 1), "wall_min": round(wall, 1),
            "extra_args": " ".join(extra), "checkpoint": c, "successes": r["succ"],
        }
        with open(out_csv, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if new_file:
                w.writeheader()
                new_file = False
            w.writerow(row)
        rows.append(row)
        print(f"[done] {label}: {r['k']}/{r['n']} = {row['pc_success']}% "
              f"(CI {row['ci95_lo']}-{row['ci95_hi']}), {row['s_per_episode']} s/ep, {row['wall_min']} min")

    if rows:
        print(f"\nSummary ({args.name}) -> {out_csv}")
        print(f"{'checkpoint':>12} {'success':>9} {'95% CI':>13} {'s/ep':>6}")
        for r in rows:
            print(f"{r['label']:>12} {r['n_success']:>3}/{r['n_episodes']:<3}  "
                  f"{r['ci95_lo']:>5}-{r['ci95_hi']:<5}  {r['s_per_episode']:>6}")


if __name__ == "__main__":
    main()
