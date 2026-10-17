#!/usr/bin/env python
"""Build the nested demonstration splits for the experiment.

Two steps:

1. Inspect (default): list every task of the suite with its demo count and lengths.
       python scripts/make_splits.py

2. Write: pick tasks by their *simulator* task id (the number --env.task_ids uses)
   and write configs/splits.json.
       python scripts/make_splits.py --tasks 0 3 7

Why this exists
- lerobot/libero mixes 4 suites (40 tasks, 1,693 episodes) and is ordered by suite,
  so "the first N episodes" are the wrong task. Episodes are selected by task here.
- The dataset numbers tasks 0-39 (task_index); the simulator numbers them 0-9 inside
  the suite (--env.task_ids). The two are matched by instruction text.
- Splits are nested: one shuffled order per task (fixed seed), then the first
  5 / 10 / 20 / all episodes of that order. 5 is inside 10, 10 inside 20, and so on.
- Only metadata is downloaded (a few MB), not the 1.9 GB of videos.

Run inside the `lerobot` conda env (needs lerobot, huggingface_hub, and libero for
the simulator task names).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import re
import statistics
import sys
from pathlib import Path

DEFAULT_SIZES = [5, 10, 20]  # plus "all"
DEBUG = False


# ----------------------------------------------------------------------------- helpers
def norm(text: str) -> str:
    """Normalize an instruction so dataset and simulator strings compare equal."""
    text = text.strip().lower().replace("_", " ")
    text = re.sub(r"\s+", " ", text)
    return text.rstrip(".")


def nested_splits(episode_ids: list[int], sizes: list[int], seed: int, key: int) -> dict:
    """One shuffled order per task; each split is a prefix of that order (=> nested)."""
    order = sorted(episode_ids)
    random.Random(seed * 1000 + key).shuffle(order)
    splits = {str(n): sorted(order[:n]) for n in sizes if n <= len(order)}
    splits["all"] = sorted(order)
    # Nesting check: every smaller split is contained in the next one.
    keys = [str(n) for n in sizes if n <= len(order)] + ["all"]
    for small, big in zip(keys, keys[1:]):
        assert set(splits[small]) <= set(splits[big]), (small, big)
    return {"shuffled_order": order, "splits": splits}


# ----------------------------------------------------------------------------- loaders
def load_dataset_meta(repo_id: str, revision: str | None):
    """Return (revision_sha, codebase_version, tasks{task_index: text}, episodes[list of dict])."""
    from huggingface_hub import HfApi

    sha = HfApi().dataset_info(repo_id, revision=revision).sha

    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    except ImportError:  # older LeRobot layout
        from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata

    meta = LeRobotDatasetMetadata(repo_id, revision=sha)
    version = meta.info.get("codebase_version", "?")

    # Tasks: v3 = DataFrame indexed by task text with a task_index column; v2.x = dict.
    tasks_obj = meta.tasks
    if hasattr(tasks_obj, "iterrows"):
        tasks = {int(row["task_index"]): str(text) for text, row in tasks_obj.iterrows()}
    else:
        tasks = {int(k): str(v) for k, v in tasks_obj.items()}
    text_to_index = {norm(t): i for i, t in tasks.items()}

    # Episodes: v3 = HF Dataset with columns episode_index, tasks, length; v2.x = dict.
    eps_obj = meta.episodes
    if hasattr(eps_obj, "to_pandas"):
        rows = eps_obj.to_pandas().to_dict("records")
    elif hasattr(eps_obj, "to_dict") and hasattr(eps_obj, "columns"):
        rows = eps_obj.to_dict("records")
    elif isinstance(eps_obj, dict):
        rows = list(eps_obj.values())
    else:
        rows = list(eps_obj)

    if DEBUG and rows:
        print("[debug] episode metadata columns:", sorted(rows[0].keys()))
        print("[debug] first row:", {k: rows[0][k] for k in list(rows[0])[:8]})

    def task_indices_of(r) -> list[int]:
        # Strategy 1: the `tasks` column (instruction text, a list or a single string).
        raw = r.get("tasks")
        if raw is not None:
            items = [raw] if isinstance(raw, str) else [str(x) for x in _flat(raw)]
            idxs = sorted({text_to_index[norm(t)] for t in items if norm(t) in text_to_index})
            if idxs:
                return idxs
        # Strategy 2: per-episode stats of the task_index feature (v3 metadata).
        lo, hi = r.get("stats/task_index/min"), r.get("stats/task_index/max")
        if lo is not None and hi is not None:
            lo, hi = int(_flat(lo)[0]), int(_flat(hi)[0])
            return [lo] if lo == hi else list(range(lo, hi + 1))
        return []

    episodes = [
        {"episode_index": int(r["episode_index"]), "length": int(r["length"]), "task_indices": task_indices_of(r)}
        for r in rows
    ]

    # Strategy 3 (fallback): read episode_index/task_index from the data parquet files
    # (tabular frames only, no videos; tens of MB).
    if all(len(e["task_indices"]) != 1 for e in episodes):
        print("  episode metadata has no usable task info; reading task_index from the data files ...")
        ep_to_tasks = _task_index_from_data(repo_id, sha)
        for e in episodes:
            e["task_indices"] = sorted(ep_to_tasks.get(e["episode_index"], []))
    return sha, version, tasks, episodes


def _flat(x) -> list:
    """Flatten numpy arrays / nested lists to a plain list."""
    try:
        import numpy as np

        return list(np.asarray(x, dtype=object).ravel())
    except Exception:
        return list(x) if isinstance(x, (list, tuple)) else [x]


def _task_index_from_data(repo_id: str, sha: str) -> dict[int, set[int]]:
    import pandas as pd
    from huggingface_hub import snapshot_download

    root = snapshot_download(repo_id, repo_type="dataset", revision=sha, allow_patterns=["data/**/*.parquet"])
    mapping: dict[int, set[int]] = {}
    for f in sorted(Path(root, "data").rglob("*.parquet")):
        df = pd.read_parquet(f, columns=["episode_index", "task_index"]).drop_duplicates()
        for ep, ti in zip(df["episode_index"], df["task_index"]):
            mapping.setdefault(int(ep), set()).add(int(ti))
    return mapping


def load_sim_tasks(suite: str) -> list[str]:
    """Instruction text for simulator task ids 0..n-1 (the order --env.task_ids uses)."""
    from libero.libero import benchmark

    s = benchmark.get_benchmark_dict()[suite]()
    return [s.get_task(i).language for i in range(s.n_tasks)]


def lerobot_version() -> str:
    try:
        from importlib.metadata import version

        return version("lerobot")
    except Exception:
        return "?"


# ----------------------------------------------------------------------------- main
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo-id", default="lerobot/libero")
    p.add_argument("--revision", default=None, help="dataset commit to pin (default: current main, recorded)")
    p.add_argument("--suite", default="libero_object")
    p.add_argument("--tasks", type=int, nargs="*", help="simulator task ids to keep, in order -> t1, t2, t3")
    p.add_argument("--sizes", type=int, nargs="*", default=DEFAULT_SIZES)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="configs/splits.json")
    p.add_argument("--debug", action="store_true", help="print the episode metadata columns")
    args = p.parse_args()
    global DEBUG
    DEBUG = args.debug

    print(f"Loading metadata for {args.repo_id} ...")
    sha, version, tasks, episodes = load_dataset_meta(args.repo_id, args.revision)
    print(f"  revision {sha}  codebase {version}  {len(tasks)} tasks  {len(episodes)} episodes")

    multi = [e["episode_index"] for e in episodes if len(e["task_indices"]) != 1]
    if multi:
        print(f"  WARNING: {len(multi)} episodes do not map to exactly one task; they are skipped: {multi[:10]}")

    sim_tasks = load_sim_tasks(args.suite)
    text_to_index = {norm(t): i for i, t in tasks.items()}

    table = []
    for env_id, text in enumerate(sim_tasks):
        ds_idx = text_to_index.get(norm(text))
        eps = [e for e in episodes if e["task_indices"] == [ds_idx]] if ds_idx is not None else []
        lengths = [e["length"] for e in eps]
        table.append(
            {
                "env_task_id": env_id,
                "dataset_task_index": ds_idx,
                "instruction": text,
                "episode_ids": sorted(e["episode_index"] for e in eps),
                "lengths": lengths,
            }
        )

    print(f"\n{args.suite}: simulator task id -> dataset task_index, demos, length (frames)")
    print(f"{'env':>3} {'ds':>3} {'demos':>5} {'len min/mean/max':>17}  instruction")
    for t in table:
        L = t["lengths"]
        lens = f"{min(L)}/{statistics.mean(L):.0f}/{max(L)}" if L else "-"
        ds = t["dataset_task_index"] if t["dataset_task_index"] is not None else "??"
        print(f"{t['env_task_id']:>3} {ds:>3} {len(L):>5} {lens:>17}  {t['instruction']}")

    unmatched = [t["env_task_id"] for t in table if t["dataset_task_index"] is None]
    if unmatched:
        print(f"\nERROR: simulator tasks {unmatched} have no matching instruction in the dataset.")
        sys.exit(1)

    if not args.tasks:
        print("\nInspect only. Choose 3 tasks, then rerun with: --tasks <id> <id> <id>")
        return

    out = {
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        "dataset": {"repo_id": args.repo_id, "revision": sha, "codebase_version": version},
        "lerobot_version": lerobot_version(),
        "suite": args.suite,
        "seed": args.seed,
        "sizes": [str(n) for n in args.sizes] + ["all"],
        "note": "Episode ids are global dataset indices for --dataset.episodes. "
        "env_task_id is the value for --env.task_ids. Splits are nested prefixes of shuffled_order.",
        "tasks": {},
    }
    for label_i, env_id in enumerate(args.tasks, start=1):
        t = table[env_id]
        if len(t["episode_ids"]) < max(args.sizes):
            print(f"WARNING: task {env_id} has only {len(t['episode_ids'])} demos (< {max(args.sizes)}).")
        s = nested_splits(t["episode_ids"], args.sizes, args.seed, key=env_id)
        out["tasks"][f"t{label_i}"] = {
            "env_task_id": env_id,
            "dataset_task_index": t["dataset_task_index"],
            "instruction": t["instruction"],
            "n_available": len(t["episode_ids"]),
            "mean_length": round(statistics.mean(t["lengths"]), 1),
            **s,
        }

    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nWrote {path}")
    for label, t in out["tasks"].items():
        sizes = ", ".join(f"{k}:{len(v)}" for k, v in t["splits"].items())
        print(f"  {label} = env task {t['env_task_id']} ({sizes})  {t['instruction']}")
        print(f'     e.g. --dataset.episodes="{t["splits"][str(args.sizes[0])]}" --env.task_ids="[{t["env_task_id"]}]"')


if __name__ == "__main__":
    main()
