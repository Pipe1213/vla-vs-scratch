#!/usr/bin/env python3
"""Per-demonstration statistics for the 3 tasks, to look for odd demos behind SmolVLA's non-monotonic curves.

    python scripts/inspect_demos.py          # laptop, env `lerobot` (reads the dataset parquet from the HF cache)

Why (plan, Phase 3b finding 3): SmolVLA's per-task curves dip in both seeds when demos are added
(t3: 95% at 5 demos -> ~67% at 10-20 -> 97% at all; t1: dip at 20). Splits are nested, so the demos added at
that size are the suspects. This script describes every demo from its recorded actions and states; it flags
outliers to watch first in the dataset viewer. It cannot prove that a demo *caused* a dip (that would need
retraining without it).

Per demo: length; gripper close events (> 1 = a re-grasp inside the demo); frame of the first close;
end-effector path length; fraction of near-idle frames (hesitation); action jerk (mean change of the arm
action between frames); end-effector position at the first close (where it grasped).
Each demo is labelled with the smallest split that contains it ("5", "10", "20", "all" = added at that size).

Writes notes/demo_stats.csv and notes/demo_inspection.md. Only the tabular data is read (no videos).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "configs" / "splits.json"
OUT_CSV, OUT_MD = ROOT / "notes" / "demo_stats.csv", ROOT / "notes" / "demo_inspection.md"
IDLE = 0.05          # |arm action| below this (normalised units, LIBERO actions are in [-1, 1]) = idle frame
Z_FLAG = 2.0         # flag a metric if it is more than 2 SD from the task's mean
SUSPECTS = {"t3": "10", "t1": "20"}  # task -> size whose added demos coincide with SmolVLA's dip
TASK_NAME = {"t1": "bbq sauce", "t2": "butter", "t3": "chocolate pudding"}
METRICS = ["length", "gripper_closes", "first_close_frac", "path_len", "idle_frac", "jerk"]


def load_frames(repo_id: str, revision: str, episodes: set[int]) -> pd.DataFrame:
    from huggingface_hub import snapshot_download
    root = snapshot_download(repo_id, repo_type="dataset", revision=revision, allow_patterns=["data/**/*.parquet"])
    import pyarrow.parquet as pq
    parts = []
    for f in sorted(Path(root, "data").rglob("*.parquet")):
        cols = pq.read_schema(f).names
        need = [c for c in ["episode_index", "frame_index", "action", "observation.state"] if c in cols]
        df = pd.read_parquet(f, columns=need)
        df = df[df["episode_index"].isin(episodes)]
        if len(df):
            parts.append(df)
    if not parts:
        sys.exit("no frames found for the split episodes: check the dataset revision / cache")
    df = pd.concat(parts, ignore_index=True)
    if "action" not in df.columns:
        sys.exit(f"no 'action' column; columns are {list(df.columns)}")
    return df.sort_values(["episode_index", "frame_index"])


def demo_stats(g: pd.DataFrame) -> dict:
    a = np.stack(g["action"].to_numpy()).astype(float)            # (T, 7): 6 arm + gripper (-1 open, +1 close)
    T = len(a)
    closed = a[:, -1] > 0
    closes = int(closed[0]) + int(np.sum(~closed[:-1] & closed[1:]))
    first = int(np.argmax(closed)) if closed.any() else -1
    arm = a[:, :6]
    out = dict(length=T, gripper_closes=closes, first_close_frac=(first / T) if first >= 0 else np.nan,
               idle_frac=float(np.mean(np.linalg.norm(arm, axis=1) < IDLE)),
               jerk=float(np.mean(np.linalg.norm(np.diff(arm, axis=0), axis=1))) if T > 1 else np.nan)
    if "observation.state" in g:
        s = np.stack(g["observation.state"].to_numpy()).astype(float)
        pos = s[:, :3]                                              # end-effector xyz
        out["path_len"] = float(np.sum(np.linalg.norm(np.diff(pos, axis=0), axis=1)))
        if first >= 0:
            out.update(grasp_x=pos[first, 0], grasp_y=pos[first, 1], grasp_z=pos[first, 2])
    else:
        out["path_len"] = np.nan
    return out


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for r in df.itertuples(index=False):
        lines.append("| " + " | ".join(f"{v:.2f}" if isinstance(v, float) else str(v) for v in r) + " |")
    return "\n".join(lines)


def main():
    sp = json.loads(SPLITS.read_text())
    added = {}
    for t, info in sp["tasks"].items():
        prev = set()
        for size in sp["sizes"]:
            cur = set(info["splits"][size])
            for ep in cur - prev:
                added[ep] = (t, size)
            prev = cur
    frames = load_frames(sp["dataset"]["repo_id"], sp["dataset"]["revision"], set(added))
    rows = []
    for ep, g in frames.groupby("episode_index"):
        t, size = added[int(ep)]
        rows.append(dict(task=t, episode=int(ep), added_at=size, **demo_stats(g)))
    st = pd.DataFrame(rows).sort_values(["task", "episode"])
    got = st.groupby("task").size().to_dict()
    want = {t: info["n_available"] for t, info in sp["tasks"].items()}
    if got != want:
        sys.exit(f"demos found per task {got} != splits.json {want}")
    if (st["gripper_closes"] != 1).mean() > 0.5:
        print("WARNING: most demos do not close the gripper exactly once: the gripper sign convention "
              "(action[-1] > 0 = close) may be inverted for this dataset; check gripper_closes before trusting flags")
    flags = []
    for m in METRICS:
        z = st.groupby("task")[m].transform(lambda x: (x - x.mean()) / (x.std(ddof=1) or np.nan))
        st[f"z_{m}"] = z
    for r in st.itertuples():
        f = [m for m in METRICS if abs(getattr(r, f"z_{m}")) > Z_FLAG]
        if r.gripper_closes > 1:
            f.append("regrasp")
        if r.gripper_closes == 0:
            f.append("never-closes")
        flags.append(", ".join(dict.fromkeys(f)))
    st["flags"] = flags
    OUT_CSV.parent.mkdir(exist_ok=True)
    st.to_csv(OUT_CSV, index=False, float_format="%.4f")

    md = ["# Demonstration inspection (generated by `scripts/inspect_demos.py`)", "",
          f"Per-demo statistics from the recorded actions/states (`notes/demo_stats.csv`). Flags: metric more than "
          f"{Z_FLAG} SD from the task mean, `regrasp` = gripper closes more than once in the demo. "
          "Descriptive only: a flagged demo is a candidate to watch, not a proven cause.", ""]
    show = ["episode", "added_at", "length", "gripper_closes", "first_close_frac", "path_len", "idle_frac", "jerk", "flags"]
    for t in sorted(st.task.unique()):
        d = st[st.task == t]
        md += [f"## {t} · {TASK_NAME.get(t, t)} ({len(d)} demos)", "",
               "Means by the split size at which the demos were added:", "",
               md_table(d.groupby("added_at", sort=False)[METRICS].mean().reindex(sp["sizes"]).reset_index()), ""]
        if t in SUSPECTS:
            s = SUSPECTS[t]
            md += [f"**Suspects:** demos added at {s} (SmolVLA's dip). Per demo:", "",
                   md_table(d[d.added_at == s][show]), ""]
        fl = d[d["flags"] != ""]
        md += [f"Flagged demos in {t}: {len(fl)}", ""]
        if len(fl):
            md += [md_table(fl[show]), ""]
    OUT_MD.write_text("\n".join(md) + "\n")
    print(f"wrote {OUT_CSV.relative_to(ROOT)} ({len(st)} demos) and {OUT_MD.relative_to(ROOT)}")
    for t, s in SUSPECTS.items():
        d = st[(st.task == t) & (st.added_at == s)]
        print(f"{t} demos added at {s}: " + "; ".join(f"{r.episode} [{r.flags or '-'}]" for r in d.itertuples()))


if __name__ == "__main__":
    main()
