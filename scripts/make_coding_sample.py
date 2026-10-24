#!/usr/bin/env python3
"""Draw the fixed, seeded sample of evaluation videos to code for the failure analysis (Phase 4).

    python scripts/make_coding_sample.py            # writes notes/coding_sample.csv (refuses to overwrite)
    python scripts/make_coding_sample.py --check    # only list which sampled videos exist on this machine

Design (agreed 24 Oct, decided before watching any of the sampled videos):
- 6 cells: {ACT, SmolVLA} x {5 demos, all demos, butter (t2) at 10 demos}; 5/all pool the 3 tasks.
- Pool = episodes 0-9 of every evaluated checkpoint in the cell (the only episodes with video), all seeds.
- Per cell: 10 failures + 5 successes (if a cell has fewer than 10 failures, all of them, and successes fill
  the cell up to 15). Outcome comes from results.csv (simulator), never from watching.
- Failures are over-represented on purpose; the pool counts are stored per row so analyze.py can weight
  each outcome back to its real share when it computes recovery / retry rates.
- All 90 rows are shuffled together and get blind ids V01..V90 (the coding page hides the run until a
  video is saved, so the coder does not know the method).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "results.csv"
OUT = ROOT / "notes" / "coding_sample.csv"
SEED = 20261024
N_FAIL, N_TOTAL, N_VIDEO_EPISODES = 10, 15, 10
# LeRobot eval layout (plan, Phase 2 brief): one folder per (run, checkpoint), videos for episodes 0..9
VIDEO = "outputs/eval/{run}/{label}_ep50/videos/libero_object_{task_id}/eval_episode_{episode}.mp4"

CELLS = {  # cell id -> (method, filter on (task, size))
    "act_n05": ("act", lambda t, s: s == "05"),
    "act_nall": ("act", lambda t, s: s == "all"),
    "act_t2_n10": ("act", lambda t, s: t == "t2" and s == "10"),
    "smolvla_n05": ("smolvla", lambda t, s: s == "05"),
    "smolvla_nall": ("smolvla", lambda t, s: s == "all"),
    "smolvla_t2_n10": ("smolvla", lambda t, s: t == "t2" and s == "10"),
}
RUN_RE = r"^(act|smolvla)_obj_(t[123])_n(05|10|20|all)(?:_s(\d))?$"


def pool() -> pd.DataFrame:
    df = pd.read_csv(RESULTS, dtype={"label": str, "successes": str})
    p = df["name"].str.extract(RUN_RE)
    df["method"], df["task"], df["size"], df["seed"] = p[0], p[1], p[2], p[3].fillna("1").astype(int)
    rows = []
    for r in df.itertuples():
        for cell, (m, keep) in CELLS.items():
            if r.method != m or not keep(r.task, r.size):
                continue
            for ep in range(N_VIDEO_EPISODES):
                rows.append(dict(cell=cell, run=r.name, method=r.method, task=r.task, size=r.size.lstrip("0"),
                                 seed=r.seed, checkpoint=r.label, task_id=r.task_id, episode=ep,
                                 success=int(r.successes.strip()[ep]),
                                 video=VIDEO.format(run=r.name, label=r.label, task_id=r.task_id, episode=ep)))
    return pd.DataFrame(rows)


def draw(pl: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    parts = []
    for cell in CELLS:  # fixed order -> reproducible
        c = pl[pl.cell == cell]
        fail, succ = c[c.success == 0], c[c.success == 1]
        nf = min(N_FAIL, len(fail))
        ns = min(N_TOTAL - nf, len(succ))
        pick = pd.concat([fail.iloc[np.sort(rng.choice(len(fail), nf, replace=False))],
                          succ.iloc[np.sort(rng.choice(len(succ), ns, replace=False))]])
        pick = pick.assign(pool_fail=len(fail), pool_succ=len(succ), sampled_fail=nf, sampled_succ=ns)
        parts.append(pick)
    s = pd.concat(parts).reset_index(drop=True)
    s = s.iloc[rng.permutation(len(s))].reset_index(drop=True)
    s.insert(0, "vid", [f"V{i + 1:02d}" for i in range(len(s))])
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="report which sampled videos exist; write nothing")
    ap.add_argument("--force", action="store_true", help="overwrite an existing sample (only before coding starts)")
    a = ap.parse_args()
    if a.check:
        s = pd.read_csv(OUT)
        missing = [v for v in s["video"] if not (ROOT / v).is_file()]
        print(f"{len(s) - len(missing)}/{len(s)} sampled videos found")
        for v in missing:
            print("  missing:", v)
        sys.exit(1 if missing else 0)
    if OUT.exists() and not a.force:
        sys.exit(f"{OUT.relative_to(ROOT)} exists: the sample is fixed once drawn (use --force only before coding)")
    s = draw(pool())
    OUT.parent.mkdir(exist_ok=True)
    s.to_csv(OUT, index=False)
    summary = s.groupby("cell")[["success"]].agg(["count", "sum"])
    print(f"wrote {OUT.relative_to(ROOT)}: {len(s)} videos")
    for cell, g in s.groupby("cell", sort=False):
        print(f"  {cell:15s} {int((g.success == 0).sum()):2d} failures + {int(g.success.sum()):2d} successes "
              f"(pool: {g.pool_fail.iloc[0]} failures, {g.pool_succ.iloc[0]} successes)")


if __name__ == "__main__":
    main()
