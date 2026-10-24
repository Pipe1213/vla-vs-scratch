#!/usr/bin/env python3
"""Make the README / site GIFs from the coded evaluation videos (Phase 4). Runs on the laptop (needs the videos + ffmpeg).

    python scripts/make_gifs.py               # writes results/figures/gifs/*.gif + gifs.csv
    python scripts/make_gifs.py --dry-run     # only print which videos were picked
    python scripts/make_gifs.py --pick act_retry_fail=V24   # override one pick (logged in gifs.csv as "manual")

Picks are drawn at random (fixed seed) among the 90 coded videos (notes/failure_codes.csv) that match each slot,
so the GIFs show typical behaviour, not the best-looking episodes. Slots:
- success / failure for each method at 5 demos and at all demos (8 GIFs)
- butter at 10 demos: SmolVLA success | ACT failure, side by side (the strongest per-task contrast)
- behaviour: an ACT failure after retrying (RETRY-FAIL) | a SmolVLA recovery (success after G-MISS/G-SLIP)
Videos are 256x256 at 80 fps (an episode lasts 1.5-3.5 s); GIFs play them at 1/4 speed (20 fps) and hold the
last frame for 1 s.
"""
from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SAMPLE, CODES = ROOT / "notes" / "coding_sample.csv", ROOT / "notes" / "failure_codes.csv"
OUT = ROOT / "results" / "figures" / "gifs"
SEED = 20261025
SLOW, OUT_FPS, HOLD_S, WIDTH = 4, 20, 1.0, 256

is_miss = lambda r: any(c in r["codes"].split("+") for c in ("G-MISS", "G-SLIP"))
SLOTS = {  # name -> filter on a coded row (dict of strings)
    "smolvla_n05_success": lambda r: r["cell"] == "smolvla_n05" and r["success"] == "1",
    "smolvla_n05_failure": lambda r: r["cell"] == "smolvla_n05" and r["success"] == "0",
    "act_n05_success": lambda r: r["cell"] == "act_n05" and r["success"] == "1",
    "act_n05_failure": lambda r: r["cell"] == "act_n05" and r["success"] == "0",
    "smolvla_nall_success": lambda r: r["cell"] == "smolvla_nall" and r["success"] == "1",
    "smolvla_nall_failure": lambda r: r["cell"] == "smolvla_nall" and r["success"] == "0",
    "act_nall_success": lambda r: r["cell"] == "act_nall" and r["success"] == "1",
    "act_nall_failure": lambda r: r["cell"] == "act_nall" and r["success"] == "0",
    "butter_n10_smolvla_success": lambda r: r["cell"] == "smolvla_t2_n10" and r["success"] == "1",
    "butter_n10_act_failure": lambda r: r["cell"] == "act_t2_n10" and r["success"] == "0",
    "act_retry_fail": lambda r: r["method"] == "act" and r["success"] == "0" and "RETRY-FAIL" in r["codes"],
    "smolvla_recovery": lambda r: r["method"] == "smolvla" and r["success"] == "1" and is_miss(r),
}
PAIRS = {"butter_n10_pair": ("butter_n10_smolvla_success", "butter_n10_act_failure"),
         "act_retry_fail_vs_smolvla_recovery_pair": ("act_retry_fail", "smolvla_recovery")}


def rows() -> list[dict]:
    sample = {r["vid"]: r for r in csv.DictReader(SAMPLE.open())}
    out = []
    for c in csv.DictReader(CODES.open()):
        r = dict(sample[c["vid"]])
        r.update(codes=c["codes"], attempts=c["attempts"], comment=c["comment"])
        out.append(r)
    if len(out) < len(sample):
        sys.exit(f"only {len(out)}/{len(sample)} videos coded: finish coding first")
    return out


def pick(rs: list[dict], manual: dict) -> dict:
    rng = np.random.default_rng(SEED)
    by_vid = {r["vid"]: r for r in rs}
    picks = {}
    for name, keep in SLOTS.items():  # fixed order -> reproducible
        cand = sorted((r for r in rs if keep(r)), key=lambda r: r["vid"])
        drawn = cand[rng.integers(len(cand))] if cand else None  # always draw, so manual picks don't shift others
        if name in manual:
            r = by_vid[manual[name]]
            if not keep(r):
                sys.exit(f"--pick {name}={r['vid']}: that video does not match the slot")
            picks[name] = dict(r, how="manual", n_candidates=len(cand))
        elif drawn:
            picks[name] = dict(drawn, how="random", n_candidates=len(cand))
        else:
            print(f"  [skip] {name}: no coded video matches")
    return picks


def duration(path: Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True, check=True).stdout
    return float(out.strip())


def ffmpeg(args: list[str]):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *args], check=True)


def chain(i: int, pad_to: float) -> str:
    """Slow down, resample, hold the last frame so every input lasts pad_to seconds."""
    return (f"[{i}:v]setpts={SLOW}*PTS,fps={OUT_FPS},scale={WIDTH}:-1:flags=neighbor,"
            f"tpad=stop_mode=clone:stop_duration={pad_to:.3f}[v{i}]")


def make_gif(inputs: list[Path], out: Path):
    durs = [duration(p) * SLOW for p in inputs]
    total = max(durs) + HOLD_S
    graph = ";".join(chain(i, total - d) for i, d in enumerate(durs))
    stack = "".join(f"[v{i}]" for i in range(len(inputs)))
    graph += f";{stack}hstack=inputs={len(inputs)}[s]" if len(inputs) > 1 else ";[v0]null[s]"
    graph += ";[s]split[a][b];[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4"
    args = sum((["-i", str(p)] for p in inputs), [])
    ffmpeg([*args, "-filter_complex", graph, "-t", f"{total:.3f}", "-loop", "0", str(out)])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--pick", action="append", default=[], metavar="SLOT=VID", help="override one slot's video")
    a = ap.parse_args()
    manual = dict(p.split("=", 1) for p in a.pick)
    bad = set(manual) - set(SLOTS)
    if bad:
        sys.exit(f"unknown slot(s) {bad}; slots: {', '.join(SLOTS)}")
    if not a.dry_run and not shutil.which("ffmpeg"):
        sys.exit("ffmpeg not found")
    picks = pick(rows(), manual)
    for name, r in picks.items():
        print(f"  {name:28s} {r['vid']} {r['how']:6s} (1 of {r['n_candidates']:2d})  {r['run']} @{r['checkpoint']} "
              f"ep{r['episode']}  {'success' if r['success'] == '1' else 'failure'}  {r['codes'] or 'clean'}")
    if a.dry_run:
        return
    missing = [r["video"] for r in picks.values() if not (ROOT / r["video"]).is_file()]
    if missing:
        sys.exit("missing videos:\n  " + "\n  ".join(missing))
    OUT.mkdir(parents=True, exist_ok=True)
    for name, r in picks.items():
        make_gif([ROOT / r["video"]], OUT / f"{name}.gif")
    for name, (left, right) in PAIRS.items():
        if left in picks and right in picks:
            make_gif([ROOT / picks[left]["video"], ROOT / picks[right]["video"]], OUT / f"{name}.gif")
    fields = ["gif", "vid", "how", "n_candidates", "cell", "run", "method", "task", "size", "seed", "checkpoint",
              "episode", "success", "codes", "attempts", "comment", "video"]
    with (OUT / "gifs.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for name, r in picks.items():
            w.writerow(dict(r, gif=f"{name}.gif"))
    sizes = {p.name: p.stat().st_size / 1e6 for p in sorted(OUT.glob("*.gif"))}
    print(f"wrote {len(sizes)} GIFs to {OUT.relative_to(ROOT)}/ ({sum(sizes.values()):.1f} MB total, "
          f"largest {max(sizes.values()):.1f} MB) + gifs.csv")


if __name__ == "__main__":
    main()
