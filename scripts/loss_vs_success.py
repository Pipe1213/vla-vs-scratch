#!/usr/bin/env python3
"""Does the training loss pick a good checkpoint? (after-publishing check, plan "Stretch", audit point 9)

For every grid run, average the training loss logged to wandb just before each evaluated checkpoint and pair it
with that checkpoint's simulator success from results/results.csv. Then ask: is the checkpoint with the lowest
loss also the most successful one, more often than a random pick would be?

    python scripts/loss_vs_success.py                    # ACT (36 runs): download from wandb once, then analyse
    python scripts/loss_vs_success.py --method smolvla   # same for SmolVLA (24 runs)
    python scripts/loss_vs_success.py --method both
    python scripts/loss_vs_success.py --refresh          # download again (otherwise the cached file is reused)
    python scripts/loss_vs_success.py --entity <name>    # if the runs are not under your default wandb entity

Needs `wandb login` and internet for the download only (wandb is in the lerobot env). Afterwards it runs offline
from the cached file, which is small and can be committed so anyone can re-run the analysis without wandb.

Reads   results/results.csv, wandb project "lerobot" (LeRobot's default; run name = run name in results.csv)
Writes  results/train_loss.csv                       training loss curves as logged (cache, one row per log step)
        results/loss_vs_success_<method>.csv         one row per (run, checkpoint): window loss + success
        results/loss_vs_success_<method>.md          summary (what to quote) + per-run table
        results/figures/loss_vs_success_<method>.{png,svg}

How the loss is measured
- LeRobot logs "train/loss" every 200 steps (log_freq), and each logged value is already the mean over those 200
  steps. The loss of checkpoint c is the mean of the values logged at steps in (c - window, c], i.e. the training
  loss over the last `--window` steps (default 1000) before the checkpoint was saved. The summary also repeats the
  main count for windows of 400 and 2000 steps, to show that the answer does not hinge on this choice.
- This is the training loss on the batches being trained on. There is no validation loss (no held-out demos),
  which is exactly the situation the README sentence is about.

How "the loss picks the best checkpoint" is scored
- Hit = the lowest-loss checkpoint is among the most successful checkpoints of that run (ties count as a hit).
- Chance level, per run = (number of tied best checkpoints) / 3, i.e. the hit rate of a random pick. Summed over
  runs it gives the expected number of hits by chance (about 12 of 36 without ties, a bit more with ties).
- p-value = probability of at least the observed hits if the pick were random (exact, Poisson-binomial).
- Regret = success of the best checkpoint minus success of the picked one (points), for the loss pick, for a random
  pick (expected), and for "always take the last checkpoint". Because the loss usually keeps falling, the loss
  pick is often simply the last checkpoint; the summary says how often.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze as A  # noqa: E402  (same loading, colours and figure style as the main analysis)

import matplotlib.pyplot as plt  # noqa: E402  (analyze already selected the Agg backend)

ROOT = A.ROOT
CACHE = ROOT / "results" / "train_loss.csv"
LOG_FREQ = 200
CKPTS = {"act": [6000, 8000, 10000], "smolvla": [3000, 4000, 5000]}
SENSITIVITY_WINDOWS = [400, 1000, 2000]
LOSS_COLS = ["train/loss", "train/l1_loss", "train/kld_loss", "train/lr"]  # ACT logs l1 + kld; SmolVLA only loss


# --------------------------------------------------------------------------------------------- download
def fetch(names: list[str], entity: str | None, project: str) -> pd.DataFrame:
    """Download the logged training curves of the given runs from wandb (one row per log step)."""
    import wandb

    api = wandb.Api(timeout=60)
    ent = entity or api.default_entity
    print(f"wandb: {ent}/{project}, looking for {len(names)} runs")
    found = {}
    for r in api.runs(f"{ent}/{project}", filters={"display_name": {"$in": names}}):
        found.setdefault(r.name, []).append(r)

    rows, problems = [], []
    for name in names:
        cands = found.get(name, [])
        last_ckpt = max(CKPTS[name.split("_")[0]])
        # Keep runs that got to the last evaluated checkpoint (drops crashed / cancelled attempts).
        good = [r for r in cands if (r.summary.get("_step") or 0) >= last_ckpt]
        if not good:
            problems.append(f"{name}: no wandb run reaching step {last_ckpt} ({len(cands)} found)")
            continue
        if len(good) > 1:
            good.sort(key=lambda r: r.created_at)
            print(f"  [warn] {name}: {len(good)} complete wandb runs ({', '.join(r.id for r in good)}); using the latest")
        run = good[-1]
        n = 0
        for row in run.scan_history():  # all logged rows; ~50 per ACT run, ~25 per SmolVLA run
            if row.get("train/loss") is None:
                continue
            rows.append({"name": name, "wandb_id": run.id, "step": int(row["_step"]),
                         **{c.split("/", 1)[1]: row.get(c) for c in LOSS_COLS}})
            n += 1
        print(f"  {name}: {run.id}, {n} loss points")
    if problems:
        print("\n".join(["Not found:"] + problems))
    return pd.DataFrame(rows)


def load_curves(names: list[str], entity, project, refresh: bool) -> pd.DataFrame:
    cache = pd.read_csv(CACHE) if CACHE.exists() and not refresh else pd.DataFrame(columns=["name"])
    missing = [n for n in names if n not in set(cache["name"])]
    if missing:
        new = fetch(missing, entity, project)
        cache = pd.concat([cache[~cache["name"].isin(missing)], new], ignore_index=True)
        cache["step"] = cache["step"].astype(int)
        cache = cache.sort_values(["name", "step"]).reset_index(drop=True)
        cache.to_csv(CACHE, index=False)
        print(f"wrote {CACHE.relative_to(ROOT)} ({cache['name'].nunique()} runs)")
    still = [n for n in names if n not in set(cache["name"])]
    if still:
        sys.exit(f"{len(still)} runs have no loss curve (see above): {still[:5]}{' ...' if len(still) > 5 else ''}")
    return cache[cache["name"].isin(names)]


# --------------------------------------------------------------------------------------------- analysis
def window_loss(curves: pd.DataFrame, name: str, ckpt: int, window: int) -> float:
    c = curves[(curves["name"] == name) & (curves["step"] > ckpt - window) & (curves["step"] <= ckpt)]
    expected = window // LOG_FREQ
    if len(c) != expected:
        print(f"  [warn] {name} @ {ckpt}: {len(c)} loss points in the {window}-step window (expected {expected})")
    return float(c["loss"].mean()) if len(c) else np.nan


def pair(df: pd.DataFrame, curves: pd.DataFrame, method: str, window: int) -> pd.DataFrame:
    rows = []
    for _, r in df[df["method"] == method].iterrows():
        step = int(r["label"])
        rows.append({"name": r["name"], "task": r["task"], "size": r["size"], "seed": r["seed"], "step": step,
                     "n_success": int(r["n_success"]), "success": 100 * r["p"],
                     "loss": window_loss(curves, r["name"], step, window)})
    return pd.DataFrame(rows).sort_values(["name", "step"]).reset_index(drop=True)


def poisson_binomial_sf(ps: list[float], k: int) -> float:
    """P(X >= k) for a sum of independent Bernoulli(p_i)."""
    dist = np.zeros(len(ps) + 1)
    dist[0] = 1.0
    for p in ps:
        dist[1:] = dist[1:] * (1 - p) + dist[:-1] * p
        dist[0] *= 1 - p
    return float(dist[k:].sum())


def score(paired: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    per_run = []
    for name, g in paired.groupby("name", sort=False):
        g = g.sort_values("step")
        succ, loss, steps = g["n_success"].to_numpy(), g["loss"].to_numpy(), g["step"].to_numpy()
        best = succ.max()
        pick = int(np.nanargmin(loss))
        per_run.append({
            "name": name, "task": g["task"].iat[0], "size": g["size"].iat[0], "seed": g["seed"].iat[0],
            "successes": "/".join(map(str, succ)), "losses": "/".join(f"{x:.4f}" for x in loss),
            "loss_pick": int(steps[pick]), "best": "/".join(str(s) for s, v in zip(steps, succ) if v == best),
            "hit": bool(succ[pick] == best), "p_chance": float((succ == best).mean()),
            "regret_loss": 2.0 * (best - succ[pick]), "regret_random": 2.0 * (best - succ.mean()),
            "regret_last": 2.0 * (best - succ[-1]), "loss_pick_is_last": pick == len(steps) - 1,
            "loss_monotone": bool(np.all(np.diff(loss) < 0)), "last_is_best": bool(succ[-1] == best),
        })
    pr = pd.DataFrame(per_run)
    hits = int(pr["hit"].sum())
    # Within-run association: centre loss and success inside each run, then correlate the 3 x n_runs points.
    c = paired.copy()
    c["loss_c"] = c["loss"] - c.groupby("name")["loss"].transform("mean")
    c["succ_c"] = c["success"] - c.groupby("name")["success"].transform("mean")
    rho, rho_p = stats.spearmanr(c["loss_c"], c["succ_c"])
    summ = {
        "n_runs": len(pr), "hits": hits, "expected_by_chance": float(pr["p_chance"].sum()),
        "p_value_at_least": poisson_binomial_sf(pr["p_chance"].tolist(), hits),
        "loss_pick_is_last": int(pr["loss_pick_is_last"].sum()), "loss_monotone": int(pr["loss_monotone"].sum()),
        "last_is_best": int(pr["last_is_best"].sum()),
        "mean_regret_loss": float(pr["regret_loss"].mean()), "mean_regret_random": float(pr["regret_random"].mean()),
        "mean_regret_last": float(pr["regret_last"].mean()),
        "within_run_spearman": float(rho), "within_run_spearman_p": float(rho_p),
    }
    return pr, summ


# --------------------------------------------------------------------------------------------- outputs
def figure(paired: pd.DataFrame, pr: pd.DataFrame, method: str) -> None:
    A.style()
    sizes, tasks = A.SIZES, A.TASKS
    seeds = sorted(paired["seed"].unique())
    cols = [(t, s) for t in tasks for s in seeds]
    fig, axes = plt.subplots(len(sizes), len(cols), figsize=(1.75 * len(cols) + 0.6, 1.75 * len(sizes) + 1.0),
                             sharey=True, squeeze=False)
    color = A.COLOR[method]
    for i, size in enumerate(sizes):
        for j, (t, s) in enumerate(cols):
            ax = axes[i][j]
            g = paired[(paired["size"] == size) & (paired["task"] == t) & (paired["seed"] == s)].sort_values("step")
            if g.empty:
                ax.set_visible(False)
                continue
            run = pr[pr["name"] == g["name"].iat[0]].iloc[0]
            ax.plot(g["loss"], g["success"], "-", color=color, lw=1, alpha=0.45, zorder=2)
            for _, r in g.iterrows():
                picked = r["step"] == run["loss_pick"]
                ax.plot(r["loss"], r["success"], "o", ms=6 if picked else 4.5, color=color if picked else A.SURFACE,
                        mec=color, mew=1.3, zorder=3)
                ax.annotate(f"{r['step'] // 1000}k", (r["loss"], r["success"]), xytext=(4, 3),
                            textcoords="offset points", fontsize=6.5, color=A.INK_2)
            ax.set_ylim(-5, 105)
            ax.tick_params(labelsize=6.5)
            ax.xaxis.set_major_locator(plt.MaxNLocator(2))
            lo, hi = g["loss"].min(), g["loss"].max()
            pad = (hi - lo) * 0.35 or 0.01
            ax.set_xlim(lo - pad, hi + pad)
            mark = "lowest loss = best" if run["hit"] else "lowest loss not best"
            ax.text(0.03, 0.04, mark, transform=ax.transAxes, fontsize=6, color=A.INK if run["hit"] else A.INK_2)
            if i == 0:
                ax.set_title(f"{A.TASK_TITLE[t]}, seed {s}", fontsize=7.5, loc="left")
            if j == 0:
                ax.set_ylabel(f"{size} demos\nsuccess (%)", fontsize=7.5)
    fig.supxlabel("training loss, mean over the 1000 steps before the checkpoint (axis range differs per panel)",
                  fontsize=8, color=A.INK_2)
    label = A.METHOD_LABEL[method]
    fig.text(0.01, 0.985, f"{label}: training loss and success at each evaluated checkpoint", ha="left", va="top",
             fontsize=11, fontweight="bold")
    fig.text(0.01, 0.95, "One panel per training run. Filled dot = checkpoint with the lowest loss; labels give the "
             "training step. Success out of 50 episodes.", fontsize=8, color=A.INK_2, ha="left", va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    A.FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(A.FIG_DIR / f"loss_vs_success_{method}.{ext}", dpi=200, bbox_inches="tight",
                    metadata={"Date": None} if ext == "svg" else {"Software": None})
    plt.close(fig)


def report(method: str, window: int, summ: dict, sens: dict, pr: pd.DataFrame) -> str:
    n = summ["n_runs"]
    ck = CKPTS[method]
    L = [f"# Training loss vs checkpoint success: {A.METHOD_LABEL[method]}", "",
         f"Generated by `scripts/loss_vs_success.py --method {method}` from `results/results.csv` and "
         f"`results/train_loss.csv` (wandb). Loss of a checkpoint = mean training loss over the {window} steps before it.",
         "", "## Summary", "",
         f"- Lowest-loss checkpoint is also the most successful in **{summ['hits']} of {n} runs**; a random pick "
         f"would do this in {summ['expected_by_chance']:.1f} of {n} on average (ties included). "
         f"Probability of {summ['hits']} or more by chance: p = {summ['p_value_at_least']:.3f}.",
         f"- The loss fell at every checkpoint in {summ['loss_monotone']} of {n} runs, so the lowest-loss checkpoint "
         f"was the last one ({ck[-1] // 1000}k) in {summ['loss_pick_is_last']} of {n} runs. The last checkpoint was "
         f"(one of) the most successful in {summ['last_is_best']} of {n} runs.",
         f"- Success lost against the best checkpoint of each run (mean over runs, points): loss pick "
         f"{summ['mean_regret_loss']:.1f}, random pick {summ['mean_regret_random']:.1f}, always the last checkpoint "
         f"{summ['mean_regret_last']:.1f}.",
         f"- Within runs, lower loss went with higher success with Spearman rho = {-summ['within_run_spearman']:.2f} "
         f"(sign flipped so that positive = lower loss, more success; p = {summ['within_run_spearman_p']:.3f}, "
         f"{3 * n} checkpoints centred per run; rough, the 3 checkpoints of a run are not independent).",
         "- Window sensitivity (hits of " + str(n) + "): " + ", ".join(f"{w} steps: {h}" for w, h in sens.items()) + ".",
         "", "Reading it: hits close to the chance level mean the training loss does not tell you which of these "
         "checkpoints is good. Clearly more hits than chance means it does help. The best checkpoint is judged on the "
         "same 50 episodes, so small success differences are partly sampling noise.", "",
         "## Per run", ""]
    rows = [[r.name, r.successes, r.losses, f"{r.loss_pick // 1000}k",
             "/".join(f"{int(b) // 1000}k" for b in r.best.split("/")), "yes" if r.hit else "no", f"{r.p_chance:.2f}"]
            for r in pr.itertuples()]
    L.append(A.md_table(["run", f"successes /50 at {'/'.join(f'{c // 1000}k' for c in ck)}", "loss",
                         "lowest loss", "best", "hit", "chance"], rows))
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--method", choices=["act", "smolvla", "both"], default="act")
    ap.add_argument("--window", type=int, default=1000, help="steps of training loss averaged before each checkpoint")
    ap.add_argument("--entity", default=None, help="wandb entity (default: your default entity)")
    ap.add_argument("--project", default="lerobot", help="wandb project (LeRobot's default)")
    ap.add_argument("--refresh", action="store_true", help="download the loss curves again")
    a = ap.parse_args()
    if a.window % LOG_FREQ:
        sys.exit(f"--window must be a multiple of {LOG_FREQ} (LeRobot's log_freq)")

    df = A.load_results(A.RESULTS)
    methods = ["act", "smolvla"] if a.method == "both" else [a.method]
    names = sorted(df[df["method"].isin(methods)]["name"].unique())
    curves = load_curves(names, a.entity, a.project, a.refresh)

    for m in methods:
        paired = pair(df, curves, m, a.window)
        if paired["loss"].isna().any():
            sys.exit(f"Missing loss values for: {sorted(paired.loc[paired['loss'].isna(), 'name'].unique())}")
        pr, summ = score(paired)
        sens = {w: score(pair(df, curves, m, w))[1]["hits"] for w in SENSITIVITY_WINDOWS}
        out = ROOT / "results" / f"loss_vs_success_{m}"
        paired.to_csv(out.with_suffix(".csv"), index=False, float_format="%.5f")
        out.with_suffix(".md").write_text(report(m, a.window, summ, sens, pr))
        figure(paired, pr, m)
        print(f"\n{A.METHOD_LABEL[m]}: lowest loss = best checkpoint in {summ['hits']} of {summ['n_runs']} runs "
              f"(chance {summ['expected_by_chance']:.1f}, p = {summ['p_value_at_least']:.3f}); "
              f"loss pick = last checkpoint in {summ['loss_pick_is_last']} runs")
        print(f"wrote {out.relative_to(ROOT)}.csv/.md, results/figures/loss_vs_success_{m}.png/.svg")


if __name__ == "__main__":
    main()
