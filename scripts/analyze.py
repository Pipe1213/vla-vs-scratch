#!/usr/bin/env python3
"""Phase 4 analysis: results.csv -> figures, Markdown tables and a summary of every number.

    python scripts/analyze.py            # from the repo root; needs pandas, numpy, scipy, matplotlib, pyyaml

Reads   results/results.csv (grid), results/diagnostics.csv (butter 30k runs, reported separately),
        results/compute_log.csv, configs/runs.yaml, configs/splits.json
Writes  results/figures/{headline,per_task,stability,scenes}.{png,svg}
        (+ failure analysis from notes/failure_codes.csv once videos are coded with scripts/code_videos.py)
        results/analysis.md     all tables, README-ready
        results/summary.json    every number quoted in the README / PDF / site (copy from here, never by hand)

Conventions
- Unit of replication = training seed. A "seed point" is one seed's success pooled over 3 tasks x 3 checkpoints
  x 50 episodes (450 episodes). Curves show the mean over seeds (ACT 3 seeds, SmolVLA 2) and the seed range.
- Wilson 95% CIs are on the pooled episodes and cover evaluation sampling only (not seeds or checkpoints;
  the 50 starting scenes are the same for every run). They are reported in the tables, not drawn on the plot.
- The script stops before computing anything if results.csv does not match the frozen grid.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS, DIAG, COMPUTE = (ROOT / "results" / f for f in ("results.csv", "diagnostics.csv", "compute_log.csv"))
RUNS_YAML, SPLITS = ROOT / "configs" / "runs.yaml", ROOT / "configs" / "splits.json"
CODING_SAMPLE, FAILURE_CODES = ROOT / "notes" / "coding_sample.csv", ROOT / "notes" / "failure_codes.csv"
CODE_NAMES = ["G-MISS", "G-SLIP", "NO-RETRY", "RETRY-FAIL", "COLLIDE", "WRONG-OBJ", "PLACE", "TIMEOUT", "OTHER"]
CELL_LABEL = {"act_n05": "ACT · 5 demos", "act_nall": "ACT · all demos", "act_t2_n10": "ACT · butter, 10 demos",
              "smolvla_n05": "SmolVLA · 5 demos", "smolvla_nall": "SmolVLA · all demos",
              "smolvla_t2_n10": "SmolVLA · butter, 10 demos"}
FIG_DIR, MD_OUT, JSON_OUT = ROOT / "results" / "figures", ROOT / "results" / "analysis.md", ROOT / "results" / "summary.json"

METHODS = ["smolvla", "act"]
METHOD_LABEL = {"smolvla": "SmolVLA (fine-tuned)", "act": "ACT (from scratch)"}
SEEDS = {"act": [1, 2, 3], "smolvla": [1, 2]}  # seed index -> runs.yaml training_seeds; ACT got a 3rd seed (Phase 3c)
SIZES = ["5", "10", "20", "all"]
TASKS = ["t1", "t2", "t3"]
TASK_NAME = {"t1": "bbq sauce", "t2": "butter", "t3": "chocolate pudding"}
TASK_TITLE = {"t1": "BBQ sauce", "t2": "Butter", "t3": "Chocolate pudding"}   # figure titles
N_EPS = 50
# Frozen 24 Oct (plan, Phase 3c): pooled success % per seed at 5/10/20/all demos. The script refuses to run if
# results.csv no longer reproduces them.
FROZEN = {
    ("act", 1): [10.4, 17.1, 38.9, 57.8], ("act", 2): [10.9, 54.0, 46.2, 53.1], ("act", 3): [32.4, 32.7, 14.2, 55.3],
    ("smolvla", 1): [76.0, 81.3, 72.0, 88.0], ("smolvla", 2): [80.0, 83.6, 76.4, 86.7],
}
RUN_RE = re.compile(r"^(act|smolvla)_obj_(t[123])_n(05|10|20|all)(?:_s(\d))?$")
DIAG_RE = re.compile(r"^act_obj_t2_nall_30k(?:_s(\d))?$")
MC_SIMS, RNG_SEED = 20000, 0

# Colours: reference categorical palette slots 1-2 (validated light mode, 2 series) + text/grid tokens.
COLOR = {"smolvla": "#2a78d6", "act": "#eb6834"}
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"


# --------------------------------------------------------------------------------------------- loading
def parse_runs(df: pd.DataFrame, regex: re.Pattern) -> pd.DataFrame:
    parts = df["name"].str.extract(regex)
    bad = df.loc[parts[0].isna(), "name"].unique()
    if len(bad):
        sys.exit(f"Unrecognised run names: {list(bad)}")
    return parts


def load_results(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"label": str, "successes": str})
    df["successes"] = df["successes"].str.strip()
    p = parse_runs(df, RUN_RE)
    df["method"], df["task"] = p[0], p[1]
    df["size"] = p[2].replace({"05": "5"})
    df["seed"] = p[3].fillna("1").astype(int)
    df["p"] = df["n_success"] / df["n_episodes"]
    return df


def load_diag(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"label": str, "successes": str})
    df["successes"] = df["successes"].str.strip()
    df["seed"] = df["name"].str.extract(DIAG_RE)[0].fillna("1").astype(int)
    return df


# --------------------------------------------------------------------------------------------- validation
def validate(df: pd.DataFrame, cfg: dict, splits: dict) -> None:
    err = []
    task_ids = cfg["tasks"]
    ckpts = {m: cfg["methods"][m]["eval_checkpoints"] for m in METHODS}
    expected_rows = sum(len(SEEDS[m]) * len(TASKS) * len(SIZES) * len(ckpts[m]) for m in METHODS)
    if len(df) != expected_rows:
        err.append(f"{len(df)} rows, expected {expected_rows}")
    if df.duplicated(["name", "label"]).any():
        err.append(f"duplicate (run, checkpoint) rows: {df[df.duplicated(['name', 'label'])]['name'].tolist()}")
    for m in METHODS:
        for t in TASKS:
            for s in SIZES:
                for seed in SEEDS[m]:
                    got = sorted(df.query("method==@m and task==@t and size==@s and seed==@seed")["label"])
                    if got != sorted(ckpts[m]):
                        err.append(f"{m} {t} n{s} seed{seed}: checkpoints {got}, expected {ckpts[m]}")
    extra_seeds = set(df.groupby("method")["seed"].unique().explode().items()) - {(m, s) for m in METHODS for s in SEEDS[m]}
    if extra_seeds:
        err.append(f"unexpected seeds {extra_seeds}")
    for r in df.itertuples():
        where = f"{r.name}@{r.label}"
        if r.task_id != task_ids[r.task]:
            err.append(f"{where}: task_id {r.task_id} != {task_ids[r.task]}")
        if splits["tasks"][r.task]["env_task_id"] != r.task_id:
            err.append(f"{where}: task_id disagrees with splits.json")
        if r.n_episodes != N_EPS or len(r.successes) != N_EPS or set(r.successes) - {"0", "1"}:
            err.append(f"{where}: not {N_EPS} episodes / bad successes string")
        elif r.successes.count("1") != r.n_success:
            err.append(f"{where}: n_success {r.n_success} != successes string {r.successes.count('1')}")
        if "--policy.n_action_steps=10" not in str(r.extra_args):
            err.append(f"{where}: n_action_steps=10 missing")
        if r.method == "smolvla" and "rename_map" not in str(r.extra_args):
            err.append(f"{where}: rename_map missing")
        if f"{r.name}/checkpoints/{r.label}" not in str(r.checkpoint).replace("\\", "/"):
            err.append(f"{where}: checkpoint path {r.checkpoint} does not match run/label")
    if not err:  # frozen numbers (only meaningful once the structure is right)
        for (m, seed), vals in FROZEN.items():
            for s, v in zip(SIZES, vals):
                g = df.query("method==@m and seed==@seed and size==@s")
                got = round(100 * g["n_success"].sum() / g["n_episodes"].sum(), 1)
                if abs(got - v) > 0.05:
                    err.append(f"{m} seed{seed} n{s}: {got}% != frozen {v}%")
    if err:
        sys.exit("results.csv failed validation:\n  " + "\n  ".join(err))
    print(f"validation OK: {len(df)} rows, {df['name'].nunique()} runs, frozen numbers reproduced")


# --------------------------------------------------------------------------------------------- statistics
def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * (c - h), 100 * (c + h))


def pct(k, n) -> float:
    return 100.0 * k / n


def headline(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for m in METHODS:
        for s in SIZES:
            g = df.query("method==@m and size==@s")
            per_seed = {seed: pct(*gs[["n_success", "n_episodes"]].sum()) for seed, gs in g.groupby("seed")}
            k, n = int(g["n_success"].sum()), int(g["n_episodes"].sum())
            lo, hi = wilson(k, n)
            vals = list(per_seed.values())
            rows.append(dict(method=m, size=s, per_seed=per_seed, mean=float(np.mean(vals)),
                             seed_min=min(vals), seed_max=max(vals), k=k, n=n, wilson_lo=lo, wilson_hi=hi))
    return pd.DataFrame(rows)


def per_task(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (m, t, s), g in df.groupby(["method", "task", "size"]):
        per_seed = {seed: pct(*gs[["n_success", "n_episodes"]].sum()) for seed, gs in g.groupby("seed")}
        k, n = int(g["n_success"].sum()), int(g["n_episodes"].sum())
        rows.append(dict(method=m, task=t, size=s, per_seed=per_seed, mean=float(np.mean(list(per_seed.values()))),
                         seed_range=max(per_seed.values()) - min(per_seed.values()), k=k, n=n))
    out = pd.DataFrame(rows)
    out["size"] = pd.Categorical(out["size"], SIZES, ordered=True)
    return out.sort_values(["method", "task", "size"]).reset_index(drop=True)


def checkpoint_test(counts: np.ndarray, n: int, rng: np.random.Generator) -> float:
    """Monte Carlo p-value: are the checkpoints' success counts more different than sampling alone predicts?
    Null: all checkpoints have the run's pooled success rate; statistic = chi-square of the k x 2 table."""
    p = counts.sum() / (n * len(counts))
    if p in (0.0, 1.0):
        return 1.0

    def chi2(c):
        e = c.sum(axis=-1, keepdims=True) / c.shape[-1]
        return (((c - e) ** 2) / (e * (1 - e / n))).sum(axis=-1)
    sims = rng.binomial(n, p, size=(MC_SIMS, len(counts))).astype(float)
    ok = (sims.sum(axis=1) > 0) & (sims.sum(axis=1) < n * len(counts))
    obs = chi2(counts.astype(float)[None, :])[0]
    sim_stat = np.zeros(MC_SIMS)
    sim_stat[ok] = chi2(sims[ok])
    return float((1 + (sim_stat >= obs - 1e-9).sum()) / (MC_SIMS + 1))


def stability(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Variance of success between checkpoints of a run, and between seeds of a (method, task, size) cell,
    each compared with what the level below (sampling noise; checkpoint + sampling noise) predicts."""
    rng = np.random.default_rng(RNG_SEED)
    run_rows = []
    for (m, t, s, seed), g in df.groupby(["method", "task", "size", "seed"]):
        g = g.sort_values("label")
        counts = g["n_success"].to_numpy()
        n_tot = len(counts) * N_EPS
        pbar = counts.sum() / n_tot
        run_rows.append(dict(method=m, task=t, size=s, seed=seed, run=g["name"].iloc[0],
                             ckpt_counts=counts.tolist(), mean=pbar,
                             ckpt_range=int(counts.max() - counts.min()),
                             var_obs=float(np.var(counts / N_EPS, ddof=1)),
                             var_sampling=pbar * (1 - pbar) * n_tot / (n_tot - 1) / N_EPS,
                             p_ckpt=checkpoint_test(counts, N_EPS, rng)))
    runs = pd.DataFrame(run_rows)
    summ = []
    for m, g in runs.groupby("method"):
        ck_obs, ck_samp = g["var_obs"].mean(), g["var_sampling"].mean()
        n_ck = 3
        cell = g.groupby(["task", "size"])
        seed_obs = cell["mean"].var(ddof=1).mean()
        seed_exp = (cell["var_obs"].mean() / n_ck).mean()  # variance of a 3-checkpoint mean if seeds were identical
        summ.append(dict(
            method=m, n_runs=len(g), n_seeds=len(SEEDS[m]),
            ckpt_sd_obs=100 * np.sqrt(ck_obs), ckpt_sd_sampling=100 * np.sqrt(ck_samp), ckpt_ratio=ck_obs / ck_samp,
            ckpt_sd_extra=100 * np.sqrt(max(0.0, ck_obs - ck_samp)),
            ckpt_range_median=float(g["ckpt_range"].median()), ckpt_range_max=int(g["ckpt_range"].max()),
            runs_ckpt_differ=int((g["p_ckpt"] < 0.05).sum()),
            seed_sd_obs=100 * np.sqrt(seed_obs), seed_sd_expected=100 * np.sqrt(seed_exp), seed_ratio=seed_obs / seed_exp,
            seed_sd_extra=100 * np.sqrt(max(0.0, seed_obs - seed_exp)),
            seed_range_max=float(cell["mean"].agg(lambda x: x.max() - x.min()).max() * 100),
        ))
    return runs, pd.DataFrame(summ).set_index("method").loc[METHODS].reset_index()


def demos_to_50(head: pd.DataFrame) -> dict:
    out = {}
    for m in METHODS:
        h = head[head.method == m].set_index("size").loc[SIZES]
        first = next((s for s in SIZES if h.loc[s, "mean"] >= 50), None)
        per_seed = {}
        for seed in SEEDS[m]:
            vals = [h.loc[s, "per_seed"][seed] for s in SIZES]
            f = next((s for s, v in zip(SIZES, vals) if v >= 50), None)
            stays = f is not None and all(v >= 50 for v in vals[SIZES.index(f):])
            per_seed[seed] = dict(first=f, stays_above=stays)
        out[m] = dict(first_size_mean=first, below_smallest=(first == SIZES[0]), per_seed=per_seed)
    return out


def robust_claim(df: pd.DataFrame, n_all: dict) -> list[dict]:
    """Smallest SmolVLA demo count whose WORST seed beats ACT's BEST seed with all demos (pooled and per task)."""
    rows = []
    for scope in ["pooled"] + TASKS:
        g = df if scope == "pooled" else df[df.task == scope]
        seed_pct = g.groupby(["method", "size", "seed"])[["n_success", "n_episodes"]].sum()
        seed_pct = 100 * seed_pct["n_success"] / seed_pct["n_episodes"]
        act_best_all = seed_pct.loc[("act", "all")].max()
        smol_worst = {s: seed_pct.loc[("smolvla", s)].min() for s in SIZES}
        holds = [s for s in SIZES if smol_worst[s] > act_best_all]
        rows.append(dict(scope=scope, act_best_seed_all=act_best_all, smolvla_worst_seed=smol_worst,
                         smolvla_size_beating_act_all=holds[0] if holds else None, holds_at=holds,
                         act_all_demos=(round(np.mean(list(n_all.values()))) if scope == "pooled" else n_all[scope])))
    return rows


def scenes(df: pd.DataFrame) -> pd.DataFrame:
    """Per starting scene (episode i = LIBERO init state i, identical in every run): success rate per method,
    whether scene difficulty exists beyond sampling noise, and whether the two methods find the same scenes hard."""
    rows = []
    for t in TASKS:
        rec = {}
        for m in METHODS:
            g = df[(df.task == t) & (df.method == m)]
            S = np.array([[int(c) for c in s] for s in g["successes"]])     # trials x 50 scenes
            p_trial = S.mean(axis=1, keepdims=True)                          # each checkpoint's success rate
            rate = S.mean(axis=0)
            # variance of per-scene rates if every scene were equally hard for a given checkpoint
            var_null = float((p_trial * (1 - p_trial)).sum() / S.shape[0] ** 2)
            rec[m] = dict(rate=rate, trials=S.shape[0], ratio=float(rate.var(ddof=1) / var_null))
        rho, pval = stats.spearmanr(rec["act"]["rate"], rec["smolvla"]["rate"])
        both_hard = int(((rec["act"]["rate"] < np.median(rec["act"]["rate"])) &
                         (rec["smolvla"]["rate"] < np.median(rec["smolvla"]["rate"]))).sum())
        rows.append(dict(task=t, act_rate=rec["act"]["rate"], smolvla_rate=rec["smolvla"]["rate"],
                         act_trials=rec["act"]["trials"], smolvla_trials=rec["smolvla"]["trials"],
                         act_scene_ratio=rec["act"]["ratio"], smolvla_scene_ratio=rec["smolvla"]["ratio"],
                         spearman=float(rho), spearman_p=float(pval), both_below_median=both_hard,
                         hardest_act=np.argsort(rec["act"]["rate"])[:5].tolist(),
                         hardest_smolvla=np.argsort(rec["smolvla"]["rate"])[:5].tolist()))
    return pd.DataFrame(rows)


def butter(diag: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    if len(diag) != 15 or (diag["n_episodes"] != N_EPS).any():
        sys.exit(f"diagnostics.csv: expected 15 rows of {N_EPS} episodes, got {len(diag)}")
    rows = []
    for seed, g in diag.groupby("seed"):
        c = dict(zip(g["label"], g["n_success"]))
        last3 = sum(c[k] for k in ["026000", "028000", "030000"])
        grid = df.query("method=='act' and task=='t2' and size=='all' and seed==@seed")
        rows.append(dict(seed=seed, ckpt=c, last3=last3, last3_pct=pct(last3, 150),
                         grid_ckpt=dict(zip(grid["label"], grid["n_success"])),
                         grid=int(grid["n_success"].sum()), grid_pct=pct(grid["n_success"].sum(), 150)))
    return pd.DataFrame(rows)


def compute_summary(df: pd.DataFrame, diag: pd.DataFrame) -> dict:
    log = pd.read_csv(COMPUTE)
    def kind(run):
        if RUN_RE.match(run):
            return "grid"
        if DIAG_RE.match(run):
            return "diagnostic"
        return "pilot"
    log["kind"] = log["run"].map(kind)
    out = {}
    for (m, k), g in log.groupby(["method", "kind"]):
        out.setdefault(m, {})[k] = dict(runs=len(g), train_hours=round(float(g["hours"].sum()), 2),
                                        cost_usd=round(float(g["cost_usd"].sum()), 2),
                                        hours_per_run=round(float(g["hours"].median()), 2),
                                        where=sorted(g["where"].unique().tolist()))
    for m in METHODS:
        out[m]["grid"]["eval_hours"] = round(float(df[df.method == m]["wall_min"].sum() / 60), 1)
        out[m]["grid"]["eval_episodes"] = int(df[df.method == m]["n_episodes"].sum())
    out["act"]["diagnostic"]["eval_hours"] = round(float(diag["wall_min"].sum() / 60), 1)
    out["total_cloud_usd"] = round(float(log["cost_usd"].sum()), 2)
    return out


def failures() -> dict | None:
    """Coded videos (notes/failure_codes.csv, from scripts/code_videos.py) -> failure types, retry and recovery
    rates. Failures were over-sampled (10 failures + 5 successes per cell), so each coded video is weighted by
    (episodes with that outcome in the cell's pool) / (coded videos with that outcome in the cell)."""
    if not FAILURE_CODES.exists():
        return None
    fc = pd.read_csv(FAILURE_CODES, dtype=str, keep_default_na=False)
    if fc.empty:
        return None
    sm = pd.read_csv(CODING_SAMPLE, dtype={"vid": str})
    d = fc[["vid", "codes", "attempts", "flag"]].merge(sm, on="vid", how="inner", validate="one_to_one")
    d["codes"] = d["codes"].map(lambda c: [x for x in c.split("+") if x])
    d["miss"] = d["codes"].map(lambda c: "G-MISS" in c or "G-SLIP" in c)
    d["retry"] = d["attempts"] != "1"
    pool_n = np.where(d["success"] == 1, d["pool_succ"], d["pool_fail"])
    coded_n = d.groupby(["cell", "success"])["vid"].transform("count")
    d["w"] = pool_n / coded_n
    rng = np.random.default_rng(RNG_SEED)

    def rates(g):
        """Weighted retry / recovery rate among episodes with a miss or drop, + stratified bootstrap 95% CI
        (resampling videos within each (cell, outcome) stratum, keeping the weights)."""
        def est(miss, w, retry, succ):  # arrays (..., n) -> rates (...)
            den = (w * miss).sum(-1)
            with np.errstate(invalid="ignore", divide="ignore"):
                return (w * miss * retry).sum(-1) / den, (w * miss * succ).sum(-1) / den
        cols = lambda x: [x[k].to_numpy(float) for k in ("miss", "w", "retry", "success")]
        point = est(*cols(g))
        B, parts = 2000, []
        for _, x in g.groupby(["cell", "success"]):
            idx = rng.integers(0, len(x), size=(B, len(x)))
            parts.append([c[idx] for c in cols(x)])
        boot = est(*[np.concatenate([p_[j] for p_ in parts], axis=1) for j in range(4)])
        ci = [tuple(np.nanpercentile(b_, [2.5, 97.5])) if np.isfinite(b_).any() else (np.nan, np.nan) for b_ in boot]
        return dict(n_coded=len(g), n_miss=int(g["miss"].sum()), n_miss_success=int((g["miss"] & (g["success"] == 1)).sum()),
                    n_miss_retry=int((g["miss"] & g["retry"]).sum()),
                    retry=float(point[0]), retry_ci=ci[0], recovery=float(point[1]), recovery_ci=ci[1])

    out = dict(n_coded=len(d), n_sample=len(sm), n_flagged=int((d["flag"] == "1").sum()), cells={}, methods={})
    for cell, g in d.groupby("cell"):
        f = g[g["success"] == 0]
        out["cells"][cell] = dict(
            failures=len(f), successes=int((g["success"] == 1).sum()),
            any={c: int(f["codes"].map(lambda x: c in x).sum()) for c in CODE_NAMES},
            first={c: int(f["codes"].map(lambda x: bool(x) and x[0] == c).sum()) for c in CODE_NAMES},
            clean_successes=int(((g["success"] == 1) & (g["codes"].map(len) == 0)).sum()),
            **rates(g))
    for m, g in d.groupby("method"):
        out["methods"][m] = rates(g)
    return out


# --------------------------------------------------------------------------------------------- figures
def style():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Inter", "DejaVu Sans"],
        "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10, "axes.edgecolor": GRID,
        "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.8, "axes.axisbelow": True, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "legend.frameon": False, "svg.fonttype": "none", "svg.hashsalt": "vla-vs-scratch",
    })


def save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(FIG_DIR / f"{name}.{ext}", dpi=200, bbox_inches="tight",
                    metadata={"Date": None} if ext == "svg" else {"Software": None})  # reproducible files
    plt.close(fig)


def x_positions(n_all):
    return {"5": 5, "10": 10, "20": 20, "all": n_all}


def fig_headline(head: pd.DataFrame, n_all_mean: float):
    xs = x_positions(n_all_mean)
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    for m in METHODS:
        h = head[head.method == m].set_index("size").loc[SIZES]
        x = [xs[s] for s in SIZES]
        ax.fill_between(x, h["seed_min"], h["seed_max"], color=COLOR[m], alpha=0.14, lw=0, zorder=1)
        for s in SIZES:
            for v in h.loc[s, "per_seed"].values():
                ax.plot(xs[s], v, "o", ms=5, color=COLOR[m], alpha=0.55, mec=SURFACE, mew=0.8, zorder=2)
        ax.plot(x, h["mean"], "-o", color=COLOR[m], lw=2, ms=8, mec=SURFACE, mew=1.5, zorder=3)
        end = h.loc["all", "mean"]
        ax.annotate(f"{METHOD_LABEL[m]}: {end:.0f}%\n{len(SEEDS[m])} training seeds", (xs["all"], end),
                    xytext=(12, 0), textcoords="offset points", va="center", fontsize=9.5, color=INK)
        for s in SIZES[:-1]:  # selective value labels on the mean line (the last one is in the direct label)
            ax.annotate(f"{h.loc[s, 'mean']:.0f}%", (xs[s], h.loc[s, "mean"]), xytext=(0, 13) if m == "smolvla" else (8, -10),
                        textcoords="offset points", ha="center" if m == "smolvla" else "left", va="center",
                        fontsize=8.5, color=INK_2)
    ax.set_xscale("log")
    ax.set_xticks([xs[s] for s in SIZES])
    ax.set_xticklabels(["5", "10", "20", f"all\n(45 to 50)"])
    ax.minorticks_off()
    ax.set_xlim(4.3, 50)
    ax.set_ylim(0, 100)
    ax.set_yticks(range(0, 101, 25))
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax.grid(axis="x", visible=False)
    ax.set_xlabel("Demonstrations per task (log scale)")
    ax.set_ylabel("Success rate")
    handles = [Line2D([], [], color=INK_2, lw=2, marker="o", ms=7, mec=SURFACE, label="mean over training seeds"),
               Line2D([], [], color=INK_2, lw=0, marker="o", ms=5, alpha=0.55, label="one training seed"),
               Patch(color=INK_2, alpha=0.18, lw=0, label="range across seeds")]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, fontsize=8.5, handlelength=1.6)
    fig.suptitle("A pretrained VLA with 5 demos beats ACT trained from scratch on all demos",
                 x=0.06, ha="left", y=1.04, fontsize=12, fontweight="bold")
    fig.text(0.06, 0.975, "LIBERO-Object, 3 tasks in simulation. Each seed point pools 3 tasks, 3 checkpoints and 50 episodes",
             ha="left", fontsize=9, color=INK_2)
    fig.subplots_adjust(right=0.78)
    save(fig, "headline")


def fig_per_task(pt: pd.DataFrame, n_all: dict):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.9), sharey=True)
    for ax, t in zip(axes, TASKS):
        xs = x_positions(n_all[t])
        x = [xs[s] for s in SIZES]
        for m in METHODS:
            g = pt[(pt.method == m) & (pt.task == t)].set_index("size").loc[SIZES]
            for seed in SEEDS[m]:
                ax.plot(x, [g.loc[s, "per_seed"][seed] for s in SIZES], color=COLOR[m], lw=1, alpha=0.4, zorder=2)
            ax.plot(x, g["mean"], "-o", color=COLOR[m], lw=2, ms=6.5, mec=SURFACE, mew=1.2, zorder=3)
        ax.set_xscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels(["5", "10", "20", f"all\n({n_all[t]})"])
        ax.minorticks_off()
        ax.grid(axis="x", visible=False)
        ax.set_xlim(4.3, 56)
        ax.set_title(TASK_TITLE[t], loc="left", fontsize=10.5)
        ax.set_xlabel("Demonstrations per task")
    axes[0].set_ylim(0, 100)
    axes[0].set_yticks(range(0, 101, 25))
    axes[0].yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    axes[0].set_ylabel("Success rate")
    handles = [Line2D([], [], color=COLOR[m], lw=2, marker="o", ms=6, mec=SURFACE, label=f"{METHOD_LABEL[m]}, mean")
               for m in METHODS] + [Line2D([], [], color=INK_2, lw=1, alpha=0.5, label="individual training seeds")]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.06, 0.93), ncol=3, fontsize=9)
    fig.suptitle("Per task: ACT does worst on butter, and SmolVLA dips at 10 or 20 demos on two tasks",
                 x=0.06, ha="left", y=1.06, fontsize=12, fontweight="bold")
    fig.subplots_adjust(left=0.07, right=0.98, top=0.86, bottom=0.17, wspace=0.08)
    save(fig, "per_task")


def fig_stability(runs: pd.DataFrame, summ: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(11.5, 4.4))
    gap, xt, xl = 7.0, [], []
    for gi, (t, s) in enumerate([(t, s) for t in TASKS for s in SIZES]):
        base = gi * gap
        offs = {("act", 1): 0, ("act", 2): 1, ("act", 3): 2, ("smolvla", 1): 3.5, ("smolvla", 2): 4.5}
        for (m, seed), o in offs.items():
            r = runs[(runs.method == m) & (runs.task == t) & (runs["size"] == s) & (runs.seed == seed)].iloc[0]
            v = np.array(r["ckpt_counts"]) * 100 / N_EPS
            ax.plot([base + o] * 2, [v.min(), v.max()], color=COLOR[m], lw=2.2, solid_capstyle="round", zorder=2)
            ax.plot([base + o] * 3, v, "o", color=COLOR[m], ms=4.2, mec=SURFACE, mew=0.7, zorder=3)
        xt.append(base + 2.25)
        xl.append(s)
        if gi % 4 == 0 and gi:
            ax.axvline(base - gap / 2 + 0.25, color=GRID, lw=1)
    for ti, t in enumerate(TASKS):
        ax.text(ti * 4 * gap + 1.5 * gap + 2.25, -17, TASK_TITLE[t], ha="center", fontsize=10, color=INK)
    ax.set_xticks(xt)
    ax.set_xticklabels(xl)
    ax.grid(axis="x", visible=False)
    ax.set_xlim(-1.5, 12 * gap - 1.5)
    ax.set_ylim(0, 100)
    ax.set_yticks(range(0, 101, 25))
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax.set_ylabel("Success rate of one checkpoint")
    ax.set_xlabel("Demonstrations per task, grouped by task", labelpad=28)
    sm = summ.set_index("method")
    handles = [Line2D([], [], color=COLOR[m], lw=2.2, marker="o", ms=4.5, mec=SURFACE,
                      label=f"{METHOD_LABEL[m]}: checkpoint variance {sm.loc[m, 'ckpt_ratio']:.1f} times sampling noise")
               for m in ["act", "smolvla"]]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=9)
    fig.suptitle("ACT's checkpoints disagree far beyond sampling noise; SmolVLA's do not",
                 x=0.06, ha="left", y=1.06, fontsize=12, fontweight="bold")
    fig.text(0.06, 0.975, "Each bar is one training run and its 3 dots are the evaluated checkpoints (50 episodes each). "
             "In each group, ACT seeds 1 to 3 are on the left and SmolVLA seeds 1 and 2 on the right.", ha="left", fontsize=9, color=INK_2)
    save(fig, "stability")


def fig_scenes(sc: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.8), sharex=True, sharey=True)
    for ax, r in zip(axes, sc.itertuples()):
        ax.plot(100 * r.act_rate, 100 * r.smolvla_rate, "o", color=COLOR["smolvla"], ms=6, alpha=0.75, mec=SURFACE,
                mew=0.8)
        ax.set_title(TASK_TITLE[r.task], loc="left", fontsize=10.5)
        ax.text(0.03, 0.04, f"Spearman ρ = {r.spearman:.2f}", transform=ax.transAxes, fontsize=9, color=INK_2)
        ax.set_xlabel("ACT success on this scene")
        ax.set_aspect("equal")
    axes[0].set_ylabel("SmolVLA success on this scene")
    for ax in axes:
        ax.set_xlim(-3, 103)
        ax.set_ylim(-3, 103)
        ax.set_xticks(range(0, 101, 50))
        ax.set_yticks(range(0, 101, 50))
        ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    fig.suptitle("Do both methods fail on the same starting scenes? Each dot is one of the 50 scenes",
                 x=0.06, ha="left", y=1.04, fontsize=12, fontweight="bold")
    fig.text(0.06, 0.955, "Pooled over all demo sizes, seeds and checkpoints (ACT 36, SmolVLA 24 trials per scene)",
             ha="left", fontsize=9, color=INK_2)
    fig.tight_layout()
    save(fig, "scenes")


# --------------------------------------------------------------------------------------------- report
def md_table(headers, rows, align=None):
    align = align or ["l"] + ["r"] * (len(headers) - 1)
    sep = ["---:" if a == "r" else ":---:" if a == "c" else "---" for a in align]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(sep) + " |"]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def f1(x):
    return f"{x:.1f}"


def size_label(s, n_all=None):
    return f"all ({n_all})" if s == "all" and n_all else s


def report(head, pt, runs, summ, d50, claim, sc, bt, comp, n_all, fl=None) -> str:
    out = ["# Results analysis", "",
           "Generated by `scripts/analyze.py` from `results/results.csv` — do not edit by hand.",
           "Unit of replication = training seed; a seed point pools 3 tasks × 3 checkpoints × 50 episodes. "
           "Wilson 95% CIs cover evaluation sampling only.", ""]

    # 1 headline
    out += ["## 1. Headline: success vs demonstrations (pooled over 3 tasks)", ""]
    rows = []
    for m in METHODS:
        for s in SIZES:
            h = head[(head.method == m) & (head["size"] == s)].iloc[0]
            seeds = " / ".join(f1(h.per_seed[k]) for k in SEEDS[m])
            rows.append([METHOD_LABEL[m], size_label(s, "45–50"), seeds, f"**{f1(h['mean'])}**",
                         f"{f1(h.seed_min)}–{f1(h.seed_max)}", f"{h.k}/{h.n}", f"{f1(h.wilson_lo)}–{f1(h.wilson_hi)}"])
    out += [md_table(["Method", "Demos", "Per seed (%)", "Mean (%)", "Seed range", "Pooled episodes",
                      "Wilson 95% CI (eval only)"], rows), ""]
    gap = [head[(head.method == "smolvla") & (head["size"] == s)]["mean"].iloc[0]
           - head[(head.method == "act") & (head["size"] == s)]["mean"].iloc[0] for s in SIZES]
    out += ["Gap (SmolVLA − ACT, mean over seeds): " + " · ".join(f"{s}: +{g:.1f}" for s, g in zip(SIZES, gap)), ""]

    # 2 robust claim + demos to 50
    out += ["## 2. Claims that do not depend on how uncertainty is computed", "",
            "Demo counts at which SmolVLA's **worst** seed beats ACT's **best** seed with all demos:", ""]
    rows = [[c["scope"] if c["scope"] == "pooled" else f"{c['scope']} · {TASK_NAME[c['scope']]}",
             f1(c["act_best_seed_all"]), " / ".join(f1(c["smolvla_worst_seed"][s]) for s in SIZES),
             ", ".join(c["holds_at"]) or "none"] for c in claim]
    out += [md_table(["Scope", "ACT best seed, all demos (%)", "SmolVLA worst seed at 5/10/20/all (%)",
                      "Holds at SmolVLA demos"], rows), ""]
    out += ["Demos needed to reach 50% (first size where the mean over seeds is ≥ 50%):", ""]
    rows = []
    for m in METHODS:
        d = d50[m]
        first = "< 5 (above 50% at the smallest size)" if d["below_smallest"] else (d["first_size_mean"] or "not reached")
        ps = "; ".join(f"s{k}: {v['first'] or '—'}{'' if v['stays_above'] or not v['first'] else ' (drops below again)'}"
                       for k, v in d["per_seed"].items())
        rows.append([METHOD_LABEL[m], first, ps])
    out += [md_table(["Method", "Mean over seeds", "Per seed"], rows, ["l", "l", "l"]), ""]

    # 3 per task
    out += ["## 3. Per task (success %, per seed and mean; 150 episodes per seed)", ""]
    rows = []
    for t in TASKS:
        for s in SIZES:
            cells = []
            for m in ["act", "smolvla"]:
                r = pt[(pt.method == m) & (pt.task == t) & (pt["size"] == s)].iloc[0]
                cells += [" / ".join(f"{r.per_seed[k]:.0f}" for k in SEEDS[m]), f"**{r['mean']:.1f}**"]
            rows.append([f"{t} · {TASK_NAME[t]}", size_label(s, n_all[t])] + cells)
    out += [md_table(["Task", "Demos", "ACT s1/s2/s3", "ACT mean", "SmolVLA s1/s2", "SmolVLA mean"], rows), ""]

    # 4 stability
    out += ["## 4. Stability: checkpoints and seeds compared with sampling noise", "",
            "*Checkpoints:* spread of the 3 evaluated checkpoints of a run vs the spread that sampling 50 episodes "
            "alone would produce if the checkpoints were equally good. *Seeds:* spread of run means across seeds vs "
            "what checkpoint + sampling noise alone would produce. \"Extra SD\" = the part not explained by the level "
            "below (percentage points). A run's checkpoints \"differ\" if a Monte Carlo test against sampling noise "
            "gives p < 0.05 (≈5% of runs expected by chance).", ""]
    rows = []
    for r in summ.itertuples():
        rows.append([METHOD_LABEL[r.method], f"{r.ckpt_sd_obs:.1f} vs {r.ckpt_sd_sampling:.1f}", f"{r.ckpt_ratio:.1f}×",
                     f"{r.ckpt_sd_extra:.1f}", f"{r.ckpt_range_median:.0f} / {r.ckpt_range_max}",
                     f"{r.runs_ckpt_differ}/{r.n_runs}", f"{r.seed_sd_obs:.1f} vs {r.seed_sd_expected:.1f}",
                     f"{r.seed_ratio:.1f}×", f"{r.seed_sd_extra:.1f}", f"{r.seed_range_max:.0f}"])
    out += [md_table(["Method", "Checkpoint SD: observed vs sampling (pts)", "Variance ratio", "Extra SD (pts)",
                      "Checkpoint range, median / max (of 50)", "Runs whose checkpoints differ",
                      "Seed SD: observed vs expected (pts)", "Variance ratio", "Extra SD (pts)",
                      "Max seed range (pts)"], rows), ""]
    out += ["<details><summary>Per run: successes /50 at each checkpoint</summary>", ""]
    rows = [[r.run, size_label(r.size), "/".join(map(str, r.ckpt_counts)), r.ckpt_range, f"{r.p_ckpt:.3f}"]
            for r in runs.assign(o=runs["size"].map(SIZES.index)).sort_values(["method", "task", "o", "seed"]).itertuples()]
    out += [md_table(["Run", "Demos", "Checkpoints", "Range", "p (checkpoints differ)"], rows), "", "</details>", ""]

    # 5 scenes
    out += ["## 5. Starting scenes (paired: episode i is the same scene in every run)", "",
            "*Scene effect* = variance of per-scene success rates ÷ the variance expected if all scenes were equally "
            "hard for a given checkpoint (1 = no scene effect). *Spearman ρ* = do the methods find the same scenes "
            "hard? Pooled over all sizes, seeds and checkpoints.", ""]
    rows = [[f"{r.task} · {TASK_NAME[r.task]}", f"{r.act_scene_ratio:.1f}×", f"{r.smolvla_scene_ratio:.1f}×",
             f"{r.spearman:.2f} (p = {r.spearman_p:.3f})", f"{r.both_below_median}/50",
             ", ".join(map(str, r.hardest_act)), ", ".join(map(str, r.hardest_smolvla))] for r in sc.itertuples()]
    out += [md_table(["Task", "Scene effect ACT", "Scene effect SmolVLA", "Spearman ρ", "Below median for both",
                      "Hardest 5 scenes ACT", "Hardest 5 scenes SmolVLA"], rows), ""]

    # 6 butter
    out += ["## 6. Diagnostic (not part of the grid): ACT on butter with 3× the training steps", "",
            "ACT t2, all 45 demos, 30k steps instead of 10k (`results/diagnostics.csv`). Successes /50.", ""]
    rows = [[r.seed, *[r.ckpt[k] for k in ["010000", "020000", "026000", "028000", "030000"]],
             f"{r.last3}/150 = {f1(r.last3_pct)}%", f"{r.grid}/150 = {f1(r.grid_pct)}%"] for r in bt.itertuples()]
    tot3, totg = int(bt["last3"].sum()), int(bt["grid"].sum())
    rows.append(["**pooled**", "", "", "", "", "", f"**{tot3}/450 = {f1(pct(tot3, 450))}%**",
                 f"**{totg}/450 = {f1(pct(totg, 450))}%**"])
    out += [md_table(["Seed", "10k", "20k", "26k", "28k", "30k", "Last 3 (26–30k)", "Grid run, 10k steps (6/8/10k)"],
                     rows), ""]

    # 7 compute
    out += ["## 7. Compute", ""]
    rows = []
    for m in METHODS:
        for k in ["grid", "diagnostic", "pilot"]:
            c = comp[m].get(k)
            if not c:
                continue
            rows.append([METHOD_LABEL[m], k, c["runs"], ", ".join(c["where"]), c["hours_per_run"], c["train_hours"],
                         f"${c['cost_usd']:.2f}", c.get("eval_hours", "")])
    out += [md_table(["Method", "Kind", "Runs", "Where", "Train h/run (median)", "Train h total", "Cost",
                      "Eval h (laptop)"], rows), "", f"Total cloud spend: **${comp['total_cloud_usd']:.2f}**", ""]

    # 8 failure analysis
    out += ["## 8. Failure analysis (coded videos)", ""]
    if fl is None:
        out += ["Not coded yet: run `scripts/code_videos.py` (sample in `notes/coding_sample.csv`).", ""]
        return "\n".join(out)
    cells = [c for c in CELL_LABEL if c in fl["cells"]]
    out += [f"{fl['n_coded']}/{fl['n_sample']} sampled videos coded blind to method ({fl['n_flagged']} flagged unsure). "
            "Sample: 10 failures + 5 successes per cell, drawn at random from episodes 0–9 (seeded, "
            "`scripts/make_coding_sample.py`). Counts are raw; rates are weighted back to each outcome's real share.", "",
            "**Failure causes** (coded failures; count where the code appears, first cause in brackets):", ""]
    rows = [[c] + [f"{fl['cells'][k]['any'][c]} ({fl['cells'][k]['first'][c]})" for k in cells] for c in CODE_NAMES]
    rows.append(["*failures coded*"] + [str(fl["cells"][k]["failures"]) for k in cells])
    out += [md_table(["Code"] + [CELL_LABEL[k] for k in cells], rows), ""]

    def r(x, ci):
        return "—" if np.isnan(x) else f"{100 * x:.0f}% ({100 * ci[0]:.0f}–{100 * ci[1]:.0f})"
    out += ["**After a missed or dropped grasp**: retry rate = tried again (≥ 2 attempts); recovery rate = still "
            "succeeded. Weighted; 95% CI from a stratified bootstrap.", ""]
    rows = [[CELL_LABEL[k], fl["cells"][k]["n_miss"], r(fl["cells"][k]["retry"], fl["cells"][k]["retry_ci"]),
             r(fl["cells"][k]["recovery"], fl["cells"][k]["recovery_ci"])] for k in cells]
    rows += [[f"**{METHOD_LABEL[m]}, all cells**", v["n_miss"], r(v["retry"], v["retry_ci"]),
              r(v["recovery"], v["recovery_ci"])] for m, v in fl["methods"].items()]
    out += [md_table(["Cell", "Videos with a miss/drop", "Retry rate", "Recovery rate"], rows), ""]
    return "\n".join(out)


# --------------------------------------------------------------------------------------------- main
def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return [jsonable(v) for v in o.tolist()]
    if isinstance(o, (np.floating, float)):
        return None if np.isnan(o) else round(float(o), 3)
    if isinstance(o, np.integer):
        return int(o)
    return o


def main():
    cfg = yaml.safe_load(RUNS_YAML.read_text())
    splits = json.loads(SPLITS.read_text())
    n_all = {t: splits["tasks"][t]["n_available"] for t in TASKS}
    df = load_results(RESULTS)
    validate(df, cfg, splits)
    diag = load_diag(DIAG)

    head = headline(df)
    pt = per_task(df)
    runs, summ = stability(df)
    d50 = demos_to_50(head)
    claim = robust_claim(df, n_all)
    sc = scenes(df)
    bt = butter(diag, df)
    comp = compute_summary(df, diag)
    fl = failures()

    style()
    n_all_mean = float(np.mean(list(n_all.values())))
    fig_headline(head, n_all_mean)
    fig_per_task(pt, n_all)
    fig_stability(runs, summ)
    fig_scenes(sc)

    MD_OUT.write_text(report(head, pt, runs, summ, d50, claim, sc, bt, comp, n_all, fl) + "\n")
    summary = dict(
        n_rows=len(df), n_runs=int(df["name"].nunique()), n_episodes=int(df["n_episodes"].sum()),
        demos_all=n_all, seeds=SEEDS,
        headline={m: {s: {k: v for k, v in r.items() if k not in ("method", "size")}
                      for s, r in head[head.method == m].set_index("size").to_dict("index").items()} for m in METHODS},
        gap_points={s: head[(head.method == "smolvla") & (head["size"] == s)]["mean"].iloc[0]
                    - head[(head.method == "act") & (head["size"] == s)]["mean"].iloc[0] for s in SIZES},
        per_task={f"{r.method}/{r.task}/{r.size}": dict(per_seed=r.per_seed, mean=r.mean) for r in pt.itertuples()},
        stability=summ.set_index("method").to_dict("index"),
        demos_to_50=d50, robust_claim=claim,
        scenes={r.task: dict(spearman=r.spearman, spearman_p=r.spearman_p, act_scene_ratio=r.act_scene_ratio,
                             smolvla_scene_ratio=r.smolvla_scene_ratio, both_below_median=r.both_below_median)
                for r in sc.itertuples()},
        butter_diagnostic=dict(per_seed=bt.drop(columns=["ckpt", "grid_ckpt"]).to_dict("records"),
                               long_pct=pct(bt["last3"].sum(), 450), grid_pct=pct(bt["grid"].sum(), 450)),
        compute=comp,
        failures=fl,
    )
    JSON_OUT.write_text(json.dumps(jsonable(summary), indent=1) + "\n")
    print(f"wrote {MD_OUT.relative_to(ROOT)}, {JSON_OUT.relative_to(ROOT)}, figures in {FIG_DIR.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
