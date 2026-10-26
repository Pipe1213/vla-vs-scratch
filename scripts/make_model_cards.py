#!/usr/bin/env python
"""Model cards and the Hub collection for the 60 trained policies (Phase 5, deliverable D2).

Every number in a card comes from the repo's files: results/results.csv (this run's successes per checkpoint),
results/summary.json (the study's pooled numbers), configs/splits.json (episode ids), configs/runs.yaml
(settings, revisions) and results/compute_log.csv (training time and cost). Nothing is typed by hand.

  cards        Write one README.md per run to outputs/model_cards/<run>/README.md (dry run, no network).
               With --apply, upload each card as README.md to Pipe1213/<run> (replaces LeRobot's
               auto-generated card on the SmolVLA repos). Runs whose repo does not exist are reported, not created.
  collection   Create (or reuse) one public Hub collection with all 60 models; prints its URL.
               Dry run lists what would be added; --apply creates it.

Usage (repo root, `lerobot` env, logged in with `hf auth login`, token with write access):
  python scripts/make_model_cards.py cards                         # write the 60 cards locally
  python scripts/make_model_cards.py cards --only act_obj_t2_n10_s3   # one card, to read it first
  python scripts/make_model_cards.py cards --apply                 # upload all 60 cards
  python scripts/make_model_cards.py collection [--apply]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shlex
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_grid import ROOT, load, runs_for, train_cmd  # noqa: E402

OWNER = "Pipe1213"
GITHUB = "https://github.com/Pipe1213/vla-vs-scratch"
OUT = ROOT / "outputs" / "model_cards"
LEROBOT_COMMIT = "b9cb121cb7d4e3e26ec5c906d08dda68d151ba52"
TASK_NAME = {"t1": "BBQ sauce", "t2": "butter", "t3": "chocolate pudding"}
METHOD = {"act": "ACT", "smolvla": "SmolVLA"}
METHOD_LONG = {"act": "ACT trained from scratch", "smolvla": "SmolVLA fine-tuned from lerobot/smolvla_base"}
SEEDS = {"act": [1, 2, 3], "smolvla": [1, 2]}
SIZES = ["5", "10", "20", "all"]
RENAME_MAP = ('{"observation.images.image": "observation.images.camera1", '
              '"observation.images.image2": "observation.images.camera2"}')
COLLECTION_TITLE = "SmolVLA and ACT on LIBERO-Object, 5 to 50 demos"
COLLECTION_DESC = ("60 policies from vla-vs-scratch: SmolVLA fine-tuned and ACT trained from scratch on 3 "
                   "LIBERO-Object tasks with 5, 10, 20 or all demonstrations.")


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def pct(x: float) -> str:
    return f"{x:.0f}%" if abs(x - round(x)) < 0.05 else f"{x:.1f}%"


def all_runs(cfg, splits):
    return [r | {"method": m} for m in ("smolvla", "act") for k in SEEDS[m] for r in runs_for(m, cfg, splits, k)]


def load_rows():
    rows = {}
    for r in csv.DictReader(open(ROOT / "results" / "results.csv")):
        rows.setdefault(r["name"], {})[r["label"]] = int(r["n_success"])
    compute = {r["run"]: r for r in csv.DictReader(open(ROOT / "results" / "compute_log.csv"))}
    return rows, compute


def card(run: dict, cfg: dict, splits: dict, rows: dict, compute: dict, summ: dict) -> str:
    m, t, size, seed_idx = run["method"], run["task"], run["size"], run["seed_index"]
    name, seed = run["name"], run["seed"]
    mc = cfg["methods"][m]
    ckpts = mc["eval_checkpoints"]
    counts = rows[name]
    missing = [c for c in ckpts if c not in counts]
    if missing:
        raise KeyError(f"{name}: no results for checkpoints {missing}")
    n_eps = cfg["evaluation"]["episodes_per_checkpoint"]
    task_id = cfg["tasks"][t]
    instr = splits["tasks"][t]["instruction"]
    n_demos = len(run["episodes"])
    size_text = f"all {n_demos}" if size == "all" else str(n_demos)
    k = sum(counts[c] for c in ckpts)
    n = n_eps * len(ckpts)
    lo, hi = wilson(k, n)
    last = ckpts[-1]
    other = "act" if m == "smolvla" else "smolvla"
    h_self, h_other = summ["headline"][m][size], summ["headline"][other][size]
    size_phrase = "all demonstrations (45 to 50 per task)" if size == "all" else f"{size} demonstrations per task"

    cmd = train_cmd(m, run, cfg)
    note_cmd = ""
    if m == "act" and seed_idx == 1:   # seed-1 ACT runs were trained before this logging-only flag existed
        cmd = [a for a in cmd if a != "--wandb.disable_artifact=true"]
        note_cmd = ("\n*Trained before `--wandb.disable_artifact=true` was added to the ACT settings "
                    "(it only stops checkpoint uploads to Weights & Biases).*\n")
    if m == "smolvla":
        note_cmd = ("\n*LeRobot ignored `--policy.repo_id` when fine-tuning with `--policy.path` and created the repo "
                    "with a timestamp in its name; it was renamed to this name and made public after training.*\n")
    cmd_text = " \\\n  ".join(shlex.quote(a) if i else a for i, a in enumerate(cmd))

    comp = compute.get(name, {})
    where = {"laptop": "a laptop RTX 4050 (6 GB)", "hf_jobs": "Hugging Face Jobs, NVIDIA L4 (24 GB)"}.get(
        comp.get("where", ""), comp.get("where", "unknown"))
    hours = comp.get("hours")
    cost = float(comp.get("cost_usd") or 0)
    time_text = f"{float(hours):.2f} h on {where}" if hours else where
    if cost:
        time_text += f", ${cost:.2f}"

    front = ["---", "license: apache-2.0", "library_name: lerobot", "pipeline_tag: robotics",
             "tags:", "- lerobot", "- robotics", "- libero", f"- {m}", "- imitation-learning",
             "datasets:", "- lerobot/libero"]
    if m == "smolvla":
        front += ["base_model: lerobot/smolvla_base"]
    front += ["---", ""]

    if m == "act":
        method_text = (
            "ACT (Zhao et al., 2023) with LeRobot's default settings: 52M parameters, chunks of 100 actions, "
            "and a ResNet-18 image encoder initialised from ImageNet weights. Everything else is trained from "
            "scratch on the demonstrations below. The only change from the defaults is at evaluation: the policy "
            "re-plans every 10 actions instead of running a whole chunk.")
        settings = f"batch size 8, 10,000 steps, a checkpoint every 2,000 steps, seed {seed}"
    else:
        method_text = (
            f"SmolVLA (Shukor et al., 2025), fine-tuned from `lerobot/smolvla_base` at revision "
            f"`{mc['base_revision']}`. Only the action expert is trained (about 100M of 450M parameters); "
            "the vision-language model stays frozen. The policy also reads the task instruction.")
        settings = (f"batch size 32, 5,000 steps, a checkpoint every 1,000 steps, seed {seed}; camera names "
                    "mapped to the base model's with `--rename_map`")

    eval_cmd = (f"lerobot-eval --policy.path={OWNER}/{name} \\\n"
                f"  --env.type=libero --env.task=libero_object --env.task_ids=\"[{task_id}]\" \\\n"
                f"  --env.init_states=true --env.observation_height=256 --env.observation_width=256 \\\n"
                f"  --eval.n_episodes={n_eps} --eval.batch_size=1 --seed=1000 --policy.n_action_steps=10")
    if m == "smolvla":
        eval_cmd += f" \\\n  --rename_map='{RENAME_MAP}'"

    lines = front + [
        f"# {METHOD[m]} on LIBERO-Object, {TASK_NAME[t]}, {size_text} demonstrations, seed {seed}",
        "",
        f"One of the 60 policies trained for [vla-vs-scratch]({GITHUB}), a comparison of a fine-tuned pretrained "
        "vision-language-action model (SmolVLA) with a policy trained from scratch (ACT) on the same few "
        "demonstrations, in simulation. This model is "
        f"{METHOD_LONG[m]}, on one task only: *\"{instr}\"*. It is published to make the study reproducible, "
        "not as a general-purpose policy.",
        "",
        "## Results of this model",
        "",
        f"Each checkpoint was tested on the {n_eps} fixed starting scenes of the task (LIBERO initial states "
        f"0 to {n_eps - 1}), the same scenes for every model in the study.",
        "",
        "| Checkpoint (training step) | Successes | Success rate |",
        "| --- | ---: | ---: |",
    ]
    for c in ckpts:
        lines.append(f"| {int(c):,} | {counts[c]} of {n_eps} | {pct(100 * counts[c] / n_eps)} |")
    lines += [
        f"| **Mean of the {len(ckpts)} checkpoints** (the study's score) | **{k} of {n}** | "
        f"**{pct(100 * k / n)}** (95% Wilson interval {lo:.0f} to {hi:.0f}%) |",
        "",
        f"The repository root holds the last checkpoint ({int(last):,} steps, {pct(100 * counts[last] / n_eps)}). "
        "The Wilson interval covers evaluation sampling only; differences between training seeds are larger, "
        "especially for ACT.",
        "",
        f"For context, pooled over the study's 3 tasks and all training seeds, {METHOD[m]} with {size_phrase} "
        f"scored {pct(h_self['mean'])}, and {METHOD[other]} with the same demonstrations "
        f"scored {pct(h_other['mean'])}. Full results: [README]({GITHUB}#results) and "
        f"[analysis]({GITHUB}/blob/main/results/analysis.md).",
        "",
        "## Training",
        "",
        method_text,
        "",
        f"- **Task:** LIBERO-Object, simulator task id {task_id}, *\"{instr}\"*.",
        f"- **Demonstrations:** {n_demos} episodes of [`lerobot/libero`](https://huggingface.co/datasets/lerobot/libero) "
        f"at revision `{cfg['dataset']['revision']}`: {', '.join(map(str, run['episodes']))}. The smaller sets of "
        "the study are nested inside the larger ones, and both methods use the same episodes.",
        f"- **Settings:** {settings}. The step budget is the same for every number of demonstrations.",
        f"- **Software:** LeRobot {cfg['lerobot_version']} at commit `{LEROBOT_COMMIT}`.",
        f"- **Compute:** {time_text}.",
        "",
        "Exact command, generated from the repository's `configs/runs.yaml` by `scripts/train_grid.py`"
        + (" (submitted to Hugging Face Jobs)" if m == "smolvla" else "") + ":",
        "",
        "```bash",
        cmd_text,
        "```",
        note_cmd,
        "## Evaluation",
        "",
        f"{n_eps} episodes per checkpoint, starting from LIBERO's fixed initial states, 256 by 256 pixel images, "
        "re-planning every 10 actions, evaluation seed 1000, MuJoCo rendering on the CPU (WSL2). To evaluate the "
        "last checkpoint:",
        "",
        "```bash",
        eval_cmd,
        "```",
        "",
        f"The other evaluated checkpoints are in `checkpoints/<step>/pretrained_model` "
        f"({', '.join(ckpts)}): download that folder and point `--policy.path` at it.",
        "",
        "## Limitations",
        "",
        "- Simulation only, one task from one benchmark. Not tested on a real robot.",
        "- The study compares two methods as they are used in practice. SmolVLA and ACT differ in pretraining but "
        "also in size, architecture and language input, so the gap between them cannot be put on pretraining alone.",
        "- The 3 tasks were chosen among LIBERO-Object tasks that a published SmolVLA LIBERO checkpoint already "
        "solved, so results describe these tasks.",
        "- States and actions are normalised with statistics from the whole `lerobot/libero` dataset (LeRobot's "
        "behaviour), not only the selected demonstrations.",
    ]
    if m == "act":
        st = summ["stability"]["act"]
        lines += [
            "- ACT used LeRobot's defaults and was not tuned for this task.",
            f"- ACT's checkpoints from the same run often perform very differently (in {st['runs_ckpt_differ']} of "
            f"{st['n_runs']} ACT runs in the study). Evaluate a checkpoint before using it.",
        ]
    lines += ["", f"Code and full results: [{GITHUB}]({GITHUB}).", ""]
    return "\n".join(lines)


def cmd_cards(args):
    cfg, splits = load()
    summ = json.loads((ROOT / "results" / "summary.json").read_text())
    rows, compute = load_rows()
    runs = all_runs(cfg, splits)
    if args.only:
        unknown = set(args.only) - {r["name"] for r in runs}
        if unknown:
            sys.exit(f"unknown run(s): {', '.join(sorted(unknown))}")
        runs = [r for r in runs if r["name"] in args.only]
    texts = {}
    for r in runs:
        texts[r["name"]] = card(r, cfg, splits, rows, compute, summ)
        d = OUT / r["name"]
        d.mkdir(parents=True, exist_ok=True)
        (d / "README.md").write_text(texts[r["name"]])
    print(f"wrote {len(texts)} cards to {OUT.relative_to(ROOT)}/<run>/README.md")
    if not args.apply:
        print("dry run only; read one card, then add --apply to upload them")
        return

    from huggingface_hub import HfApi

    api = HfApi()
    missing, done = [], 0
    for name, text in texts.items():
        repo = f"{OWNER}/{name}"
        if not api.repo_exists(repo):
            missing.append(repo)
            print(f"  [missing] {repo}: not uploaded (run hub_publish.py first)")
            continue
        api.upload_file(path_or_fileobj=text.encode(), path_in_repo="README.md", repo_id=repo,
                        commit_message="Model card: results, training and evaluation details")
        done += 1
        print(f"  [card] {repo}")
    print(f"uploaded {done} cards; {len(missing)} repos missing")


def cmd_collection(args):
    cfg, splits = load()
    repos = [f"{OWNER}/{r['name']}" for r in all_runs(cfg, splits)]
    if not args.apply:
        print(f"would create the public collection \"{COLLECTION_TITLE}\" with {len(repos)} models:")
        for r in repos:
            print("  ", r)
        print("dry run only; add --apply")
        return
    from huggingface_hub import HfApi

    api = HfApi()
    col = api.create_collection(COLLECTION_TITLE, namespace=OWNER, description=COLLECTION_DESC,
                                private=False, exists_ok=True)
    missing = 0
    for r in repos:
        if not api.repo_exists(r):
            print(f"  [missing] {r}")
            missing += 1
            continue
        api.add_collection_item(col.slug, item_id=r, item_type="model", exists_ok=True)
    print(f"collection: https://huggingface.co/collections/{col.slug} ({len(repos) - missing} models, "
          f"{missing} missing)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cards", help="write (and with --apply upload) the 60 model cards")
    c.add_argument("--apply", action="store_true", help="upload the cards (default: write them locally only)")
    c.add_argument("--only", nargs="+", help="only these run names")
    k = sub.add_parser("collection", help="create the Hub collection with all 60 models")
    k.add_argument("--apply", action="store_true", help="create it (default: dry run)")
    args = ap.parse_args()
    {"cards": cmd_cards, "collection": cmd_collection}[args.cmd](args)


if __name__ == "__main__":
    main()
