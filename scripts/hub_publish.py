#!/usr/bin/env python
"""Publish the experiment's checkpoints on the Hugging Face Hub (Phase 5, deliverable D2).

Two jobs, both dry runs unless --apply is given:

  act             Upload the 36 ACT grid runs (3 seeds x 3 tasks x 4 sizes) as public repos
                  <user>/<run>, e.g. Pipe1213/act_obj_t1_n05_s2. Same layout as the SmolVLA repos:
                    checkpoints/{006000,008000,010000}/pretrained_model/   (the 3 evaluated checkpoints)
                    <root>                                                 (= the 010000 weights, so
                                                                            --policy.path=<repo> works)
                  Files are hard-linked into outputs/hub_staging/<run>/ (no extra disk space) and sent
                  with HfApi.upload_large_folder, which resumes after an interruption: re-run the same
                  command and it continues where it stopped. A run whose repo already holds every
                  expected file is skipped. The root copy of 010000 has the same bytes as
                  checkpoints/010000, so the Hub's chunk deduplication should not upload it twice.

  rename-smolvla  Rename the 24 SmolVLA grid repos from <run>_<YYYY-MM-DD_HH-MM-SS> to <run>
                  (the Hub redirects the old names). The Phase 1 pilot repo is not touched.
                  eval_grid.py --download already prefers the exact <user>/<run> name.

Model cards are written later by a separate script (they need the README to link to).

Usage (repo root, `lerobot` env, logged in with `hf auth login`, token with write access):
  python scripts/hub_publish.py act                              # dry run: runs, files, sizes
  python scripts/hub_publish.py act --apply --limit 1            # speed test: first run only
  python scripts/hub_publish.py act --apply 2>&1 | tee -a outputs/logs/hub_publish.log
  python scripts/hub_publish.py act --apply --only act_obj_t2_nall_s3
  python scripts/hub_publish.py rename-smolvla                   # dry run: old -> new names
  python scripts/hub_publish.py rename-smolvla --apply
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_grid import ROOT, ckpt_dir, eval_ckpts, load, runs_for  # noqa: E402

STAGING = ROOT / "outputs/hub_staging"
STAMP = r"_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$"
SMOLVLA_GRID = re.compile(r"^(smolvla_obj_t[123]_n(?:05|10|20|all)(?:_s\d+)?)" + STAMP)


def gb(n: float) -> str:
    return f"{n / 1e9:.2f} GB"


def hms(seconds: float) -> str:
    m = int(seconds // 60)
    return f"{m // 60} h {m % 60:02d} min"


# ---------------------------------------------------------------------------------------- ACT

def act_runs(cfg, splits):
    seeds = sorted(cfg.get("training_seeds", {1: cfg["training_seed"]}))
    return [r for k in seeds for r in runs_for("act", cfg, splits, k)]


def expected_files(run, cfg) -> dict[str, Path]:
    """Repo path -> local file, for one run. Raises if a local checkpoint is missing."""
    steps = eval_ckpts(run, cfg)
    out: dict[str, Path] = {}
    for step in steps:
        src = ckpt_dir(run) / step / "pretrained_model"
        files = sorted(p for p in src.rglob("*") if p.is_file())
        if not any(p.name == "model.safetensors" for p in files):
            raise FileNotFoundError(f"{run['name']}: no model.safetensors in {src}")
        for p in files:
            out[f"checkpoints/{step}/pretrained_model/{p.relative_to(src).as_posix()}"] = p
    last = ckpt_dir(run) / steps[-1] / "pretrained_model"
    for p in sorted(q for q in last.rglob("*") if q.is_file()):
        out[p.relative_to(last).as_posix()] = p
    return out


def stage(run_name: str, files: dict[str, Path]) -> Path:
    """Hard-link (or symlink if hard links are impossible) every file into the staging folder.
    Keeps the folder's .cache/ so upload_large_folder can resume."""
    dst_root = STAGING / run_name
    for rel, src in files.items():
        dst = dst_root / rel
        if dst.exists() and dst.stat().st_ino == src.stat().st_ino:
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        try:
            os.link(src, dst)
        except OSError:
            dst.symlink_to(src.resolve())
    # remove anything in staging that is not expected (except the upload cache)
    for p in sorted(dst_root.rglob("*"), reverse=True):
        rel = p.relative_to(dst_root).as_posix()
        if rel.startswith(".cache"):
            continue
        if p.is_file() and rel not in files:
            p.unlink()
        elif p.is_dir() and not any(p.iterdir()):
            p.rmdir()
    return dst_root


def cmd_act(args):
    cfg, splits = load()
    runs = act_runs(cfg, splits)
    if args.only:
        unknown = set(args.only) - {r["name"] for r in runs}
        if unknown:
            sys.exit(f"unknown run(s): {', '.join(sorted(unknown))}")
        runs = [r for r in runs if r["name"] in args.only]

    plan = []
    for r in runs:
        files = expected_files(r, cfg)
        unique = sum(p.stat().st_size for rel, p in files.items() if rel.startswith("checkpoints/"))
        plan.append((r, files, unique))
    total = sum(u for *_, u in plan)
    print(f"{len(plan)} ACT runs, 3 checkpoints each, {gb(total)} to upload "
          f"(the root copy of 010000 is deduplicated by the Hub and not counted)")

    if not args.apply:
        for r, files, unique in plan:
            print(f"  [dry] {r['name']:<22} {len(files):>3} files  {gb(unique)}")
        print("dry run only; add --apply to upload (try --limit 1 first to measure the speed)")
        return

    from huggingface_hub import HfApi

    api = HfApi()
    me = api.whoami()["name"]
    done_bytes, t_start, uploaded = 0, time.time(), 0
    todo = plan[: args.limit] if args.limit else plan
    remaining = sum(u for *_, u in todo)
    for i, (r, files, unique) in enumerate(todo, 1):
        repo = f"{me}/{r['name']}"
        print(f"\n[{i}/{len(todo)}] {repo}  ({gb(unique)})", flush=True)
        if api.repo_exists(repo) and set(files) <= set(api.list_repo_files(repo)):
            print("  already on the Hub with every expected file: skipped")
            remaining -= unique
            continue
        folder = stage(r["name"], files)
        api.create_repo(repo, private=False, exist_ok=True)
        t0 = time.time()
        api.upload_large_folder(repo_id=repo, folder_path=folder, repo_type="model",
                                private=False, print_report_every=120)
        missing = set(files) - set(api.list_repo_files(repo))
        if missing:
            print(f"  [warn] {len(missing)} files missing after upload, e.g. {sorted(missing)[0]}; "
                  "re-run the same command to resume")
            continue
        dt_run = time.time() - t0
        done_bytes += unique
        remaining -= unique
        uploaded += 1
        speed = done_bytes / (time.time() - t_start)
        print(f"  done in {hms(dt_run)} ({unique / max(dt_run, 1) / 1e6:.1f} MB/s); "
              f"average {speed / 1e6:.1f} MB/s; left {gb(remaining)}, about {hms(remaining / speed)}",
              flush=True)
        if not args.keep_staging:
            shutil.rmtree(folder)            # hard links only: frees no data, just tidies up
    print(f"\nfinished: {uploaded} runs uploaded in {hms(time.time() - t_start)}")


# ------------------------------------------------------------------------------------ SmolVLA

def cmd_rename(args):
    from huggingface_hub import HfApi

    api = HfApi()
    me = api.whoami()["name"]
    existing = {m.id.split("/")[1] for m in api.list_models(author=me, search="smolvla_obj_")}
    by_run: dict[str, list[str]] = {}
    for name in sorted(existing):
        m = SMOLVLA_GRID.match(name)
        if m:
            by_run.setdefault(m.group(1), []).append(name)

    cfg, splits = load()
    seeds = [k for k in sorted(cfg["training_seeds"]) if k in (1, 2)]   # SmolVLA has seeds 1 and 2
    expected = [r["name"] for k in seeds for r in runs_for("smolvla", cfg, splits, k)]
    moves = []
    for run in expected:
        if run in existing:
            print(f"  [ok]   {me}/{run} already has the clean name")
            continue
        olds = by_run.get(run, [])
        if not olds:
            print(f"  [warn] no repo found for {run}")
            continue
        if len(olds) > 1:
            print(f"  [warn] {run}: several stamped repos {olds}; using the newest (as eval_grid.py did)")
        moves.append((f"{me}/{olds[-1]}", f"{me}/{run}"))

    for old, new in moves:
        print(f"  {'[move]' if args.apply else '[dry] '} {old}  ->  {new}")
        if args.apply:
            api.move_repo(from_id=old, to_id=new, repo_type="model")
    if not args.apply:
        print(f"dry run only: {len(moves)} repos would be renamed; add --apply")
    else:
        print(f"renamed {len(moves)} repos")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("act", help="upload the ACT grid checkpoints")
    a.add_argument("--apply", action="store_true", help="really upload (default: dry run)")
    a.add_argument("--only", nargs="+", help="only these run names")
    a.add_argument("--limit", type=int, help="upload at most this many runs (speed test: --limit 1)")
    a.add_argument("--keep-staging", action="store_true", help="keep outputs/hub_staging/<run> after upload")
    s = sub.add_parser("rename-smolvla", help="drop the timestamp from the SmolVLA grid repo names")
    s.add_argument("--apply", action="store_true", help="really rename (default: dry run)")
    args = ap.parse_args()
    {"act": cmd_act, "rename-smolvla": cmd_rename}[args.cmd](args)


if __name__ == "__main__":
    main()
