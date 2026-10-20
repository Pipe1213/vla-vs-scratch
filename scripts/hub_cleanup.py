#!/usr/bin/env python
"""Free Hugging Face private storage used by the SmolVLA training jobs.

Each HF Job pushed every checkpoint (1k..5k), each with its optimizer `training_state`.
Evaluation only needs `checkpoints/{003000,004000,005000}/pretrained_model`, so this script:
  - deletes checkpoints/<step>/ folders that are not evaluation checkpoints (grid repos only;
    the pilot repo keeps all its steps, which were evaluated in the pilot sweep);
  - deletes every checkpoints/<step>/training_state/ folder;
  - squashes the repo history (super_squash_history): deleted files stay in git history and keep
    counting toward the quota until the history is squashed.
The root model (pushed at the end of training, = the 5k weights) is kept.

Only repos named like the experiment are touched:
  smolvla_obj_t<1-3>_n<05|10|20|all>[_s<k>]_<YYYY-MM-DD_HH-MM-SS>   (grid runs, any seed)
  smolvla_obj_t1_nall_pilot_<YYYY-MM-DD_HH-MM-SS>                   (Phase 1 pilot)

Usage (repo root, `lerobot` env, logged in with `hf auth login`):
  python scripts/hub_cleanup.py                 # dry run: sizes and what would be deleted
  python scripts/hub_cleanup.py --apply         # delete + squash
  python scripts/hub_cleanup.py --apply --make-public   # also make the seed-1 grid repos public
                                                # (public storage has its own, much larger quota)
Run it only when no job is still pushing to these repos.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_grid import load  # noqa: E402

STAMP = r"_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$"
GRID = re.compile(r"^smolvla_obj_t[123]_n(05|10|20|all)(_s\d+)?" + STAMP)
SEED1 = re.compile(r"^smolvla_obj_t[123]_n(05|10|20|all)" + STAMP)
PILOT = re.compile(r"^smolvla_obj_t1_nall_pilot" + STAMP)
CKPT = re.compile(r"^checkpoints/(\d{6})/")


def gb(n: int) -> str:
    return f"{n / 1e9:6.2f} GB"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true", help="really delete (default: dry run)")
    ap.add_argument("--make-public", action="store_true", help="make the cleaned seed-1 grid repos public")
    args = ap.parse_args()

    from huggingface_hub import CommitOperationDelete, HfApi

    cfg, _ = load()
    keep_steps = set(cfg["methods"]["smolvla"]["eval_checkpoints"])
    api = HfApi()
    me = api.whoami()["name"]
    repos = sorted(m.id for m in api.list_models(author=me, search="smolvla_obj_"))
    repos = [r for r in repos if GRID.match(r.split("/")[1]) or PILOT.match(r.split("/")[1])]
    if not repos:
        print("no matching repos"); return

    tot_before = tot_freed = 0
    for repo in repos:
        name = repo.split("/")[1]
        is_pilot = bool(PILOT.match(name))
        files = [f for f in api.list_repo_tree(repo, recursive=True) if hasattr(f, "size") and f.size is not None]
        size = sum(f.size for f in files)
        to_delete = defaultdict(int)          # folder -> bytes
        for f in files:
            m = CKPT.match(f.path)
            if not m:
                continue
            step = m.group(1)
            if not is_pilot and step not in keep_steps:
                to_delete[f"checkpoints/{step}/"] += f.size
            elif "/training_state/" in f.path:
                to_delete[f"checkpoints/{step}/training_state/"] += f.size
        freed = sum(to_delete.values())
        tot_before += size; tot_freed += freed
        print(f"{name:<52} {gb(size)}  -> delete {gb(freed)}  ({len(to_delete)} folders){'  [pilot]' if is_pilot else ''}")
        if not args.apply:
            continue
        if to_delete:
            ops = [CommitOperationDelete(path_in_repo=p, is_folder=True) for p in sorted(to_delete)]
            api.create_commit(repo, operations=ops, repo_type="model",
                              commit_message="Remove non-evaluated checkpoints and optimizer state")
            api.super_squash_history(repo, repo_type="model")
            print("    deleted + history squashed")
        if args.make_public and SEED1.match(name):
            try:
                api.update_repo_settings(repo, private=False)
            except AttributeError:            # older huggingface_hub
                api.update_repo_visibility(repo, private=False)
            print("    now public")

    print(f"\n{len(repos)} repos, {gb(tot_before)} total, {gb(tot_freed)} {'freed' if args.apply else 'would be freed'}.")
    if not args.apply:
        print("Dry run. Re-run with --apply (and optionally --make-public).")
    else:
        print("The storage figure on huggingface.co/settings/billing can take a while to update.")


if __name__ == "__main__":
    sys.exit(main())
