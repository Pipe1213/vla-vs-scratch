#!/usr/bin/env python3
"""Free laptop disk space after publishing: delete local checkpoints that are verified to be on the Hub.

    python scripts/cleanup_local.py                     # dry run: sizes + Hub check, deletes nothing
    python scripts/cleanup_local.py --apply             # delete the grid checkpoints that passed the check
    python scripts/cleanup_local.py --apply --wandb     # ... and the local wandb/ run folders (already synced)
    python scripts/cleanup_local.py --apply --unpublished   # ... and runs NOT on the Hub (pilots, 30k diagnostic)

A grid run (outputs/train/<run>, names as in results.csv) is deleted only if every local file under
checkpoints/<step>/pretrained_model/ exists in the public repo Pipe1213/<run> at the same path with the same
size. Anything else under outputs/train is "unpublished" and is kept unless --unpublished is given.
outputs/eval (rollout videos), results/, notes/ and the HF cache are never touched (sizes are reported).
Needs internet (Hub API). Deleting files inside WSL frees space for WSL; to give it back to Windows, compact
the WSL disk afterwards (Windows side: wsl --shutdown, then diskpart "compact vdisk" on the distro's ext4.vhdx).
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "outputs" / "train"
GRID_RE = re.compile(r"^(act|smolvla)_obj_t[123]_n(05|10|20|all)(_s[23])?$")
USER = "Pipe1213"


def size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file() and not f.is_symlink())


def gb(n: int) -> str:
    return f"{n / 1e9:.2f} GB"


def hub_files(api, repo: str) -> dict[str, int] | None:
    try:
        return {f.path: f.size for f in api.list_repo_tree(repo, recursive=True) if hasattr(f, "size")}
    except Exception as e:  # missing repo, no network, ...
        print(f"  [hub] {repo}: {type(e).__name__}: {str(e).splitlines()[0][:100]}")
        return None


def verify(api, run_dir: Path) -> tuple[bool, str]:
    local = {p.relative_to(run_dir).as_posix(): p.stat().st_size
             for p in run_dir.glob("checkpoints/*/pretrained_model/**/*") if p.is_file()}
    if not local:
        return False, "no local pretrained_model files"
    remote = hub_files(api, f"{USER}/{run_dir.name}")
    if remote is None:
        return False, "Hub repo not reachable"
    missing = [k for k in local if k not in remote]
    differ = [k for k in local if k in remote and remote[k] != local[k]]
    if missing or differ:
        return False, f"{len(missing)} files missing on the Hub, {len(differ)} with a different size (e.g. {(missing + differ)[0]})"
    return True, f"{len(local)} files match"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true", help="actually delete (default: dry run)")
    ap.add_argument("--wandb", action="store_true", help="also delete ./wandb (local copies of synced runs)")
    ap.add_argument("--unpublished", action="store_true", help="also delete outputs/train runs that are not on the Hub")
    a = ap.parse_args()
    from huggingface_hub import HfApi
    api = HfApi()

    runs = sorted(p for p in TRAIN.iterdir() if p.is_dir()) if TRAIN.exists() else []
    grid = [p for p in runs if GRID_RE.match(p.name)]
    other = [p for p in runs if not GRID_RE.match(p.name)]
    delete, keep = [], []
    print(f"Checking {len(grid)} grid runs against the Hub ({USER}/<run>)...")
    for p in grid:
        ok, why = verify(api, p)
        s = size(p)
        print(f"  {'OK  ' if ok else 'KEEP'} {p.name:24s} {gb(s):>9s}  {why}")
        (delete if ok else keep).append((p, s))
    print("\nNot part of the published grid (pilots, diagnostics, failed or smoke runs):")
    for p in other:
        s = size(p)
        print(f"  {'DEL ' if a.unpublished else 'KEEP'} {p.name:40s} {gb(s):>9s}")
        (delete if a.unpublished else keep).append((p, s))
    wb = ROOT / "wandb"
    if wb.exists():
        s = size(wb)
        print(f"\n  {'DEL ' if a.wandb else 'KEEP'} wandb/ (local copies of synced runs) {gb(s)}")
        (delete if a.wandb else keep).append((wb, s))
    print("\nNever touched (for information):")
    for p in [ROOT / "outputs" / "eval", Path.home() / ".cache" / "huggingface", Path.home() / "lerobot"]:
        if p.exists():
            print(f"  {str(p).replace(str(Path.home()), '~'):40s} {gb(size(p)):>9s}")

    total = sum(s for _, s in delete)
    print(f"\n{'Deleting' if a.apply else 'Would delete'} {len(delete)} folders, {gb(total)}; keeping {len(keep)} "
          f"({gb(sum(s for _, s in keep))}).")
    if not a.apply:
        print("Dry run only. Re-run with --apply to delete.")
        return
    if not delete:
        return
    for p, _ in delete:
        shutil.rmtree(p)
    print("Done. Now compact the WSL disk to give the space back to Windows.")


if __name__ == "__main__":
    sys.exit(main())
