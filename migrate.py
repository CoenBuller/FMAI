import argparse
import os
import shutil
from pathlib import Path


def migrate(base_dir: str, dry_run: bool = True):
    base = Path(base_dir)
    src_root = base / "flocking_wind"
    dst_root = base / "flocking"

    if not src_root.exists():
        raise FileNotFoundError(f"Source directory not found: {src_root}")

    moved_count = 0

    # Iterate through <native|robustness> / <category e.g. traces, models, eval> / <wind|no_wind>
    for mode_dir in sorted(src_root.iterdir()):
        if not mode_dir.is_dir():
            continue

        for sub_dir in sorted(mode_dir.iterdir()):
            if not sub_dir.is_dir():
                continue

            for run_id_dir in sorted(sub_dir.iterdir()):
                if not run_id_dir.is_dir():
                    continue

                # Force target run_id folder to be 'wind' so it separates cleanly from 'no_wind'
                target_dir = dst_root / mode_dir.name / sub_dir.name / "wind"

                for item in sorted(run_id_dir.iterdir()):
                    dst_item = target_dir / item.name
                    if dst_item.exists():
                        print(f"[SKIP - EXISTS] {item} -> {dst_item}")
                        continue

                    print(f"[{'DRY-RUN' if dry_run else 'MOVE'}] {item} -> {dst_item}")
                    if not dry_run:
                        target_dir.mkdir(parents=True, exist_ok=True)
                        shutil.move(str(item), str(dst_item))
                    moved_count += 1

    print(f"\nDone. {'Would move' if dry_run else 'Moved'} {moved_count} items.")
    if dry_run:
        print("Re-run with --execute to actually move the files.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Move flocking_wind runs into flocking/.../wind/")
    p.add_argument("--base", default="results", help="Parent results directory containing flocking and flocking_wind")
    p.add_argument("--execute", action="store_true", help="Actually move files (default is dry-run)")
    args = p.parse_args()
    migrate(args.base, dry_run=not args.execute)