"""Convert cluttered subset episodes to HDF5 in the same layout as `clean/`.

Wraps `data/convert_to_hdf5.episode_to_hdf5` but writes the output to
`<ep_dir>/episode<n>.hdf5` (matching the clean layout) instead of the
default `<task_dir>/<n>.hdf5`. Idempotent: skips episodes whose hdf5 is
already present.

Usage:
    .venv/bin/python convert_ep_hdf5.py \
        --task put_book_in_box \
        --subsets d1,d2,d3,d4
"""
import argparse
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "data"))
from convert_to_hdf5 import episode_to_hdf5, validate_hdf5  # noqa: E402


DATA_ROOT = "/home/joshua/Desktop/ur5e_vla/data"


def convert_subset(task, subset):
    data_dir = f"{DATA_ROOT}/{task}/{subset}/data"
    if not os.path.isdir(data_dir):
        print(f"[{subset}] missing dir: {data_dir}")
        return 0, 0, 0
    eps = sorted(
        [e for e in os.listdir(data_dir)
         if e.isdigit() and os.path.isdir(f"{data_dir}/{e}")],
        key=int,
    )
    print(f"\n=== {task}/{subset}: {len(eps)} episodes ===")
    done = skipped = failed = 0
    for i, ep in enumerate(eps, 1):
        ep_dir = f"{data_dir}/{ep}"
        out = f"{ep_dir}/episode{ep}.hdf5"
        if os.path.isfile(out):
            try:
                validate_hdf5(out, ep_dir)
                size_mb = os.path.getsize(out) / 1e6
                print(f"  [{i}/{len(eps)}] ep{ep}: SKIP (valid, {size_mb:.1f} MB)")
                skipped += 1
                continue
            except Exception as error:
                print(f"  [{i}/{len(eps)}] ep{ep}: REBUILD invalid file ({error})")
        t0 = time.time()
        try:
            episode_to_hdf5(ep_dir, out, compress=True)
            dt = time.time() - t0
            sz = os.path.getsize(out) / 1e6
            print(f"  [{i}/{len(eps)}] ep{ep}: OK ({sz:.1f} MB, {dt:.1f}s)")
            done += 1
        except Exception as e:
            print(f"  [{i}/{len(eps)}] ep{ep}: FAIL: {type(e).__name__}: {e}")
            traceback.print_exc()
            failed += 1
    return done, skipped, failed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--subsets", default="d1,d2,d3,d4",
                    help="Comma-separated subset names (default: d1,d2,d3,d4)")
    args = ap.parse_args()

    subsets = [s.strip() for s in args.subsets.split(",") if s.strip()]
    total_done = total_skip = total_fail = 0
    for s in subsets:
        d, k, f = convert_subset(args.task, s)
        total_done += d
        total_skip += k
        total_fail += f
    print(f"\nTOTAL: converted={total_done} skipped={total_skip} failed={total_fail}")
    sys.exit(1 if total_fail else 0)


if __name__ == "__main__":
    main()
