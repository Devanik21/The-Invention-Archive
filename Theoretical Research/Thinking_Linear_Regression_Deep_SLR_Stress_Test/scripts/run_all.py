"""Reproduce every number, table and figure in the paper.
Usage: python scripts/run_all.py [--seeds 30]
"""
import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from deep_slr.experiments import run_all  # noqa: E402
from deep_slr.plots import make_all  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=30)
    args = ap.parse_args()
    main, params = run_all(n_seeds=args.seeds, out_dir=str(ROOT / "results"))
    make_all(root=ROOT)
    print("done:", len(main), "rows")
