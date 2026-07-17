"""Rescore a results JSONL with correspondence-free RMSD, in parallel.

For every valid-atom-count structure, computes BOTH:
  rmsd_graph       - RDKit GetBestRMS with bond perception (chemically strict;
                     None when the geometry can't be bond-perceived), and
  rmsd_assignment  - min RMSD over element-preserving atom permutations + rigid
                     alignment (a lower bound; always available).

This rescues structures RDKit can't bond-perceive, which the graph-only scorer
discarded as "Wrong syntax". Sets:
  rmsd            = rmsd_graph if available else rmsd_assignment
  rmsd_method     = "graph" | "assignment" | None
  rmsd_graph, rmsd_assignment

Writes in place (keep a backup). CPU-bound; runs across a process pool.

Usage:
    python rescore_bruteforce.py results.jsonl [more.jsonl ...] [--jobs N] [-n N]
"""
import argparse
import os

# Pin BLAS to a single thread per worker BEFORE numpy is imported. With a process
# pool this both (a) avoids core oversubscription (8 procs x N BLAS threads) and
# (b) makes results reproducible: multithreaded Accelerate/OpenBLAS reduce in a
# nondeterministic order, and the discrete Hungarian refinement amplifies those
# ~1e-12 differences into different local minima on high-symmetry molecules.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import warnings
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

from tqdm import tqdm

from parse import parse_result
from bench_tools import classify_geometry, parse_model_coordinates
from rmsd_align import best_rmsd

# Quiet the noisy geometry libs inside every worker (module re-imported on spawn).
warnings.filterwarnings("ignore")
from rdkit import RDLogger  # noqa: E402
RDLogger.DisableLog("rdApp.*")


def score_row(row):
    """Return the row with rmsd/rmsd_graph/rmsd_assignment/rmsd_method set."""
    r = parse_result(row)
    fmt = r.get("format", "xyz")
    row["rmsd_graph"] = None
    row["rmsd_assignment"] = None
    row["rmsd_method"] = None

    if not r["parsed"]:
        row["rmsd"] = "Wrong syntax"
        return row

    cls = classify_geometry(r["ground_truth_coordinates"], r["parsed"], fmt=fmt)
    if cls == "unparsable":
        row["rmsd"] = "Wrong syntax"
    elif cls == "wrong_atoms":
        row["rmsd"] = "Wrong number of atoms"
    else:
        coords = parse_model_coordinates(r["parsed"], fmt=fmt)
        res = best_rmsd(r["ground_truth_coordinates"], coords)
        row["rmsd"] = res["rmsd"]
        row["rmsd_graph"] = res["graph"]
        row["rmsd_assignment"] = res["assignment"]
        row["rmsd_method"] = res["method"]
    return row


def rescore_file(path, jobs, limit=None):
    rows = [json.loads(line) for line in open(path) if line.strip()]
    if limit:
        rows = rows[:limit]

    chunk = max(1, len(rows) // (jobs * 4))
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        scored = list(tqdm(ex.map(score_row, rows, chunksize=chunk),
                           total=len(rows), desc=os.path.basename(path), unit="mol"))

    with open(path, "w") as f:
        for r in scored:
            f.write(json.dumps(r) + "\n")

    _summarize(path, scored)


def _summarize(path, rows):
    method = Counter(r.get("rmsd_method") for r in rows)
    floats = [r["rmsd"] for r in rows if isinstance(r.get("rmsd"), float)]
    graph = [r["rmsd_graph"] for r in rows if isinstance(r.get("rmsd_graph"), float)]
    assign = [r["rmsd_assignment"] for r in rows if isinstance(r.get("rmsd_assignment"), float)]
    rescued = method.get("assignment", 0)

    def stat(xs):
        if not xs:
            return "n/a"
        xs = sorted(xs)
        return f"mean={sum(xs)/len(xs):.4f} median={xs[len(xs)//2]:.4f}"

    print(f"\n{path}")
    print(f"  scored (valid atoms):     {len(floats)}/{len(rows)}")
    print(f"    via graph:              {method.get('graph', 0)}  ({stat(graph)})")
    print(f"    via assignment (rescue):{rescued}  ({stat(assign)})")
    print(f"  wrong atoms:              {sum(1 for r in rows if r.get('rmsd') == 'Wrong number of atoms')}")
    print(f"  unparsable:               {sum(1 for r in rows if r.get('rmsd') == 'Wrong syntax')}")
    print(f"  RMSD over all scored:     {stat(floats)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="results JSONL files (rewritten in place)")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2),
                    help="worker processes (default: cores - 2)")
    ap.add_argument("-n", type=int, default=None, help="only first N rows (testing)")
    args = ap.parse_args()
    for path in args.paths:
        rescore_file(path, jobs=args.jobs, limit=args.n)


if __name__ == "__main__":
    main()
