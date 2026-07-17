"""Two-table evaluation report over result JSONLs, one row per model x format.

Table 1 - "most complete" RMSD: graph RMSD when RDKit can bond-perceive, else the
          correspondence-free assignment method at high quality. Every valid-atom
          generation is scored, so each generation is exactly one of:
          {has RMSD, wrong atom count, invalid syntax}.
Table 2 - strict RDKit-only view: same syntax/atom columns, plus how often RDKit
          can score (graph%), and RMSD stats over only those.

Denominators are nested:
    syntax% = parsable / total
    atom%   = valid_atoms / parsable
    graph%  = graph_scorable / valid_atoms

LaTeX rows by default (each ends with ` \\`); --plain for an aligned text table.

Usage:
    python eval_tables.py [file1.jsonl ...] [--plain] [--jobs N]
"""
import os

# Pin BLAS to one thread per worker BEFORE numpy is imported: determinism (the
# discrete Hungarian refinement amplifies nondeterministic BLAS reductions) and
# no core oversubscription under the process pool.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import glob
import json
import math
import warnings
from concurrent.futures import ProcessPoolExecutor

from tqdm import tqdm

from parse import parse_result
from bench_tools import classify_geometry, parse_model_coordinates
from rmsd_align import graph_rmsd, assignment_rmsd

warnings.filterwarnings("ignore")
from rdkit import RDLogger  # noqa: E402
RDLogger.DisableLog("rdApp.*")

# Table-1 assignment config (the high-quality "asymptote" settings).
ASSIGN_KW = dict(top_k=50, n_random=200, refine_iters=25)

# Default results directory (override with RESULTS_DIR env var or --results-dir).
RESULTS_DIR = os.environ.get("RESULTS_DIR", ".")


def _discover_jsonl(results_dir: str) -> list[str]:
    """Auto-discover all *.jsonl files in results_dir, sorted by name."""
    return sorted(glob.glob(os.path.join(results_dir, "*.jsonl")))

FORMAT_LABEL = {
    "xyz": "xyz",
    "zmat": "z-matrix",
    "fh": "z-matrix (FH)",
    "geomllama_xyz": "xyz",
    "geomllama_zmat": "z-matrix (FH)",
}


def score_row(row):
    """Per-generation stage + RMSDs. Top-level so it is picklable under spawn.

    Returns {stage, graph, assign}. `assign` is computed only when the graph
    RMSD is unavailable (Table 1 prefers graph, so assignment on graph-scorable
    rows would be wasted work).
    """
    r = parse_result(row)
    fmt = r.get("format", "xyz")
    if not r["parsed"]:
        return {"stage": "unparsable"}
    cls = classify_geometry(r["ground_truth_coordinates"], r["parsed"], fmt=fmt)
    if cls != "valid_atoms":
        return {"stage": cls}  # "unparsable" or "wrong_atoms"

    coords = parse_model_coordinates(r["parsed"], fmt=fmt)
    g = graph_rmsd(r["ground_truth_coordinates"], coords)
    a = None if g is not None else assignment_rmsd(
        r["ground_truth_coordinates"], coords, **ASSIGN_KW)
    return {"stage": "valid_atoms", "graph": g, "assign": a}


def _model_name(row):
    """Display label for a run: the explicit 'model_name' if the run recorded
    one, else the last path component of 'model'.

    Local checkpoints share a basename ('.../outputs/merged'), so they must set
    model_name at generation time or they all collide in the table. The path
    may carry a trailing slash; rstrip it so the name doesn't come out empty.
    """
    name = row.get("model_name")
    if name:
        return name
    return (row.get("model") or "?").rstrip("/").split("/")[-1] or "?"


def _stats(xs):
    """(mean, median) of a list, or (None, None) if empty."""
    if not xs:
        return None, None
    xs = sorted(xs)
    return sum(xs) / len(xs), xs[len(xs) // 2]


def eval_file(path, jobs):
    rows = [json.loads(line) for line in open(path) if line.strip()]
    chunk = max(1, len(rows) // (jobs * 4))
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        scored = list(tqdm(ex.map(score_row, rows, chunksize=chunk),
                           total=len(rows), desc=os.path.basename(path), unit="mol"))

    first = rows[0] if rows else {}
    total = len(rows)
    parsable = sum(1 for s in scored if s["stage"] != "unparsable")
    valid = [s for s in scored if s["stage"] == "valid_atoms"]

    # Table 1: graph-or-assignment for every valid-atom row (drop stray NaN).
    t1, dropped = [], 0
    for s in valid:
        v = s["graph"] if s["graph"] is not None else s["assign"]
        if v is None or math.isnan(v):
            dropped += 1
        else:
            t1.append(v)
    if dropped:
        print(f"  [warn] {path}: dropped {dropped} non-finite RMSD(s) from Table 1")

    # Table 2: graph RMSD over graph-scorable rows only.
    t2 = [s["graph"] for s in valid if s["graph"] is not None]

    t1_mean, t1_med = _stats(t1)
    t2_mean, t2_med = _stats(t2)
    return {
        "model": _model_name(first),
        "format": FORMAT_LABEL.get(first.get("format", "xyz"), first.get("format", "xyz")),
        "total": total,
        "parsable": parsable,
        "valid": len(valid),
        "graph_scorable": len(t2),
        "t1_mean": t1_mean, "t1_med": t1_med,
        "t2_mean": t2_mean, "t2_med": t2_med,
    }


def _pct(num, denom):
    return 100 * num / denom if denom else 0.0


def _fnum(x, prec):
    return f"{x:.{prec}f}" if x is not None else "--"


def _pct_tex(num, denom):
    """Percentage to two decimals, with the final (uncertain) digit in parens:
    99.85 -> "99.8(5)", 100.0 -> "100.0(0)".
    """
    s = f"{_pct(num, denom):.2f}"
    return f"{s[:-1]}({s[-1]})"


def print_latex(recs):
    print("% Table 1: Model & Format & Syntax% & Atom% & RMSD_mean & RMSD_median")
    for r in recs:
        print(f"\\textbf{{{r['model']}}} & {r['format']} & "
              f"{_pct_tex(r['parsable'], r['total'])} & "
              f"{_pct_tex(r['valid'], r['parsable'])} & "
              f"{_fnum(r['t1_mean'], 3)} & {_fnum(r['t1_med'], 3)} \\\\")
    print("\n% Table 2: Model & Format & Syntax% & Atom% & Graph% & RMSD_mean & RMSD_median")
    for r in recs:
        print(f"\\textbf{{{r['model']}}} & {r['format']} & "
              f"{_pct_tex(r['parsable'], r['total'])} & "
              f"{_pct_tex(r['valid'], r['parsable'])} & "
              f"{_pct_tex(r['graph_scorable'], r['valid'])} & "
              f"{_fnum(r['t2_mean'], 3)} & {_fnum(r['t2_med'], 3)} \\\\")


def print_plain(recs):
    h1 = f"{'Model':<32}{'Format':<10}{'Syntax%':>8}{'Atom%':>8}{'RMSD mean':>11}{'RMSD median':>12}"
    print("Table 1 - graph RMSD where possible, else assignment method\n" + h1)
    print("-" * len(h1))
    for r in recs:
        print(f"{r['model']:<32}{r['format']:<10}{_pct(r['parsable'],r['total']):>8.1f}"
              f"{_pct(r['valid'],r['parsable']):>8.1f}{_fnum(r['t1_mean'],3):>11}{_fnum(r['t1_med'],3):>12}")

    h2 = (f"{'Model':<32}{'Format':<10}{'Syntax%':>8}{'Atom%':>8}{'Graph%':>8}"
          f"{'RMSD mean':>11}{'RMSD median':>12}")
    print("\nTable 2 - RDKit graph RMSD only (Graph% = of valid-atom generations)\n" + h2)
    print("-" * len(h2))
    for r in recs:
        print(f"{r['model']:<32}{r['format']:<10}{_pct(r['parsable'],r['total']):>8.1f}"
              f"{_pct(r['valid'],r['parsable']):>8.1f}{_pct(r['graph_scorable'],r['valid']):>8.1f}"
              f"{_fnum(r['t2_mean'],3):>11}{_fnum(r['t2_med'],3):>12}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*", default=None,
                    help="result JSONL files (default: auto-discover *.jsonl in --results-dir)")
    ap.add_argument("--results-dir", type=str, default=RESULTS_DIR,
                    help="directory to scan for *.jsonl when no paths given (default: cwd or $RESULTS_DIR)")
    ap.add_argument("--plain", action="store_true", help="human-readable table instead of LaTeX")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2),
                    help="worker processes (default: cores - 2)")
    args = ap.parse_args()

    paths = args.paths if args.paths else _discover_jsonl(args.results_dir)
    if not paths:
        ap.error(f"no *.jsonl files found in {args.results_dir!r}; "
                 "pass paths explicitly or set --results-dir")

    recs = [eval_file(p, args.jobs) for p in paths]
    print()
    (print_plain if args.plain else print_latex)(recs)


if __name__ == "__main__":
    main()
