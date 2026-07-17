"""Tabulate benchmark results across one or more results JSONL files.

For each file reports, in addition to RMSD stats:
  - % parseable: fraction of the whole test set that yields a structurally
    valid geometry (for zmat: converts to Cartesian with valid element symbols
    and numeric coordinates; for xyz: all lines are valid `element x y z`).
  - % correct atoms | parseable: of the parseable geometries, the fraction
    whose atom multiset matches the ground truth.
  - valid RMSD: parseable + correct atom count + bond perception/alignment
    succeeds, giving a numeric RMSD.

Usage:
    python report.py results_a.jsonl results_b.jsonl ...
"""
import json
import sys
import warnings
from collections import Counter

from rdkit import Chem

from bench_tools import zmat_string_to_cartesian_string, evaluate_xyz
from prompts import resolve_eval_format

_PT = Chem.GetPeriodicTable()
VALID_ELEMENTS = {_PT.GetElementSymbol(i) for i in range(1, 119)}


def to_coords(parsed, fmt):
    """Return list[(el, x, y, z)] if structurally valid, else None."""
    if not parsed:
        return None
    if resolve_eval_format(fmt) in ("zmat", "fh"):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                s = zmat_string_to_cartesian_string(parsed).strip()
        except Exception:
            return None
    else:
        s = parsed
    coords = []
    for line in s.split("\n"):
        parts = line.split()
        if len(parts) != 4:
            return None
        el, x, y, z = parts
        if el not in VALID_ELEMENTS:
            return None
        try:
            coords.append((el, float(x), float(y), float(z)))
        except ValueError:
            return None
    return coords or None


def analyze_row(r):
    fmt = r.get("format", "xyz")
    gt = r["ground_truth_coordinates"]
    coords = to_coords(r.get("parsed"), fmt)
    parseable = coords is not None
    correct = False
    rmsd = None
    if parseable:
        correct = Counter(c[0] for c in coords) == Counter(c[0] for c in gt)
        if correct:
            val = evaluate_xyz(gt, r["parsed"], fmt=fmt)
            if isinstance(val, float):
                rmsd = val
    return parseable, correct, rmsd


def analyze_file(path):
    rows = [json.loads(l) for l in open(path)]
    total = len(rows)
    parseable = correct = 0
    rmsds = []
    fmts = set()
    for r in rows:
        fmts.add(r.get("format", "xyz"))
        p, c, rm = analyze_row(r)
        parseable += p
        correct += c
        if rm is not None:
            rmsds.append(rm)
    rmsds.sort()
    return {
        "path": path,
        "fmt": "/".join(sorted(fmts)),
        "total": total,
        "parseable": parseable,
        "correct": correct,
        "valid": len(rmsds),
        "mean": sum(rmsds) / len(rmsds) if rmsds else None,
        "median": rmsds[len(rmsds) // 2] if rmsds else None,
    }


def main():
    rows = [analyze_file(p) for p in sys.argv[1:]]
    hdr = (f"{'file':<34} {'fmt':<5} {'N':>5} {'parseable':>13} "
           f"{'correct-atoms':>16} {'validRMSD':>10} {'mean':>7} {'median':>7}")
    print(hdr)
    print("-" * len(hdr))
    for s in rows:
        pp = f"{s['parseable']:>4} ({100*s['parseable']/s['total']:>4.1f}%)"
        if s["parseable"]:
            ca = f"{s['correct']:>4} ({100*s['correct']/s['parseable']:>4.1f}%)"
        else:
            ca = f"{s['correct']:>4} ( n/a )"
        mean = f"{s['mean']:.3f}" if s["mean"] is not None else "  -  "
        median = f"{s['median']:.3f}" if s["median"] is not None else "  -  "
        name = s["path"].rsplit("/", 1)[-1]
        print(f"{name:<34} {s['fmt']:<5} {s['total']:>5} {pp:>13} "
              f"{ca:>16} {s['valid']:>10} {mean:>7} {median:>7}")
    print("\nparseable = structurally valid geometry / whole test set")
    print("correct-atoms = correct atom multiset / parseable geometries")
    print("validRMSD = parseable + correct atoms + bond-align OK (numeric RMSD)")


if __name__ == "__main__":
    main()
