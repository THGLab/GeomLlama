"""Re-score an existing results JSONL through the current evaluator, in place.

Re-parses each stored raw_response and re-runs evaluate_xyz (which now has
DetermineBonds ON), updating the 'rmsd' field. No inference needed. Prints
before/after aggregate stats.

Usage:
    python rescore.py results.jsonl [results2.jsonl ...]
"""
import json
import sys

from parse import parse_result
from bench_tools import evaluate_xyz


def summarize(rows):
    stats = {"success": 0, "wrong_syntax": 0, "wrong_atoms": 0}
    rmsds = []
    for r in rows:
        v = r.get("rmsd")
        if isinstance(v, float):
            stats["success"] += 1
            rmsds.append(v)
        elif v == "Wrong syntax":
            stats["wrong_syntax"] += 1
        elif v == "Wrong number of atoms":
            stats["wrong_atoms"] += 1
    mean = sum(rmsds) / len(rmsds) if rmsds else None
    rmsds.sort()
    median = rmsds[len(rmsds) // 2] if rmsds else None
    return stats, mean, median


def rescore_file(path):
    rows = [json.loads(line) for line in open(path)]
    before = summarize(rows)

    for idx, r in enumerate(rows):
        r = parse_result(r)  # re-derive 'parsed' from raw_response (returns a new dict)
        fmt = r.get("format", "xyz")
        if r["parsed"]:
            r["rmsd"] = evaluate_xyz(r["ground_truth_coordinates"], r["parsed"], fmt=fmt)
        else:
            r["rmsd"] = "Wrong syntax"
        rows[idx] = r  # write the reparsed/rescored dict back into the list

    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    after = summarize(rows)

    def fmt_stats(label, s):
        stats, mean, median = s
        m = f"mean={mean:.4f} median={median:.4f}" if mean is not None else "no valid RMSD"
        print(f"  {label}: valid={stats['success']} "
              f"wrong_syntax={stats['wrong_syntax']} "
              f"wrong_atoms={stats['wrong_atoms']} | {m}")

    print(f"\n{path}")
    fmt_stats("before", before)
    fmt_stats("after ", after)


if __name__ == "__main__":
    for path in sys.argv[1:]:
        rescore_file(path)
