"""Stage breakdown for a results JSONL: parse rate and atom-count validity.

Reports, per file, two metrics the collapsed 'rmsd' field can't cleanly show
(it labels both early parse failures and late bond-perception failures as
"Wrong syntax"):

  1. % of outputs that are PARSABLE   (valid coordinates; z-matrices converted)
  2. of those parsable, % with VALID ATOM COUNTS (composition matches ground truth)

Re-derives 'parsed' from raw_response, so it reflects the current parser/evaluator.
No inference needed.

Usage:
    python eval_stats.py results.jsonl [results2.jsonl ...]
"""
import json
import sys

from parse import parse_result
from bench_tools import classify_geometry


def stage_stats(path):
    counts = {"unparsable": 0, "wrong_atoms": 0, "valid_atoms": 0}
    total = 0
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        total += 1
        r = parse_result(json.loads(line))  # re-derive 'parsed' from raw_response
        fmt = r.get("format", "xyz")
        if not r["parsed"]:
            counts["unparsable"] += 1
        else:
            counts[classify_geometry(r["ground_truth_coordinates"], r["parsed"], fmt=fmt)] += 1
    return total, counts


def print_stats(path):
    total, counts = stage_stats(path)
    parsable = counts["wrong_atoms"] + counts["valid_atoms"]
    valid = counts["valid_atoms"]

    def pct(num, denom):
        return f"{100 * num / denom:.1f}%" if denom else "n/a"

    print(f"\n{path}")
    print(f"  Total outputs:              {total}")
    print(f"  (1) Parsable:               {parsable}/{total} ({pct(parsable, total)})")
    print(f"  (2) Valid atoms | parsable: {valid}/{parsable} ({pct(valid, parsable)})")
    print(f"      Valid atoms | total:    {valid}/{total} ({pct(valid, total)})")
    print(f"      breakdown: unparsable={counts['unparsable']} "
          f"wrong_atoms={counts['wrong_atoms']} valid_atoms={counts['valid_atoms']}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    for path in sys.argv[1:]:
        print_stats(path)
