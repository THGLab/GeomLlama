#!/usr/bin/env python3
"""Build the figshare archives from a populated data/ and results/ tree.

Maintainer tool -- the inverse of download_data.py. Run it after the data
pipeline has produced everything, then upload the resulting zips to figshare.

    python make_archives.py --dry-run     # list what would be packed
    python make_archives.py               # write dist/*.zip

Why this exists instead of `zip -r data/qm9 ...`: the archives split along
pipeline stages -- prepare_*.py output, then create_sft_data.py output -- but on
disk the files are organized by dataset. So data/qm9/ holds test_set.json
(datasets archive) alongside ori_fh.jsonl (sft archive). You cannot zip a
folder; you have to select:

    .json / .pkl        -> datasets archive (everything prepare_*.py emits)
    JSONL in SFT_KEEP   -> sft archive      (create_sft_data.py output)
    results/**          -> results archive  (cached outputs)

SFT_KEEP is an explicit allowlist, NOT "*.jsonl". A working data/ tree can also
hold JSONL for formats that were never released (feedback_fh, formula_fh,
connectivity_*, fh_e2e) or that are cheaply regenerable (the template_fh
ablations). Globbing every .jsonl would ship ~3x the data, much of it in formats
this package's data_formats.py cannot parse. The template_fh family is
deliberately excluded: create_template_fh_from_json.py and
create_ablation_data.py regenerate it from the processed JSON.

IMPORTANT: every path inside a zip is relative to the REPO ROOT -- entries look
like "data/qm9/test_set.json", not "qm9/test_set.json". download_data.py calls
extractall(".") from the repo root, so the layout is reproduced verbatim. Two
reasons it works this way:

  * synth expects a TOP-LEVEL cache/ (config.CACHE_DIR = {REPO}/cache/<dataset>)
    alongside results/. A results-rooted zip physically cannot place it.
  * `unzip geomllama_datasets.zip` at the repo root then does the right thing.
    Under a data-rooted scheme it would scatter qm9/ into the repo root instead.

data/alpaca/ is excluded on purpose: Alpaca is CC BY-NC 4.0 and is fetched from
upstream by `download_data.py --alpaca` rather than redistributed here. See
"Data Provenance and Licensing" in README.md.
"""

import argparse
import os
import sys
import zipfile
from pathlib import Path

# The only JSONL that ship: the formats the paper actually trains on. Anything
# else in a working tree is either an unreleased format or regenerable.
SFT_KEEP = {
    "ori_fh.jsonl",        # Tables 1, 2, 3 (Z-matrix / Fenske-Hall)
    "ori_xyz.jsonl",       # Tables 2, 3 (Cartesian)
    "roundto3_xyz.jsonl",  # Table 1 (rounded Cartesian)
}

# Figure 3 inputs.
#   --source published (the default): cache/<dataset>/xtb_<name>.pkl, the
#       artifact of record -- this is what reproduces the paper's figure.
#   --source notebook (cross-check):  full_results_<name>.pkl, from the original
#       notebook. Only the plotted methods exist here; plots.py reads
#       full_results_<name>.pkl for each entry in config.METHODS, plus rdkit.
# Left out: full_results_templated_z-matrix.pkl (template_fh is commented out of
# METHODS, so nothing reads it), the specialized_heavy run, stale PNGs, metrics
# CSVs, and correction/.
SMILEY_KEEP = {
    "ro5_smiles.json",
    "xtb_rdkit.pkl", "xtb_z-matrix.pkl",
    "full_results_rdkit.pkl", "full_results_z-matrix.pkl",
} | {
    f"ori_fh_8b_4e_{tag}_free_T1{suf}.json"
    for tag in ("ro4", "ro5") for suf in ("", "_raw")
}


def _results_pred(rel):
    """rel is relative to the repo root: results/... or cache/..."""
    if rel.parts[0] == "cache":
        return rel.name in SMILEY_KEEP
    if len(rel.parts) > 1 and rel.parts[1] == "pure_qm9":
        return rel.suffix == ".jsonl"
    return rel.name in SMILEY_KEEP


# name -> (source roots, predicate over the path relative to the REPO ROOT)
ARCHIVES = {
    "datasets": (["data"], lambda p: p.suffix in {".json", ".pkl"}),
    "sft": (["data"], lambda p: p.name in SFT_KEEP),
    "results": (["results", "cache"], _results_pred),
}

# Never ship: Alpaca is CC BY-NC and fetched from upstream instead.
EXCLUDE_PATHS = {("data", "alpaca")}


def _human(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f}{unit}" if unit == "B" else f"{n:.1f}{unit}"
        n /= 1024


def collect(name):
    """Return ([(absolute path, name inside the zip)], skipped_jsonl_names).

    Archive names are repo-root-relative ("data/qm9/test_set.json"), so the zip
    reproduces the layout verbatim when extracted at the repo root.
    """
    roots, pred = ARCHIVES[name]
    out, skipped = [], set()

    for root_name in roots:
        root = Path(root_name)
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            rel = Path(root_name) / path.relative_to(root)  # from repo root
            if any(rel.parts[:len(ex)] == ex for ex in EXCLUDE_PATHS):
                continue
            if not pred(rel):
                # Surface JSONL we deliberately drop, so a tree with extra
                # formats never silently ships less than the operator expects.
                if name == "sft" and rel.suffix == ".jsonl":
                    skipped.add(rel.name)
                continue
            out.append((path, rel))
    return out, skipped


def main():
    ap = argparse.ArgumentParser(description="Build figshare archives")
    ap.add_argument("--out-dir", default="dist", help="where to write the zips")
    ap.add_argument("--dry-run", action="store_true",
                    help="list contents without writing anything")
    ap.add_argument("--only", choices=sorted(ARCHIVES),
                    help="build a single archive")
    args = ap.parse_args()

    names = [args.only] if args.only else list(ARCHIVES)
    problems = []

    for name in names:
        items, skipped = collect(name)
        total = sum(p.stat().st_size for p, _ in items)
        print(f"\n=== {name}: {len(items)} files, {_human(total)} ===")
        if not items:
            roots = "/, ".join(ARCHIVES[name][0])
            problems.append(f"{name}: no files found under {roots}/")
            print(f"  (nothing found under {roots}/)")
            continue

        for _, rel in items[:12]:
            print(f"  {rel}")
        if len(items) > 12:
            print(f"  ... and {len(items) - 12} more")
        if skipped:
            print(f"  skipped {len(skipped)} JSONL format(s) not in SFT_KEEP: "
                  f"{', '.join(sorted(skipped))}")

        if args.dry_run:
            continue

        os.makedirs(args.out_dir, exist_ok=True)
        zip_path = os.path.join(args.out_dir, f"geomllama_{name}.zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for path, rel in items:
                zf.write(path, arcname=str(rel))
        print(f"  -> {zip_path} ({_human(os.path.getsize(zip_path))})")

    if problems:
        print("\nWARNING:")
        for p in problems:
            print(f"  {p}")
        print("Populate data/ and results/ first (see README).")
        return 1

    if not args.dry_run:
        print("\nUpload each zip to figshare, then put its URL in download_data.py.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
