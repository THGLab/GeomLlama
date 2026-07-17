#!/usr/bin/env python3
"""Prepare GEOM-Drugs-large dataset from pre-split per-molecule pickles.

Reads from a directory containing train/, valid/, test/ subdirectories of
per-molecule GEOM pickles (as produced by build_split_dirs.py), and creates
the intermediate JSON + test_smiles_dict.pkl needed by the JSONL pipeline.

Usage:
    python scripts/prepare_drugs_large.py \
        --split-dir /path/to/dmcg-split-replication \
        --output-dir /tmp/geom_drugs_large \
        --convert-fh --save-test-dict --n-jobs 56
"""

import argparse
import json
import os
import pickle
import sys
import tempfile
from glob import glob

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from prepare_geom import build_smiles_dict, process_split


def list_pickles(directory):
    """List .pickle files in a directory, sorted for determinism."""
    paths = sorted(glob(os.path.join(directory, "*.pickle")))
    if not paths:
        raise FileNotFoundError(f"No .pickle files in {directory}")
    return paths


def main():
    parser = argparse.ArgumentParser(description="Prepare GEOM-Drugs-large dataset")
    parser.add_argument("--split-dir", required=True,
                        help="Directory with train/, valid/, test/ pickle subdirs")
    parser.add_argument("--output-dir", required=True,
                        help="Output directory for JSON + test dict")
    parser.add_argument("--convert-fh", action="store_true",
                        help="Convert to Fenske-Hall Z-matrix format")
    parser.add_argument("--save-test-dict", action="store_true",
                        help="Save test SMILES dict for benchmarking")
    parser.add_argument("--n-jobs", type=int, default=None,
                        help="Parallel workers (default: all CPUs)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for split_name in ("train", "val", "test"):
        sub = "valid" if split_name == "val" else split_name
        split_dir = os.path.join(args.split_dir, sub)
        paths = list_pickles(split_dir)
        print(f"\n{split_name}: {len(paths)} pickle files from {split_dir}")

        if split_name == "test" and not args.convert_fh:
            print(f"  Skipping test JSON (no --convert-fh and not train/val)")
            continue

        with tempfile.TemporaryDirectory(prefix=f"drugs_large_{split_name}_") as tmp_dir:
            data = process_split(
                paths,
                base_path="",
                remove_hydrogens=False,
                convert_fh=args.convert_fh,
                tmp_dir=tmp_dir if args.convert_fh else None,
                split_name=split_name,
                n_jobs=args.n_jobs,
            )

        out_path = os.path.join(args.output_dir, f"{split_name}_set.json")
        print(f"  Writing {out_path} ({len(data)} conformers)...")
        with open(out_path, 'w') as f:
            json.dump(data, f)

    if args.save_test_dict:
        test_dir = os.path.join(args.split_dir, "test")
        test_paths = list_pickles(test_dir)
        print(f"\nBuilding test SMILES dict from {len(test_paths)} pickles...")
        smiles_dict = build_smiles_dict(test_paths, base_path="")
        dict_path = os.path.join(args.output_dir, "test_smiles_dict.pkl")
        print(f"  Writing {dict_path} ({len(smiles_dict)} unique SMILES, "
              f"{sum(len(v) for v in smiles_dict.values())} conformers)...")
        with open(dict_path, 'wb') as f:
            pickle.dump(smiles_dict, f)

    print("\nDone!")


if __name__ == "__main__":
    main()
