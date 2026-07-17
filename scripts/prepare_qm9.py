#!/usr/bin/env python3
"""
Prepare the QM9 dataset for fine-tuning.

Pipeline:
  1. Extract xyz files from the QM9 tarball to a temp directory
  2. Parse all xyz files to extract SMILES and coordinates
  3. Split into 99% train / 1% test (random_state=42)
  4. Optionally convert xyz -> Fenske-Hall Z-matrix via openbabel
  5. Save train_set.json and test_set.json

Usage:
  python prepare_qm9.py --tarball path/to/dsgdb9nsd.xyz.tar.bz2 --output-dir data/qm9/
  python prepare_qm9.py --xyz-dir path/to/extracted/xyz/ --output-dir data/qm9/
"""

import argparse
import json
import os
import subprocess
import tarfile
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List

from sklearn.model_selection import train_test_split
from tqdm import tqdm


def parse_qm9_xyz(file_path):
    """Parse a single QM9 .xyz file.

    Returns:
        Dict with 'filename', 'smiles', 'coordinates' keys.
    """
    with open(file_path) as f:
        lines = f.readlines()

    num_atoms = int(lines[0].strip())
    coords = []
    for k in range(2, num_atoms + 2):
        # QM9 xyz files use tabs; last field is partial charge (drop it)
        coords.append(lines[k].strip().split('\t')[:-1])

    # SMILES is on line num_atoms + 3 (0-indexed: after atoms + 2 header lines)
    smiles = lines[num_atoms + 3].strip().split('\t')[0]

    return {
        "filename": str(file_path),
        "smiles": smiles,
        "coordinates": coords,
    }


def scrape_qm9_dataset(directory: Path) -> List[Dict]:
    """Parse all QM9 xyz files in a directory."""
    results = []
    for f in tqdm(sorted(directory.glob("dsgdb9nsd_*.xyz")),
                  desc="Parsing QM9 xyz files"):
        try:
            results.append(parse_qm9_xyz(f))
        except Exception as e:
            print(f"Warning: Failed to parse {f.name}: {e}")
    return results


def convert_xyz_to_fh(xyz_path, fh_path):
    """Convert a single xyz file to Fenske-Hall format using openbabel."""
    subprocess.run(
        ["obabel", "-ixyz", str(xyz_path), "-ofh", "-O", str(fh_path)],
        capture_output=True, check=True,
    )


def read_fh_coordinates(fh_path):
    """Read Fenske-Hall coordinates from an .fh file."""
    with open(fh_path) as f:
        lines = f.readlines()
    num_atoms = int(lines[1].strip())
    coords = []
    for k in range(2, num_atoms + 2):
        coords.append(lines[k].strip().split())
    return coords


def _convert_one(args):
    """Convert a single molecule to FH (worker function for parallel execution)."""
    idx, basename, xyz_path, fh_path = args
    try:
        subprocess.run(
            ["obabel", "-ixyz", str(xyz_path), "-ofh", "-O", str(fh_path)],
            capture_output=True, check=True,
        )
        coords = read_fh_coordinates(fh_path)
        return idx, coords, None
    except Exception as e:
        return idx, None, str(e)


def add_fh_coordinates(results, xyz_dir, tmp_dir, n_jobs=None):
    """Add Fenske-Hall coordinates to each result dict.

    Writes temp xyz files, converts with openbabel, reads back, cleans up.
    Uses parallel workers for speed.

    Args:
        n_jobs: Number of parallel workers (default: number of CPUs).
    """
    fh_dir = Path(tmp_dir) / "fh"
    fh_dir.mkdir(exist_ok=True)

    if n_jobs is None:
        n_jobs = os.cpu_count() or 4

    # Build work items
    work = []
    for i, r in enumerate(results):
        basename = Path(r["filename"]).stem
        xyz_path = Path(xyz_dir) / f"{basename}.xyz"
        fh_path = fh_dir / f"{basename}.fh"
        work.append((i, basename, str(xyz_path), str(fh_path)))

    # Initialize all to None
    for r in results:
        r["fh_coordinates"] = None

    # Run in parallel
    print(f"Converting to Fenske-Hall with {n_jobs} workers...")
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures = {executor.submit(_convert_one, w): w for w in work}
        for future in tqdm(as_completed(futures), total=len(work),
                           desc="Converting to Fenske-Hall"):
            idx, coords, err = future.result()
            if err:
                basename = work[idx][1]
                print(f"Warning: FH conversion failed for {basename}: {err}")
            else:
                results[idx]["fh_coordinates"] = coords

    return results


def main():
    parser = argparse.ArgumentParser(description="Prepare QM9 dataset")
    parser.add_argument("--tarball", type=str, default=None,
                        help="Path to dsgdb9nsd.xyz.tar.bz2")
    parser.add_argument("--xyz-dir", type=str, default=None,
                        help="Path to directory with extracted xyz files")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for train/test JSON files")
    parser.add_argument("--convert-fh", action="store_true",
                        help="Also convert to Fenske-Hall Z-matrix format")
    parser.add_argument("--test-size", type=float, default=0.01,
                        help="Test set fraction (default: 0.01)")
    parser.add_argument("--random-state", type=int, default=42,
                        help="Random seed for split (default: 42)")
    parser.add_argument("--n-jobs", type=int, default=None,
                        help="Parallel workers for FH conversion (default: all CPUs)")
    args = parser.parse_args()

    if args.tarball is None and args.xyz_dir is None:
        parser.error("Provide either --tarball or --xyz-dir")

    os.makedirs(args.output_dir, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="qm9_") as tmp_dir:
        # Extract tarball if needed
        if args.xyz_dir is not None:
            xyz_dir = Path(args.xyz_dir)
        else:
            print(f"Extracting {args.tarball} to {tmp_dir}...")
            xyz_dir = Path(tmp_dir) / "xyz"
            xyz_dir.mkdir()
            with tarfile.open(args.tarball, "r:bz2") as tar:
                tar.extractall(path=xyz_dir)
            print(f"Extracted to {xyz_dir}")

        # Parse all xyz files
        results = scrape_qm9_dataset(xyz_dir)
        print(f"Parsed {len(results)} molecules")

        # Optionally convert to FH
        if args.convert_fh:
            results = add_fh_coordinates(results, xyz_dir, tmp_dir,
                                            n_jobs=args.n_jobs)

        # Split
        train, test = train_test_split(
            results,
            test_size=args.test_size,
            random_state=args.random_state,
        )
        print(f"Train: {len(train)}, Test: {len(test)}")

        # Save
        train_path = os.path.join(args.output_dir, "train_set.json")
        test_path = os.path.join(args.output_dir, "test_set.json")

        print(f"Writing {train_path}...")
        with open(train_path, 'w') as f:
            json.dump(train, f)

        print(f"Writing {test_path}...")
        with open(test_path, 'w') as f:
            json.dump(test, f)

        print("Done!")


if __name__ == "__main__":
    main()
