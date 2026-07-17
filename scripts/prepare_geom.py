#!/usr/bin/env python3
"""
Prepare the GEOM-QM9 dataset for fine-tuning.

Loads the GEOM rdkit_folder data, splits into train/val/test, extracts
coordinates, and optionally converts to Fenske-Hall Z-matrix format.

The split logic reproduces the exact split used in training:
  random.seed(19970327)
  random.shuffle(pickle_path_list)
  train = first 80%, val = next 10%, test = last 10%

Usage:
  python prepare_geom.py \\
      --rdkit-folder path/to/rdkit_folder/ \\
      --output-dir data/geomqm9/ \\
      --remove-hydrogens
"""

import argparse
import faulthandler
import json
import os
import pickle
import random
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

from rdkit import Chem
from tqdm import tqdm

# Print a traceback on segfault / abort
faulthandler.enable()


def load_pickle_paths(summary_path):
    """Load and filter pickle paths from the GEOM summary JSON.

    Filters out:
      - Molecules with no uniqueconfs
      - Molecules with no pickle_path
      - Multi-fragment molecules (SMILES containing '.')
    """
    with open(summary_path) as f:
        summary = json.load(f)

    paths = []
    for smiles, meta in summary.items():
        if meta.get("uniqueconfs") is None:
            continue
        if meta.get("pickle_path") is None:
            continue
        if "." in smiles:
            continue
        paths.append(meta["pickle_path"])

    return paths


def split_indices(pickle_paths):
    """Split pickle paths into train/val/test using the canonical split.

    CRITICAL: This must exactly reproduce the split used in the original
    DataLoader.ipynb to ensure consistency with existing trained models.
    """
    random.seed(19970327)
    random.shuffle(pickle_paths)

    train_size = int(len(pickle_paths) * 0.8)
    valid_size = int(len(pickle_paths) * 0.9)

    train_paths = pickle_paths[:train_size]
    valid_paths = pickle_paths[train_size:valid_size]
    test_paths = pickle_paths[valid_size:]

    return train_paths, valid_paths, test_paths


def extract_mol_coordinates(mol, remove_hydrogens=True):
    """Extract (element, x, y, z) from an RDKit Mol conformer.

    Args:
        mol: RDKit Mol with at least one conformer.
        remove_hydrogens: If True, skip hydrogen atoms.

    Returns:
        List of [element, x_str, y_str, z_str] entries.
    """
    conformer = mol.GetConformer(0)
    positions = conformer.GetPositions()
    coords = []
    for i, atom in enumerate(mol.GetAtoms()):
        symbol = atom.GetSymbol()
        if remove_hydrogens and symbol == 'H':
            continue
        x, y, z = positions[i]
        coords.append([symbol, f"{x:.6f}", f"{y:.6f}", f"{z:.6f}"])
    return coords


def get_smiles_atom_order(mol, heavy_only=True):
    """Get the atom reordering from mol order to canonical SMILES order.

    Args:
        mol: RDKit Mol (after RemoveHs, may still have explicit H).
        heavy_only: If True, filter H and remap to heavy-atom indices.

    Returns:
        order: list where order[i] is the heavy-atom coordinate index
               for SMILES position i.
    """
    _ = Chem.MolToSmiles(mol, canonical=True)
    order_str = mol.GetProp("_smilesAtomOutputOrder")
    raw_order = list(eval(order_str))

    if not heavy_only:
        return raw_order

    # Map mol atom idx → heavy-atom coordinate index (skipping H)
    mol_idx_to_heavy = {}
    heavy_idx = 0
    for i in range(mol.GetNumAtoms()):
        if mol.GetAtomWithIdx(i).GetSymbol() != 'H':
            mol_idx_to_heavy[i] = heavy_idx
            heavy_idx += 1

    return [mol_idx_to_heavy[i] for i in raw_order if i in mol_idx_to_heavy]


def get_canonical_smiles(mol, remove_hs=True):
    """Get canonical SMILES from an RDKit Mol."""
    if remove_hs:
        try:
            mol = Chem.RemoveHs(mol)
        except Exception:
            # Some molecules (e.g. hypervalent S) fail RemoveHs;
            # fall back to sanitize=False
            mol = Chem.RemoveHs(mol, sanitize=False)
    return Chem.MolToSmiles(mol, canonical=True)


def _write_xyz_from_coords(coords):
    """Generate an XYZ-format string from a coordinate list."""
    lines = [f"{len(coords)}", ""]
    for atom in coords:
        lines.append(f"{atom[0]:2s} {float(atom[1]):12.6f} {float(atom[2]):12.6f} {float(atom[3]):12.6f}")
    return '\n'.join(lines)


def write_xyz_string(mol, remove_hydrogens=True):
    """Generate an XYZ-format string for a molecule."""
    coords = extract_mol_coordinates(mol, remove_hydrogens)
    lines = [f"{len(coords)}"]
    smiles = get_canonical_smiles(mol)
    lines.append(smiles)
    for symbol, x, y, z in coords:
        lines.append(f"{symbol:2s} {float(x):12.6f} {float(y):12.6f} {float(z):12.6f}")
    return '\n'.join(lines)


def _process_one_pickle(args):
    """Worker: load one pickle, extract all conformers, return entries.

    Returns (entries, xyz_strings, smiles_order_xyz_strings, skipped, error).
    entry_idx fields are left as None — assigned globally after merge.
    """
    pkl_path, base_path, remove_hydrogens, convert_fh, smiles_order = args
    full_path = os.path.join(base_path, pkl_path)
    try:
        with open(full_path, 'rb') as f:
            mol_data = pickle.load(f)
    except Exception as e:
        return [], [], [], 0, f"load fail {pkl_path}: {e}"

    entries = []
    xyz_strings = []
    so_xyz_strings = []

    conformers = mol_data.get("conformers", [])
    if mol_data.get("uniqueconfs") != len(conformers):
        return [], [], [], 1, None
    if mol_data.get("uniqueconfs", 0) <= 0:
        return [], [], [], 1, None
    if conformers[0]["rd_mol"].GetNumBonds() < 1:
        return [], [], [], 1, None
    if "." in Chem.MolToSmiles(conformers[0]["rd_mol"]):
        return [], [], [], 1, None

    for conf_data in conformers:
        mol = conf_data.get("rd_mol")
        if mol is None:
            continue

        coords = extract_mol_coordinates(mol, remove_hydrogens)
        smiles = get_canonical_smiles(mol)

        entry = {
            "filename": None,  # set at merge time
            "smiles": smiles,
            "coordinates": coords,
        }

        if smiles_order:
            try:
                mol_copy = Chem.Mol(mol)
                if remove_hydrogens:
                    try:
                        mol_for_order = Chem.RemoveHs(mol_copy)
                    except Exception:
                        mol_for_order = Chem.RemoveHs(mol_copy, sanitize=False)
                else:
                    mol_for_order = mol_copy
                order = get_smiles_atom_order(mol_for_order, heavy_only=remove_hydrogens)
                entry["smiles_order_coordinates"] = [coords[order[i]] for i in range(len(order))]
            except Exception:
                entry["smiles_order_coordinates"] = coords

        local_idx = len(entries)
        entries.append(entry)

        if convert_fh:
            xyz_strings.append((local_idx, write_xyz_string(mol, remove_hydrogens)))
        if smiles_order and convert_fh:
            so_xyz_strings.append(
                (local_idx, _write_xyz_from_coords(entry.get("smiles_order_coordinates", coords))))

    return entries, xyz_strings, so_xyz_strings, 0, None


def process_split(pickle_paths, base_path, remove_hydrogens=True,
                  convert_fh=False, smiles_order=False,
                  tmp_dir=None, split_name="train",
                  n_jobs=None):
    """Process a split of pickle paths into a JSON dataset.

    Args:
        pickle_paths: List of pickle file paths (relative to base_path).
        base_path: Base directory containing pickle files.
        remove_hydrogens: Whether to exclude hydrogen atoms.
        convert_fh: Whether to also convert to Fenske-Hall format.
        tmp_dir: Temporary directory for FH conversion.
        split_name: Name of the split (for progress bars).
        n_jobs: Number of parallel workers (default: all CPUs).

    Returns:
        List of dicts with 'filename', 'smiles', 'coordinates' keys.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed

    if n_jobs is None:
        n_jobs = os.cpu_count() or 4

    results = []
    xyz_strings = []
    smiles_order_xyz_strings = []
    skipped = 0

    work = [(p, base_path, remove_hydrogens, convert_fh, smiles_order)
            for p in pickle_paths]

    print(f"  Loading {len(work)} pickles with {n_jobs} workers ({split_name})...")
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures = {executor.submit(_process_one_pickle, w): i
                   for i, w in enumerate(work)}
        # Collect in submission order so the split is deterministic
        collected = [None] * len(work)
        for future in tqdm(as_completed(futures), total=len(work),
                           desc=f"Processing {split_name}"):
            i = futures[future]
            collected[i] = future.result()

    for result in collected:
        if result is None:
            continue
        entries, xs, so_xs, sk, err = result
        if err:
            print(f"Warning: {err}")
        skipped += sk
        base_idx = len(results)
        for j, entry in enumerate(entries):
            entry["filename"] = f"{split_name}/{base_idx + j}.xyz"
            results.append(entry)
        for local_idx, s in xs:
            xyz_strings.append((base_idx + local_idx, s))
        for local_idx, s in so_xs:
            smiles_order_xyz_strings.append((base_idx + local_idx, s))

    if skipped:
        print(f"  Skipped {skipped} molecules (mismatched conformers, no bonds, or fragments)")

    # Parallel FH conversion as a second pass (batched to limit disk usage)
    if convert_fh and tmp_dir is not None and xyz_strings:
        if n_jobs is None:
            n_jobs = os.cpu_count() or 4

        for r in results:
            r["fh_coordinates"] = None

        BATCH_SIZE = 50000
        total = len(xyz_strings)
        print(f"  Converting {total} conformers to FH with {n_jobs} workers (batches of {BATCH_SIZE})...")
        for batch_start in range(0, total, BATCH_SIZE):
            batch = xyz_strings[batch_start:batch_start + BATCH_SIZE]
            fh_dir = os.path.join(tmp_dir, f"fh_{split_name}")
            os.makedirs(fh_dir, exist_ok=True)

            work = []
            for entry_idx, xyz_str in batch:
                xyz_path = os.path.join(fh_dir, f"{entry_idx}.xyz")
                fh_path = os.path.join(fh_dir, f"{entry_idx}.fh")
                with open(xyz_path, 'w') as f:
                    f.write(xyz_str)
                work.append((entry_idx, xyz_path, fh_path))

            with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                futures = {executor.submit(_convert_one_fh, w): w for w in work}
                for future in tqdm(as_completed(futures), total=len(work),
                                   desc=f"  FH batch {batch_start//BATCH_SIZE+1}/{(total-1)//BATCH_SIZE+1} ({split_name})"):
                    entry_idx, coords, err = future.result()
                    if coords is not None:
                        results[entry_idx]["fh_coordinates"] = coords

    # SMILES-ordered FH conversion as a third pass (batched)
    if smiles_order and convert_fh and tmp_dir is not None and smiles_order_xyz_strings:
        if n_jobs is None:
            n_jobs = os.cpu_count() or 4

        for r in results:
            r["smiles_order_fh_coordinates"] = None

        BATCH_SIZE = 50000
        total = len(smiles_order_xyz_strings)
        print(f"  Converting {total} conformers to SMILES-order FH with {n_jobs} workers (batches of {BATCH_SIZE})...")
        for batch_start in range(0, total, BATCH_SIZE):
            batch = smiles_order_xyz_strings[batch_start:batch_start + BATCH_SIZE]
            so_fh_dir = os.path.join(tmp_dir, f"so_fh_{split_name}")
            os.makedirs(so_fh_dir, exist_ok=True)

            work = []
            for entry_idx, xyz_str in batch:
                xyz_path = os.path.join(so_fh_dir, f"{entry_idx}.xyz")
                fh_path = os.path.join(so_fh_dir, f"{entry_idx}.fh")
                with open(xyz_path, 'w') as f:
                    f.write(xyz_str)
                work.append((entry_idx, xyz_path, fh_path))

            with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                futures = {executor.submit(_convert_one_fh, w): w for w in work}
                for future in tqdm(as_completed(futures), total=len(work),
                                   desc=f"  SMILES-order FH batch {batch_start//BATCH_SIZE+1}/{(total-1)//BATCH_SIZE+1} ({split_name})"):
                    entry_idx, coords, err = future.result()
                    if coords is not None:
                        results[entry_idx]["smiles_order_fh_coordinates"] = coords

    return results


def _convert_one_fh(args):
    """Convert a single xyz file to FH (worker function for parallel execution)."""
    entry_idx, xyz_path, fh_path = args
    try:
        subprocess.run(
            ["obabel", "-ixyz", xyz_path, "-ofh", "-O", fh_path],
            capture_output=True, check=True,
        )
        with open(fh_path) as f:
            lines = f.readlines()
        num_atoms = int(lines[1].strip())
        coords = [lines[k].strip().split() for k in range(2, num_atoms + 2)]

        # Clean up temp files
        os.remove(xyz_path)
        os.remove(fh_path)

        return entry_idx, coords, None
    except Exception as e:
        # Clean up on failure too
        for p in (xyz_path, fh_path):
            try:
                os.remove(p)
            except OSError:
                pass
        return entry_idx, None, str(e)


def build_smiles_dict(pickle_paths, base_path):
    """Build a SMILES -> [Mol, ...] dict for the test set (for GEOM benchmarking).

    This is the equivalent of test_smiles_dict.pkl used by the benchmark scripts.
    """
    smiles_dict = {}
    for pkl_path in tqdm(pickle_paths, desc="Building test SMILES dict"):
        full_path = os.path.join(base_path, pkl_path)
        try:
            with open(full_path, 'rb') as f:
                mol_data = pickle.load(f)
        except Exception:
            continue

        # Apply same per-pickle filters as process_split
        conformers = mol_data.get("conformers", [])
        if mol_data.get("uniqueconfs") != len(conformers):
            continue
        if mol_data.get("uniqueconfs", 0) <= 0:
            continue
        if conformers[0]["rd_mol"].GetNumBonds() < 1:
            continue
        if "." in Chem.MolToSmiles(conformers[0]["rd_mol"]):
            continue

        for conf_data in conformers:
            mol = conf_data.get("rd_mol")
            if mol is None:
                continue
            smiles = get_canonical_smiles(mol)
            if smiles not in smiles_dict:
                smiles_dict[smiles] = []
            smiles_dict[smiles].append(mol)

    return smiles_dict


def main():
    parser = argparse.ArgumentParser(description="Prepare GEOM-QM9 dataset")
    parser.add_argument("--rdkit-folder", type=str, required=True,
                        help="Path to rdkit_folder/ from GEOM download")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for JSON files")
    parser.add_argument("--remove-hydrogens", action="store_true",
                        help="Exclude hydrogen atoms (recommended for GEOM)")
    parser.add_argument("--convert-fh", action="store_true",
                        help="Also convert to Fenske-Hall format")
    parser.add_argument("--smiles-order", action="store_true",
                        help="Also store coordinates in SMILES atom order "
                             "(for connectivity formats)")
    parser.add_argument("--save-test-dict", action="store_true",
                        help="Save test SMILES dict as pickle (for benchmarking)")
    parser.add_argument("--n-jobs", type=int, default=None,
                        help="Parallel workers for FH conversion (default: all CPUs)")
    parser.add_argument("--summary", type=str, default="summary_qm9.json",
                        help="Summary JSON filename (default: summary_qm9.json, "
                             "use summary_drugs.json for GEOM-drugs)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    base_path = args.rdkit_folder
    summary_path = os.path.join(base_path, args.summary)

    # Load and filter pickle paths
    print("Loading summary...")
    pickle_paths = load_pickle_paths(summary_path)
    print(f"Found {len(pickle_paths)} valid molecules")

    # Split
    train_paths, valid_paths, test_paths = split_indices(pickle_paths)
    print(f"Split: train={len(train_paths)}, val={len(valid_paths)}, test={len(test_paths)}")

    with tempfile.TemporaryDirectory(prefix="geom_") as tmp_dir:
        # Process each split
        suffix = "_heavy" if args.remove_hydrogens else ""

        for name, paths in [("train", train_paths), ("val", valid_paths),
                            ("test", test_paths)]:
            data = process_split(
                paths, base_path,
                remove_hydrogens=args.remove_hydrogens,
                convert_fh=args.convert_fh,
                smiles_order=args.smiles_order,
                tmp_dir=tmp_dir if (args.convert_fh or args.smiles_order) else None,
                split_name=name,
                n_jobs=args.n_jobs,
            )
            out_path = os.path.join(args.output_dir, f"{name}_xyz{suffix}.json")
            print(f"Writing {out_path} ({len(data)} conformers)...")
            with open(out_path, 'w') as f:
                json.dump(data, f)

        # Optionally save test SMILES dict for benchmarking
        if args.save_test_dict:
            smiles_dict = build_smiles_dict(test_paths, base_path)
            dict_path = os.path.join(args.output_dir, "test_smiles_dict.pkl")
            print(f"Writing {dict_path} ({len(smiles_dict)} unique SMILES)...")
            with open(dict_path, 'wb') as f:
                pickle.dump(smiles_dict, f)

    print("Done!")


if __name__ == "__main__":
    main()
