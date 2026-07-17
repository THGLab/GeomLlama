#!/usr/bin/env python3
"""
Prepare the GEOM-drugs "small" pre-split subset for fine-tuning.

Source: data/raw/drugs-small/{train_data_39k,val_data_5k,test_data_200}.pkl
Each pkl is a list of torch_geometric Data objects (one per conformer) with
fields .rdmol, .smiles, .pos, .atom_type, .totalenergy, .boltzmannweight, etc.
The pickles were created with an older PyG release; attributes are read via
__dict__ to bypass the new __getattr__ that raises on legacy objects.

Output layout (mirrors data/qm9/ and what prepare_geom.py produces):
  data/drugs_small/
    train_set.json        # heavy-atom coords; {filename, smiles, coordinates,
    val_set.json          #   [fh_coordinates], [smiles_order_coordinates],
    test_set.json         #   [smiles_order_fh_coordinates]}
    test_smiles_dict.pkl  # {regenerated_smiles: [Mol, ...]}, for benchmark_geom.py

Heavy-atom only: H is stripped from coords (heavy-atom RMSD is the eval target,
and dropping H shortens training sequences). Canonical SMILES is regenerated
from rdmol after RemoveHs so that prompts at train and inference time share one
canonicalisation; the pkl's stored .smiles string is discarded (it disagrees
with current rdkit's canonical form for ~30-99% of conformers across splits).

Notes on the source data (preserved as-is, not corrected):
  - 5 SMILES appear in both val and test pkls; splits are kept as provided,
    so val metrics on those 5 are not independent of test.
  - For 5 of the 200 test molecules, current rdkit's E/Z bond perception
    assigns different cis/trans labels to different conformers of the same
    flat molecule, so regenerated SMILES splits them across 2-4 keys each.
    test_smiles_dict.pkl therefore has 208 keys (not 200), and benchmark
    MAT/COV for those 5 molecules is computed over partial conformer
    ensembles. Affects 2.5% of test.

Usage:
  python prepare_drugs_small.py --raw-dir data/raw/drugs-small/ \\
      --output-dir data/drugs_small/ \\
      --convert-fh --smiles-order --save-test-dict
"""

import argparse
import json
import os
import pickle
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed

from rdkit import Chem
from tqdm import tqdm

# Reuse FH-conversion / atom-order helpers from prepare_geom.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare_geom import (
    _convert_one_fh,
    _write_xyz_from_coords,
    extract_mol_coordinates,
    get_smiles_atom_order,
    write_xyz_string,
)


SPLIT_FILES = {
    "train": "train_data_39k.pkl",
    "val": "val_data_5k.pkl",
    "test": "test_data_200.pkl",
}


def _get_attr(data_obj, name):
    """Read an attribute from a (possibly legacy-PyG) Data object via __dict__."""
    return data_obj.__dict__.get(name)


def _heavy_only_mol(rdmol):
    """Return a heavy-atom copy of rdmol; falls back to sanitize=False on failure."""
    try:
        return Chem.RemoveHs(rdmol)
    except Exception:
        return Chem.RemoveHs(rdmol, sanitize=False)


def process_pkl_split(pkl_path, split_name, smiles_order=False, convert_fh=False,
                      keep_hydrogens=False):
    """Walk one pre-split pkl into the JSON-friendly entry shape.

    Returns:
        results: list of entry dicts.
        xyz_strings: list of (entry_idx, xyz_str) for FH conversion.
        smiles_order_xyz_strings: list of (entry_idx, xyz_str) for smiles-order FH.
        skipped: count of conformers dropped by the defensive filters.
    """
    with open(pkl_path, "rb") as f:
        data_list = pickle.load(f)

    results = []
    xyz_strings = []
    so_xyz_strings = []
    skipped = 0
    remove_h = not keep_hydrogens

    for data_obj in tqdm(data_list, desc=f"Loading {split_name}"):
        rdmol = _get_attr(data_obj, "rdmol")
        if rdmol is None:
            skipped += 1
            continue

        # Canonical SMILES is always derived from the heavy-atom mol
        # (canonical SMILES is heavy-only by convention).
        mol_no_h = _heavy_only_mol(rdmol)
        smiles = Chem.MolToSmiles(mol_no_h, canonical=True)

        # Defensive filters (no-ops on the current pkls; matches prepare_geom)
        if "." in smiles:
            skipped += 1
            continue
        if mol_no_h.GetNumBonds() < 1:
            skipped += 1
            continue

        # Coordinates in original-mol atom order.
        # If keep_hydrogens, includes H rows; otherwise heavy-only.
        coords = extract_mol_coordinates(rdmol, remove_hydrogens=remove_h)

        idx = len(results)
        entry = {
            "filename": f"{split_name}/{idx}.xyz",
            "smiles": smiles,
            "coordinates": coords,
        }

        # SMILES-order coords are heavy-only by definition (canonical SMILES
        # only enumerates heavy atoms); skip when keeping H to avoid an
        # inconsistent mix of full-atom 'coordinates' + heavy-only smiles_order_*.
        if smiles_order and not keep_hydrogens:
            try:
                order = get_smiles_atom_order(mol_no_h, heavy_only=True)
                entry["smiles_order_coordinates"] = [
                    coords[order[i]] for i in range(len(order))
                ]
            except Exception:
                entry["smiles_order_coordinates"] = coords

        results.append(entry)

        if convert_fh:
            xyz_strings.append((idx, write_xyz_string(rdmol, remove_hydrogens=remove_h)))
            if smiles_order and not keep_hydrogens:
                so_xyz_strings.append(
                    (idx, _write_xyz_from_coords(entry["smiles_order_coordinates"]))
                )

    return results, xyz_strings, so_xyz_strings, skipped


def run_fh_conversion(results, xyz_strings, target_key, tmp_dir, split_name,
                      n_jobs, batch_size=50000):
    """Populate target_key in each entry by running obabel on /tmp xyz files.

    Mirrors the batched parallel FH pass in prepare_geom.process_split.
    """
    for r in results:
        r[target_key] = None

    if not xyz_strings:
        return

    total = len(xyz_strings)
    print(f"  Converting {total} conformers -> {target_key} with {n_jobs} workers "
          f"(batches of {batch_size})...")

    for batch_start in range(0, total, batch_size):
        batch = xyz_strings[batch_start:batch_start + batch_size]
        sub_dir = os.path.join(tmp_dir, f"{target_key}_{split_name}")
        os.makedirs(sub_dir, exist_ok=True)

        work = []
        for entry_idx, xyz_str in batch:
            xyz_path = os.path.join(sub_dir, f"{entry_idx}.xyz")
            fh_path = os.path.join(sub_dir, f"{entry_idx}.fh")
            with open(xyz_path, "w") as f:
                f.write(xyz_str)
            work.append((entry_idx, xyz_path, fh_path))

        n_batches = (total - 1) // batch_size + 1
        batch_num = batch_start // batch_size + 1
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = {executor.submit(_convert_one_fh, w): w for w in work}
            for future in tqdm(as_completed(futures), total=len(work),
                               desc=f"  {target_key} batch {batch_num}/{n_batches} ({split_name})"):
                entry_idx, coords, _ = future.result()
                if coords is not None:
                    results[entry_idx][target_key] = coords


def build_test_smiles_dict(pkl_path):
    """Group test conformers by regenerated canonical SMILES.

    Mirrors prepare_geom.build_smiles_dict but reads from the pre-split pkl
    and keys on the regenerated SMILES (so eval lookups match training prompts).
    Stores the original full-H rdmol as the value, like prepare_geom does;
    benchmark_geom.py's mol_to_xyz_data + evaluate_geom_molecule(mode=
    'remove_hydrogens') handles H-stripping at eval time.
    """
    with open(pkl_path, "rb") as f:
        data_list = pickle.load(f)

    smiles_dict = {}
    for data_obj in tqdm(data_list, desc="Building test SMILES dict"):
        rdmol = _get_attr(data_obj, "rdmol")
        if rdmol is None:
            continue
        smiles = Chem.MolToSmiles(_heavy_only_mol(rdmol), canonical=True)
        if "." in smiles:
            continue
        smiles_dict.setdefault(smiles, []).append(rdmol)

    return smiles_dict


def main():
    parser = argparse.ArgumentParser(description="Prepare GEOM-drugs-small dataset")
    parser.add_argument("--raw-dir", type=str, required=True,
                        help="Directory containing the three pre-split pkl files")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for {train,val,test}_set.json")
    parser.add_argument("--convert-fh", action="store_true",
                        help="Also convert to Fenske-Hall format")
    parser.add_argument("--smiles-order", action="store_true",
                        help="Also store coordinates in canonical-SMILES atom order "
                             "(needed for connectivity_* formats)")
    parser.add_argument("--save-test-dict", action="store_true",
                        help="Save test_smiles_dict.pkl for benchmark_geom.py")
    parser.add_argument("--keep-hydrogens", action="store_true",
                        help="Include H atoms in coordinates and FH (default: heavy-only). "
                             "Disables --smiles-order outputs since SMILES atom order "
                             "is heavy-only by convention.")
    parser.add_argument("--n-jobs", type=int, default=None,
                        help="Parallel workers for FH conversion (default: all CPUs)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    n_jobs = args.n_jobs or os.cpu_count() or 4

    with tempfile.TemporaryDirectory(prefix="drugs_small_") as tmp_dir:
        for split_name, pkl_name in SPLIT_FILES.items():
            pkl_path = os.path.join(args.raw_dir, pkl_name)
            print(f"\n=== {split_name} ({pkl_name}) ===")

            results, xyz_strings, so_xyz_strings, skipped = process_pkl_split(
                pkl_path, split_name,
                smiles_order=args.smiles_order,
                convert_fh=args.convert_fh,
                keep_hydrogens=args.keep_hydrogens,
            )
            if skipped:
                print(f"  Skipped {skipped} conformers (filtered by defensive checks)")

            if args.convert_fh:
                run_fh_conversion(results, xyz_strings, "fh_coordinates",
                                  tmp_dir, split_name, n_jobs)
                if args.smiles_order:
                    run_fh_conversion(results, so_xyz_strings,
                                      "smiles_order_fh_coordinates",
                                      tmp_dir, split_name, n_jobs)

            out_path = os.path.join(args.output_dir, f"{split_name}_set.json")
            print(f"  Writing {out_path} ({len(results)} conformers)...")
            with open(out_path, "w") as f:
                json.dump(results, f)

        if args.save_test_dict:
            smiles_dict = build_test_smiles_dict(
                os.path.join(args.raw_dir, SPLIT_FILES["test"])
            )
            dict_path = os.path.join(args.output_dir, "test_smiles_dict.pkl")
            print(f"\nWriting {dict_path} ({len(smiles_dict)} unique SMILES)...")
            with open(dict_path, "wb") as f:
                pickle.dump(smiles_dict, f)

    print("\nDone!")


if __name__ == "__main__":
    main()
