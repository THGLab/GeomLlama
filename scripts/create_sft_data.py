#!/usr/bin/env python3
"""
Generate .jsonl training data from prepared JSON datasets.

Uses the pluggable format system in geomllama.data_formats.

Usage:
  # Single format
  python create_sft_data.py --input data/qm9/train_set.json \\
      --format ori_xyz --output data/qm9/train_xyz/ori_xyz.jsonl

  # Fenske-Hall format (requires fh_coordinates in JSON)
  python create_sft_data.py --input data/qm9/train_set.json \\
      --format ori_fh --output data/qm9/train_fh/ori_fh.jsonl \\
      --fh-key fh_coordinates

  # All XYZ formats at once
  python create_sft_data.py --input data/qm9/train_set.json \\
      --all-xyz-formats --output-dir data/qm9/train_xyz/

  # Formula variant (requires rdkit)
  python create_sft_data.py --input data/qm9/train_set.json \\
      --format formula_roundto_strict3_xyz \\
      --output data/qm9/train_xyz/formula_roundto_strict3_xyz.jsonl
"""

import argparse
import json
import os
import sys

from tqdm import tqdm

# Ensure geomllama is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from geomllama.data_formats import get_format, list_formats
from geomllama.prompts import make_sft_datapoint, write_jsonl


def get_formula(smiles):
    """Compute molecular formula from SMILES (requires rdkit)."""
    from rdkit import Chem
    from rdkit.Chem.rdMolDescriptors import CalcMolFormula
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    return CalcMolFormula(mol)


def create_datapoints(data, format_name, fh_key=None):
    """Create SFT datapoints for a dataset.

    Args:
        data: List of dicts with 'smiles', 'coordinates', and optionally fh_key.
        format_name: Name of the geometry format.
        fh_key: Key for Fenske-Hall coordinates (if format uses FH data).

    Returns:
        List of SFT datapoint dicts.
    """
    fmt = get_format(format_name)
    needs_formula = 'formula' in format_name
    is_fh = 'fh' in format_name
    is_connectivity = 'connectivity' in format_name

    datapoints = []
    for r in tqdm(data, desc=f"Creating {format_name}"):
        kwargs = {}

        if needs_formula:
            kwargs['formula'] = get_formula(r['smiles'])

        # Pick the right coordinate key
        if is_connectivity and is_fh:
            coords = r.get("smiles_order_fh_coordinates")
            if coords is None:
                continue
        elif is_connectivity:
            coords = r.get("smiles_order_coordinates")
            if coords is None:
                continue
        elif is_fh and fh_key:
            coords = r.get(fh_key)
            if coords is None:
                continue
        else:
            coords = r['coordinates']

        dp = make_sft_datapoint(r['smiles'], coords, format_name, **kwargs)
        datapoints.append(dp)

    return datapoints


# All XYZ format names (not FH)
XYZ_FORMATS = [
    "ori_xyz",
    "roundto1_xyz", "roundto2_xyz", "roundto3_xyz", "roundto6_xyz",
    "roundto_strict2_xyz", "roundto_strict3_xyz",
    "formula_roundto_strict2_xyz", "formula_roundto_strict3_xyz",
    "spacedsmiles_roundto_strict2_xyz", "spacedsmiles_roundto_strict3_xyz",
]


def main():
    parser = argparse.ArgumentParser(description="Create SFT training data")
    parser.add_argument("--input", type=str, required=True,
                        help="Input JSON file (from prepare_qm9.py or prepare_geom.py)")
    parser.add_argument("--format", type=str, default=None,
                        help=f"Format name. Available: {list_formats()}")
    parser.add_argument("--output", type=str, default=None,
                        help="Output .jsonl file path")
    parser.add_argument("--fh-key", type=str, default="fh_coordinates",
                        help="JSON key for FH coordinates (default: fh_coordinates)")
    parser.add_argument("--all-xyz-formats", action="store_true",
                        help="Generate all XYZ format variants")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (for --all-xyz-formats)")
    args = parser.parse_args()

    print(f"Loading {args.input}...")
    with open(args.input) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} entries")

    if args.all_xyz_formats:
        if args.output_dir is None:
            parser.error("--output-dir required with --all-xyz-formats")
        os.makedirs(args.output_dir, exist_ok=True)

        for fmt_name in XYZ_FORMATS:
            print(f"\n--- {fmt_name} ---")
            datapoints = create_datapoints(data, fmt_name)
            out_path = os.path.join(args.output_dir, f"{fmt_name}.jsonl")
            write_jsonl(datapoints, out_path)
            print(f"Wrote {len(datapoints)} examples to {out_path}")

    elif args.format is not None:
        if args.output is None:
            parser.error("--output required with --format")
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

        datapoints = create_datapoints(data, args.format, args.fh_key)
        write_jsonl(datapoints, args.output)
        print(f"Wrote {len(datapoints)} examples to {args.output}")

    else:
        parser.error("Provide either --format or --all-xyz-formats")


if __name__ == "__main__":
    main()
