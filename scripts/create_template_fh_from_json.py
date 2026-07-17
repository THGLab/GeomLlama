#!/usr/bin/env python3
"""Build template_fh JSONL from QM9-style JSON (smiles + coordinates).

Reconstructs an rdmol from each entry's XYZ coordinates (via
DetermineBonds), then runs the same pipeline as prepare_geom_drugs.py:
canonical SMILES → H-inclusive order → Z-matrix rows → template_fh
datapoint.

Usage:
    python scripts/create_template_fh_from_json.py \
        --input data/qm9/train_set.json \
        --output data/qm9/train_fh/template_fh.jsonl \
        --n-jobs 8
"""
import argparse, json, os, sys, time
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from rdkit import Chem
from rdkit import RDLogger; RDLogger.DisableLog('rdApp.*')
from rdkit.Chem.rdDetermineBonds import DetermineConnectivity, DetermineBondOrders
from geomllama.prompts import make_sft_datapoint
from geomllama.connectivity import (
    get_hydrogen_atom_order, mol_with_hs_from_smiles, mol_to_zmat_rows,
)


def _mol_from_xyz(xyz_coords):
    """Build an rdmol with bond graph from raw XYZ coordinates."""
    mol = Chem.RWMol()
    conf = Chem.Conformer(len(xyz_coords))
    for i, (el, x, y, z) in enumerate(xyz_coords):
        mol.AddAtom(Chem.Atom(el))
        conf.SetAtomPosition(i, (float(x), float(y), float(z)))
    mol.AddConformer(conf)
    DetermineConnectivity(mol)
    DetermineBondOrders(mol)
    Chem.SanitizeMol(mol)
    return mol


def _canonical_smiles(mol):
    try:
        m = Chem.RemoveHs(mol)
    except Exception:
        m = Chem.RemoveHs(mol, sanitize=False)
    return Chem.MolToSmiles(m, canonical=True)


def _convert_one(xyz_coords):
    ordering = "D"
    try:
        mol = _mol_from_xyz(xyz_coords)
        smiles = _canonical_smiles(mol)

        m_inf = mol_with_hs_from_smiles(smiles)
        if m_inf is None:
            return None, None, "smiles_unparseable"
        el_inf = [m_inf.GetAtomWithIdx(j).GetSymbol()
                  for j in get_hydrogen_atom_order(m_inf, ordering)]

        rows, el_data = mol_to_zmat_rows(mol, ordering, sanitize=True, ref_base=1)
        if el_data != el_inf:
            return None, None, "order_not_reproducible"

        return smiles, rows, None
    except Exception as e:
        return None, None, f"exception:{type(e).__name__}"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="Input JSON (from prepare_qm9.py)")
    ap.add_argument("--output", required=True, help="Output JSONL")
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    t0 = time.time()
    print(f"Loading {args.input}...")
    with open(args.input) as f:
        data = json.load(f)
    n = len(data) if not args.limit else min(len(data), args.limit)
    print(f"  {n} entries, loaded in {time.time()-t0:.0f}s")

    work = [r['coordinates'] for r in data[:n]]

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    written = failed = 0
    skip = {}
    t1 = time.time()

    with open(args.output, "w") as out, ProcessPoolExecutor(max_workers=args.n_jobs) as ex:
        for smiles, payload, err in ex.map(_convert_one, work, chunksize=64):
            if payload is None:
                failed += 1
                skip[err] = skip.get(err, 0) + 1
            else:
                dp = make_sft_datapoint(smiles, payload, "template_fh")
                out.write(json.dumps(dp) + "\n")
                written += 1
            total = written + failed
            if total % 10000 == 0:
                el = time.time() - t1
                print(f"  {total}/{n} ({written} ok, {failed} skip) "
                      f"{el:.0f}s {total/max(el,1):.0f}/s", flush=True)

    print(f"DONE: {written} written, {failed} skipped in {time.time()-t0:.0f}s")
    if skip:
        print(f"  skip breakdown: {skip}")


if __name__ == "__main__":
    main()
