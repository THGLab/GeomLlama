import json
import math
from collections import Counter
from rdkit import Chem
from rdkit.Chem import rdMolAlign, rdDetermineBonds
from rdkit.Geometry import Point3D
import numpy as np

from converter import Converter
from prompts import resolve_eval_format


def zmat_string_to_cartesian_string(zmat_string):
    """Convert a Fenske-Hall style Z-matrix string to an XYZ coordinate string.

    Expected per-line format (1-indexed refs, angles in degrees):
        <element>
        <element> <ref1> <distance>
        <element> <ref1> <distance> <ref2> <angle>
        <element> <ref1> <distance> <ref2> <angle> <ref3> <dihedral>
        ...

    Returns an ``element x y z`` block (one line per atom). Raises on malformed
    input; callers are expected to treat that as a syntax error.
    """
    c = Converter()
    zmat = []
    lines = zmat_string.strip().split("\n")
    for i, line in enumerate(lines):
        items = line.split()
        name = items[0]
        mass = c.masses[name]
        if i == 0:
            zmat.append([name, [], mass])
        elif i == 1:
            name, atom1, distance = items
            zmat.append([name, [[int(atom1) - 1, float(distance)], [], []], mass])
        elif i == 2:
            name, atom1, distance, atom2, angle = items
            zmat.append([name, [[int(atom1) - 1, float(distance)],
                                [int(atom2) - 1, np.radians(float(angle))], []], mass])
        else:
            name, atom1, distance, atom2, angle, atom3, dihedral = items
            zmat.append([name, [[int(atom1) - 1, float(distance)],
                                [int(atom2) - 1, np.radians(float(angle))],
                                [int(atom3) - 1, np.radians(float(dihedral))]], mass])
    c.zmatrix = zmat
    c.zmatrix_to_cartesian()
    return c.str_cartesian()


def coordinates_to_mol(coordinates):
    mol = Chem.RWMol()
    conformer = Chem.Conformer()
    for i, parts in enumerate(coordinates):
        element, x, y, z = parts
        x, y, z = float(str(x).replace("*^", "e")), float(str(y).replace("*^", "e")), float(str(z).replace("*^", "e"))
        atom = Chem.Atom(element)
        mol.AddAtom(atom)
        conformer.SetAtomPosition(i, Point3D(x, y, z))
    conformer.SetId(0)
    mol.AddConformer(conformer)
    return mol.GetMol()

def parse_model_coordinates(model_str, fmt="xyz", verbose=False):
    """Parse a model output block into [[element, x, y, z], ...].

    For z-matrix formats ("zmat"/"fh") the block is first converted to
    Cartesian coordinates. Returns None if the output is not syntactically
    parsable: z-matrix conversion failure, a line with the wrong number of
    components, an unknown element, or non-float coordinates.

    Prompt formats such as "geomllama_zmat" are resolved to their coordinate
    dialect first, so every caller downstream of here gets it right.
    """
    fmt = resolve_eval_format(fmt)

    # Z-matrix output must first be converted to Cartesian coordinates.
    # ("zmat" is this repo's name; "fh" matches the reference benchmark.)
    if fmt in ("zmat", "fh"):
        try:
            model_str = zmat_string_to_cartesian_string(model_str).strip()
        except Exception as e:
            if verbose:
                print(f"Wrong syntax: z-matrix conversion failed: {e}")
            return None

    pt = Chem.GetPeriodicTable()
    model_coordinates = []
    for line in model_str.split("\n"):
        parts = line.strip().split()
        if len(parts) != 4:
            if verbose:
                print(f"Wrong syntax: line with the wrong number of components: {line}")
            return None
        element, x, y, z = parts
        try:
            anum = pt.GetAtomicNumber(element)
        except RuntimeError:
            anum = -1
        if anum <= 0:
            if verbose:
                print(f"Wrong syntax: invalid element in {line}")
            return None
        try:
            x, y, z = float(x), float(y), float(z)
        except ValueError:
            if verbose:
                print(f"Wrong syntax: coordiantes are not valid floats in {line}")
            return None
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
            # e.g. a degenerate z-matrix conversion yielding nan/inf -> unscorable
            if verbose:
                print(f"Wrong syntax: non-finite coordinates in {line}")
            return None
        model_coordinates.append([element, x, y, z])
    return model_coordinates


def classify_geometry(ground_truth_coordinates, model_str, fmt="xyz"):
    """Classify a model output by pipeline stage, independent of RMSD:

      "unparsable"  - could not be parsed into valid coordinates (bad syntax,
                      unknown element, or z-matrix conversion failure)
      "wrong_atoms" - parsable, but atom composition != ground truth
      "valid_atoms" - parsable and atom composition matches ground truth

    Note: unlike evaluate_xyz, a valid-atom-count geometry that later fails
    bond perception is still "valid_atoms" here — bond perception is an RMSD
    concern, not a parse/atom-count one.
    """
    coords = parse_model_coordinates(model_str, fmt=fmt)
    if coords is None:
        return "unparsable"
    if Counter(c[0] for c in coords) != Counter(c[0] for c in ground_truth_coordinates):
        return "wrong_atoms"
    return "valid_atoms"


def evaluate_xyz(ground_truth_coordinates, model_str, fmt="xyz", verbose=False):
    model_coordinates = parse_model_coordinates(model_str, fmt=fmt, verbose=verbose)
    if model_coordinates is None:
        return "Wrong syntax"

    ground_truth_atoms = [c[0] for c in ground_truth_coordinates]
    model_atoms = [c[0] for c in model_coordinates]
    if Counter(model_atoms) != Counter(ground_truth_atoms):
        if verbose:
            print(f"Ground truth has {Counter(ground_truth_atoms)}, model generated {Counter(model_atoms)}")
        return "Wrong number of atoms"

    # Get RMSD
    ground_truth_mol = coordinates_to_mol(ground_truth_coordinates)
    model_mol = coordinates_to_mol(model_coordinates)
    # Perceive bonds from geometry so GetBestRMS can graph-match atoms.
    # Matches the reference evaluator. Two failure modes are treated as an
    # unscorable (wrong-syntax) result rather than crashing the whole run:
    #   - bond perception fails (geometry too implausible for DetermineBonds)
    #   - the perceived connectivity differs from the ground truth, so
    #     GetBestRMS finds no sub-structure match (wrong molecule, not just
    #     wrong geometry).
    try:
        rdDetermineBonds.DetermineBonds(ground_truth_mol, charge=0)
        rdDetermineBonds.DetermineBonds(model_mol, charge=0)
        rmsd = rdMolAlign.GetBestRMS(ground_truth_mol, model_mol)
    except (ValueError, RuntimeError) as e:
        if verbose:
            print(f"Wrong syntax: bond perception / alignment failed: {e}")
        return "Wrong syntax"
    return rmsd

def coordinates_to_string(coordinates):
    atoms = []
    for c in coordinates:
        atoms.append(' '.join(c))
    return '\n'.join(atoms)


def print_stats_from_jsonl(jsonl_path: str) -> None:
    """Print aggregate stats from a results JSONL (success/syntax/atom-count counts, RMSD distribution)."""
    stats = {"success": 0, "wrong_syntax": 0, "wrong_atoms": 0}
    rmsds = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            rmsd = r.get("rmsd")
            if isinstance(rmsd, float):
                stats["success"] += 1
                rmsds.append(rmsd)
            elif rmsd == "Wrong syntax":
                stats["wrong_syntax"] += 1
            elif rmsd == "Wrong number of atoms":
                stats["wrong_atoms"] += 1

    total = sum(stats.values())
    print(f"\n{'='*50}")
    print(f"Results: {jsonl_path}")
    print(f"Total: {total}")
    print(f"  Valid RMSD:          {stats['success']}")
    print(f"  Wrong syntax:        {stats['wrong_syntax']}")
    print(f"  Wrong atom count:    {stats['wrong_atoms']}")
    if rmsds:
        rmsds.sort()
        print(f"\nRMSD (Angstroms):")
        print(f"  Mean:   {sum(rmsds) / len(rmsds):.4f}")
        print(f"  Median: {rmsds[len(rmsds) // 2]:.4f}")
        print(f"  Min:    {min(rmsds):.4f}")
        print(f"  Max:    {max(rmsds):.4f}")
    print(f"{'='*50}")
