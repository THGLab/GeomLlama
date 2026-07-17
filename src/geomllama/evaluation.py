"""
Evaluation metrics for generated molecular geometries.

QM9 benchmark: RMSD between single generated geometry and ground truth.
GEOM benchmark: MAT (coverage) and COV (mean RMSD) across conformer ensembles.
"""

import copy
from collections import Counter

import numpy as np
from joblib import Parallel, delayed
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds, rdMolAlign
from rdkit.Geometry import Point3D
from tqdm import tqdm

from geomllama.converter import fh_string_to_xyz_string
from geomllama.connectivity import get_smiles_atom_order


# ---------------------------------------------------------------------------
# Coordinate / molecule helpers
# ---------------------------------------------------------------------------

def coordinates_to_mol(coordinates):
    """Create an RDKit Mol from a list of (element, x, y, z).

    Args:
        coordinates: List of (element, x, y, z) tuples or lists.

    Returns:
        RDKit Mol with a single conformer.
    """
    mol = Chem.RWMol()
    conformer = Chem.Conformer()
    for i, parts in enumerate(coordinates):
        element, x, y, z = parts
        x = float(str(x).replace("*^", "e"))
        y = float(str(y).replace("*^", "e"))
        z = float(str(z).replace("*^", "e"))
        mol.AddAtom(Chem.Atom(element))
        conformer.SetAtomPosition(i, Point3D(x, y, z))
    conformer.SetId(0)
    mol.AddConformer(conformer)
    return mol.GetMol()


def remove_all_hydrogens(mol):
    """Remove all hydrogen atoms from a molecule."""
    em = Chem.EditableMol(mol)
    for atom in reversed(mol.GetAtoms()):
        if atom.GetAtomicNum() == 1:
            em.RemoveAtom(atom.GetIdx())
    return em.GetMol()


def mol_to_xyz_data(mol):
    """Extract (element, x, y, z) tuples from an RDKit Mol."""
    conformer = mol.GetConformer(0)
    positions = conformer.GetPositions()
    return [
        (atom.GetSymbol(), positions[i][0], positions[i][1], positions[i][2])
        for i, atom in enumerate(mol.GetAtoms())
    ]


# ---------------------------------------------------------------------------
# Parsing generated text into molecules
# ---------------------------------------------------------------------------

def _parse_xyz_text(text):
    """Parse XYZ coordinate text into [(element, x, y, z), ...]."""
    coords = []
    pt = Chem.GetPeriodicTable()
    for line in text.strip().split('\n'):
        parts = line.strip().split()
        if len(parts) != 4:
            return None
        element, x, y, z = parts
        # GetAtomicNumber raises RuntimeError on unknown symbols (e.g. lowercase
        # 'c' from a model leaking SMILES aromatic notation into xyz output);
        # treat that as an invalid line, same as a non-positive atomic number.
        try:
            atomic_num = pt.GetAtomicNumber(element)
        except RuntimeError:
            return None
        if atomic_num <= 0:
            return None
        try:
            coords.append((element, float(x), float(y), float(z)))
        except ValueError:
            return None
    return coords


def parse_generated_text(text, fmt='fh'):
    """Parse LLM-generated text into coordinates.

    Args:
        text: Generated text string.
        fmt: 'fh' for Fenske-Hall Z-matrix, 'xyz' for Cartesian.

    Returns:
        List of (element, x, y, z) tuples, or None on failure.
    """
    text = text.strip()
    if fmt == 'template_fh':
        from geomllama.data_formats import get_format
        return get_format('template_fh').parse_output(text)
    elif fmt == 'fh':
        try:
            xyz_text = fh_string_to_xyz_string(text)
        except Exception:
            return None
        return _parse_xyz_text(xyz_text)
    elif fmt == 'xyz':
        return _parse_xyz_text(text)
    else:
        # Fall back to the format registry for any registered format name
        # (e.g. 'ori_fh', 'ori_xyz', 'connectivity_fh', ...).
        from geomllama.data_formats import get_format, list_formats
        if fmt in list_formats():
            return get_format(fmt).parse_output(text)
        raise ValueError(f"Unknown format: {fmt}")


def text_to_mol(text, true_atom_counts, fmt='fh', mode='remove_hydrogens'):
    """Convert generated text to an RDKit Mol, with validation.

    Args:
        text: Generated geometry string.
        true_atom_counts: Counter of expected atom symbols.
        fmt: 'fh' or 'xyz'.
        mode: 'remove_hydrogens' to ignore H count mismatch.

    Returns:
        RDKit Mol, or a string error message.
    """
    coords = parse_generated_text(text, fmt)
    if coords is None:
        return "Wrong syntax"

    gen_counts = Counter(c[0] for c in coords)
    expected = Counter(true_atom_counts)
    if mode == 'remove_hydrogens':
        gen_counts['H'] = 0
        expected['H'] = 0
    if gen_counts != expected:
        return "Wrong number of atoms"

    return coordinates_to_mol(coords)


# ---------------------------------------------------------------------------
# QM9 benchmark: single-geometry RMSD
# ---------------------------------------------------------------------------

def evaluate_qm9_single(ground_truth_coordinates, generated_text, fmt='fh'):
    """Evaluate a single generated geometry against ground truth.

    Args:
        ground_truth_coordinates: List of (element, x, y, z).
        generated_text: LLM-generated geometry string.
        fmt: 'fh' or 'xyz'.

    Returns:
        RMSD (float), or error string ("Wrong syntax" / "Wrong number of atoms").
    """
    if fmt == 'fh':
        try:
            generated_text = fh_string_to_xyz_string(generated_text).strip()
        except Exception:
            return "Wrong syntax"

    coords = _parse_xyz_text(generated_text)
    if coords is None:
        return "Wrong syntax"

    gt_atoms = Counter(c[0] for c in ground_truth_coordinates)
    gen_atoms = Counter(c[0] for c in coords)
    if gen_atoms != gt_atoms:
        return "Wrong number of atoms"

    gt_mol = coordinates_to_mol(ground_truth_coordinates)
    gen_mol = coordinates_to_mol(coords)

    # Try with bond perception first (allows smarter atom matching for
    # symmetric atoms); fall back to raw alignment without bonds
    try:
        rdDetermineBonds.DetermineBonds(gt_mol, charge=0)
        rdDetermineBonds.DetermineBonds(gen_mol, charge=0)
        return rdMolAlign.GetBestRMS(gt_mol, gen_mol)
    except Exception:
        pass

    try:
        gt_mol = coordinates_to_mol(ground_truth_coordinates)
        gen_mol = coordinates_to_mol(coords)
        return rdMolAlign.GetBestRMS(gt_mol, gen_mol)
    except Exception:
        return "RMSD failed"


def _eval_qm9_single_worker(args):
    """Worker function for parallel QM9 evaluation (must be top-level for pickling)."""
    i, generated_text, gt_coords, fmt = args
    return i, evaluate_qm9_single(gt_coords, generated_text.strip(), fmt)


def evaluate_qm9_parallel(results, test_data, fmt='fh', n_processes=None):
    """Evaluate all QM9 test molecules in parallel.

    Args:
        results: List of (prompt, [generated_texts]) from inference.
        test_data: List of dicts with 'coordinates' key.
        fmt: 'fh' or 'xyz'.
        n_processes: Number of parallel workers (None = auto).

    Returns:
        List of RMSD values or error strings, one per molecule.
    """
    import multiprocessing as mp

    if n_processes is None:
        n_processes = mp.cpu_count() - 1

    args_list = [
        (i, results[i][1][0], test_data[i]["coordinates"], fmt)
        for i in range(len(results))
    ]

    with mp.Pool(processes=n_processes) as pool:
        results_indexed = list(tqdm(
            pool.imap(_eval_qm9_single_worker, args_list),
            total=len(args_list),
            desc="Evaluating molecules",
        ))

    results_indexed.sort(key=lambda x: x[0])
    return [r[1] for r in results_indexed]


# ---------------------------------------------------------------------------
# GEOM benchmark: MAT / COV over conformer ensembles
# ---------------------------------------------------------------------------

def _get_best_rmsd_heavy(gen_mol, ref_mol):
    """RMSD between two molecules after removing hydrogens."""
    gen_mol = remove_all_hydrogens(copy.deepcopy(gen_mol))
    ref_mol = remove_all_hydrogens(copy.deepcopy(ref_mol))
    return rdMolAlign.GetBestRMS(gen_mol, ref_mol)


def _compute_rmsd_row(i, gen_mol, ref_mols, mode):
    """Compute RMSD for one generated mol against all reference mols."""
    gen_mol = copy.deepcopy(gen_mol)
    row = np.zeros(len(ref_mols), dtype=np.float32)
    for j, ref_mol in enumerate(ref_mols):
        ref_mol = copy.deepcopy(ref_mol)
        if mode == 'remove_hydrogens':
            row[j] = _get_best_rmsd_heavy(gen_mol, ref_mol)
        else:
            row[j] = rdMolAlign.GetBestRMS(gen_mol, ref_mol)
    return i, row


def compute_mat_cov(gen_mols, ref_mols, threshold=0.5,
                    mode='remove_hydrogens', n_jobs=-1):
    """Compute MAT (coverage) and mean RMSD for generated vs reference conformers.

    Args:
        gen_mols: List of generated RDKit Mol objects.
        ref_mols: List of reference RDKit Mol objects.
        threshold: RMSD threshold for coverage (Angstroms).
        mode: 'remove_hydrogens' or 'with_hydrogens'.
        n_jobs: Number of parallel jobs (-1 = all cores).

    Returns:
        (coverage, mean_rmsd, rmsd_matrix)
        - coverage: fraction of ref conformers matched within threshold
        - mean_rmsd: mean of min RMSD per ref conformer
        - rmsd_matrix: full (n_ref, n_gen) RMSD matrix
    """
    # Use 2x reference conformers as generation cap
    idx = len(ref_mols) * 2
    gen_mols = gen_mols[:idx]

    results = Parallel(n_jobs=n_jobs)(
        delayed(_compute_rmsd_row)(i, gen_mol, ref_mols, mode)
        for i, gen_mol in tqdm(enumerate(gen_mols), total=len(gen_mols))
    )

    rmsd_mat = np.zeros([len(ref_mols), len(gen_mols)], dtype=np.float32)
    for i, row in results:
        rmsd_mat[:, i] = row

    rmsd_min = rmsd_mat.min(axis=-1)
    coverage = float((rmsd_min <= threshold).mean())
    mean_rmsd = float(rmsd_min.mean())
    return coverage, mean_rmsd, rmsd_mat


def _mol_heavy_coords(mol):
    """(elements, (A,3) ndarray) for the non-H atoms of an RDKit Mol."""
    conf = mol.GetConformer(0)
    pos = conf.GetPositions()
    el, xyz = [], []
    for i, atom in enumerate(mol.GetAtoms()):
        if atom.GetAtomicNum() == 1:
            continue
        el.append(atom.GetSymbol())
        xyz.append(pos[i])
    return el, np.asarray(xyz, dtype=np.float64)


# ---------------------------------------------------------------------------
# Order-free (no canonical atom order) evaluation
#
# The metric is the minimum RMSD over element-preserving atom permutations.
# rdMolAlign.GetBestRMS on bond-less mols computes it EXACTLY when the number of
# such permutations fits under its `maxMatches` cap, and silently does not when
# it doesn't: it enumerates mappings in atom-index order, truncates, and returns
# the best of a vanishing prefix. On GEOM-Drugs (median ~1e19 permutations) that
# value does not even respond to raising the cap 10,000x, and is inflated by a
# median 0.15 A -- enough to move MAT-R by 0.145 A and COV-R by 8 points.
#
# So dispatch on the permutation count, which is exactly computable:
#   count <= RDKIT_MAX_MATCHES  -> GetBestRMS, provably exact (all GEOM-QM9)
#   count >  RDKIT_MAX_MATCHES  -> Kabsch-Hungarian, a validated upper bound
# Nothing exact is computable in the second regime; KH matched the exact minimum
# on 315/320 QM9 pairs (mean abs err 3.8e-4 A) and never dipped below it.
# ---------------------------------------------------------------------------

RDKIT_MAX_MATCHES = 10 ** 6   # rdMolAlign.GetBestRMS default


def element_permutation_count(elements):
    """Number of element-preserving permutations of an atom list: prod(n_e!)."""
    import math
    n = 1
    for c in Counter(elements).values():
        n *= math.factorial(c)
    return n


def _kh_rmsd_row(j, gen_el, gen_xyz, ref_heavy):
    """Column j of the RMSD matrix: one generated mol vs every reference."""
    from geomllama.kabsch_hungarian import kh_rmsd
    row = np.zeros(len(ref_heavy), dtype=np.float32)
    for i, (ref_el, ref_xyz) in enumerate(ref_heavy):
        row[i] = kh_rmsd(gen_el, gen_xyz, ref_el, ref_xyz)
    return j, row


def compute_mat_cov_orderfree(gen_mols, ref_mols, threshold=0.5,
                              mode='remove_hydrogens', n_jobs=-1, engine="auto"):
    """COV/MAT for formats with no canonical atom order (ori_fh).

    Picks the RMSD engine per molecule by the exact element-permutation count
    (see the block comment above). ``engine`` forces one of 'auto' (default),
    'exact' (GetBestRMS, may be a capped non-minimum on large molecules) or
    'kh' (Kabsch-Hungarian).

    Same contract as :func:`compute_mat_cov`: ``(coverage, mean_rmsd, matrix)``
    with an (n_ref, n_gen) matrix. Both sides must already be bond-less point
    clouds sharing one heavy-atom element multiset.
    """
    if mode != 'remove_hydrogens':
        raise ValueError("compute_mat_cov_orderfree supports only "
                         "mode='remove_hydrogens'")
    gen_mols = gen_mols[:len(ref_mols) * 2]

    ref_heavy = [_mol_heavy_coords(m) for m in ref_mols]
    n_perm = element_permutation_count(ref_heavy[0][0])

    if engine == "auto":
        engine = "exact" if n_perm <= RDKIT_MAX_MATCHES else "kh"
    if engine == "exact":
        return compute_mat_cov(gen_mols, ref_mols, threshold=threshold,
                               mode=mode, n_jobs=n_jobs)
    if engine != "kh":
        raise ValueError(f"unknown engine {engine!r}")

    gen_heavy = [_mol_heavy_coords(m) for m in gen_mols]
    results = Parallel(n_jobs=n_jobs)(
        delayed(_kh_rmsd_row)(j, el, xyz, ref_heavy)
        for j, (el, xyz) in enumerate(gen_heavy))

    rmsd_mat = np.zeros([len(ref_mols), len(gen_mols)], dtype=np.float32)
    for j, row in results:
        rmsd_mat[:, j] = row

    rmsd_min = rmsd_mat.min(axis=-1)
    return (float((rmsd_min <= threshold).mean()),
            float(rmsd_min.mean()), rmsd_mat)


def compute_mat_cov_fast(gen_mols, ref_mols, threshold=0.5,
                         mode='remove_hydrogens', n_jobs=-1):
    """Fast, faithful drop-in for :func:`compute_mat_cov` on bond-less mols.

    Reproduces the exact bond-less heavy-atom ``GetBestRMS`` permutation minimum
    that ``compute_mat_cov`` computes per (gen, ref) pair -- the minimum RMSD over
    all element-preserving atom permutations -- but amortizes the permutation set
    across the whole (n_ref x n_gen) grid and evaluates it with a vectorized,
    GPU-batched Kabsch instead of a per-pair ``rdMolAlign.GetBestRMS`` call. This
    makes the ori_fh ensemble benchmark (no canonical atom order -> full
    permutation search) tractable: days -> minutes.

    Same contract as :func:`compute_mat_cov`: returns
    ``(coverage, mean_rmsd, rmsd_matrix)`` with the identical (n_ref, n_gen)
    matrix. ``mode``/``n_jobs`` are accepted for signature parity; only
    ``'remove_hydrogens'`` (heavy-atom) matching is supported (the sole mode used
    by the GEOM ensemble path).

    Note: garbage generations must already be filtered out upstream (as in
    :func:`evaluate_geom_molecule`); this function assumes every gen mol shares
    the reference heavy-atom element multiset.
    """
    if mode != 'remove_hydrogens':
        raise ValueError("compute_mat_cov_fast only supports "
                         "mode='remove_hydrogens'")
    from geomllama.fast_rmsd import best_rms_grid_from_coords

    idx = len(ref_mols) * 2
    gen_mols = gen_mols[:idx]

    gen_heavy = [_mol_heavy_coords(m) for m in gen_mols]
    ref_heavy = [_mol_heavy_coords(m) for m in ref_mols]
    # Per-conformer element lists: ori_fh gens (and refs) can each list atoms in
    # a different order, so grouping must be per-conformer, not shared.
    gen_elems = [el for el, _ in gen_heavy]
    ref_elems = [el for el, _ in ref_heavy]
    gen_coords = [c for _, c in gen_heavy]
    ref_coords = [c for _, c in ref_heavy]

    rmsd_mat = best_rms_grid_from_coords(
        gen_elems, gen_coords, ref_elems, ref_coords).astype(np.float32)

    rmsd_min = rmsd_mat.min(axis=-1)
    coverage = float((rmsd_min <= threshold).mean())
    mean_rmsd = float(rmsd_min.mean())
    return coverage, mean_rmsd, rmsd_mat


# ---------------------------------------------------------------------------
# Known-ordering evaluation (for connectivity formats)
#
# Connectivity-format models are trained to emit atoms in SMILES-canonical
# order. When both reference and generated coords share that ordering, we can
# skip the expensive GetBestRMS permutation search and do a direct
# Kabsch-aligned RMSD — which is also what the format is actually optimizing.
# ---------------------------------------------------------------------------

def reorder_mol_to_smiles_coords(mol, remove_hs=True):
    """Return (element, x, y, z) tuples for `mol` in SMILES-canonical order.

    Args:
        mol: RDKit Mol with a conformer.
        remove_hs: If True, strip hydrogens before reordering.

    Returns:
        List of (element, x, y, z) tuples in SMILES atom order, or None on
        failure.
    """
    try:
        m = mol
        if remove_hs:
            try:
                m = Chem.RemoveHs(m)
            except Exception:
                m = Chem.RemoveHs(m, sanitize=False)
        order = get_smiles_atom_order(m, heavy_only=False)
        positions = m.GetConformer(0).GetPositions()
        atoms = list(m.GetAtoms())
        return [
            (atoms[i].GetSymbol(),
             float(positions[i][0]), float(positions[i][1]), float(positions[i][2]))
            for i in order
        ]
    except Exception:
        return None


def reorder_mol_to_labeled_coords(mol, ordering="D"):
    """Reference coords in labeled_xyz canonical H-inclusive order (sanitized).

    The eval-side counterpart to prepare_chembl3d.py's coordinate reorder: both
    call connectivity.reorder_mol_coords_canonical, so the ground-truth atom
    order used for direct (Kabsch) RMSD matches the order the labeled_xyz model
    was trained to emit. Returns None on failure.
    """
    from geomllama.connectivity import reorder_mol_coords_canonical
    try:
        return reorder_mol_coords_canonical(mol, ordering=ordering, sanitize=True)
    except Exception:
        return None


def kabsch_rmsd(P, Q):
    """RMSD after optimal rigid-body alignment (Kabsch), no atom permutation."""
    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    if P.shape != Q.shape:
        return None
    P = P - P.mean(axis=0)
    Q = Q - Q.mean(axis=0)
    H = P.T @ Q
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    P_rot = P @ R.T
    return float(np.sqrt(((P_rot - Q) ** 2).sum(axis=1).mean()))


def direct_positional_rmsd(ref_coords, gen_coords, strict_symbols=True):
    """Direct RMSD between two coord lists assumed to share atom ordering.

    Args:
        ref_coords: [(element, x, y, z), ...]
        gen_coords: [(element, x, y, z), ...]
        strict_symbols: If True, require matching element at every position.

    Returns:
        RMSD (float) after Kabsch alignment, or None on mismatch.
    """
    if len(ref_coords) != len(gen_coords):
        return None
    if strict_symbols:
        for (r, *_), (g, *_) in zip(ref_coords, gen_coords):
            if r != g:
                return None
    P = np.array([[x, y, z] for _, x, y, z in ref_coords], dtype=np.float64)
    Q = np.array([[x, y, z] for _, x, y, z in gen_coords], dtype=np.float64)
    return kabsch_rmsd(P, Q)


def _ordered_rmsd_row(i, gen_coords, ref_coords_list):
    row = np.full(len(ref_coords_list), np.inf, dtype=np.float32)
    for j, ref in enumerate(ref_coords_list):
        r = direct_positional_rmsd(ref, gen_coords)
        if r is not None:
            row[j] = r
    return i, row


def evaluate_geom_molecule_ordered(ref_mols, generated_texts, fmt='fh',
                                   threshold=0.5, remove_hs=True, n_jobs=-1):
    """Known-ordering evaluation for connectivity-format models.

    Reorders each reference conformer to SMILES-canonical atom order, parses
    each generated output as-is (connectivity-trained models emit SMILES
    order), and computes direct Kabsch-aligned RMSD. Returns MAT/COV in the
    same shape as `evaluate_geom_molecule`.

    Args:
        ref_mols: List of RDKit Mol conformers (ground truth).
        generated_texts: List of generated geometry strings.
        fmt: 'fh' or 'xyz' (parser for generated text).
        threshold: RMSD threshold (Angstroms) for coverage.
        remove_hs: If True, heavy-atom-only evaluation.
        n_jobs: Parallel workers for RMSD rows.

    Returns:
        (num_valid, (coverage, mean_rmsd, rmsd_matrix))
        or (0, ('None', 'None', 'None')) if no valid generations.
    """
    ref_coords_list = []
    for m in ref_mols:
        c = reorder_mol_to_smiles_coords(m, remove_hs=remove_hs)
        if c is not None:
            ref_coords_list.append(c)
    if not ref_coords_list:
        return 0, ('None', 'None', 'None')

    expected_count = len(ref_coords_list[0])
    expected_symbols = [c[0] for c in ref_coords_list[0]]

    gen_coords_list = []
    for text in generated_texts:
        try:
            coords = parse_generated_text(text.strip(), fmt)
        except Exception as ex:
            print(f"Warning: parse_generated_text raised {type(ex).__name__}: "
                  f"{ex}; skipping one generation")
            continue
        if coords is None:
            continue
        if len(coords) != expected_count:
            continue
        if [c[0] for c in coords] != expected_symbols:
            continue
        gen_coords_list.append(coords)

    if not gen_coords_list:
        return 0, ('None', 'None', 'None')

    gen_coords_list = gen_coords_list[:len(ref_coords_list) * 2]

    results = Parallel(n_jobs=n_jobs)(
        delayed(_ordered_rmsd_row)(i, g, ref_coords_list)
        for i, g in tqdm(enumerate(gen_coords_list), total=len(gen_coords_list))
    )

    rmsd_mat = np.zeros([len(ref_coords_list), len(gen_coords_list)],
                        dtype=np.float32)
    for i, row in results:
        rmsd_mat[:, i] = row

    rmsd_min = rmsd_mat.min(axis=-1)
    coverage = float((rmsd_min <= threshold).mean())
    mean_rmsd = float(rmsd_min.mean())
    return len(gen_coords_list), (coverage, mean_rmsd, rmsd_mat)


def evaluate_geom_molecule(ground_truth_conformers, generated_texts, fmt='fh',
                           threshold=0.5, mode='remove_hydrogens', n_jobs=-1):
    """Evaluate generated conformers for a single molecule (GEOM benchmark).

    Args:
        ground_truth_conformers: List of [(element, x, y, z), ...] per conformer.
        generated_texts: List of generated geometry strings.
        fmt: 'fh' or 'xyz'.
        threshold: RMSD threshold for coverage.
        mode: 'remove_hydrogens' or 'with_hydrogens'.
        n_jobs: Parallel jobs for RMSD computation.

    Returns:
        (num_valid, (coverage, mean_rmsd, rmsd_matrix))
        or (0, ('None', 'None', 'None')) if no valid generations.
    """
    true_atoms = Counter(c[0] for c in ground_truth_conformers[0])

    gen_mols = []
    for text in generated_texts:
        try:
            mol = text_to_mol(text.strip(), true_atoms, fmt, mode)
        except Exception as ex:
            # Defense in depth: a parse path inside text_to_mol unexpectedly
            # raised. Treat as an invalid generation rather than aborting a
            # multi-hour eval over one bad output.
            print(f"Warning: text_to_mol raised {type(ex).__name__}: {ex}; "
                  f"skipping one generation")
            continue
        if isinstance(mol, Chem.Mol):
            gen_mols.append(mol)

    ref_mols = [coordinates_to_mol(conf) for conf in ground_truth_conformers]

    if len(gen_mols) == 0:
        return 0, ('None', 'None', 'None')

    assessments = compute_mat_cov(
        gen_mols, ref_mols, threshold, mode, n_jobs
    )
    return len(gen_mols), assessments
