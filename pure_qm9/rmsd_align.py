"""Correspondence-free RMSD between a model geometry and a ground-truth geometry.

Two RMSDs, both symmetry-aware in different senses:

  graph_rmsd       - RDKit GetBestRMS after bond perception on BOTH molecules.
                     Chemically strict: only permutes graph-automorphic atoms.
                     Returns None when bond perception fails (implausible geometry).

  assignment_rmsd  - min RMSD over all ELEMENT-preserving atom bijections + rigid
                     alignment. A lower bound on graph_rmsd (they coincide when the
                     geometry is good). Always available for matching atom counts,
                     so it rescues structures RDKit can't bond-perceive.

The assignment RMSD is computed by anchoring the rigid alignment on the heavy
atoms (few enough to brute-force the element-blocked permutations -> global
optimum for the heavy frame), then, for each candidate alignment, solving the
exact optimal atom assignment via the Hungarian algorithm (linear_sum_assignment)
-- which, for a FIXED alignment, is the global optimum over all element-preserving
permutations, hydrogens included. A short Kabsch<->Hungarian refinement polishes
each candidate toward the joint optimum; a few random restarts cover degenerate
heavy frames (<3 or collinear heavy atoms).

Atom counts must already match by element (caller's responsibility; see
bench_tools.classify_geometry -> "valid_atoms").
"""
import itertools
import math

import numpy as np
from scipy.optimize import linear_sum_assignment

from bench_tools import coordinates_to_mol
from rdkit.Chem import rdMolAlign, rdDetermineBonds


# ----- primitives ---------------------------------------------------------

def _coord_float(v):
    """Parse a coordinate, handling QM9's Mathematica notation (4.98*^-6 -> 4.98e-6).

    Matches bench_tools.coordinates_to_mol; model coords are already floats (no-op).
    """
    return float(str(v).replace("*^", "e"))


def _kabsch(P, Q):
    """Optimal rotation R (3x3) s.t. P @ R.T best aligns to Q.

    P, Q are centered (N, 3) arrays with row i of P corresponding to row i of Q.
    """
    H = P.T @ Q
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    return Vt.T @ D @ U.T


def _random_rotation(rng):
    """Uniformish random rotation via QR of a Gaussian matrix (det forced +1)."""
    A = rng.standard_normal((3, 3))
    Q, _ = np.linalg.qr(A)
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


# ----- exact assignment for a fixed alignment -----------------------------

def _element_assignment(GT, TM, elem_idx):
    """Given ground-truth coords GT and transformed-model coords TM, solve the
    optimal element-blocked atom assignment (Hungarian per element).

    elem_idx maps element symbol -> (gt_indices, model_indices) arrays.
    Returns (total_squared_deviation, assign) where assign[gt_i] = model_j.
    """
    total = 0.0
    assign = np.empty(len(GT), dtype=int)
    for gi, mi in elem_idx.values():
        # pairwise squared distances, gt rows x model cols
        diff = GT[gi][:, None, :] - TM[mi][None, :, :]
        cost = np.einsum("ijk,ijk->ij", diff, diff)
        r, c = linear_sum_assignment(cost)
        total += cost[r, c].sum()
        assign[gi[r]] = mi[c]
    return total, assign


def _refine(GT, MODEL, elem_idx, TM, max_iter=25):
    """Alternate exact Hungarian assignment and full-atom Kabsch until the
    assignment stops changing. Returns (rmsd, TM_final).

    TM is the initial transformed model (N,3); MODEL is the raw model coords.
    """
    n = len(GT)
    prev = None
    for _ in range(max_iter):
        total, assign = _element_assignment(GT, TM, elem_idx)
        if np.array_equal(assign, prev):
            break
        prev = assign
        # full-atom Kabsch on the current correspondence, then re-apply to all
        Pcorr = MODEL[assign]                 # model point matched to each gt atom
        cP, cQ = Pcorr.mean(0), GT.mean(0)
        R = _kabsch(Pcorr - cP, GT - cQ)
        TM = (MODEL - cP) @ R.T + cQ
    total, assign = _element_assignment(GT, TM, elem_idx)
    return float(np.sqrt(total / n)), TM


# ----- assignment RMSD ----------------------------------------------------

def assignment_rmsd(gt_coords, model_coords, top_k=5, heavy_perm_cap=20000,
                    n_random=12, seed=0, use_heavy=True, refine_iters=25):
    """Min RMSD over element-preserving atom bijections + rigid alignment.

    Anchors alignment on heavy atoms (brute-forced when the element-blocked
    permutation count is <= heavy_perm_cap, else skipped in favour of random
    restarts), keeps the top_k heavy frames, refines each with Kabsch/Hungarian,
    and returns the best RMSD found. n_random random restarts add robustness for
    degenerate heavy frames.
    """
    elems = np.array([c[0] for c in gt_coords])
    GT = np.array([[_coord_float(c[1]), _coord_float(c[2]), _coord_float(c[3])] for c in gt_coords])
    m_elems = np.array([c[0] for c in model_coords])
    MODEL = np.array([[_coord_float(c[1]), _coord_float(c[2]), _coord_float(c[3])] for c in model_coords])

    # Non-finite coordinates (e.g. a degenerate z-matrix conversion) are
    # unscorable; guard so Kabsch's SVD never sees nan/inf.
    if not (np.isfinite(GT).all() and np.isfinite(MODEL).all()):
        return float("nan")

    # element -> (gt indices, model indices); model side reordered so counts line up
    elem_idx = {}
    for e in sorted(set(elems)):
        gi = np.where(elems == e)[0]
        mi = np.where(m_elems == e)[0]
        elem_idx[e] = (gi, mi)

    heavy = [e for e in elem_idx if e != "H"]
    candidates = []  # list of initial TM arrays

    # --- heavy-anchored candidate alignments ---
    heavy_gi = np.concatenate([elem_idx[e][0] for e in heavy]) if heavy else np.array([], int)
    heavy_perm_count = 1
    for e in heavy:
        heavy_perm_count *= math.factorial(len(elem_idx[e][1]))

    if use_heavy and heavy and len(heavy_gi) >= 3 and heavy_perm_count <= heavy_perm_cap:
        GTh = GT[heavy_gi]
        cQh = GTh.mean(0)
        # enumerate element-blocked permutations of the model heavy atoms
        per_elem_perms = [list(itertools.permutations(elem_idx[e][1])) for e in heavy]
        scored = []
        for combo in itertools.product(*per_elem_perms):
            model_heavy_order = np.concatenate([np.array(p, int) for p in combo])
            Ph = MODEL[model_heavy_order]
            cPh = Ph.mean(0)
            R = _kabsch(Ph - cPh, GTh - cQh)
            aligned = (Ph - cPh) @ R.T + cQh
            hr = np.sqrt(((aligned - GTh) ** 2).sum(1).mean())
            scored.append((hr, cPh, R))
        scored.sort(key=lambda t: t[0])
        for hr, cPh, R in scored[:top_k]:
            candidates.append((MODEL - cPh) @ R.T + cQh)

    # --- random restarts (also the sole source when heavy frame is degenerate) ---
    rng = np.random.default_rng(seed)
    cP_all, cQ_all = MODEL.mean(0), GT.mean(0)
    for _ in range(n_random):
        R = _random_rotation(rng)
        candidates.append((MODEL - cP_all) @ R.T + cQ_all)

    best = np.inf
    for TM0 in candidates:
        rmsd, _ = _refine(GT, MODEL, elem_idx, TM0, max_iter=refine_iters)
        if rmsd < best:
            best = rmsd
    return best


# ----- graph RMSD (chemically strict) -------------------------------------

def graph_rmsd(gt_coords, model_coords):
    """RDKit GetBestRMS after bond perception on both mols. None if it fails."""
    try:
        g = coordinates_to_mol(gt_coords)
        m = coordinates_to_mol(model_coords)
        rdDetermineBonds.DetermineBonds(g, charge=0)
        rdDetermineBonds.DetermineBonds(m, charge=0)
        return float(rdMolAlign.GetBestRMS(m, g))
    except (ValueError, RuntimeError):
        return None


def best_rmsd(gt_coords, model_coords):
    """Both RMSDs plus a method tag for the primary value.

    Returns dict: {graph, assignment, rmsd, method}. `rmsd` prefers the
    chemically-strict graph value and falls back to the assignment lower bound;
    `method` is "graph" or "assignment".
    """
    g = graph_rmsd(gt_coords, model_coords)
    a = assignment_rmsd(gt_coords, model_coords)
    return {
        "graph": g,
        "assignment": a,
        "rmsd": g if g is not None else a,
        "method": "graph" if g is not None else "assignment",
    }
