"""Order-free RMSD via alternating Hungarian assignment and Kabsch rotation.

The order-free RMSD between two point clouds is the minimum, over all
element-preserving atom permutations, of the Kabsch-aligned RMSD. Formats with
no canonical atom order (ori_fh) require it: nothing tells us which generated
atom corresponds to which reference atom.

Why not rdMolAlign.GetBestRMS on bond-less mols? It enumerates atom mappings in
index order and stops at ``maxMatches`` (default 1e6). GEOM-QM9 molecules have
at most 10 heavy atoms (<= 362,880 permutations), so the cap never binds and
GetBestRMS is exact. GEOM-Drugs molecules have a MEDIAN of ~1e19 permutations;
the cap truncates the search to a ~1e-13 fraction of it, never leaves the
neighborhood of the identity mapping, and returns a value that does not respond
to raising the cap (1e3 and 1e7 give the same answer). Measured against this
method on real ori_fh Drugs generations, that value is inflated by a median
0.17 A -- 0.05 A on 11-24 heavy atoms, rising monotonically to 0.27 A on 29-34.

Method (Kromann's "reorder RMSD"): for a fixed rotation, the optimal
element-preserving assignment is a linear sum assignment on the squared-distance
cost matrix, solved exactly per element block; for a fixed assignment, the
optimal rotation is Kabsch. Alternate to a fixed point, restarting from the 24
proper alignments of the two principal-axis frames, and keep the best.

This is a heuristic: it returns the RMSD of an actual valid permutation, hence
an UPPER BOUND on the true minimum -- never below it. Validated against the
exactly-computable QM9 minimum on 320 pairs: exact to <1e-4 A on 315, mean
absolute error 3.8e-4 A, never below the true minimum. On Drugs it was never
worse than capped GetBestRMS on any of 144 pairs, and ~57x faster.
"""
import itertools

import numpy as np
from scipy.optimize import linear_sum_assignment


def _kabsch(P, Q):
    """Rotation taking centered P onto centered Q (proper, det=+1)."""
    V, _, Wt = np.linalg.svd(P.T @ Q)
    d = np.sign(np.linalg.det(V @ Wt))
    return V @ np.diag([1.0, 1.0, d]) @ Wt


def _rmsd(P, Q):
    return float(np.sqrt(((P - Q) ** 2).sum() / len(P)))


def _inertia_frames(X):
    """The 24 proper principal-axis frames (6 axis orders x 4 sign flips)."""
    _, V = np.linalg.eigh(X.T @ X)
    V = V[:, ::-1]
    out = []
    for order in itertools.permutations(range(3)):
        for s1 in (1, -1):
            for s2 in (1, -1):
                A = V[:, list(order)].copy()
                A[:, 0] *= s1
                A[:, 1] *= s2
                A[:, 2] = np.cross(A[:, 0], A[:, 1])
                out.append(A)
    return out


def kh_rmsd(el_p, P, el_q, Q, max_iter=60):
    """Order-free RMSD between point clouds (el_p, P) and (el_q, Q).

    Args:
        el_p, el_q: element symbol sequences; must share an element multiset.
        P, Q: (A, 3) coordinates.

    Returns:
        float RMSD of the best element-preserving permutation found (an upper
        bound on the true permutation minimum).
    """
    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    # A generated z-matrix can parse with the right formula yet decode to
    # non-finite coordinates (degenerate reference atoms -> nan). Those are
    # unusable geometries, not matches: report +inf so they lose every min()
    # and never contribute a spurious low RMSD. (The bonded GetBestRMS path
    # returns ~1e154 here, which silently overflows float32 to inf -- same
    # end state, by accident rather than design.)
    if not (np.isfinite(P).all() and np.isfinite(Q).all()):
        return float("inf")
    P = P - P.mean(0)
    Q = Q - Q.mean(0)
    el_p = np.asarray(el_p)
    el_q = np.asarray(el_q)

    blocks = []
    for e in np.unique(el_p):
        ip = np.where(el_p == e)[0]
        iq = np.where(el_q == e)[0]
        if len(ip) != len(iq):
            raise ValueError(f"element multiset mismatch for {e!r}: "
                             f"{len(ip)} vs {len(iq)}")
        blocks.append((ip, iq))

    # Degenerate point clouds (collinear, coincident atoms) can make eigh/svd
    # fail to converge. Such a start is simply unusable; skip it rather than
    # kill the caller. The identity start needs no decomposition, so `starts`
    # is never empty.
    starts = [np.eye(3)]
    try:
        fp, fq = _inertia_frames(P), _inertia_frames(Q)
        starts += [fq[0] @ f.T for f in fp] + [f @ fp[0].T for f in fq]
    except np.linalg.LinAlgError:
        pass

    best = np.inf
    for R in starts:
        try:
            perm = prev = None
            for _ in range(max_iter):
                Pr = P @ R
                perm = np.empty(len(P), dtype=int)
                for ip, iq in blocks:
                    cost = ((Pr[ip][:, None, :] - Q[iq][None, :, :]) ** 2).sum(-1)
                    rr, cc = linear_sum_assignment(cost)
                    perm[ip[rr]] = iq[cc]
                if prev is not None and np.array_equal(perm, prev):
                    break
                prev = perm
                Qm = Q[perm]
                Pc, Qc = P - P.mean(0), Qm - Qm.mean(0)
                R = _kabsch(Pc, Qc)
            Qm = Q[perm]
            Pc, Qc = P - P.mean(0), Qm - Qm.mean(0)
            best = min(best, _rmsd(Pc @ _kabsch(Pc, Qc), Qc))
        except np.linalg.LinAlgError:
            continue
    return best


def kh_rmsd_grid(gen_elems, gen_coords, ref_elems, ref_coords):
    """(n_ref, n_gen) order-free RMSD matrix, matching compute_mat_cov's layout."""
    mat = np.zeros((len(ref_coords), len(gen_coords)), dtype=np.float32)
    for j, (eg, Xg) in enumerate(zip(gen_elems, gen_coords)):
        for i, (er, Xr) in enumerate(zip(ref_elems, ref_coords)):
            mat[i, j] = kh_rmsd(eg, Xg, er, Xr)
    return mat
