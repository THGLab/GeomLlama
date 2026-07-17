"""Fast bond-less element-permutation Kabsch RMSD core.

Reproduces rdMolAlign.GetBestRMS on BOND-LESS point clouds (as built by
coordinates_to_mol + RemoveHs in evaluation.compute_mat_cov): the minimum RMSD
over ALL element-preserving atom permutations. Because the point clouds carry no
bonds, GetBestRMS minimizes over every permutation that maps each atom to another
of the same element -- NOT graph automorphisms. Default maxMatches=1e6 means QM9
(<= 9! perms) is uncapped, so this is the exact permutation minimum.

Speedup vs per-pair GetBestRMS:
  * the element-permutation set depends ONLY on the element multiset, so it is
    identical for every (gen, ref) pair of a molecule -> enumerate once per mol.
  * closed-form 3x3 symmetric eigenvalues instead of np.linalg.svd on batches of
    tiny matrices.
  * no per-pair deepcopy / RemoveHs Python overhead.
"""
import itertools
import os
from collections import defaultdict

import numpy as np

try:
    import torch
    _HAS_TORCH = torch.cuda.is_available()
except Exception:  # pragma: no cover
    torch = None
    _HAS_TORCH = False

# Route to the GPU backend when the per-molecule Kabsch workload
# (n_perm * n_gen * n_ref) exceeds this; below it, NumPy launch overhead is
# lower than a CUDA dispatch. The workload -- not the raw permutation count --
# is what matters: a molecule with only 1440 perms but 98x49 conformers is
# ~7M solves and belongs on the GPU.
_GPU_WORK_THRESHOLD = int(os.environ.get("FAST_RMSD_GPU_WORK", "30000"))


def perm_set(elems):
    """Element-grouped permutation set for a heavy-atom element list.

    Returns (order, perms):
      order : indices that sort atoms into contiguous element groups.
      perms : (P, A) int array. Row r is a permutation of range(A) acting on the
              *grouped* coordinate array (i.e. coords already reordered by
              `order`); within each element block it applies one permutation of
              that block, holding other blocks fixed.
    """
    groups = defaultdict(list)
    for i, e in enumerate(elems):
        groups[e].append(i)
    order = [i for g in groups.values() for i in g]
    sizes = [len(g) for g in groups.values()]
    offs = np.cumsum([0] + sizes)
    axes = [list(itertools.permutations(range(s))) for s in sizes]
    perms = [
        [offs[k] + j for k, combo in enumerate(prod) for j in combo]
        for prod in itertools.product(*axes)
    ]
    return order, np.array(perms, dtype=np.int64)


def _sym3x3_eigvals(M):
    """Descending eigenvalues of a batch of 3x3 symmetric PSD matrices (N,3,3)."""
    p1 = M[:, 0, 1] ** 2 + M[:, 0, 2] ** 2 + M[:, 1, 2] ** 2
    q = (M[:, 0, 0] + M[:, 1, 1] + M[:, 2, 2]) / 3.0
    p2 = ((M[:, 0, 0] - q) ** 2 + (M[:, 1, 1] - q) ** 2 + (M[:, 2, 2] - q) ** 2
          + 2.0 * p1)
    p = np.sqrt(np.clip(p2 / 6.0, 1e-30, None))
    I = np.eye(3)
    B = (M - q[:, None, None] * I) / p[:, None, None]
    dB = (B[:, 0, 0] * (B[:, 1, 1] * B[:, 2, 2] - B[:, 1, 2] * B[:, 2, 1])
          - B[:, 0, 1] * (B[:, 1, 0] * B[:, 2, 2] - B[:, 1, 2] * B[:, 2, 0])
          + B[:, 0, 2] * (B[:, 1, 0] * B[:, 2, 1] - B[:, 1, 1] * B[:, 2, 0]))
    phi = np.arccos(np.clip(dB / 2.0, -1.0, 1.0)) / 3.0
    e1 = q + 2.0 * p * np.cos(phi)
    e3 = q + 2.0 * p * np.cos(phi + 2.0 * np.pi / 3.0)
    e2 = 3.0 * q - e1 - e3
    eig = np.stack([e1, e2, e3], axis=1)
    # Degenerate (near-diagonal) case: eigenvalues are the diagonal entries.
    diag = p1 < 1e-20
    if diag.any():
        d = np.sort(np.stack([M[:, 0, 0], M[:, 1, 1], M[:, 2, 2]], axis=1),
                    axis=1)[:, ::-1]
        eig[diag] = d[diag]
    return np.clip(eig, 0.0, None)


def _det3(H):
    return (H[:, 0, 0] * (H[:, 1, 1] * H[:, 2, 2] - H[:, 1, 2] * H[:, 2, 1])
            - H[:, 0, 1] * (H[:, 1, 0] * H[:, 2, 2] - H[:, 1, 2] * H[:, 2, 0])
            + H[:, 0, 2] * (H[:, 1, 0] * H[:, 2, 1] - H[:, 1, 1] * H[:, 2, 0]))


def _kabsch_batch(G, R):
    """Closed-form Kabsch RMSD for a batch of permuted gens vs one ref.

    G : (P, A, 3) permuted+centered gen coords (P permutations).
    R : (A, 3)    centered ref coords.
    Returns (P,) RMSD values.
    """
    A = R.shape[0]
    E0 = (G ** 2).sum((1, 2)) + (R ** 2).sum()
    H = np.einsum('pai,aj->pij', G, R)
    sv = np.sqrt(_sym3x3_eigvals(np.einsum('pij,pik->pjk', H, H)))
    sv[:, 2] *= np.sign(_det3(H))
    return np.sqrt(np.clip((E0 - 2.0 * sv.sum(1)) / A, 0.0, None))


def _group_by_element(elems):
    """Stable grouping indices so atoms are contiguous per element.

    Returns (order, sizes): `order` reindexes atoms into element blocks (blocks
    ordered by first appearance of each element); `sizes` are the block sizes.
    """
    groups = defaultdict(list)
    for i, e in enumerate(elems):
        groups[e].append(i)
    keys = list(groups.keys())
    order = [i for k in keys for i in groups[k]]
    sizes = [len(groups[k]) for k in keys]
    return np.asarray(order, dtype=np.int64), keys, sizes


def _perms_for_sizes(sizes):
    """Within-block permutation index array (P, A) for element block sizes."""
    offs = np.cumsum([0] + list(sizes))
    axes = [list(itertools.permutations(range(s))) for s in sizes]
    perms = [
        [offs[k] + j for k, combo in enumerate(prod) for j in combo]
        for prod in itertools.product(*axes)
    ]
    return np.array(perms, dtype=np.int64)


def _canon_stack(elems_list, coords_list, canon_elems):
    """Reorder each conformer's coords into canonical element-block order.

    CRITICAL for ori_fh: every generation (and every reference conformer) may
    emit its heavy atoms in a DIFFERENT arbitrary order, so each conformer must
    be grouped by ITS OWN element list -- not by conformer 0's. After this, all
    conformers share the identical element sequence `canon_elems`, so a single
    within-block permutation set applies uniformly.
    """
    out = np.empty((len(coords_list), len(canon_elems), 3), dtype=np.float64)
    for k, (el, xyz) in enumerate(zip(elems_list, coords_list)):
        order = sorted(range(len(el)), key=lambda i: el[i])  # stable by element
        out[k] = np.asarray(xyz, dtype=np.float64)[order]
    return out


def best_rms_grid_from_coords(gen_elems, gen_coords, ref_elems, ref_coords,
                              chunk=200_000):
    """Bond-less permutation-min RMSD grid, amortizing the perm set per molecule.

    Every gen and every ref conformer may list heavy atoms in a DIFFERENT
    arbitrary order (ori_fh has no canonical order); their element multisets are
    equal (guaranteed by the upstream atom-count filter). Each conformer is
    grouped by its own element list into a shared canonical block order, so a
    single within-block permutation set applied to gens (refs held fixed) yields
    exactly the bond-less GetBestRMS minimum over element-preserving
    permutations.

    Args:
        gen_elems: per-gen heavy-atom element symbol lists (list of lists), OR a
            single element list if every gen shares one order (back-compat).
        gen_coords: list of (A,3) arrays, gen heavy coords.
        ref_elems: per-ref element symbol lists, OR a single list.
        ref_coords: list of (A,3) arrays, ref heavy coords.
    Returns:
        (n_ref, n_gen) float32 RMSD matrix (min over element permutations).
    """
    # Accept either a single shared element list or per-conformer lists.
    def _as_per_conformer(elems, coords):
        if len(elems) and isinstance(elems[0], (list, tuple, np.ndarray)):
            return list(elems)
        return [elems] * len(coords)

    gen_el_list = _as_per_conformer(gen_elems, gen_coords)
    ref_el_list = _as_per_conformer(ref_elems, ref_coords)

    canon_elems = sorted(gen_el_list[0])          # element-sorted block order
    sizes = [len(list(v)) for _, v in itertools.groupby(canon_elems)]
    perms = _perms_for_sizes(sizes)
    P = perms.shape[0]
    A = len(canon_elems)

    G = _canon_stack(gen_el_list, gen_coords, canon_elems)   # (nG,A,3)
    R = _canon_stack(ref_el_list, ref_coords, canon_elems)   # (nR,A,3)
    G = G - G.mean(axis=1, keepdims=True)
    R = R - R.mean(axis=1, keepdims=True)
    nG, nR = len(G), len(R)

    if _HAS_TORCH and P * nG * nR >= _GPU_WORK_THRESHOLD:
        return _grid_torch(G, R, perms)

    out = np.zeros((nR, nG), dtype=np.float32)
    for gi in range(nG):
        Gp = G[gi][perms]  # (P, A, 3): all permutations of this gen conformer
        for ri in range(nR):
            Rr = R[ri]
            if P <= chunk:
                out[ri, gi] = float(_kabsch_batch(Gp, Rr).min())
            else:
                best = np.inf
                for s in range(0, P, chunk):
                    best = min(best, float(_kabsch_batch(Gp[s:s + chunk],
                                                         Rr).min()))
                out[ri, gi] = best
    return out


# ---------------------------------------------------------------------------
# GPU (torch) backend -- same permutation-minimum, batched on device.
# ---------------------------------------------------------------------------

def _grid_torch(G, R, perms, cell_cap=16_000_000):
    """(nR, nG) permutation-min RMSD grid on GPU.

    G : (nG, A, 3) grouped+centered gen coords (numpy).
    R : (nR, A, 3) grouped+centered ref coords (numpy).
    perms : (P, A) within-block permutation indices (numpy).

    All (gen, perm) rows are flattened and processed in large chunks (one big
    kernel per chunk, evaluated against every ref at once) so the GPU stays
    saturated; per-chunk row count is bounded by cell_cap/nR to cap memory.

    The RMSD of each (permuted-gen, ref) alignment is computed by the QCP
    (quaternion characteristic polynomial) method of Theobald (2005): the
    optimal RMSD is (E0 - 2*lambda_max)/A where lambda_max is the largest
    eigenvalue of the 4x4 key matrix built from the 3x3 correlation matrix H.
    lambda_max is found by Newton-Raphson on the (depressed, cubic-term-free)
    characteristic quartic -- pure arithmetic, no SVD or trig, so it is both
    numerically stable in fp64 (no catastrophic cancellation on small RMSDs,
    unlike the trace-of-singular-values form) and fast (no fp64 transcendentals,
    which throttle GPU SFU throughput).
    """
    dev = torch.device("cuda")
    dt = torch.float64
    A = G.shape[1]
    Gt = torch.as_tensor(G, dtype=dt, device=dev)   # (nG,A,3)
    Rt = torch.as_tensor(R, dtype=dt, device=dev)   # (nR,A,3)
    pt = torch.as_tensor(perms, dtype=torch.long, device=dev)  # (P,A)
    P, nR, nG = perms.shape[0], R.shape[0], G.shape[0]
    E0r = (Rt ** 2).sum((1, 2))                     # (nR,)

    # Bound the number of (perm x ref) 4x4 systems held at once by cell_cap.
    # For a single gen we may still need to chunk over the permutation axis
    # (a high-symmetry gen against many refs can be tens of millions of rows),
    # so iterate gens in groups AND perms in chunks, keeping a running min.
    rows_cap = max(1, cell_cap // max(nR, 1))        # max (g*pchunk) rows/kernel
    gchunk = max(1, min(nG, rows_cap // P))          # >=1 whole gens if they fit
    pchunk = P if gchunk >= 1 and gchunk * P <= rows_cap else max(1, rows_cap)
    qcp_fn = _get_qcp_fn()
    out = torch.empty((nR, nG), dtype=dt, device=dev)
    for gs in range(0, nG, gchunk):
        gsl = Gt[gs:gs + gchunk]                     # (g, A, 3)
        g = gsl.shape[0]
        best = torch.full((g, nR), float("inf"), dtype=dt, device=dev)
        for ps in range(0, P, pchunk):
            pc = pt[ps:ps + pchunk]                  # (pcn, A)
            pcn = pc.shape[0]
            Gp = gsl[:, pc].reshape(g * pcn, A, 3)   # (g*pcn, A, 3)
            E0g = (Gp ** 2).sum((1, 2))              # (g*pcn,)
            # H[n,r] = Gp[n]^T @ R[r]  -> (g*pcn, nR, 3, 3), then flatten the
            # (row, ref) axes into ONE batch dim so the compiled QCP kernel sees
            # a single dynamic leading dim (compiles exactly once, no reshape
            # storms / per-molecule recompiles).
            H = torch.einsum('nai,raj->nrij', Gp, Rt)
            E0 = E0g[:, None] + E0r[None, :]         # (g*pcn, nR)
            M = H.shape[0] * nR
            lam = qcp_fn(H.reshape(M, 3, 3), E0.reshape(M))
            rmsd = torch.sqrt(torch.clamp((E0.reshape(M) - 2.0 * lam) / A,
                                          min=0.0)).reshape(g, pcn, nR)
            best = torch.minimum(best, rmsd.amin(dim=1))
        out[:, gs:gs + g] = best.transpose(0, 1)
    return out.cpu().numpy().astype(np.float64)


_QCP_FN = None


def _get_qcp_fn():
    """Return _qcp_lambda_max, torch.compiled once (single dynamic batch dim).

    The elementwise QCP work (K build, power traces, 20-iter Newton) is ~150
    separate passes over large tensors in eager; fusing it collapses that to a
    couple of kernels, leaving only the irreducible correlation matmul. Flatten
    upstream guarantees one dynamic dim, so this compiles once for the whole run.
    """
    global _QCP_FN
    if _QCP_FN is None:
        if os.environ.get("FAST_RMSD_NO_COMPILE") == "1":
            _QCP_FN = _qcp_lambda_max
        else:
            try:
                _QCP_FN = torch.compile(_qcp_lambda_max, dynamic=True)
            except Exception:
                _QCP_FN = _qcp_lambda_max
    return _QCP_FN


def _det3_scalar(a, b, c, d, e, f, g, h, i):
    """Determinant of the 3x3 [[a,b,c],[d,e,f],[g,h,i]] from scalar/batched
    components (elementwise). Distinct from the matrix-arg _det3 used by the
    NumPy Kabsch path."""
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)


def _qcp_lambda_max(H, E0, iters=20):
    """Largest eigenvalue of the QCP 4x4 key matrix for correlation matrices H.

    H : (..., 3, 3) correlation matrices (sum_a gen_a (x) ref_a).
    E0: (...)       per-pair ||gen||^2 + ||ref||^2; the Newton seed lambda0 =
                    E0/2 is a rigorous upper bound on lambda_max.

    The characteristic polynomial of the traceless symmetric key matrix K is the
    depressed quartic p(l) = l^4 + c2 l^2 + c1 l + c0. Its coefficients are
    computed analytically straight from the nine components of H -- purely
    elementwise scalar ops, so the whole thing fuses under torch.compile and
    avoids materializing/matmul-ing a (N,4,4) K tensor (batched 4x4 matmul over
    tens of millions of tiny matrices is what made the trace form slow):
        c2 = -2*||H||_F^2,  c1 = -8*det(H),  c0 = det(K).
    These match the K power-trace identities to machine precision. Newton-Raphson
    from lambda0 = E0/2 converges monotonically down to lambda_max.
    """
    Sxx = H[..., 0, 0]; Sxy = H[..., 0, 1]; Sxz = H[..., 0, 2]
    Syx = H[..., 1, 0]; Syy = H[..., 1, 1]; Syz = H[..., 1, 2]
    Szx = H[..., 2, 0]; Szy = H[..., 2, 1]; Szz = H[..., 2, 2]

    sumS2 = (Sxx * Sxx + Sxy * Sxy + Sxz * Sxz
             + Syx * Syx + Syy * Syy + Syz * Syz
             + Szx * Szx + Szy * Szy + Szz * Szz)
    c2 = -2.0 * sumS2
    detH = (Sxx * (Syy * Szz - Syz * Szy)
            - Sxy * (Syx * Szz - Syz * Szx)
            + Sxz * (Syx * Szy - Syy * Szx))
    c1 = -8.0 * detH
    # c0 = det(K), K the traceless symmetric quaternion key matrix.
    a = Sxx + Syy + Szz
    b = Syz - Szy; c = Szx - Sxz; d = Sxy - Syx
    e = Sxx - Syy - Szz; f = Sxy + Syx; gg = Szx + Sxz
    hh = -Sxx + Syy - Szz; ii = Syz + Szy; jj = -Sxx - Syy + Szz
    M00 = _det3_scalar(e, f, gg, f, hh, ii, gg, ii, jj)
    M01 = _det3_scalar(b, f, gg, c, hh, ii, d, ii, jj)
    M02 = _det3_scalar(b, e, gg, c, f, ii, d, gg, jj)
    M03 = _det3_scalar(b, e, f, c, f, hh, d, gg, ii)
    c0 = a * M00 - b * M01 + c * M02 - d * M03

    lam = E0 / 2.0
    for _ in range(iters):
        lam2 = lam * lam
        p = lam2 * lam2 + c2 * lam2 + c1 * lam + c0
        dp = 4.0 * lam2 * lam + 2.0 * c2 * lam + c1
        lam = lam - p / dp
    return lam
