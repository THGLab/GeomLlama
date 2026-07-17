"""Chemistry helpers shared by the stages.

IMPORTANT: this module must never import torch/vllm. The xTB stage forks a process pool,
and forking a CUDA-initialized process deadlocks -- which is the likeliest cause of the
notebook hangs. Keeping CUDA out of this import path is what makes the pool safe.
"""
from collections import Counter, defaultdict

import numpy as np
from rdkit import Chem, RDLogger

RDLogger.DisableLog('rdApp.*')

from . import config  # noqa: E402


def true_counts(smiles):
    """Element counts implied by the SMILES, with explicit hydrogens."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Counter(a.GetSymbol() for a in Chem.AddHs(mol).GetAtoms())


def matching_num_atoms(entry):
    """(counts_match, smiles_counts, coords_counts) for one parsed entry."""
    smi, coords = entry['smiles'], entry.get('coords')
    truth = true_counts(smi)
    if truth is None or coords is None:
        return False, Counter(), Counter()
    gen = Counter(row[0] for row in coords)
    return (truth == gen), truth, gen


def is_clash_free(positions, threshold=config.CLASH_THRESHOLD):
    """No two atoms closer than `threshold` Angstrom."""
    p = np.asarray(positions, dtype=float)
    if p.ndim != 2 or len(p) < 2:
        return False
    d = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    return bool(d.min() >= threshold)


def xyz_of(entry):
    """Coordinates of a parsed entry as an (N, 3) array, or None."""
    coords = entry.get('coords')
    if coords is None:
        return None
    try:
        return np.array([c[1:] for c in coords], dtype=float)
    except (TypeError, ValueError):
        return None


def group_by_smiles(parsed):
    """by_smiles[smiles] = [(trial_index, entry, xyz_or_None), ...] in generation order."""
    by = defaultdict(list)
    counter = defaultdict(int)
    for p in parsed:
        smi = p['smiles']
        t = counter[smi]
        counter[smi] += 1
        by[smi].append((t, p, xyz_of(p)))
    return by


def _clean(entry, xyz):
    return xyz is not None and is_clash_free(xyz) and matching_num_atoms(entry)[0]


def select_conformers(by_smiles, strategy=None):
    """Pick which generated conformers feed the comparison. See config.CONFORMER_SELECTION."""
    strategy = strategy or config.CONFORMER_SELECTION
    if callable(strategy):
        return strategy(by_smiles)

    out = []
    for _, trials in by_smiles.items():
        trials = sorted(trials, key=lambda twp: twp[0])
        if strategy == 'all':
            out += [p for _, p, _ in trials]
        elif strategy == 'all_clashfree':
            out += [p for _, p, xyz in trials if _clean(p, xyz)]
        elif strategy == 'first':
            if trials:
                out.append(trials[0][1])
        elif strategy == 'first_clashfree':
            pick = next((p for _, p, xyz in trials if _clean(p, xyz)), None)
            if pick is not None:
                out.append(pick)
        elif strategy == 'best_effort':
            # Give the model its best shot, but never hide a real failure:
            #   1. first conformer that is atom-matched AND clash-free
            #   2. else the LAST atom-matched one (they all clash)
            #   3. else the 10th (last) structure -- nothing matched, so this molecule
            #      is the reported atom mismatch
            # Atom-matching outranks clash-freedom on purpose: an atom mismatch is only
            # reported when ALL trials miss, so the mismatch bar means "the model never
            # once built the right molecule", not "the first try was wrong".
            pick = next((p for _, p, xyz in trials if _clean(p, xyz)), None)
            if pick is None:
                matched = [p for _, p, xyz in trials
                           if xyz is not None and matching_num_atoms(p)[0]]
                pick = matched[-1] if matched else trials[-1][1]
            out.append(pick)
        else:
            raise ValueError(f'unknown CONFORMER_SELECTION: {strategy!r}')
    return out


def report_quality(by_smiles, n_trials=None):
    """Per-trial geometry quality: mean +/- SE across the trial slots."""
    n_trials = n_trials or config.N_CONFORMERS_PER_SMILES
    cats = ['good', 'clash', 'mismatched_atoms', 'mismatched_hydrogens']
    counts = [{c: 0 for c in cats} for _ in range(n_trials)]
    for trials in by_smiles.values():
        for t, p, xyz in trials:
            if xyz is None or t >= n_trials:
                continue
            clashfree = is_clash_free(xyz)
            match, sc, cc = matching_num_atoms(p)
            if not clashfree:
                counts[t]['clash'] += 1
            if not match:
                counts[t]['mismatched_atoms'] += 1
                if sc.get('H', 0) != cc.get('H', 0):
                    counts[t]['mismatched_hydrogens'] += 1
            if clashfree and match:
                counts[t]['good'] += 1
    for cat in cats:
        arr = np.array([counts[t][cat] for t in range(n_trials)])
        se = arr.std(ddof=1) / np.sqrt(n_trials) if n_trials > 1 else 0.0
        print(f'  {cat:22s} {arr.mean():6.1f} +/- {se:.1f}')
    return counts
