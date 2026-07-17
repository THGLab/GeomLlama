"""Stage 3: metrics + figures. No GPU, no model, no forking after CUDA -- safe anywhere.

Builds the combined 2x2 panel published as Figure 3 of arXiv:2607.13350v1 (Ro4) and its
Ro5 counterpart: RMSD histogram + validity bars, energy drop, and RMSD-vs-dE scatter.

    python -m synth.plots                                  # ro4 (config.DATASET default)
    SYNTH_DATASET=1k_smiles_ro5 python -m synth.plots      # ro5
    python -m synth.plots --source notebook                # cross-check vs the notebook

The xTB source is explicit and never auto-substituted -- see load_entries().
Writes figures + CSVs + a provenance sidecar into figures/<dataset>/.
"""
import argparse
import json
import os
import pickle

import matplotlib
matplotlib.use('Agg')          # headless: no kernel, no display, cannot hang on a GUI
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from rdkit import Chem  # noqa: E402
from rdkit.Chem import AllChem  # noqa: E402
from rdkit.Geometry import Point3D  # noqa: E402

from . import chem, config  # noqa: E402
from .xtb_stage import xtb_cache_path  # noqa: E402

plt.rcParams.update({
    'figure.dpi': 150,
    'axes.spines.top': False,     # recessive axes: the data is the ink
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.25,
    'grid.linewidth': 0.6,
    'font.size': 10,
})


# --------------------------------------------------------------------------- loading

def load_entries(name, source='published'):
    """xTB-annotated entries for one method. Returns (entries, source_path).

    Sources are explicit and never silently substituted for one another. The old 'auto'
    mode preferred our cache and fell back to the notebook's pickle -- which is how the
    published panels ended up sourcing z-matrix from this pipeline and rdkit from the
    notebook, two provenances in one figure with nothing recording it.

      published : cache/<dataset>/xtb_<name>.pkl -- the artifact of record (the paper)
      notebook  : the original notebook's full_results_<name>.pkl (kept for cross-checking)

    Both now agree for rdkit (verified: median RMSD 0.7152, dE 0.0593, identical coords on
    1000/1000), so 'published' is the single source for every series.
    """
    paths = {'published': xtb_cache_path(name),
             'notebook': f'{config.RESULTS_DIR}/full_results_{name}.pkl'}
    if source not in paths:
        raise ValueError(f'unknown source {source!r}; choose from {sorted(paths)}')

    path = paths[source]
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'no {source!r} xTB results for {name!r} at:\n  {path}\n'
            f'Run:  python -m synth.xtb_stage --method {name}\n'
            f'(Not falling back to another source -- that would mix provenances silently.)')
    with open(path, 'rb') as f:
        entries = pickle.load(f)
    print(f'[{name}] {len(entries)} entries <- {path}')
    return entries, path


# --------------------------------------------------------------------------- metrics

def _mol_with_coords(smiles, positions):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    positions = np.asarray(positions, dtype=float)
    if mol.GetNumAtoms() != len(positions):
        return None
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i, (x, y, z) in enumerate(positions):
        conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))
    mol.AddConformer(conf, assignId=True)
    return mol


def best_rmsd(smiles, a, b):
    """Symmetry-aware RMSD with optimal rigid-body alignment.

    Falls back to direct positional RMSD when RDKit cannot build the molecule -- which is
    exactly the atom-count-mismatch case, so those rows are still scored.
    """
    ma, mb = _mol_with_coords(smiles, a), _mol_with_coords(smiles, b)
    if ma is None or mb is None:
        d = np.asarray(a) - np.asarray(b)
        return float(np.sqrt((d ** 2).sum(axis=1).mean()))
    return float(AllChem.GetBestRMS(ma, mb))


def stats_frame(entries, name):
    """Per-molecule metrics + the validity counters that drive the overflow bars."""
    rows = []
    for e in entries:
        x = e.get('xtb', {}) or {}
        row = {'smiles': e['smiles'], 'parse_ok': e.get('parse_ok', False),
               'xtb_ok': bool(x.get('ok'))}
        if x.get('ok'):
            row.update(
                rmsd=best_rmsd(e['smiles'], x['positions_init'], x['positions_final']),
                energy_init=x['energy_init'], energy_final=x['energy_final'],
                energy_drop=x['energy_drop'], converged=x['converged'], n_iter=x['n_iter'])
        rows.append(row)

    df = pd.DataFrame(rows)
    valid = df[df['xtb_ok']]

    # invalid  = no successful xTB run (parse failure, or xTB errored/timed out)
    # mismatch = parse-ok but atom count != SMILES. xTB happily optimizes these, so they
    #            DO appear in the histograms -- the bar surfaces that hidden share.
    mismatch = sum(1 for e in entries
                   if e.get('parse_ok') and e.get('coords') is not None
                   and not chem.matching_num_atoms(e)[0])

    print(f'\n=== {name} ===')
    print(f'  selected:      {len(df)}')
    print(f'  parse OK:      {int(df["parse_ok"].sum())} ({100 * df["parse_ok"].mean():.1f}%)')
    print(f'  xTB ran:       {len(valid)} ({100 * len(valid) / max(len(df), 1):.1f}%)')
    if len(valid):
        print(f'  xTB converged: {int(valid["converged"].sum())} '
              f'({100 * valid["converged"].mean():.1f}%)')
        print(f'  median RMSD:   {valid["rmsd"].median():.4f} A')
        print(f'  median E drop: {valid["energy_drop"].median():.4f} Hartree')
    print(f'  atom mismatch: {mismatch} ({100 * mismatch / max(len(df), 1):.1f}%)')

    os.makedirs(config.FIG_DIR, exist_ok=True)
    df.to_csv(f'{config.FIG_DIR}/metrics_{name}.csv', index=False)
    return {'valid': valid, 'total': len(df),
            'invalid': len(df) - len(valid), 'mismatch': mismatch}


# --------------------------------------------------------------------------- figures

def shared_log_bins(series, n_bins=30, pad=0.05):
    vals = np.concatenate([np.asarray(s)[np.asarray(s) > 0] for s in series])
    lo, hi = np.log10(vals.min()), np.log10(vals.max())
    span = hi - lo
    lo, hi = lo - pad * span, hi + pad * span
    return np.logspace(lo, hi, n_bins + 1), (10 ** lo, 10 ** hi)


# Typography for the combined panel. Bumped well above matplotlib defaults so the figure
# stays legible at half-column width in a paper.
PANEL_LABEL_FS = 17
PANEL_TICK_FS = 15
PANEL_LEGEND_FS = 15


def _hist_panel(ax, stats, names, column, xlabel, n_bins=30, legend=False):
    """One normalized log-x histogram with per-method medians. Returns True if it drew."""
    colors = config.method_colors()
    series = [stats[n]['valid'][column][stats[n]['valid'][column] > 0] for n in names]
    series = [s for s in series if len(s)]
    if not series:
        return False
    bins, xlim = shared_log_bins(series, n_bins=n_bins)

    for n in names:
        st = stats[n]
        v, tot = st['valid'], st['total']
        vals = v[column][v[column] > 0].to_numpy()
        c = colors[n]
        if not len(vals):
            continue
        med = float(np.median(vals))
        ax.hist(vals, bins=bins, weights=np.ones(len(vals)) / tot,
                histtype='stepfilled', alpha=0.40, color=c, edgecolor=c, linewidth=1.6,
                label=f'{n}  (median={med:.3g})')
        ax.axvline(med, color=c, linestyle='--', linewidth=1.4, alpha=0.9)

    ax.set_xscale('log')
    ax.set_xlim(xlim)
    ax.set_xlabel(xlabel, fontsize=PANEL_LABEL_FS)
    ax.set_ylabel('Fraction of generations', fontsize=PANEL_LABEL_FS)
    ax.tick_params(labelsize=PANEL_TICK_FS)
    if legend:
        ax.legend(fontsize=PANEL_LEGEND_FS, loc='upper left', frameon=False)
    return True


def plot_combined(stats, fname='combined_panel'):
    """One 2x2 figure:

        top-left     RMSD pre->post opt
        top-right    failure modes (no xTB / atom mismatch)
        bottom-left  energy drop
        bottom-right RMSD vs energy drop correlation

    The two histograms sit in the same (left) column, so their FRAMES align. Their x-axes stay
    independent (A vs Hartree, different decades), so tick marks do NOT align -- that is
    intended, not an oversight.
    """
    names = [n for n in config.plot_order() if n in stats and len(stats[n]['valid'])]
    if not names:
        print('  no data for the combined panel')
        return
    colors = config.method_colors()

    fig, axes = plt.subplots(2, 2, figsize=(14, 9.5), constrained_layout=True,
                             gridspec_kw={'width_ratios': [1.55, 1]})
    (ax_rmsd, ax_fail), (ax_edrop, ax_corr) = axes

    # --- top-right: failure modes ------------------------------------------------
    cats = [('no\nxTB', 'invalid'), ('atom\nmismatch', 'mismatch')]
    nb = len(names)
    bw = 0.8 / max(nb, 1)
    for ci, (_, key) in enumerate(cats):
        for j, n in enumerate(names):
            tot = max(stats[n]['total'], 1)
            frac = stats[n][key] / tot
            x = ci + (j - (nb - 1) / 2) * bw
            ax_fail.bar(x, frac, width=bw * 0.9, color=colors[n], alpha=0.75,
                        edgecolor=colors[n])
            if frac > 0:   # direct-label the bars; a bare axis makes these hard to read
                ax_fail.text(x, frac, f'{100 * frac:.1f}%', ha='center', va='bottom',
                             fontsize=PANEL_TICK_FS, color='#333333')
    ax_fail.set_xticks(range(len(cats)))
    ax_fail.set_xticklabels([c for c, _ in cats])
    ax_fail.set_xlim(-0.5, len(cats) - 0.5)
    ax_fail.set_ylabel('Fraction of generations', fontsize=PANEL_LABEL_FS)
    ax_fail.tick_params(labelsize=PANEL_TICK_FS)
    ax_fail.grid(axis='x', visible=False)
    ax_fail.margins(y=0.18)

    # --- top-left: RMSD ----------------------------------------------------------
    _hist_panel(ax_rmsd, stats, names, 'rmsd', 'RMSD pre -> post opt (A)', legend=True)

    # --- bottom-left: energy drop (frame aligned with RMSD above) -----------------
    _hist_panel(ax_edrop, stats, names, 'energy_drop', 'Energy drop (Hartree)', legend=True)

    # --- bottom-right: correlation -----------------------------------------------
    for n in names:
        v = stats[n]['valid']
        d = v[(v['rmsd'] > 0) & (v['energy_drop'] > 0)]
        x, y = d['rmsd'].to_numpy(), d['energy_drop'].to_numpy()
        ax_corr.scatter(x, y, s=16, alpha=0.45, edgecolor='none', color=colors[n], label=n)
    ax_corr.set_xscale('log')
    ax_corr.set_yscale('log')
    ax_corr.set_xlabel('RMSD pre -> post opt (A)', fontsize=PANEL_LABEL_FS)
    ax_corr.set_ylabel('Energy drop (Hartree)', fontsize=PANEL_LABEL_FS)
    ax_corr.tick_params(labelsize=PANEL_TICK_FS)
    ax_corr.legend(frameon=False, fontsize=PANEL_LEGEND_FS, loc='upper left')

    # No metadata/timestamp is written into the PNG on purpose: it keeps the figure
    # byte-reproducible against the published one. Provenance goes in the sidecar
    # (write_provenance) so it can never silently alter the artifact it describes.
    out = f'{config.FIG_DIR}/{fname}.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {out}')


# --------------------------------------------------------------------------- stage

def write_provenance(sources, stats):
    """Sidecar recording exactly how this figure was built.

    Deliberately written next to the figure rather than into it, so the PNG stays
    byte-comparable against the published one.
    """
    prov = {
        'dataset': config.DATASET,
        'conformer_selection': config.CONFORMER_SELECTION,
        'reproduces_published_paper': config.CONFORMER_SELECTION == config.PUBLISHED_SELECTION,
        'published_paper': 'arXiv:2607.13350v1',
        'sources': sources,
        'xtb': {'max_iter': config.XTB_MAX_ITER, 'accuracy': config.XTB_ACCURACY,
                'level': config.XTB_LEVEL, 'clash_threshold': config.CLASH_THRESHOLD},
        'rdkit_seed': config.RDKIT_SEED,
        'n_conformers_per_smiles': config.N_CONFORMERS_PER_SMILES,
        'results': {n: {'total': s['total'], 'no_xtb': s['invalid'],
                        'atom_mismatch': s['mismatch'],
                        'median_rmsd': (round(float(s['valid']['rmsd'].median()), 4)
                                        if len(s['valid']) else None),
                        'median_energy_drop': (round(float(s['valid']['energy_drop'].median()), 4)
                                               if len(s['valid']) else None)}
                    for n, s in stats.items()},
    }
    out = f'{config.FIG_DIR}/provenance.json'
    with open(out, 'w') as f:
        json.dump(prov, f, indent=2)
    print(f'  wrote {out}')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--source', choices=['published', 'notebook'], default='published',
                    help="'published' = cache/<dataset>/xtb_<name>.pkl (the paper's data). "
                         "'notebook' = the original notebook's full_results_*.pkl. "
                         "No auto-fallback: a missing source is an error, not a substitution.")
    args = ap.parse_args()

    os.makedirs(config.FIG_DIR, exist_ok=True)
    print(f'reading xTB results (source={args.source})')
    print(f'writing figures -> {config.FIG_DIR}')
    print(f'conformer selection: {config.CONFORMER_SELECTION}')
    if config.CONFORMER_SELECTION != config.PUBLISHED_SELECTION:
        print(f'  *** WARNING: this is NOT the published setting '
              f'({config.PUBLISHED_SELECTION!r}). These figures will NOT match arXiv:2607.13350v1.')
    print()

    stats, sources = {}, {}
    for name in config.plot_order():
        entries, path = load_entries(name, args.source)
        sources[name] = path
        stats[name] = stats_frame(entries, name)

    print('\n=== validity summary (fraction of the selected set) ===')
    for name in config.plot_order():
        st = stats[name]
        tot = max(st['total'], 1)
        print(f'  {name:20s} valid={len(st["valid"])}/{st["total"]}  '
              f'no-xTB={st["invalid"]} ({100 * st["invalid"] / tot:.0f}%)  '
              f'atom-mismatch={st["mismatch"]} ({100 * st["mismatch"] / tot:.0f}%)')

    print('\n=== figures ===')
    plot_combined(stats)
    write_provenance(sources, stats)


if __name__ == '__main__':
    main()
