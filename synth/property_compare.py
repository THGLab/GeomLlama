"""Property distributions: GEOM-Drugs vs SmileyLlama-Ro5 vs SmileyLlama-Ro4.

Based on `compare_to_geom()` from smileyllama-synthesis-template-fh.ipynb (cell 31), generalized
from two sets to three and stripped of the notebook globals.

GEOM-Drugs is the distribution the geometry model ("geom-llama") was TRAINED on. The two
SmileyLlama sets are what actually get fed to it at synthesis time, under two different prompts.
Where these disagree is where the geometry model is being asked to work off-distribution.

No filtering: all 1000 molecules per generated set.

    python -m synth.property_compare
"""
import argparse
import json
import os
import warnings
from collections import Counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from rdkit import Chem, RDLogger  # noqa: E402
from rdkit.Chem import QED, Lipinski, rdMolDescriptors  # noqa: E402
from tqdm import tqdm  # noqa: E402

from . import config  # noqa: E402

warnings.filterwarnings('ignore')
RDLogger.DisableLog('rdApp.*')

# Vendored into the repo so this figure is reproducible by anyone.
# 39,990 GEOM-Drugs training SMILES, md5 54a27c93403cb45b2e94bbec306844a5.
GEOM_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         'reference', 'geom_drugs_train_smiles.txt')
GEOM_SUBSAMPLE = 10_000
SEED = 42                 # fixes the 10k subsample of the 39,990 -- do not change

_RESULTS = f'{config.REPO}/results'

# (label, smiles_json | None for GEOM, color). Order is fixed -- never cycled or recolored.
#
# NOTE: the Ro4 SMILES really do live in a file called `ro5_smiles.json`. The notebook hardcoded
# that filename for BOTH runs, so the two sets differ only by parent directory. Not a typo.
#
# Palette validated with synth/_validate_palette.py: passes normal / deuteranopia / protanopia /
# tritanopia (CIEDE2000 >= 15 on every pair) and >= 3:1 contrast on white. The dark green works
# because it separates by LIGHTNESS -- a mid-green fails against blue under tritanopia (dE=1.6),
# and amber fails against the pink (dE=11.3). Do not swap these without re-running the validator.
SETS = [
    ('GEOM-Drugs',      None,                                        '#2196F3'),
    ('SmileyLlama-Ro5', f'{_RESULTS}/1k_smiles_ro5/ro5_smiles.json', '#E91E63'),
    ('SmileyLlama-Ro4', f'{_RESULTS}/1k_smiles_ro4/ro5_smiles.json', '#1B5E20'),
]
LABELS = [s[0] for s in SETS]
COLORS = {label: color for label, _, color in SETS}

PROPS = ['fraction_csp3', 'heavy_atoms', 'hbond_donor', 'hbond_acceptor',
         'num_ring_aliphatic', 'num_ring_aromatic', 'num_rotatable_bond', 'qed_default',
         'MW', 'ALOGP', 'PSA', 'ALERTS', 'hetero_prop', 'max_ring_size', 'tpsa']

XLIMS = dict(
    fraction_csp3=(0, 1), heavy_atoms=(5, 70), hbond_donor=(-1, 10), hbond_acceptor=(-1, 15),
    num_ring_aliphatic=(-1, 8), num_ring_aromatic=(-1, 8), num_rotatable_bond=(-1, 20),
    qed_default=(0, 1), MW=(0, 1000), ALOGP=(-5, 12), PSA=(0, 300), ALERTS=(-1, 5),
    hetero_prop=(0, 0.7), max_ring_size=(0, 10), tpsa=(0, 300))

TITLES = dict(
    fraction_csp3='Fraction of sp3 Carbons', heavy_atoms='Number of Heavy Atoms',
    hbond_donor='H-Bond Donors', hbond_acceptor='H-Bond Acceptors',
    num_ring_aliphatic='Aliphatic Rings', num_ring_aromatic='Aromatic Rings',
    num_rotatable_bond='Rotatable Bonds', qed_default='QED',
    MW='Molecular Weight [g/mol]', ALOGP='ALOGP', PSA='PSA [A^2]',
    ALERTS='Structural Alerts', hetero_prop='Heteroatom Proportion',
    max_ring_size='Max Ring Size', tpsa='TPSA [A^2]')

# Integer-valued counts -> dodged discrete bars. An overlapping step-histogram over integers is
# misleading (it implies a continuum between the bins), so every count lives here.
BAR_PROPS = {'hbond_donor', 'hbond_acceptor', 'num_ring_aliphatic', 'num_ring_aromatic',
             'num_rotatable_bond', 'ALERTS', 'max_ring_size'}


def _max_ring(mol):
    rings = [len(r) for r in mol.GetRingInfo().AtomRings()]
    return max(rings) if rings else 0


def props(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    try:
        qp = QED.properties(mol)
        qd = QED.default(mol)
    except Exception:
        return None
    ha = Lipinski.HeavyAtomCount(mol)
    return dict(
        SMILES=smi, fraction_csp3=Lipinski.FractionCSP3(mol), heavy_atoms=ha,
        hbond_donor=Lipinski.NumHDonors(mol), hbond_acceptor=Lipinski.NumHAcceptors(mol),
        num_ring_aliphatic=Lipinski.NumAliphaticRings(mol),
        num_ring_aromatic=Lipinski.NumAromaticRings(mol),
        num_rotatable_bond=Lipinski.NumRotatableBonds(mol),
        qed_default=qd, MW=qp.MW, ALOGP=qp.ALOGP, PSA=qp.PSA, ALERTS=qp.ALERTS,
        hetero_prop=Lipinski.NumHeteroatoms(mol) / ha if ha else 0.0,
        max_ring_size=_max_ring(mol), tpsa=rdMolDescriptors.CalcTPSA(mol))


def build_df(smiles_list, label):
    rows = []
    for s in tqdm(smiles_list, desc=label, leave=False):
        p = props(s.strip())
        if p is not None:
            p['dataset'] = label
            rows.append(p)
    return pd.DataFrame(rows)


def count_atoms(smiles_list):
    counts = Counter()
    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        if mol:
            for a in mol.GetAtoms():
                counts[a.GetSymbol()] += 1
    return counts


def load_geom():
    with open(GEOM_PATH) as f:
        allg = [line.strip() for line in f if line.strip()]
    rng = np.random.RandomState(SEED)
    idx = rng.choice(len(allg), min(GEOM_SUBSAMPLE, len(allg)), replace=False)
    return [allg[i] for i in idx]


def load_sets():
    """{label: [smiles, ...]} in the fixed SETS order."""
    out = {}
    for label, path, _ in SETS:
        if path is None:
            out[label] = load_geom()
            print(f'  {label:18s} {len(out[label]):>6,d} sampled of 39,990  <- {GEOM_PATH}')
        else:
            with open(path) as f:
                out[label] = json.load(f)
            print(f'  {label:18s} {len(out[label]):>6,d}  <- {path}')
    return out


def plot_grid(df, fname='property_comparison_3way'):
    sns.set_context('paper', font_scale=1.5)
    fig, axes = plt.subplots(5, 3, figsize=(20, 20))
    for ax, prop in zip(axes.flat, PROPS):
        if prop in BAR_PROPS:
            sns.histplot(data=df, x=prop, hue='dataset', hue_order=LABELS, discrete=True,
                         multiple='dodge', shrink=0.8, stat='probability', common_norm=False,
                         palette=COLORS, ax=ax)
        else:
            sns.histplot(data=df, x=prop, hue='dataset', hue_order=LABELS, element='step',
                         stat='density', common_norm=False, palette=COLORS, ax=ax)
        ax.set_xlim(XLIMS[prop])
        ax.set_title('')
        ax.set_xlabel(TITLES[prop], fontsize=18)
        ax.set_ylabel('')          # 'Density' / 'Probability' carry no information here
        ax.tick_params(axis='x', labelsize=14)
        ax.set_yticks([])
        ax.grid(True, ls='--', lw=0.6, alpha=0.7)
        if ax.legend_:
            ax.legend_.remove()

    # common_norm=False -> each set is normalized to ITSELF, so 10k GEOM molecules and 1k
    # generated ones are directly comparable despite the 10x size difference.
    fig.legend(handles=[plt.Line2D([0], [0], color=COLORS[n], lw=6, label=n) for n in LABELS],
               loc='upper center', ncol=3, fontsize=19, bbox_to_anchor=(0.5, 1.015))
    plt.tight_layout()

    # Top-level figures/, not config.FIG_DIR: this figure spans BOTH datasets, so it must
    # not live under figures/<dataset>/ and is unaffected by SYNTH_DATASET.
    out = f'{config.REPO}/figures/{fname}.png'
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'\nwrote {out}')


def report_elements(sets):
    """Element composition. The F / S / B / I enrichment lives here -- keep it even without pies."""
    counts = {label: count_atoms(smiles) for label, smiles in sets.items()}
    totals = {label: sum(c.values()) for label, c in counts.items()}
    elems = sorted(set().union(*[set(c) for c in counts.values()]))

    print('\n=== element composition (% of all atoms) ===')
    print(f'  {"elem":<6s}' + ''.join(f'{label:>20s}' for label in LABELS))
    for e in elems:
        row = f'  {e:<6s}'
        for label in LABELS:
            c, t = counts[label].get(e, 0), totals[label]
            row += f'{c:>11,d} {100 * c / t:>6.2f}%'
        print(row)

    geom_elems = set(counts[LABELS[0]])
    for label in LABELS[1:]:
        only = sorted(set(counts[label]) - geom_elems)
        if only:
            print(f'  !! in {label} but NOT in GEOM-Drugs: {only}  '
                  f'(the model has never seen these)')


def report_medians(df):
    print('\n=== medians ===')
    print(f'  {"property":<22s}' + ''.join(f'{label:>18s}' for label in LABELS))
    for p in PROPS:
        row = f'  {p:<22s}'
        for label in LABELS:
            row += f'{df[df.dataset == label][p].median():>18.2f}'
        print(row)


def main():
    argparse.ArgumentParser(description=__doc__.split('\n')[0]).parse_args()

    print('loading (no filtering -- all molecules)')
    sets = load_sets()

    df = pd.concat([build_df(smiles, label) for label, smiles in sets.items()],
                   ignore_index=True)

    report_medians(df)
    report_elements(sets)
    plot_grid(df)


if __name__ == '__main__':
    main()
