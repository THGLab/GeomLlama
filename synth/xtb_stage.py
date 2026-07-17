"""Stage 2: GFN2-xTB optimization of the selected conformers. CACHED.

The notebook never cached xTB output -- it recomputed hours of optimization on every run
and only wrote results at the metrics step, where a re-run silently overwrote them. Here
each method's xTB results land in cache/xtb_<name>.pkl right after optimization, so a
re-run is free and a prior run is never destroyed.

This module deliberately does NOT import torch/vllm. It forks a process pool, and forking
a CUDA-initialized process deadlocks -- run this in its own process, never in a kernel
that has loaded a model.

    python -m synth.xtb_stage --method z-matrix
    python -m synth.xtb_stage --all
"""
import argparse
import json
import multiprocessing as mp
import os
import pickle
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np
from rdkit import Chem
from tqdm import tqdm

from . import chem, config

# xTB binary: use XTB_BIN env var if set, otherwise assume 'xtb' is on PATH.
_XTB_BIN = os.environ.get('XTB_BIN', 'xtb')

_PT = Chem.GetPeriodicTable()


def xtb_cache_path(name):
    return f'{config.CACHE_DIR}/xtb_{name}.pkl'


# --------------------------------------------------------------------------- xtb driver

def _write_xyz(path, coords):
    with open(path, 'w') as f:
        f.write(f'{len(coords)}\n\n')
        for el, x, y, z in coords:
            f.write(f'{el} {float(x):.8f} {float(y):.8f} {float(z):.8f}\n')


def _read_xyz(path):
    with open(path) as f:
        lines = f.readlines()
    n = int(lines[0])
    out = []
    for line in lines[2:2 + n]:
        p = line.split()
        out.append((p[0], float(p[1]), float(p[2]), float(p[3])))
    return out


def _parse_opt_log(path):
    """xtbopt.log is a multi-frame xyz trajectory; the comment line carries 'energy: E'."""
    energies = []
    with open(path) as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        try:
            n = int(lines[i].strip())
        except (ValueError, IndexError):
            break
        comment = lines[i + 1] if i + 1 < len(lines) else ''
        m = re.search(r'energy:\s*(-?\d+\.\d+)', comment)
        if m:
            energies.append(float(m.group(1)))
        i += 2 + n
    return energies


def xtb_optimize(coords):
    """Run `xtb --opt` in a temp dir. Returns a result dict (schema matches the notebook)."""
    workdir = tempfile.mkdtemp(prefix='xtb_')
    try:
        _write_xyz(os.path.join(workdir, 'mol.xyz'), coords)
        cmd = [_XTB_BIN, 'mol.xyz', '--opt', config.XTB_LEVEL, '--gfn', '2',
               '--iterations', str(config.XTB_MAX_ITER), '--acc', str(config.XTB_ACCURACY)]
        env = os.environ.copy()
        env['OMP_NUM_THREADS'] = '1'   # never oversubscribe: the pool owns the parallelism
        env['MKL_NUM_THREADS'] = '1'
        try:
            proc = subprocess.run(cmd, cwd=workdir, env=env, capture_output=True,
                                  text=True, timeout=config.XTB_TIMEOUT)
        except subprocess.TimeoutExpired:
            return {'ok': False, 'error': 'xtb timeout'}

        stdout = proc.stdout
        opt_xyz = os.path.join(workdir, 'xtbopt.xyz')
        opt_log = os.path.join(workdir, 'xtbopt.log')
        if not os.path.exists(opt_xyz):
            return {'ok': False, 'error': f'no xtbopt.xyz (rc={proc.returncode}): {stdout[-300:]}'}

        final = _read_xyz(opt_xyz)
        energies = _parse_opt_log(opt_log) if os.path.exists(opt_log) else []
        if len(energies) >= 2:
            e0, ef, n_iter = energies[0], energies[-1], len(energies) - 1
        else:
            totals = [float(m.group(1))
                      for m in re.finditer(r'TOTAL ENERGY\s+(-?\d+\.\d+)', stdout)]
            if len(totals) >= 2:
                e0, ef, n_iter = totals[0], totals[-1], len(totals) - 1
            elif len(totals) == 1:
                e0 = ef = totals[0]
                n_iter = 0
            else:
                return {'ok': False, 'error': 'could not parse energies'}

        return {
            'ok': True,
            'energy_init': e0,
            'energy_final': ef,
            'energy_drop': e0 - ef,
            'converged': 'GEOMETRY OPTIMIZATION CONVERGED' in stdout,
            'n_iter': n_iter,
            'positions_init': np.array([[float(x), float(y), float(z)]
                                        for _, x, y, z in coords]),
            'positions_final': np.array([[x, y, z] for _, x, y, z in final]),
            'numbers': np.array([_PT.GetAtomicNumber(el) for el, *_ in coords], dtype=np.int32),
        }
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def _worker(args):
    idx, entry = args
    if not entry.get('parse_ok'):
        return idx, {'ok': False, 'error': 'parse failed'}
    try:
        return idx, xtb_optimize(entry['coords'])
    except Exception as ex:  # a single bad molecule must not kill the batch
        return idx, {'ok': False, 'error': str(ex)}


def xtb_all(entries, desc='xTB'):
    """Optimize every entry, in parallel. Uses a SPAWN pool -- fork would inherit any CUDA
    context and deadlock, which is what hangs the notebook."""
    if not entries:
        return entries
    n_workers = min(config.XTB_WORKERS, max(1, len(entries)))
    ctx = mp.get_context('spawn')
    with ctx.Pool(processes=n_workers) as pool:
        for idx, result in tqdm(pool.imap_unordered(_worker, list(enumerate(entries))),
                                total=len(entries), desc=desc):
            entries[idx]['xtb'] = result

    ok = sum(1 for e in entries if e['xtb'].get('ok'))
    conv = sum(1 for e in entries if e['xtb'].get('ok') and e['xtb'].get('converged'))
    print(f'  xTB ran:       {ok}/{len(entries)}')
    print(f'  xTB converged: {conv}/{len(entries)}')
    return entries


# --------------------------------------------------------------------------- rdkit baseline

def _rdkit_worker(args):
    from rdkit.Chem import AllChem
    idx, smiles = args
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return idx, None
    mol = Chem.AddHs(mol)
    p = AllChem.ETKDGv3()
    p.randomSeed = config.RDKIT_SEED
    if AllChem.EmbedMolecule(mol, p) != 0:
        p.useRandomCoords = True   # helps awkward / macrocyclic systems
        if AllChem.EmbedMolecule(mol, p) != 0:
            return idx, None
    conf = mol.GetConformer()
    return idx, [(a.GetSymbol(), *(float(v) for v in (conf.GetAtomPosition(a.GetIdx()).x,
                                                      conf.GetAtomPosition(a.GetIdx()).y,
                                                      conf.GetAtomPosition(a.GetIdx()).z)))
                 for a in mol.GetAtoms()]


def build_rdkit_entries(smiles_list):
    """One deterministic ETKDGv3 conformer per unique SMILES. Invalid = embedding failure."""
    unique = list(dict.fromkeys(smiles_list))
    out = [None] * len(unique)
    ctx = mp.get_context('spawn')
    with ctx.Pool(processes=min(config.XTB_WORKERS, max(1, len(unique)))) as pool:
        for idx, coords in tqdm(pool.imap_unordered(_rdkit_worker, list(enumerate(unique))),
                                total=len(unique), desc='RDKit embed'):
            out[idx] = {'smiles': unique[idx], 'raw_text': '', 'coords': coords,
                        'parse_ok': coords is not None}
    ok = sum(1 for e in out if e['parse_ok'])
    print(f'  RDKit embedded: {ok}/{len(out)} unique SMILES')
    return out


# --------------------------------------------------------------------------- stage

def load_selected(method):
    """Parsed generations -> grouped by SMILES -> the selected conformer set."""
    with open(method['cache']) as f:
        parsed = json.load(f)
    ok = sum(1 for p in parsed if p['parse_ok'])
    print(f'  loaded {len(parsed)} parsed generations, {ok} parse OK '
          f'({100 * ok / max(len(parsed), 1):.1f}%)')

    by = chem.group_by_smiles(parsed)
    print(f'  per-trial quality over {len(by)} SMILES:')
    chem.report_quality(by)

    sel = chem.select_conformers(by)
    print(f"  selection '{config.CONFORMER_SELECTION}': {len(sel)} conformers "
          f'from {len(by)} unique SMILES')
    return sel


def run_method(method, force=False):
    name = method['name']
    path = xtb_cache_path(name)
    if os.path.exists(path) and not force:
        print(f'[{name}] cached -> {path}  (--force to recompute)')
        return
    print(f'\n===== {name} =====')
    # No hardness filter here on purpose: the xTB cache stays COMPLETE, and the filter is
    # applied as a view at plot time. That way toggling it never costs a recompute.
    sel = load_selected(method)
    sel = xtb_all(sel, desc=f'xTB {name}')
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(sel, f)
    print(f'[{name}] wrote {path}')


def run_rdkit(force=False):
    path = xtb_cache_path(config.RDKIT_NAME)
    if os.path.exists(path) and not force:
        print(f'[{config.RDKIT_NAME}] cached -> {path}  (--force to recompute)')
        return
    print(f'\n===== {config.RDKIT_NAME} =====')
    with open(config.SMILES_PATH) as f:
        smiles = json.load(f)
    entries = build_rdkit_entries(smiles)   # unfiltered; see run_method
    entries = xtb_all(entries, desc='xTB rdkit')
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(entries, f)
    print(f'[{config.RDKIT_NAME}] wrote {path}')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--method', help='method name from config.METHODS, or "rdkit"')
    ap.add_argument('--all', action='store_true', help='every configured method + rdkit')
    ap.add_argument('--force', action='store_true', help='recompute even if cached')
    args = ap.parse_args()

    if args.all:
        for m in config.METHODS:
            run_method(m, args.force)
        if config.INCLUDE_RDKIT:
            run_rdkit(args.force)
        return

    if not args.method:
        ap.error('pass --method <name> or --all')
    if args.method == config.RDKIT_NAME:
        run_rdkit(args.force)
        return
    for m in config.METHODS:
        if m['name'] == args.method:
            run_method(m, args.force)
            return
    ap.error(f'unknown method {args.method!r}; '
             f'known: {[m["name"] for m in config.METHODS] + [config.RDKIT_NAME]}')


if __name__ == '__main__':
    sys.exit(main())
