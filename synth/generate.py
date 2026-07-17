"""Stage 1: SMILES -> 3D geometry with the fine-tuned geometry model. GPU.

Runs vLLM batch inference and parses each completion back to coordinates. Writes entries in
the SAME schema the xTB stage consumes ({smiles, raw_text, coords, parse_ok}), so output here
can be fed straight to `synth.xtb_stage`.

Run this as its OWN process, never in a notebook kernel: vLLM does not reliably release GPU
memory between in-process model loads, and a later fork of a CUDA-initialized process
deadlocks. Both are why the notebook hangs -- see README.md.

    python -m synth.generate --smiles "CC(=O)Oc1ccccc1C(=O)O" --n 3
    python -m synth.generate --from-smiles-file --limit 8      # first 8 of the ro4 set
"""
import argparse
import json
import os

from . import chem, config

# A few recognizable drug-like molecules, spanning small/rigid -> larger/floppier.
DEMO_SMILES = [
    ('aspirin',      'CC(=O)Oc1ccccc1C(=O)O'),
    ('caffeine',     'Cn1cnc2c1c(=O)n(C)c(=O)n2C'),
    ('paracetamol',  'CC(=O)Nc1ccc(O)cc1'),
    ('ibuprofen',    'CC(C)Cc1ccc(cc1)C(C)C(=O)O'),
]


def get_method(name):
    for m in config.METHODS:
        if m['name'] == name:
            return m
    raise SystemExit(f'unknown method {name!r}; known: {[m["name"] for m in config.METHODS]}')


def canonicalize(smiles_list):
    """RDKit canonical SMILES from the heavy-atom mol -- the form the model was trained on.

    Canonical SMILES are always derived from the heavy-atom mol (RemoveHs first) for prompt
    consistency. Feeding a non-canonical string is an off-distribution prompt, so do this
    before building the prompt, not after.
    """
    from rdkit import Chem
    out = []
    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            raise SystemExit(f'RDKit could not parse SMILES: {s!r}')
        out.append(Chem.MolToSmiles(Chem.RemoveHs(mol)))
    return out


def build_entries(smiles_list, texts_per_smiles, fmt):
    """Parse each completion into the {smiles, raw_text, coords, parse_ok} entry schema."""
    entries = []
    for smi, texts in zip(smiles_list, texts_per_smiles):
        for text in texts:
            try:
                coords = fmt.parse_output(text)
            except Exception:
                coords = None
            entries.append({'smiles': smi, 'raw_text': text, 'coords': coords,
                            'parse_ok': coords is not None})
    return entries


def report(entries, labels=None):
    """Per-generation validity: did it parse, is it the right molecule, does it clash?"""
    print(f'\n{"molecule":<14s} {"parse":<6s} {"atoms (SMILES/gen)":<20s} {"clash-free":<11s}')
    print('-' * 60)
    for i, e in enumerate(entries):
        name = (labels or {}).get(e['smiles'], e['smiles'])
        if not e['parse_ok']:
            print(f'{name[:13]:<14s} {"FAIL":<6s} {"-":<20s} {"-":<11s}')
            continue
        match, truth, gen = chem.matching_num_atoms(e)
        xyz = chem.xyz_of(e)
        clash_free = xyz is not None and chem.is_clash_free(xyz)
        counts = f'{sum(truth.values())}/{sum(gen.values())}'
        print(f'{name[:13]:<14s} {"ok":<6s} {counts:<20s} '
              f'{("yes" if clash_free else "NO"):<11s}'
              f'{"" if match else "   <- ATOM MISMATCH"}')

    ok = [e for e in entries if e['parse_ok']]
    matched = [e for e in ok if chem.matching_num_atoms(e)[0]]
    clean = [e for e in matched
             if chem.xyz_of(e) is not None and chem.is_clash_free(chem.xyz_of(e))]
    n = max(len(entries), 1)
    print(f'\n  parsed:      {len(ok)}/{len(entries)} ({100 * len(ok) / n:.0f}%)')
    print(f'  atom-matched:{len(matched):>3d}/{len(entries)} ({100 * len(matched) / n:.0f}%)')
    print(f'  + clash-free:{len(clean):>3d}/{len(entries)} ({100 * len(clean) / n:.0f}%)')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--method', default='z-matrix', help='method name from config.METHODS')
    ap.add_argument('--smiles', action='append', help='a SMILES string (repeatable)')
    ap.add_argument('--from-smiles-file', action='store_true',
                    help=f'draw SMILES from {config.SMILES_PATH}')
    ap.add_argument('--limit', type=int, default=4, help='how many SMILES to use')
    ap.add_argument('--n', type=int, default=1, help='conformers per SMILES')
    ap.add_argument('--temperature', type=float, default=config.GEOM_TEMPERATURE)
    ap.add_argument('--top-p', type=float, default=config.GEOM_TOP_P)
    ap.add_argument('--max-new-tokens', type=int, default=4096)
    ap.add_argument('--gpu-mem', type=float, default=0.90)
    ap.add_argument('--show-raw', action='store_true', help='print the raw Z-matrix text')
    ap.add_argument('--no-canonical', action='store_true',
                    help='prompt with the SMILES exactly as given (off-distribution; see '
                         'canonicalize())')
    ap.add_argument('--out', help='write entries to this JSON (xtb_stage-compatible)')
    args = ap.parse_args()

    method = get_method(args.method)
    labels = {}

    if args.smiles:
        smiles_list = args.smiles
    elif args.from_smiles_file:
        with open(config.SMILES_PATH) as f:
            smiles_list = list(dict.fromkeys(json.load(f)))[:args.limit]
    else:
        picked = DEMO_SMILES[:args.limit]
        smiles_list = [s for _, s in picked]
        labels = {s: n for n, s in picked}

    if not args.no_canonical:
        original = list(smiles_list)
        smiles_list = canonicalize(smiles_list)
        changed = [(o, c) for o, c in zip(original, smiles_list) if o != c]
        if changed:
            print(f'canonicalized {len(changed)}/{len(original)} SMILES:')
            for o, c in changed:
                print(f'  {o}\n  -> {c}')
            print()
        labels = {c: labels.get(o, labels.get(c, c)) for o, c in zip(original, smiles_list)}

    # Import here, not at module scope: keeps `python -m synth.generate --help` (and any
    # accidental import) from touching CUDA.
    from geomllama.data_formats import get_format
    from geomllama.inference import InferenceEngine
    from geomllama.prompts import make_inference_prompt

    fmt = get_format(method['format'])
    model_path = f'{config.REPO}/{method["path"]}'
    prompts = [make_inference_prompt(s, method['format']) for s in smiles_list]

    print(f'method:  {method["name"]}  (format={method["format"]}, mode={method["mode"]})')
    print(f'model:   {model_path}')
    print(f'SMILES:  {len(smiles_list)}   conformers each: {args.n}')
    print(f'sampling: T={args.temperature}, top_p={args.top_p}\n')

    engine = InferenceEngine(model_path, tensor_parallel_size=1, data_parallel_size=1,
                             gpu_memory_utilization=args.gpu_mem)
    results = engine.generate(prompts, max_new_tokens=args.max_new_tokens,
                              temperature=args.temperature, top_p=args.top_p, n=args.n)

    texts_per_smiles = [texts for _, texts in results]
    entries = build_entries(smiles_list, texts_per_smiles, fmt)

    if args.show_raw:
        for smi, texts in zip(smiles_list, texts_per_smiles):
            print(f'\n===== {labels.get(smi, smi)} =====')
            for i, t in enumerate(texts):
                print(f'--- conformer {i} ---\n{t.strip()}')

    report(entries, labels)

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, 'w') as f:
            json.dump(entries, f)
        print(f'\nwrote {len(entries)} entries -> {args.out}')


if __name__ == '__main__':
    main()
