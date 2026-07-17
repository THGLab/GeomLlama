#!/usr/bin/env python3
"""COV / MAT benchmark for a template_fh (or labeled_xyz) model on the
GEOM-Drugs test set (drugs_processed/test_data_200.pkl).

Standard GEOM ensemble protocol: for each test molecule, generate 2x its number
of reference conformers, then compute COV / MAT (Recall and Precision) from the
heavy-atom best-fit RMSD matrix.

COV threshold defaults to 1.25 A (the GEOM-Drugs convention; GEOM-QM9 uses 0.5).

Generated conformers are turned into BONDED RDKit mols (SMILES template + the
generated geometry, atoms placed in the model's canonical order) so GetBestRMS
performs correct symmetry-aware atom matching.

Two generation modes:
  free       (default) one-shot batched generation across all visible GPUs
             (data-parallel); fast. Parses each output; wrong-atom-count
             generations are dropped and counted (atom_match_pct reported).
  scaffolded (--scaffolded) harness emits the label+refs scaffold, model fills
             only the internal coordinates; 0% atom miscount by construction.
             Single replica (slower).

Usage (run AFTER merging the LoRA adapter to a full model dir):
  python benchmark_geom_drugs.py --model <merged_dir> \
      --test-pkl data/geom_drugs_small/drugs_processed/test_data_200.pkl \
      --format template_fh --threshold 1.25 --output results/geom_drugs.json
"""
import argparse, json, os, pickle, sys, time
from collections import Counter
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from rdkit import Chem
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")
from rdkit.Geometry import Point3D

from geomllama.data_formats import get_format
from geomllama.connectivity import get_hydrogen_atom_order, mol_with_hs_from_smiles
from geomllama.evaluation import (compute_mat_cov, compute_mat_cov_fast,
                                      compute_mat_cov_orderfree)
from geomllama.prompts import make_inference_prompt


def canonical_smiles(mol):
    try:
        m = Chem.RemoveHs(mol)
    except Exception:
        m = Chem.RemoveHs(mol, sanitize=False)
    return Chem.MolToSmiles(m, canonical=True)


def load_test_molecules(pkl_path, neutral_only=False, max_molecules=None):
    """Group test conformers by canonical SMILES -> [{'smiles', 'ref_mols'}].

    Accepts either:
      - raw GeoMol pkl: list of PyG Data objects with .rdmol;
      - pre-grouped dict: {canonical_smiles: [Mol, ...]} (test_smiles_dict.pkl).

    neutral_only defaults to False — template_fh trains on the full
    distribution including charged species.
    """
    data = pickle.load(open(pkl_path, "rb"))

    if isinstance(data, dict):
        mols = []
        for smi, ref_mols in data.items():
            if neutral_only and any(
                any(a.GetFormalCharge() != 0 for a in m.GetAtoms())
                for m in ref_mols
            ):
                continue
            mols.append({"smiles": smi, "ref_mols": list(ref_mols)})
        if max_molecules:
            mols = mols[:max_molecules]
        return mols

    groups = {}
    for e in data:
        mol = getattr(e, "rdmol", None) or getattr(e, "rd_mol", None)
        if mol is None or mol.GetNumConformers() == 0:
            continue
        smi = canonical_smiles(mol)
        groups.setdefault(smi, []).append(mol)
    mols = [{"smiles": s, "ref_mols": ms} for s, ms in groups.items()]
    if max_molecules:
        mols = mols[:max_molecules]
    return mols


def gen_text_to_bonded_mol(text, smiles, fmt, ordering="D"):
    """Bonded RDKit mol = SMILES template with the generated conformer.

    Atoms of the freshly built AddHs(MolFromSmiles) template are positioned from
    the parsed (canonical-order) coordinates, so the result carries the real
    bond graph (-> symmetry-aware GetBestRMS) AND the generated geometry.
    Returns None if the generation is unparseable or atom-count/elements mismatch.
    """
    coords = fmt.parse_output(text)
    if coords is None:
        return None
    template = mol_with_hs_from_smiles(smiles)
    if template is None:
        return None
    order = get_hydrogen_atom_order(template, ordering)
    if len(coords) != len(order):
        return None
    conf = Chem.Conformer(template.GetNumAtoms())
    for k, mol_idx in enumerate(order):
        el, x, y, z = coords[k]
        if template.GetAtomWithIdx(mol_idx).GetSymbol() != el:
            return None
        conf.SetAtomPosition(mol_idx, Point3D(float(x), float(y), float(z)))
    template.RemoveAllConformers()
    template.AddConformer(conf, assignId=True)
    return template


def _drop_degenerate(gen_mols):
    """Split off generations whose conformer coordinates are not finite.

    A z-matrix can parse with the correct formula yet decode to nan/inf
    coordinates (e.g. collinear reference atoms make a dihedral undefined).
    These are unusable geometries, not molecules: on the bonded path
    rdMolAlign.GetBestRMS returns ~1e154 for them, which overflows float32 to
    +inf -- harmless for COV-R/MAT-R (an inf never wins a min) but it poisons
    the precision mean (MAT-P). Reject them up front and account for them in
    their own bucket instead of silently scoring garbage.

    Returns (kept_mols, n_degenerate).
    """
    kept, n_bad = [], 0
    for m in gen_mols:
        try:
            pos = m.GetConformer(0).GetPositions()
        except Exception:
            n_bad += 1
            continue
        if np.isfinite(pos).all():
            kept.append(m)
        else:
            n_bad += 1
    return kept, n_bad


def _classify_known_order_failure(text, smiles, fmt, ordering="D"):
    """Bucket a rejected known-order generation as 'syntax' (unparseable text /
    no template) or 'atom_mismatch' (wrong atom count or element), mirroring the
    staged checks in gen_text_to_bonded_mol. Called only on generations that
    already failed to build, so it never returns 'ok'."""
    coords = fmt.parse_output(text)
    if coords is None:
        return "syntax"
    template = mol_with_hs_from_smiles(smiles)
    if template is None:
        return "syntax"
    order = get_hydrogen_atom_order(template, ordering)
    # length OR per-position element mismatch both mean the formula is wrong
    return "atom_mismatch"


def cov_mat_from_matrix(rmsd_mat, threshold):
    """(COV-R, MAT-R, COV-P, MAT-P) from an (n_ref, n_gen) RMSD matrix."""
    # Recall: best generated for each reference (min over gen / axis=1)
    r_min = rmsd_mat.min(axis=1)
    cov_r = float((r_min <= threshold).mean())
    mat_r = float(r_min.mean())
    # Precision: best reference for each generated (min over ref / axis=0)
    p_min = rmsd_mat.min(axis=0)
    cov_p = float((p_min <= threshold).mean())
    mat_p = float(p_min.mean())
    return cov_r, mat_r, cov_p, mat_p


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_free(engine, mols, fmt_name, n_mult, temperature, top_p,
                  max_new_tokens):
    """One-shot batched generation (data-parallel). Attaches 'gen_texts'."""
    prompts = []
    counts = []
    for m in mols:
        n = n_mult * len(m["ref_mols"])
        counts.append(n)
        prompts.extend([make_inference_prompt(m["smiles"], fmt_name)] * n)
    print(f"free generation: {len(prompts)} prompts", flush=True)
    results = engine.generate(prompts=prompts, max_new_tokens=max_new_tokens,
                              temperature=temperature, top_p=top_p, n=1)
    idx = 0
    for m, n in zip(mols, counts):
        m["gen_texts"] = [t for _, texts in results[idx:idx + n] for t in texts]
        idx += n


def generate_scaffolded(engine, mols, fmt, fmt_name, n_mult, temperature, top_p):
    """Per-molecule scaffolded generation (single replica). Attaches 'gen_texts'."""
    from tqdm import tqdm
    for m in tqdm(mols, desc="scaffolded gen"):
        smi = m["smiles"]
        prompt = make_inference_prompt(smi, fmt_name)
        n = n_mult * len(m["ref_mols"])
        if fmt_name.startswith("template_fh"):
            scaffold = fmt.scaffold_for(smi)
            m["gen_texts"] = engine.generate_scaffolded_zmat(
                prompt, scaffold, n=n, temperature=temperature, top_p=top_p)
        else:
            labels = fmt.labels_for(smi)
            m["gen_texts"] = engine.generate_scaffolded(
                prompt, labels, n=n, temperature=temperature, top_p=top_p)


def generate_grammar(engine, mols, fmt, fmt_name, n_mult, temperature, top_p,
                     max_new_tokens):
    """Grammar-constrained scaffolded generation (template_fh only).

    Single llm.generate call across every molecule; vLLM's continuous batcher
    schedules conformers like free mode while xgrammar masks logits at every
    step so the scaffold tokens are pinned and only numeric values are
    sampled. Wall clock should approach free mode rather than the per-field
    scaffolded harness.
    """
    if not fmt_name.startswith("template_fh"):
        raise ValueError("grammar-constrained generation only supports "
                         "template_fh formats")
    prompts = [make_inference_prompt(m["smiles"], fmt_name) for m in mols]
    scaffolds = [fmt.scaffold_for(m["smiles"]) for m in mols]
    ns = [n_mult * len(m["ref_mols"]) for m in mols]
    all_gens = engine.generate_grammar_zmat(
        prompts, scaffolds, ns,
        temperature=temperature, top_p=top_p, max_tokens=max_new_tokens)
    for m, gens in zip(mols, all_gens):
        m["gen_texts"] = gens


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

# Formats with no canonical atom order: the generated atom i does NOT correspond
# to reference atom i, so RMSD must be minimised over element-preserving atom
# permutations (see compute_mat_cov_orderfree). ori_xyz is order-free for the same
# reason ori_fh is -- its training targets use openbabel's atom order, verifiably
# not the canonical 'D' order that the template_fh family emits.
ORDER_FREE_FORMATS = ("ori_fh", "fh", "ori_xyz", "xyz")


def _parse_kind(fmt_name):
    """The text-parser kind ('fh' or 'xyz') that text_to_mol needs for a format."""
    return "xyz" if fmt_name in ("ori_xyz", "xyz") else "fh"


def _bondless_gen_mols(texts, ref_mols, parse_kind="fh"):
    """Bond-less gen mols for order-free formats (ori_fh / ori_xyz), mirroring
    evaluation.evaluate_geom_molecule's staged filtering: text->coords parse,
    heavy-atom count match vs the reference, then a bond-less point-cloud mol."""
    from geomllama.evaluation import text_to_mol
    true_atoms = Counter(a.GetSymbol() for a in ref_mols[0].GetAtoms())
    out = []
    for t in texts:
        try:
            mol = text_to_mol(t.strip(), true_atoms, parse_kind, "remove_hydrogens")
        except Exception:
            continue
        if isinstance(mol, Chem.Mol):
            out.append(mol)
    return out


def _bondless_gen_mols_counted(texts, ref_mols, parse_kind="fh"):
    """Like _bondless_gen_mols but also returns (n_syntax, n_atom_mismatch)
    failure counts, read from text_to_mol's reason strings ('Wrong syntax' /
    'Wrong number of atoms'). Used for the order-free failure-mode accounting."""
    from geomllama.evaluation import text_to_mol
    true_atoms = Counter(a.GetSymbol() for a in ref_mols[0].GetAtoms())
    out, n_syntax, n_atom = [], 0, 0
    for t in texts:
        try:
            r = text_to_mol(t.strip(), true_atoms, parse_kind, "remove_hydrogens")
        except Exception:
            r = "Wrong syntax"
        if isinstance(r, Chem.Mol):
            out.append(r)
        elif r == "Wrong number of atoms":
            n_atom += 1
        else:
            n_syntax += 1
    return out, n_syntax, n_atom


def _bondless_ref_mols(ref_mols):
    """Rebuild reference mols as bond-less point clouds from their coordinates.

    evaluate_geom_molecule (which produced the validated reference numbers) uses
    coordinates_to_mol for BOTH gen and ref, so GetBestRMS minimizes over element
    permutations with no bond constraints on either side. Loaded test mols may
    carry a bond graph, so rebuild them bond-less to match that method exactly.
    """
    from geomllama.evaluation import coordinates_to_mol, mol_to_xyz_data
    return [coordinates_to_mol(mol_to_xyz_data(m)) for m in ref_mols]


def evaluate(mols, fmt, threshold, ordering, n_jobs, return_permol=False,
             fast_rmsd=False, order_free=False, return_details=False,
             rmsd_engine="auto", parse_kind="fh"):
    """Score COV/MAT per molecule.

    order_free (ori_fh and other formats with no canonical atom order): build
    BOTH gen and ref as bond-less point clouds and match by the element
    permutation minimum. Scored by compute_mat_cov_orderfree, which picks the
    engine per molecule from the exact permutation count -- exact GetBestRMS
    where it fits under RDKit's maxMatches cap (all of GEOM-QM9), else the
    Kabsch-Hungarian upper bound (GEOM-Drugs, median ~1e19 permutations, where
    capped GetBestRMS silently returns a non-minimum). Override with
    rmsd_engine='exact'|'kh'.

    Otherwise use the known-order bonded path (gen_text_to_bonded_mol), where
    fast_rmsd chooses compute_mat_cov_fast (batched permutation-Kabsch) vs
    compute_mat_cov (per-pair rdMolAlign.GetBestRMS). fast_rmsd is ignored for
    order_free formats.
    """
    cov_r, mat_r, cov_p, mat_p = [], [], [], []
    permol = []
    details = []
    n_gen_total = n_parsed_total = 0
    n_syntax_total = n_atom_total = n_geom_total = 0
    from tqdm import tqdm
    for m in tqdm(mols, desc=f"COV/MAT @ {threshold}A"):
        texts = m.get("gen_texts", [])
        n_gen = len(texts)
        n_gen_total += n_gen
        n_conf = len(m["ref_mols"])          # the 50-500 table-filter key
        n_syntax = n_atom = 0
        if order_free:
            gen_mols, n_syntax, n_atom = _bondless_gen_mols_counted(
                texts, m["ref_mols"], parse_kind=parse_kind)
            ref_mols = _bondless_ref_mols(m["ref_mols"])
        else:
            ref_mols = m["ref_mols"]
            gen_mols = []
            for t in texts:
                gm = gen_text_to_bonded_mol(t, m["smiles"], fmt, ordering)
                if gm is not None:
                    gen_mols.append(gm)
                elif _classify_known_order_failure(
                        t, m["smiles"], fmt, ordering) == "atom_mismatch":
                    n_atom += 1
                else:
                    n_syntax += 1
        # Correct syntax + correct formula, but the coordinates decode to
        # nan/inf -> unusable geometry. Own bucket; never RMSD-scored.
        gen_mols, n_geom = _drop_degenerate(gen_mols)
        n_parsed = len(gen_mols)             # valid syntax, correct atoms, finite coords
        n_parsed_total += n_parsed
        n_syntax_total += n_syntax
        n_atom_total += n_atom
        n_geom_total += n_geom
        vf = n_parsed / n_gen if texts else 0.0
        # Score RMSD only when >=1 generation is valid. Molecules with zero valid
        # gens are STILL recorded (None metrics, no matrix): they carry
        # n_syntax/n_atom and a conformer count that the 50-500 table filter and
        # the syn%/atom% denominators must see. Skipping them would silently bias
        # the aggregates.
        if gen_mols:
            gen_scored = gen_mols[:2 * len(ref_mols)]   # cap to 2x refs (standard)
            if order_free:
                # Per-molecule engine choice on the exact permutation count:
                # exact GetBestRMS under RDKit's maxMatches cap, else the
                # Kabsch-Hungarian upper bound. See compute_mat_cov_orderfree.
                _, _, rmsd_mat = compute_mat_cov_orderfree(
                    gen_scored, ref_mols, threshold=threshold,
                    mode="remove_hydrogens", n_jobs=n_jobs,
                    engine=rmsd_engine)
            elif fast_rmsd:
                _, _, rmsd_mat = compute_mat_cov_fast(
                    gen_scored, ref_mols, threshold=threshold,
                    mode="remove_hydrogens")
            else:
                _, _, rmsd_mat = compute_mat_cov(
                    gen_scored, ref_mols, threshold=threshold,
                    mode="remove_hydrogens", n_jobs=n_jobs)
            cr, mr, cp, mp = cov_mat_from_matrix(rmsd_mat, threshold)
            cov_r.append(cr); mat_r.append(mr); cov_p.append(cp); mat_p.append(mp)
        else:
            gen_scored, rmsd_mat = [], None
            cr = mr = cp = mp = None
        permol.append({"smiles": m["smiles"], "n_conformers": n_conf,
                       "cov_r": cr, "mat_r": mr, "cov_p": cp, "mat_p": mp,
                       "valid_frac": vf, "n_gen": n_gen, "n_parsed": n_parsed,
                       "n_syntax_fail": n_syntax, "n_atom_mismatch": n_atom,
                       "n_degenerate_geom": n_geom})
        if return_details:
            # Heavy per-molecule record: the actual RDKit mols (with conformers,
            # in real atom order) + the (n_ref x n_gen) RMSD matrix. gen_mols are
            # the RMSD-scored set (matrix columns); rmsd_mat is None if no valid
            # gens. This is the "never recompute" artifact (esp. for order-free).
            details.append({"smiles": m["smiles"], "n_conformers": n_conf,
                            "n_gen": n_gen, "n_parsed": n_parsed,
                            "n_degenerate_geom": n_geom,
                            "n_syntax_fail": n_syntax, "n_atom_mismatch": n_atom,
                            "cov_r": cr, "mat_r": mr, "cov_p": cp, "mat_p": mp,
                            "gen_texts": texts, "gen_mols": gen_scored,
                            "ref_mols": ref_mols, "rmsd_mat": rmsd_mat})

    def stat(xs):
        return (float(np.mean(xs)), float(np.median(xs))) if xs else (None, None)

    cr_mean, cr_med = stat(cov_r)
    mr_mean, mr_med = stat(mat_r)
    cp_mean, cp_med = stat(cov_p)
    mp_mean, mp_med = stat(mat_p)
    res = {
        "threshold_A": threshold,
        "n_molecules_scored": len(cov_r),
        "n_molecules_total": len(mols),
        "n_gen_total": n_gen_total,
        "n_parsed_total": n_parsed_total,
        "n_syntax_fail": n_syntax_total,
        "n_atom_mismatch": n_atom_total,
        # Parsed + correct formula, but coords decode to nan/inf. Never scored.
        "n_degenerate_geom": n_geom_total,
        "atom_match_pct": 100.0 * n_parsed_total / max(n_gen_total, 1),
        "syntax_fail_pct": 100.0 * n_syntax_total / max(n_gen_total, 1),
        "atom_mismatch_pct": 100.0 * n_atom_total / max(n_gen_total, 1),
        "degenerate_geom_pct": 100.0 * n_geom_total / max(n_gen_total, 1),
        "COV_R_mean": cr_mean, "COV_R_median": cr_med,
        "MAT_R_mean": mr_mean, "MAT_R_median": mr_med,
        "COV_P_mean": cp_mean, "COV_P_median": cp_med,
        "MAT_P_mean": mp_mean, "MAT_P_median": mp_med,
    }
    if return_permol and return_details:
        return res, permol, details
    if return_permol:
        return res, permol
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="merged model dir (omit with --eval-only)")
    ap.add_argument("--test-pkl", required=True,
                    help="Path to test pickle (test_data_200.pkl or "
                         "test_smiles_dict.pkl)")
    ap.add_argument("--format", default="template_fh")
    ap.add_argument("--ordering", default="D")
    ap.add_argument("--threshold", type=float, default=1.25,
                    help="COV RMSD threshold in Angstroms (GEOM-Drugs: 1.25)")
    ap.add_argument("--n-mult", type=int, default=2)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-new-tokens", type=int, default=3072)
    ap.add_argument("--scaffolded", action="store_true",
                    help="use scaffolded generation (guaranteed atom count, slower)")
    ap.add_argument("--scaffold-grammar", action="store_true",
                    help="use grammar-constrained generation (template_fh): "
                         "free-mode-style single pass per conformer, with "
                         "regex-pinned scaffold tokens. Mutually exclusive "
                         "with --scaffolded.")
    ap.add_argument("--data-parallel-size", type=int, default=None)
    ap.add_argument("--tensor-parallel-size", type=int, default=None)
    ap.add_argument("--max-model-len", type=int, default=None)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--max-molecules", type=int, default=None)
    ap.add_argument("--min-conformers", type=int, default=None,
                    help="keep only molecules with >= this many ref conformers")
    ap.add_argument("--max-conformers", type=int, default=None,
                    help="keep only molecules with <= this many ref conformers")
    ap.add_argument("--num-shards", type=int, default=1,
                    help="split the test molecules into N strided shards for "
                         "independent multi-GPU runs; merge with merge_shards.py")
    ap.add_argument("--shard-id", type=int, default=0,
                    help="which shard (0..num_shards-1) this process scores")
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--fast-rmsd", dest="fast_rmsd", action="store_true",
                    default=None,
                    help="use the bond-less permutation-minimum GPU evaluator "
                         "(compute_mat_cov_fast). Default: auto-on for ori_fh.")
    ap.add_argument("--no-fast-rmsd", dest="fast_rmsd", action="store_false",
                    help="force the per-pair GetBestRMS evaluator even for ori_fh")
    ap.add_argument("--rmsd-engine", choices=["auto", "exact", "kh"],
                    default="auto",
                    help="order-free RMSD engine. auto (default): exact "
                         "GetBestRMS when the element-permutation count fits "
                         "under RDKit's 1e6 maxMatches cap, else "
                         "Kabsch-Hungarian. exact/kh force one. Ignored for "
                         "known-order formats.")
    ap.add_argument("--output", required=True)
    ap.add_argument("--gen-cache", default=None,
                    help="pickle to save/reuse generations (skip model if present)")
    ap.add_argument("--eval-only", action="store_true")
    args = ap.parse_args()

    fmt = get_format(args.format)
    # ori_fh (and other order-free FH formats) have no canonical atom order, so
    # BOTH gen and ref must be scored bond-less (evaluate_geom_molecule's method).
    order_free = args.format in ORDER_FREE_FORMATS
    parse_kind = _parse_kind(args.format)
    # fast_rmsd only selects among the KNOWN-ORDER engines now; order-free
    # scoring dispatches per molecule inside compute_mat_cov_orderfree.
    if args.fast_rmsd is None:
        args.fast_rmsd = False
    t0 = time.time()
    print(f"loading test molecules from {args.test_pkl} ...", flush=True)
    mols = load_test_molecules(args.test_pkl, max_molecules=args.max_molecules)
    if args.min_conformers is not None or args.max_conformers is not None:
        lo = args.min_conformers or 0
        hi = args.max_conformers if args.max_conformers is not None else 10**9
        before = len(mols)
        mols = [m for m in mols if lo <= len(m["ref_mols"]) <= hi]
        print(f"  conformer filter [{lo},{hi}]: {before} -> {len(mols)} molecules",
              flush=True)
    if args.num_shards > 1:
        if not (0 <= args.shard_id < args.num_shards):
            sys.exit(f"--shard-id must be in [0, {args.num_shards})")
        mols = mols[args.shard_id::args.num_shards]
        print(f"  shard {args.shard_id}/{args.num_shards}", flush=True)
    n_conf = sum(len(m["ref_mols"]) for m in mols)
    print(f"  {len(mols)} molecules, {n_conf} reference conformers", flush=True)

    cache = args.gen_cache
    if args.eval_only or (cache and os.path.exists(cache)):
        if not (cache and os.path.exists(cache)):
            sys.exit("--eval-only needs --gen-cache pointing at saved generations")
        print(f"loading cached generations from {cache}", flush=True)
        gen = pickle.load(open(cache, "rb"))
        by_smi = {m["smiles"]: m for m in mols}
        for g in gen:
            if g["smiles"] in by_smi:
                by_smi[g["smiles"]]["gen_texts"] = g["gen_texts"]
    else:
        if args.model is None:
            sys.exit("--model is required for generation")
        from geomllama.inference import InferenceEngine
        if args.scaffolded and args.scaffold_grammar:
            sys.exit("use at most one of --scaffolded / --scaffold-grammar")
        if args.scaffolded or args.scaffold_grammar:
            dp = 1
            tp = args.tensor_parallel_size
        else:
            dp = args.data_parallel_size
            if dp is None:
                vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
                if vis:
                    dp = len(vis.split(","))
                else:
                    import torch
                    dp = torch.cuda.device_count() or 1
            tp = args.tensor_parallel_size
        print(f"loading model {args.model} (dp={dp}, scaffolded={args.scaffolded})", flush=True)
        engine = InferenceEngine(
            model_path=args.model, data_parallel_size=dp,
            tensor_parallel_size=tp, max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization)
        t_gen = time.time()
        if args.scaffolded:
            generate_scaffolded(engine, mols, fmt, args.format, args.n_mult,
                                args.temperature, args.top_p)
            mode_label = "scaffolded"
        elif args.scaffold_grammar:
            generate_grammar(engine, mols, fmt, args.format, args.n_mult,
                             args.temperature, args.top_p, args.max_new_tokens)
            mode_label = "scaffold_grammar"
        else:
            generate_free(engine, mols, args.format, args.n_mult,
                          args.temperature, args.top_p, args.max_new_tokens)
            mode_label = "free"
        print(f"[bench] {mode_label} gen ({len(mols)} mols) took "
              f"{time.time()-t_gen:.1f}s", flush=True)
        if cache:
            pickle.dump([{"smiles": m["smiles"], "gen_texts": m["gen_texts"]}
                         for m in mols], open(cache, "wb"))
            print(f"saved generations to {cache}", flush=True)

    print("scoring COV/MAT ...", flush=True)
    res, permol, details = evaluate(mols, fmt, args.threshold, args.ordering,
                                    args.n_jobs, return_permol=True,
                                    return_details=True,
                                    rmsd_engine=args.rmsd_engine,
                                    parse_kind=parse_kind,
                                    fast_rmsd=args.fast_rmsd,
                                    order_free=order_free)
    res.update({"model": args.model, "format": args.format,
                "generation": ("scaffold_grammar" if args.scaffold_grammar
                               else "scaffolded" if args.scaffolded
                               else "free"),
                "temperature": args.temperature, "n_mult": args.n_mult,
                "num_shards": args.num_shards, "shard_id": args.shard_id,
                "elapsed_s": round(time.time() - t0, 1)})
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    json.dump(res, open(args.output, "w"), indent=2)
    # Always persist per-molecule records (lightweight JSON: scalars + buckets +
    # n_conformers, for fast table aggregation / re-filtering) AND the full
    # details pickle (RDKit gen+ref mols + RMSD matrices, so nothing is ever
    # recomputed -- critical for the order-free case). Previously the sidecar was
    # only written under sharding; both are now unconditional.
    permol_path = args.output + ".permol.json"
    json.dump({"per_molecule": permol,
               "n_gen_total": res["n_gen_total"],
               "n_parsed_total": res["n_parsed_total"],
               "n_syntax_fail": res["n_syntax_fail"],
               "n_atom_mismatch": res["n_atom_mismatch"],
               "n_degenerate_geom": res["n_degenerate_geom"],
               "n_molecules_total": res["n_molecules_total"],
               "threshold_A": args.threshold,
               "shard_id": args.shard_id, "num_shards": args.num_shards},
              open(permol_path, "w"))
    print(f"  wrote per-molecule records {permol_path}", flush=True)
    details_path = args.output + ".details.pkl"
    with open(details_path, "wb") as fh:
        pickle.dump(details, fh)
    print(f"  wrote full details (mols + RMSD matrices) {details_path}", flush=True)
    print("=== RESULT ===")
    for k, v in res.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
