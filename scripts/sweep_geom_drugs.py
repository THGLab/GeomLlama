#!/usr/bin/env python3
"""Hyperparameter sweep over inference sampling params on GEOM-Drugs-Small.

Loads test molecules and the vllm model ONCE, then iterates over a grid of
(temperature, top_p, top_k) combos. For each combo: free-mode batched
generation across all visible GPUs (data-parallel), then COV/MAT evaluation.

Reuses benchmark_geom_drugs.py for the heavy lifting (load_test_molecules,
generate_free, evaluate, gen_text_to_bonded_mol). Resumable: combos whose
result JSON already exists are skipped.

Usage:
  python sweep_geom_drugs.py --model <merged_dir> --test-pkl <path> \
    --out-dir results/sweep_template_fh \
    --temperatures 0.7,1.0,1.3 --top-ps 0.90,0.95 --top-ks 0,50

  # Smoke test: one combo
  python sweep_geom_drugs.py --model <merged_dir> --test-pkl <path> \
    --out-dir results/sweep_smoke \
    --temperatures 1.0 --top-ps 0.95
"""
import argparse, copy, itertools, json, os, pickle, sys, time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
import benchmark_geom_drugs as bgd  # noqa: E402

from geomllama.data_formats import get_format  # noqa: E402


def load_mols(test_pkl, neutral_only=False, max_molecules=None):
    """Auto-detect test_data_200.pkl (PyG Data list) vs. test_smiles_dict.pkl
    (already grouped {smiles: [Mol,...]}); return the [{smiles, ref_mols}, ...]
    shape that benchmark_geom_drugs.evaluate expects. Charged species are kept
    by default — template_fh was trained on the full distribution.
    """
    import pickle
    data = pickle.load(open(test_pkl, "rb"))

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

    return bgd.load_test_molecules(
        test_pkl, neutral_only=neutral_only, max_molecules=max_molecules)


def parse_csv_floats(s):
    return [float(x) for x in s.split(",") if x.strip()]


def parse_csv_ints(s):
    return [int(x) for x in s.split(",") if x.strip()]


def combo_tag(t, p, k):
    parts = [f"T{t:g}", f"p{p:g}"]
    if k and k > 0:
        parts.append(f"k{k}")
    return "_".join(parts)


def detect_dp():
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if vis:
        return len(vis.split(","))
    try:
        import torch
        return torch.cuda.device_count() or 1
    except Exception:
        return 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--test-pkl", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--format", default="template_fh")
    ap.add_argument("--ordering", default="D")
    ap.add_argument("--threshold", type=float, default=1.25)
    ap.add_argument("--n-mult", type=int, default=2)
    ap.add_argument("--max-new-tokens", type=int, default=3072)
    ap.add_argument("--temperatures", type=parse_csv_floats, default=[1.0])
    ap.add_argument("--top-ps", type=parse_csv_floats, default=[0.95])
    ap.add_argument("--top-ks", type=parse_csv_ints, default=[0],
                    help="0 disables top-k. Use e.g. '0,50' to compare.")
    ap.add_argument("--max-molecules", type=int, default=None)
    ap.add_argument("--min-conformers", type=int, default=None,
                    help="keep only molecules with >= this many ref conformers")
    ap.add_argument("--max-conformers", type=int, default=None,
                    help="keep only molecules with <= this many ref conformers")
    ap.add_argument("--fast-rmsd", dest="fast_rmsd", action="store_true",
                    default=None, help="force the fast permutation-Kabsch RMSD "
                    "evaluator (default: auto-on for order-free formats)")
    ap.add_argument("--no-fast-rmsd", dest="fast_rmsd", action="store_false",
                    help="force the slow validated per-pair GetBestRMS evaluator")
    ap.add_argument("--rmsd-engine", choices=["auto", "exact", "kh"],
                    default="auto",
                    help="order-free RMSD engine (see benchmark_geom_drugs.py)")
    ap.add_argument("--max-model-len", type=int, default=None)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--data-parallel-size", type=int, default=None)
    ap.add_argument("--tensor-parallel-size", type=int, default=None)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--save-gens", action="store_true",
                    help="Pickle per-combo generations next to result JSON.")
    ap.add_argument("--scaffold-grammar", action="store_true",
                    help="Grammar-constrained scaffolded gen (template_fh).")
    ap.add_argument("--force", action="store_true",
                    help="Re-run combos even if result JSON exists.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    fmt = get_format(args.format)
    # ori_fh (and other order-free FH formats) have no canonical atom order, so
    # BOTH gen and ref are scored bond-less, and the engine is chosen per
    # molecule by --rmsd-engine (default auto). fast_rmsd only selects among the
    # known-order engines.
    order_free = args.format in bgd.ORDER_FREE_FORMATS
    parse_kind = bgd._parse_kind(args.format)
    if args.fast_rmsd is None:
        args.fast_rmsd = False

    combos = [(t, p, k) for t, p, k
              in itertools.product(args.temperatures, args.top_ps, args.top_ks)]
    print(f"[sweep] {len(combos)} combo(s) to run: "
          f"{[combo_tag(t, p, k) for t, p, k in combos]}", flush=True)

    # Skip already-done combos before engine load.
    pending = []
    for t, p, k in combos:
        tag = combo_tag(t, p, k)
        out_json = os.path.join(args.out_dir, f"{tag}.json")
        if not args.force and os.path.exists(out_json):
            print(f"[sweep] skip {tag}: result exists ({out_json})", flush=True)
            continue
        pending.append((t, p, k, tag, out_json))
    if not pending:
        print("[sweep] nothing to do.", flush=True)
        return

    t0 = time.time()
    print(f"[sweep] loading test molecules from {args.test_pkl}", flush=True)
    base_mols = load_mols(args.test_pkl, max_molecules=args.max_molecules)
    if args.min_conformers is not None or args.max_conformers is not None:
        lo = args.min_conformers or 0
        hi = args.max_conformers if args.max_conformers is not None else 10**9
        before = len(base_mols)
        base_mols = [m for m in base_mols if lo <= len(m["ref_mols"]) <= hi]
        print(f"[sweep]   conformer filter [{lo},{hi}]: "
              f"{before} -> {len(base_mols)} molecules", flush=True)
    n_conf = sum(len(m["ref_mols"]) for m in base_mols)
    print(f"[sweep]   {len(base_mols)} molecules, {n_conf} ref conformers",
          flush=True)

    dp = args.data_parallel_size or detect_dp()
    if args.scaffold_grammar and dp != 1:
        print(f"[sweep] grammar mode requires dp=1; clamping (was {dp})",
              flush=True)
        dp = 1
    print(f"[sweep] loading model {args.model} (dp={dp})", flush=True)
    from geomllama.inference import InferenceEngine
    engine = InferenceEngine(
        model_path=args.model,
        data_parallel_size=dp,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    print(f"[sweep] engine ready ({time.time()-t0:.0f}s elapsed)", flush=True)

    for t, p, k, tag, out_json in pending:
        ct = time.time()
        print(f"\n[sweep] === {tag} (T={t}, top_p={p}, top_k={k}) ===",
              flush=True)
        mols = copy.deepcopy(base_mols)  # fresh per combo (gen_texts attached)
        if args.scaffold_grammar:
            if k and k > 0:
                print(f"[sweep]   warn: top_k={k} ignored in grammar mode",
                      flush=True)
            print(f"[sweep]   grammar generation: {len(mols)} mols", flush=True)
            bgd.generate_grammar(engine, mols, fmt, args.format, args.n_mult,
                                 temperature=t, top_p=p,
                                 max_new_tokens=args.max_new_tokens)
        else:
            # bgd.generate_free does not support top_k; we accept that for now —
            # InferenceEngine.generate does, so use it directly to honor top_k.
            prompts, counts = [], []
            from geomllama.prompts import make_inference_prompt
            for m in mols:
                n = args.n_mult * len(m["ref_mols"])
                counts.append(n)
                prompts.extend([make_inference_prompt(m["smiles"], args.format)] * n)
            print(f"[sweep]   free generation: {len(prompts)} prompts",
                  flush=True)
            results = engine.generate(
                prompts=prompts,
                max_new_tokens=args.max_new_tokens,
                temperature=t, top_p=p,
                top_k=(k if k and k > 0 else None),
                n=1)
            idx = 0
            for m, n in zip(mols, counts):
                m["gen_texts"] = [tx for _, texts in results[idx:idx + n]
                                  for tx in texts]
                idx += n
        gen_t = time.time() - ct

        if args.save_gens:
            gens_path = os.path.join(args.out_dir, f"{tag}_gens.pkl")
            pickle.dump([{"smiles": m["smiles"], "gen_texts": m["gen_texts"]}
                         for m in mols], open(gens_path, "wb"))
            print(f"[sweep]   saved gens -> {gens_path}", flush=True)

        ev_t0 = time.time()
        res, permol, details = bgd.evaluate(
            mols, fmt, args.threshold, args.ordering, args.n_jobs,
            return_permol=True, return_details=True,
            fast_rmsd=args.fast_rmsd, order_free=order_free,
            rmsd_engine=args.rmsd_engine, parse_kind=parse_kind)
        ev_t = time.time() - ev_t0

        res.update({
            "tag": tag, "model": args.model, "format": args.format,
            "generation": ("scaffold_grammar" if args.scaffold_grammar
                           else "free"),
            "temperature": t, "top_p": p, "top_k": k,
            "n_mult": args.n_mult, "max_new_tokens": args.max_new_tokens,
            "fast_rmsd": args.fast_rmsd,
            "rmsd_engine": args.rmsd_engine,
            "gen_seconds": round(gen_t, 1),
            "eval_seconds": round(ev_t, 1),
        })
        json.dump(res, open(out_json, "w"), indent=2)
        # Per-molecule records (scalars + buckets + n_conformers) and the full
        # details pickle (RDKit gen+ref mols + per-molecule RMSD matrices) so
        # nothing is discarded -- everything lands next to the model.
        json.dump({"per_molecule": permol,
                   "n_gen_total": res["n_gen_total"],
                   "n_parsed_total": res["n_parsed_total"],
                   "n_syntax_fail": res["n_syntax_fail"],
                   "n_atom_mismatch": res["n_atom_mismatch"],
                   "n_molecules_total": res["n_molecules_total"],
                   "threshold_A": args.threshold, "tag": tag},
                  open(os.path.join(args.out_dir, f"{tag}.permol.json"), "w"))
        with open(os.path.join(args.out_dir, f"{tag}.details.pkl"), "wb") as fh:
            pickle.dump(details, fh)
        print(f"[sweep]   wrote {out_json} + {tag}.permol.json + "
              f"{tag}.details.pkl (gen {gen_t:.0f}s, eval {ev_t:.0f}s)",
              flush=True)
        for kk in ("atom_match_pct", "COV_R_mean", "MAT_R_mean",
                   "COV_P_mean", "MAT_P_mean"):
            print(f"    {kk}: {res.get(kk)}", flush=True)

    print(f"\n[sweep] DONE in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
