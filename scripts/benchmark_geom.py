#!/usr/bin/env python3
"""
Benchmark a fine-tuned LLM on generating GEOM-QM9 conformer ensembles.

Computes MAT (coverage) and COV (mean RMSD) scores.

For each molecule in the test set, generates 2x the number of ground-truth
conformers, then compares using RMSD with a threshold of 0.5 Angstroms.

Supports split-mode operation for running inference on GPU nodes and
evaluation on CPU nodes:

  # GPU node: generate conformers only
  python benchmark_geom.py --inference-only \\
      --model-path path/to/model --test-dict data/test_smiles_dict.pkl \\
      --format fh --output results/run.pkl

  # CPU node: evaluate from saved generations
  python benchmark_geom.py --eval-only \\
      --output results/run.pkl --format fh

  # Combined (original behavior)
  python benchmark_geom.py \\
      --model-path path/to/model --test-dict data/test_smiles_dict.pkl \\
      --format fh --output results/run.pkl
"""

import argparse
import json
import os
import pickle
import sys

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import CalcMolFormula

from geomllama.evaluation import (
    evaluate_geom_molecule,
    evaluate_geom_molecule_ordered,
    mol_to_xyz_data,
)
from geomllama.prompts import make_inference_prompt


_SCAFFOLDED_FORMATS = {'template_fh', 'template_fh_no_graph'}


def run_inference(args, geom, prompt_format):
    """Run LLM inference and attach generated texts to each molecule."""
    from geomllama.inference import InferenceEngine

    if prompt_format in _SCAFFOLDED_FORMATS:
        return run_inference_template_fh(args, geom, prompt_format)

    needs_formula = 'formula' in prompt_format
    prompts = []
    for mol_data in geom:
        smiles = mol_data['smiles']
        prompt_kwargs = {}
        if needs_formula:
            mol = Chem.MolFromSmiles(smiles)
            prompt_kwargs['formula'] = CalcMolFormula(mol) if mol else ''
        prompt = make_inference_prompt(smiles, prompt_format, **prompt_kwargs)
        prompts.extend([prompt] * (2 * len(mol_data['conf_xyz'])))

    print(f"Total prompts: {len(prompts)}")

    print(f"Loading model from {args.model_path}...")
    engine = InferenceEngine(
        model_path=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        data_parallel_size=args.data_parallel_size,
        trust_remote_code=args.trust_remote_code,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        quantization=args.quantization,
    )

    print("Generating conformers...")
    results = engine.generate(
        prompts=prompts,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        n=1,
    )

    idx = 0
    for mol_data in tqdm(geom, desc="Collecting results"):
        n_prompts = 2 * len(mol_data['conf_xyz'])
        mol_results = results[idx:idx + n_prompts]
        mol_data['gen_zmat'] = []
        for _, texts in mol_results:
            mol_data['gen_zmat'].extend(texts)
        idx += n_prompts


def run_inference_template_fh(args, geom, prompt_format):
    """Run scaffolded template_fh inference (per-molecule, not batch)."""
    from geomllama.inference import InferenceEngine
    from geomllama.data_formats import get_format

    fmt = get_format(prompt_format)

    print(f"Loading model from {args.model_path}...")
    engine = InferenceEngine(
        model_path=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        data_parallel_size=args.data_parallel_size,
        trust_remote_code=args.trust_remote_code,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        quantization=args.quantization,
    )

    total_gen = sum(2 * len(m['conf_xyz']) for m in geom)
    print(f"Scaffolded template_fh inference: {len(geom)} molecules, "
          f"{total_gen} total conformers")

    for mol_data in tqdm(geom, desc="Generating (scaffolded)"):
        smiles = mol_data['smiles']
        n_gen = 2 * len(mol_data['conf_xyz'])
        prompt = make_inference_prompt(smiles, prompt_format)
        scaffold = fmt.scaffold_for(smiles)
        if scaffold is None:
            mol_data['gen_zmat'] = [""] * n_gen
            continue
        texts = engine.generate_scaffolded_zmat(
            prompt, scaffold, n=n_gen,
            temperature=args.temperature, top_p=args.top_p,
        )
        mol_data['gen_zmat'] = texts


def run_evaluation(args, geom, eval_fmt, use_ordered_eval):
    """Evaluate generated conformers and attach assessments."""
    cov_scores = []
    mat_scores = []
    fraction_valid = []

    eval_mode_str = "ordered (direct RMSD)" if use_ordered_eval else "permutation (GetBestRMS)"
    print(f"Evaluation mode: {eval_mode_str}")

    for k, mol_data in tqdm(enumerate(geom), total=len(geom),
                            desc="Evaluating"):
        if getattr(args, 'resume', False) and 'assessment' in mol_data:
            num_valid, assessment = mol_data['assessment']
            total_gen = len(mol_data['gen_zmat'])
            if assessment[0] != 'None':
                cov_scores.append(assessment[0])
                mat_scores.append(assessment[1])
            fraction_valid.append(num_valid / total_gen if total_gen > 0 else 0)
            continue
        try:
            if use_ordered_eval:
                num_valid, assessment = evaluate_geom_molecule_ordered(
                    ref_mols=mol_data['ref_mols'],
                    generated_texts=mol_data['gen_zmat'],
                    fmt=eval_fmt,
                    threshold=0.5,
                    remove_hs=True,
                    n_jobs=args.n_jobs,
                )
            else:
                num_valid, assessment = evaluate_geom_molecule(
                    ground_truth_conformers=mol_data['conf_xyz'],
                    generated_texts=mol_data['gen_zmat'],
                    fmt=eval_fmt,
                    threshold=0.5,
                    mode='remove_hydrogens',
                    n_jobs=args.n_jobs,
                )
        except Exception as ex:
            print(f"\nWarning: evaluate_geom_molecule raised "
                  f"{type(ex).__name__} on molecule {k} "
                  f"({mol_data.get('smiles', '')[:60]}): {ex}; "
                  f"recording as no-valid and continuing")
            num_valid, assessment = 0, ('None', 'None', 'None')

        mol_data['assessment'] = (num_valid, assessment)
        total_gen = len(mol_data['gen_zmat'])

        if assessment[0] != 'None':
            cov_scores.append(assessment[0])
            mat_scores.append(assessment[1])
        fraction_valid.append(num_valid / total_gen if total_gen > 0 else 0)

        with open(args.output, 'wb') as f:
            pickle.dump(geom, f)

    print(f"\n{'='*50}")
    print(f"Results ({len(geom)} molecules)")
    print(f"{'='*50}")
    if cov_scores:
        print(f"COV mean:  {np.mean(cov_scores):.4f}")
        print(f"COV median: {np.median(cov_scores):.4f}")
        print(f"MAT mean:  {np.mean(mat_scores):.4f}")
        print(f"MAT median: {np.median(mat_scores):.4f}")
    print(f"Mean fraction valid: {np.mean(fraction_valid):.4f}")


def main():
    parser = argparse.ArgumentParser(description="GEOM-QM9 MAT/COV Benchmark")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Path to model (local or HuggingFace ID)")
    parser.add_argument("--test-dict", type=str, default=None,
                        help="Path to test_smiles_dict.pkl")
    parser.add_argument("--format", type=str, default="fh",
                        choices=["fh", "xyz", "propertyprompt_fh",
                                 "connectivity_fh", "connectivity_xyz",
                                 "formula_fh", "formula_xyz",
                                 "fh_e2e", "feedback_fh",
                                 "template_fh",
                                 "template_fh_no_labels",
                                 "template_fh_no_graph"],
                        help="Geometry format (default: fh)")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--output", type=str, required=True,
                        help="Output pickle file for detailed results")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--n-jobs", type=int, default=-1,
                        help="Parallel jobs for RMSD computation (-1 = all)")
    parser.add_argument("--max-molecules", type=int, default=None,
                        help="Limit to first N molecules (for quick testing)")
    parser.add_argument("--max-model-len", type=int, default=None,
                        help="Max context length (reduce for large models)")
    parser.add_argument("--max-new-tokens", type=int, default=1024,
                        help="Max tokens to generate per prompt. Bump for "
                             "feedback_fh on drug-sized molecules (~4096).")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90,
                        help="Fraction of GPU memory to use (default: 0.90)")
    parser.add_argument("--quantization", type=str, default=None,
                        choices=["awq", "gptq", "squeezellm"],
                        help="Quantization method (for pre-quantized models)")
    parser.add_argument("--tensor-parallel-size", type=int, default=None,
                        help="GPUs per model replica for tensor parallelism "
                             "(default: 1 when dp>1, else all GPUs)")
    parser.add_argument("--data-parallel-size", type=int, default=None,
                        help="Number of model replicas. "
                             "(default: number of visible GPUs)")
    parser.add_argument("--ordered-eval", type=str, default="auto",
                        choices=["auto", "on", "off"],
                        help="Use direct positional RMSD assuming SMILES-order "
                             "atom output (auto = on for connectivity formats)")
    parser.add_argument("--inference-only", action="store_true",
                        help="Run inference only; save generations without "
                             "computing MAT/COV. Use --eval-only later on a "
                             "CPU node to evaluate.")
    parser.add_argument("--eval-only", action="store_true",
                        help="Evaluate only; load an existing pickle with "
                             "generated texts and compute MAT/COV scores. "
                             "No GPU or model required.")
    parser.add_argument("--resume", action="store_true",
                        help="Skip molecules that already have an "
                             "'assessment' field in the pickle (use to "
                             "resume a partial eval run).")
    args = parser.parse_args()

    if args.inference_only and args.eval_only:
        parser.error("Cannot use --inference-only and --eval-only together")

    if args.eval_only:
        # --- Eval-only mode: load existing pickle and evaluate ---
        if not os.path.exists(args.output):
            parser.error(f"--eval-only requires an existing pickle at "
                         f"{args.output}")

        print(f"Loading existing results from {args.output}...")
        with open(args.output, 'rb') as f:
            geom = pickle.load(f)
        print(f"Loaded {len(geom)} molecules")

        for mol_data in geom:
            if 'gen_zmat' not in mol_data:
                parser.error(
                    f"Pickle at {args.output} has no generated texts "
                    f"('gen_zmat'). Run inference first.")

        if args.ordered_eval == "auto":
            use_ordered_eval = (args.format.startswith("connectivity_")
                                or args.format in _SCAFFOLDED_FORMATS)
        else:
            use_ordered_eval = args.ordered_eval == "on"
        if args.format in _SCAFFOLDED_FORMATS:
            eval_fmt = "template_fh"
        elif "fh" in args.format:
            eval_fmt = "fh"
        else:
            eval_fmt = "xyz"

        run_evaluation(args, geom, eval_fmt, use_ordered_eval)
        print(f"\nResults saved to {args.output}")
        return

    # --- Inference path (inference-only or combined) ---
    if args.model_path is None:
        parser.error("--model-path is required for inference")
    if args.test_dict is None:
        parser.error("--test-dict is required for inference")

    if args.data_parallel_size is None:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        args.data_parallel_size = len(visible.split(",")) if visible else 1

    if args.ordered_eval == "auto":
        use_ordered_eval = (args.format.startswith("connectivity_")
                            or args.format in _SCAFFOLDED_FORMATS)
    else:
        use_ordered_eval = args.ordered_eval == "on"

    format_map = {"fh": "ori_fh", "xyz": "ori_xyz",
                   "propertyprompt_fh": "propertyprompt_fh",
                   "connectivity_fh": "connectivity_fh",
                   "connectivity_xyz": "connectivity_xyz",
                   "formula_fh": "formula_fh",
                   "formula_xyz": "formula_xyz",
                   "fh_e2e": "fh_e2e",
                   "feedback_fh": "feedback_fh",
                   "template_fh": "template_fh",
                   "template_fh_no_labels": "template_fh_no_labels",
                   "template_fh_no_graph": "template_fh_no_graph"}
    prompt_format = format_map[args.format]
    if args.format in _SCAFFOLDED_FORMATS:
        eval_fmt = "template_fh"
    elif "fh" in args.format:
        eval_fmt = "fh"
    else:
        eval_fmt = "xyz"

    # Load test data
    print(f"Loading test dict from {args.test_dict}...")
    with open(args.test_dict, 'rb') as f:
        mol_dict = pickle.load(f)
    print(f"Loaded {len(mol_dict)} unique SMILES")

    if args.max_molecules is not None:
        items = list(mol_dict.items())[:args.max_molecules]
        mol_dict = dict(items)
        print(f"Truncated to {len(mol_dict)} molecules for testing")

    # Build molecule list with ground truth conformer coordinates
    geom = []
    for smiles, mols in tqdm(mol_dict.items(), desc="Extracting coordinates"):
        conf_xyz = [mol_to_xyz_data(m) for m in mols]
        entry = {
            'smiles': smiles,
            'conf_xyz': conf_xyz,
        }
        if use_ordered_eval:
            entry['ref_mols'] = mols
        geom.append(entry)

    run_inference(args, geom, prompt_format)

    # Save after inference
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'wb') as f:
        pickle.dump(geom, f)

    if args.inference_only:
        print(f"\nInference complete. Generations saved to {args.output}")
        print(f"Run evaluation later with: python benchmark_geom.py "
              f"--eval-only --format {args.format} --output {args.output}")
        return

    # Combined mode: also evaluate
    run_evaluation(args, geom, eval_fmt, use_ordered_eval)
    print(f"\nDetailed results saved to {args.output}")


if __name__ == "__main__":
    main()
