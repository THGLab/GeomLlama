#!/usr/bin/env python3
"""
Benchmark a fine-tuned LLM on generating QM9 molecular geometries.

Loads the test set, generates geometries, and evaluates RMSD.

Usage:
  python benchmark_qm9.py \\
      --model-path path/to/merged/model \\
      --test-set data/qm9/test_set.json \\
      --format fh \\
      --output results/qm9_benchmark.txt
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import CalcMolFormula

from geomllama.evaluation import evaluate_qm9_parallel
from geomllama.inference import InferenceEngine
from geomllama.prompts import make_inference_prompt


def main():
    parser = argparse.ArgumentParser(description="QM9 RMSD Benchmark")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to model (local or HuggingFace ID)")
    parser.add_argument("--test-set", type=str, required=True,
                        help="Path to test_set.json")
    parser.add_argument("--format", type=str, default="fh",
                        choices=["fh", "xyz", "ori_fh", "ori_xyz",
                                 "roundto_strict3_xyz", "roundto_strict2_xyz",
                                 "roundto3_xyz",
                                 "formula_fh", "formula_xyz",
                                 "fh_e2e", "feedback_fh"],
                        help="Geometry format used in training")
    parser.add_argument("--output", type=str, required=True,
                        help="Output file for results")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--tensor-parallel-size", type=int, default=None,
                        help="GPUs per model replica for tensor parallelism "
                             "(default: 1 when dp>1, else all GPUs)")
    parser.add_argument("--data-parallel-size", type=int, default=None,
                        help="Number of model replicas. "
                             "(default: number of visible GPUs)")
    args = parser.parse_args()

    if args.data_parallel_size is None:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        args.data_parallel_size = len(visible.split(",")) if visible else 1


    # Map short format names to data_formats names for prompts
    fmt_to_prompt_format = {
        "fh": "ori_fh",
        "xyz": "ori_xyz",
    }
    prompt_format = fmt_to_prompt_format.get(args.format, args.format)

    # Map to evaluation format ('fh' or 'xyz')
    eval_fmt = "fh" if "fh" in args.format else "xyz"

    # Load test set
    print(f"Loading test set from {args.test_set}...")
    with open(args.test_set) as f:
        test = json.load(f)
    print(f"Loaded {len(test)} test molecules")

    # Create prompts
    needs_formula = 'formula' in prompt_format
    prompts = []
    for t in test:
        prompt_kwargs = {}
        if needs_formula:
            mol = Chem.MolFromSmiles(t['smiles'])
            prompt_kwargs['formula'] = CalcMolFormula(mol) if mol else ''
        prompts.append(make_inference_prompt(t['smiles'], prompt_format,
                                            **prompt_kwargs))

    # Run inference
    print(f"Loading model from {args.model_path}...")
    engine = InferenceEngine(
        model_path=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        data_parallel_size=args.data_parallel_size,
        trust_remote_code=args.trust_remote_code,
    )

    print("Generating geometries...")
    results = engine.generate(
        prompts=prompts,
        max_new_tokens=1024,
        temperature=None,
        do_sample=False,
        n=1,
    )

    # Evaluate
    print("Evaluating...")
    assessments = evaluate_qm9_parallel(results, test, fmt=eval_fmt)

    # Tally results
    error_types = ("Wrong syntax", "Wrong number of atoms", "RMSD failed")
    wrong_syntax = sum(1 for r in assessments if r == "Wrong syntax")
    wrong_atoms = sum(1 for r in assessments if r == "Wrong number of atoms")
    rmsd_failed = sum(1 for r in assessments if r == "RMSD failed")
    rmsds = [float(r) for r in assessments if r not in error_types]

    total = len(assessments)
    report_lines = [
        f"Wrong syntax: {wrong_syntax}, or {100*wrong_syntax/total:.1f}%",
        f"Wrong number of atoms: {wrong_atoms}, or {100*wrong_atoms/total:.1f}%",
        f"RMSD failed: {rmsd_failed}, or {100*rmsd_failed/total:.1f}%",
        f"Valid molecules: {len(rmsds)}, or {100*len(rmsds)/total:.1f}%",
    ]
    if rmsds:
        report_lines.extend([
            f"RMSD mean: {np.mean(rmsds):.6f}",
            f"RMSD median: {np.median(rmsds):.6f}",
        ])

    # Print and save
    for line in report_lines:
        print(line)

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w') as f:
        f.write('\n'.join(report_lines) + '\n')
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
