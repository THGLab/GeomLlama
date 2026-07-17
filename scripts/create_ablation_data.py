#!/usr/bin/env python3
"""Create ablation datasets from template_fh JSONL files.

Ablation 1 ("no_labels"): strips atom labels (C1→C, H2→H) from both the
    connectivity graph in the prompt and the Z-matrix completion. The
    connectivity lines become "C is connected to: H, C., C." with no
    cross-referencing between lines. Output is a standard unlabeled FH
    Z-matrix. Tests whether explicit atom labeling/ordering matters.

Ablation 2 ("no_graph"): removes the connectivity graph from the prompt
    entirely (just SMILES), keeps the labeled Z-matrix completion unchanged.
    Tests whether the explicit connectivity graph matters when the model
    already has labeled output.

Ablation 3 ("order_only"): makes the example identical to ori_fh in BOTH the
    prompt (regenerated ori_fh SMILES prompt) and the output (labels stripped
    -> standard unlabeled FH Z-matrix). The ONLY thing left distinguishing it
    from ori_fh is the atom ordering, which is inherited from template_fh's
    deterministic "D" ordering (heavy-first, H-after-parent) instead of
    openbabel's order. Isolates the effect of atom ordering alone. Eval as
    ori_fh (permutation / GetBestRMS).

Usage:
    python scripts/create_ablation_data.py \\
        --input data/geom_drugs_small/train_fh/template_fh.jsonl \\
        --ablation no_labels \\
        --output data/geom_drugs_small/train_fh/template_fh_no_labels.jsonl

    python scripts/create_ablation_data.py \\
        --input data/geom_drugs_small/train_fh/template_fh.jsonl \\
        --ablation no_graph \\
        --output data/geom_drugs_small/train_fh/template_fh_no_graph.jsonl

    python scripts/create_ablation_data.py \\
        --input data/geom_drugs_small/train_fh/template_fh.jsonl \\
        --ablation order_only \\
        --output data/geom_drugs_small/train_fh/template_fh_order_only.jsonl
"""

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


# -- Instructions for each ablation ------------------------------------------

# Ablation 1: connectivity graph present but no atom labels anywhere
NO_LABELS_INSTRUCTION = (
    "You will be given the SMILES string of a molecule and a list of its atoms "
    "(including hydrogens) with their bond connectivity. Generate a realistic "
    "conformer as a Z-matrix: one atom per line in the listed order, each line "
    "giving the reference atom numbers and the bond distance, angle, and "
    "dihedral that place the atom. "
    "Bond notation: = double, # triple, . aromatic."
)

# Ablation 2: no connectivity graph, just SMILES, but labeled output
NO_GRAPH_INSTRUCTION = (
    "You will be given the SMILES string of a molecule. Generate a realistic "
    "conformer as a Z-matrix with labeled atoms: one atom per line, each line "
    "starting with the atom label (e.g. C1, H2) followed by the reference "
    "atom numbers and the bond distance, angle, and dihedral that place the atom."
)


def strip_label(token):
    """'C1' -> 'C', 'Cl3' -> 'Cl', 'C1.' -> 'C.'. Strips trailing digits,
    preserving any trailing bond-type suffix (., =, #)."""
    suffix = ""
    t = token
    if t and t[-1] in ".=#":
        suffix = t[-1]
        t = t[:-1]
    i = len(t)
    while i > 0 and t[i - 1].isdigit():
        i -= 1
    return t[:i] + suffix


def transform_connectivity_no_labels(input_text):
    """Strip atom labels from the connectivity block.

    'C1 is connected to: H2, C3., C33.' -> 'C is connected to: H, C., C.'
    """
    lines = input_text.split("\n")
    out = []
    for line in lines:
        if " is connected to: " in line:
            subject, rest = line.split(" is connected to: ", 1)
            subject_stripped = strip_label(subject)
            neighbors = [strip_label(n.strip()) for n in rest.split(", ")]
            out.append(f"{subject_stripped} is connected to: {', '.join(neighbors)}")
        else:
            out.append(line)
    return "\n".join(out)


def transform_output_no_labels(output_text):
    """Strip atom labels from Z-matrix output lines.

    'C1' -> 'C'
    'H2 1 1.0793' -> 'H 1 1.0793'
    """
    lines = output_text.split("\n")
    out = []
    for line in lines:
        parts = line.strip().split()
        if not parts:
            if line.strip() == "":
                out.append("")
            continue
        parts[0] = strip_label(parts[0])
        out.append(" ".join(parts))
    result = "\n".join(out)
    if output_text.endswith("\n") and not result.endswith("\n"):
        result += "\n"
    return result


def transform_input_no_graph(input_text):
    """Remove the connectivity block, keep only the SMILES line."""
    for line in input_text.split("\n"):
        if line.startswith("SMILES:"):
            return line
    return input_text


def extract_smiles(input_text):
    """Pull the SMILES string out of a template_fh 'SMILES: <smiles>' line."""
    for line in input_text.split("\n"):
        if line.startswith("SMILES:"):
            return line.split("SMILES:", 1)[1].strip()
    raise ValueError("no 'SMILES:' line found in template_fh input")


# The ori_fh prompt is regenerated via the registered format class so that the
# (instruction, input) pair is byte-identical to what ori_fh.jsonl contains for
# the same SMILES. Imported lazily so the no_labels/no_graph paths keep working
# without the geomllama package installed.
_ORI_FH = {"fmt": None, "instruction": None}


def _ori_fh_prompt(smiles):
    if _ORI_FH["fmt"] is None:
        from geomllama.data_formats import get_format
        from geomllama.prompts import INSTRUCTION
        fmt = get_format("ori_fh")
        _ORI_FH["fmt"] = fmt
        _ORI_FH["instruction"] = fmt.instruction or INSTRUCTION
    return _ORI_FH["instruction"], _ORI_FH["fmt"].format_prompt(smiles)


def process_entry(entry, ablation):
    """Transform a single JSONL entry for the given ablation."""
    new = dict(entry)

    if ablation == "no_labels":
        new["instruction"] = NO_LABELS_INSTRUCTION
        new["input"] = transform_connectivity_no_labels(entry["input"])
        new["output"] = transform_output_no_labels(entry["output"])

    elif ablation == "no_graph":
        new["instruction"] = NO_GRAPH_INSTRUCTION
        new["input"] = transform_input_no_graph(entry["input"])
        # output stays the same (labeled Z-matrix)

    elif ablation == "order_only":
        smiles = extract_smiles(entry["input"])
        instruction, input_text = _ori_fh_prompt(smiles)
        new["instruction"] = instruction
        new["input"] = input_text
        new["output"] = transform_output_no_labels(entry["output"])

    return new


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="Input template_fh JSONL file")
    parser.add_argument("--ablation", required=True,
                        choices=["no_labels", "no_graph", "order_only"],
                        help="Which ablation to create")
    parser.add_argument("--output", required=True, help="Output JSONL file path")
    args = parser.parse_args()

    n = 0
    with open(args.input) as fin, open(args.output, "w") as fout:
        for line in fin:
            entry = json.loads(line)
            transformed = process_entry(entry, args.ablation)
            fout.write(json.dumps(transformed, ensure_ascii=False) + "\n")
            n += 1

    print(f"Wrote {n} entries to {args.output}")

    # Show a sample
    with open(args.output) as f:
        sample = json.loads(f.readline())
    print(f"\n=== Sample instruction ===\n{sample['instruction']}")
    print(f"\n=== Sample input (first 500 chars) ===\n{sample['input'][:500]}")
    print(f"\n=== Sample output (first 500 chars) ===\n{sample['output'][:500]}")


if __name__ == "__main__":
    main()
