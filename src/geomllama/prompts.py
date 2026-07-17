"""
Prompt templates for SFT data creation and inference.

Uses the Alpaca format (instruction / input / output).
"""

import json
from geomllama.data_formats import get_format

INSTRUCTION = (
    "You can generate accurate molecular coordinates from a prompt "
    "containing a SMILES string."
)


def make_sft_datapoint(smiles, coordinates, format_name, **kwargs):
    """Create a single SFT training example in Alpaca format.

    Args:
        smiles: SMILES string for the molecule.
        coordinates: Coordinate data (format depends on the geometry format).
        format_name: Name of the registered geometry format.
        **kwargs: Extra arguments passed to format_prompt and format_output
                  (e.g., formula= for formula variants).

    Returns:
        Dict with 'instruction', 'input', and 'output' keys.
    """
    fmt = get_format(format_name)
    return {
        "instruction": fmt.instruction or INSTRUCTION,
        "input": fmt.format_prompt(smiles, **kwargs),
        "output": fmt.format_output(coordinates, **kwargs),
    }


def make_inference_prompt(smiles, format_name, **kwargs):
    """Create a prompt string for inference (no output).

    Args:
        smiles: SMILES string for the molecule.
        format_name: Name of the registered geometry format.
        **kwargs: Extra arguments passed to format_prompt.

    Returns:
        Formatted prompt string ready for model input.
    """
    fmt = get_format(format_name)
    instruction = fmt.instruction or INSTRUCTION
    input_text = fmt.format_prompt(smiles, **kwargs)
    return (
        f"### Instruction:\n{instruction}\n\n"
        f"### Input:\n{input_text}\n\n"
        f"### Response:\n"
    )


def write_jsonl(datapoints, output_path):
    """Write a list of SFT datapoints to a JSONL file.

    Args:
        datapoints: List of dicts (from make_sft_datapoint).
        output_path: Path to write the .jsonl file.
    """
    with open(output_path, 'w') as f:
        for dp in datapoints:
            f.write(json.dumps(dp) + '\n')
