XYZ_SYSTEM_PROMPT = """\
You are a computational chemistry assistant. When given a SMILES string, you \
generate plausible 3D molecular geometries in XYZ format purely from your \
training knowledge. Include all Hydrogen atoms.

Rules:
- Do NOT write or execute any code.
- Do NOT use RDKit, Open Babel, or any cheminformatics toolkit.
- Do NOT use any external tools or calculators.
- Respond ONLY with the XYZ coordinate block. No explanation, no commentary.

The format is:
<element> <x> <y> <z>
...

Coordinates should be in Angstroms with reasonable bond lengths and angles."""

ZMAT_SYSTEM_PROMPT = """\
You are a computational chemistry assistant. When given a SMILES string, you \
generate plausible 3D molecular geometries in Z-matrix (internal coordinates) \
format purely from your training knowledge. Include all Hydrogen atoms.

Rules:
- Do NOT write or execute any code.
- Do NOT use RDKit, Open Babel, or any cheminformatics toolkit.
- Do NOT use any external tools or calculators.
- Respond ONLY with the Z-matrix block. No explanation, no commentary.

Use standard Z-matrix format:
<element>
<element> <ref1> <distance>
<element> <ref1> <distance> <ref2> <angle>
<element> <ref1> <distance> <ref2> <angle> <ref3> <dihedral>
...

<element> MUST be an atomic symbol, and <ref1> MUST be 1-indexed. For example,
"C 3 1.234 4 109.47 5 120.0" is a valid line

Distances in Angstroms, angles in degrees."""

# --- geomllama: the prompts that match the GeomLlama fine-tuned models
# (ori_xyz.jsonl / ori_fh.jsonl, axolotl `type: alpaca`).
#
# These models are completion models, not chat models: they saw a bare alpaca
# template with no chat special tokens. Build the prompt with
# make_geomllama_prompt() and send it as a raw completion -- wrapping it in a
# chat template puts the model off-distribution.
#
# "geomllama_zmat" emits Fenske-Hall z-matrices, which is the "fh" dialect in
# bench_tools.parse_model_coordinates, not the "zmat" one used above.

GEOMLLAMA_INSTRUCTION = (
    "You can generate accurate molecular coordinates from a prompt containing "
    "a SMILES string."
)

# Wording is taken verbatim from the training data. Note that qm9-benchmark.py
# instead says "Z-matrix fmt:" for the Fenske-Hall case -- a stray rename that
# the fine-tuned model tolerated, but which never appeared during training.
GEOMLLAMA_INPUTS = {
    "geomllama_xyz": (
        "Generate a realistic equilibrium geometry for the molecule with the "
        "following SMILES string in xyz format: {smiles}"
    ),
    "geomllama_zmat": (
        "Generate a realistic equilibrium geometry for the molecule with the "
        "following SMILES string in Fenske-Hall Z-matrix format: {smiles}"
    ),
}

GEOMLLAMA_FORMATS = frozenset(GEOMLLAMA_INPUTS)

# The z-matrix dialect each geomllama format emits, for the evaluator.
GEOMLLAMA_EVAL_FORMAT = {"geomllama_xyz": "xyz", "geomllama_zmat": "fh"}


def resolve_eval_format(fmt: str) -> str:
    """Map a prompt format onto the coordinate dialect the parsers understand.

    Prompt formats name *how the model was asked*; parsers only care about what
    comes back. "geomllama_zmat" -> "fh", "geomllama_xyz" -> "xyz"; anything
    else passes through unchanged.
    """
    return GEOMLLAMA_EVAL_FORMAT.get(fmt, fmt)


def make_geomllama_prompt(smiles: str, fmt: str) -> str:
    """Build the raw alpaca completion prompt for a geomllama_* format."""
    try:
        input_text = GEOMLLAMA_INPUTS[fmt].format(smiles=smiles)
    except KeyError:
        raise ValueError(
            f"unknown geomllama format {fmt!r}; "
            f"expected one of {sorted(GEOMLLAMA_INPUTS)}"
        ) from None
    return (
        f"### Instruction:\n{GEOMLLAMA_INSTRUCTION}\n\n"
        f"### Input:\n{input_text}\n\n"
        f"### Response:\n"
    )


def make_user_prompt(smiles: str) -> str:
    return smiles
