"""
Pluggable geometry format system for molecular coordinate representations.

Each format defines how to:
- Create the prompt input string from a SMILES string
- Format coordinate data into the expected output string
- Parse an LLM-generated output string back into coordinates

To add a new format, subclass GeometryFormat and register it with
register_format() or use the @geometry_format decorator.
"""

from geomllama.connectivity import (
    build_connectivity_prompt_h,
)
from geomllama.converter import fh_string_to_coordinates


# ---------------------------------------------------------------------------
# Format registry
# ---------------------------------------------------------------------------

_FORMAT_REGISTRY = {}


def register_format(fmt):
    """Register a GeometryFormat instance in the global registry."""
    _FORMAT_REGISTRY[fmt.name] = fmt
    return fmt


def get_format(name):
    """Look up a registered format by name."""
    if name not in _FORMAT_REGISTRY:
        raise KeyError(
            f"Unknown format '{name}'. "
            f"Available: {list(_FORMAT_REGISTRY.keys())}"
        )
    return _FORMAT_REGISTRY[name]


def list_formats():
    """Return names of all registered formats."""
    return list(_FORMAT_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class GeometryFormat:
    """Base class for geometry formats.

    Subclass and implement format_prompt, format_output, and parse_output.
    """

    name = None  # override in subclass
    instruction = None  # override to use a custom instruction for this format

    def format_prompt(self, smiles, **kwargs):
        """Create the 'input' field for the SFT datapoint."""
        raise NotImplementedError

    def format_output(self, coordinates, **kwargs):
        """Create the 'output' field from coordinate data.

        Args:
            coordinates: list of [element, x_str, y_str, z_str] or similar.
        """
        raise NotImplementedError

    def parse_output(self, text):
        """Parse LLM-generated text back into [(element, x, y, z), ...].

        Returns None if parsing fails.
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# XYZ formats
# ---------------------------------------------------------------------------

class OriginalXYZ(GeometryFormat):
    name = "ori_xyz"

    def format_prompt(self, smiles, **kwargs):
        return (
            "Generate a realistic equilibrium geometry for the molecule "
            f"with the following SMILES string in xyz format: {smiles}"
        )

    def format_output(self, coordinates, **kwargs):
        lines = []
        for atom in coordinates:
            element = atom[0]
            x, y, z = atom[1], atom[2], atom[3]
            lines.append(f"{element} {x} {y} {z}")
        return '\n'.join(lines) + '\n'

    def parse_output(self, text):
        try:
            coords = []
            for line in text.strip().split('\n'):
                parts = line.strip().split()
                if len(parts) != 4:
                    return None
                coords.append((parts[0], float(parts[1]),
                               float(parts[2]), float(parts[3])))
            return coords
        except (ValueError, IndexError):
            return None


class RoundToXYZ(GeometryFormat):
    """XYZ with coordinates rounded to N decimal places (no trailing zeros)."""

    def __init__(self, digits):
        self.digits = digits
        self.name = f"roundto{digits}_xyz"

    def format_prompt(self, smiles, **kwargs):
        return (
            "Generate a realistic equilibrium geometry for the molecule "
            f"with the following SMILES string in xyz format: {smiles}"
        )

    def format_output(self, coordinates, **kwargs):
        lines = []
        for atom in coordinates:
            element = atom[0]
            x = str(round(float(str(atom[1]).replace("*^", "e")), self.digits))
            y = str(round(float(str(atom[2]).replace("*^", "e")), self.digits))
            z = str(round(float(str(atom[3]).replace("*^", "e")), self.digits))
            lines.append(f"{element} {x} {y} {z}")
        return '\n'.join(lines) + '\n'

    def parse_output(self, text):
        return OriginalXYZ().parse_output(text)


# ---------------------------------------------------------------------------
# Fenske-Hall Z-matrix format
# ---------------------------------------------------------------------------

class OriginalFH(GeometryFormat):
    name = "ori_fh"

    def format_prompt(self, smiles, **kwargs):
        return (
            "Generate a realistic equilibrium geometry for the molecule "
            "with the following SMILES string in Fenske-Hall Z-matrix "
            f"format: {smiles}"
        )

    def format_output(self, fh_coordinates, **kwargs):
        """Format FH coordinate lines.

        Args:
            fh_coordinates: list of lists, e.g. [['C', '1'], ['C', '1', '1.534', ...]]
        """
        lines = []
        for atom in fh_coordinates:
            cleaned = [atom[0]] + [a.replace("*^", "e") for a in atom[1:]]
            lines.append(' '.join(cleaned))
        return '\n'.join(lines) + '\n'

    def parse_output(self, text):
        """Parse FH z-matrix text back to Cartesian coordinates."""
        try:
            return fh_string_to_coordinates(text)
        except Exception:
            return None


# ---------------------------------------------------------------------------
# Template FH — scaffolded internal-coordinate geometry. Atom
# identities/count/order from the SMILES graph; each atom is placed by
# (distance, angle, dihedral) to graph-deterministic reference atoms; the model
# fills only the internal coordinates. Decodes via the existing FH decoder.
# ---------------------------------------------------------------------------

_TEMPLATE_FH_INSTRUCTION = (
    "You will be given the SMILES string of a molecule and a numbered list of "
    "its atoms (including hydrogens) with their bond connectivity. Generate a "
    "realistic conformer as a Z-matrix: one atom per line in the listed order, "
    "each line giving the reference atom numbers and the bond distance, angle, "
    "and dihedral that place the atom. "
    "Bond notation: = double, # triple, . aromatic."
)


# Z-matrices use the classic 1-based convention: atom labels are numbered from
# 1, and a reference value equals the referenced atom's label number.
_ZMAT_INDEX_BASE = 1


def _strip_label(token):
    """'C12' -> 'C', 'Cl3' -> 'Cl'. Strips the trailing position index."""
    i = len(token)
    while i > 0 and token[i - 1].isdigit():
        i -= 1
    return token[:i]


class TemplateFH(GeometryFormat):
    """Scaffolded Z-matrix where every atom carries an '{Element}{index}' label.

    1-based throughout (classic Z-matrix convention): atom labels start at 1, and
    every reference value is the 1-based number of the atom it points to. Output
    line for the n-th atom (label number n = position+1):
        n==1: 'C1'
        n==2: 'C2 r1 dist'
        n==3: 'O3 r1 dist r2 ang'
        n>=4: 'C4 r1 dist r2 ang r3 dih'
    The label + reference numbers are the deterministic scaffold (from the SMILES
    graph); the model fills only dist/ang/dih. Coordinates passed to
    format_output are the Z-matrix rows from connectivity.mol_to_zmat_rows
    (ref_base=1).

    Args:
        ordering: atom-ordering knob "D" (default), "B", or "C".
    """

    instruction = _TEMPLATE_FH_INSTRUCTION

    def __init__(self, ordering="D"):
        self.ordering = ordering
        self.name = "template_fh" if ordering == "D" else f"template_fh_{ordering.lower()}"

    def format_prompt(self, smiles, **kwargs):
        prompt = kwargs.get('connectivity')
        if prompt is None:
            prompt, _, _ = build_connectivity_prompt_h(
                smiles, ordering=self.ordering, index_base=_ZMAT_INDEX_BASE)
        if prompt is None:
            return f"SMILES: {smiles}"
        return prompt

    def format_output(self, zmat_rows, **kwargs):
        """zmat_rows: rows from connectivity.mol_to_zmat_rows (ref_base=1).
        Row i -> '{El}{i+1} <rest>' (1-based label)."""
        lines = []
        for i, row in enumerate(zmat_rows):
            element = row[0]
            rest = " ".join(str(t).replace("*^", "e") for t in row[1:])
            lines.append(f"{element}{i + _ZMAT_INDEX_BASE} {rest}".rstrip())
        return '\n'.join(lines) + '\n'

    def parse_output(self, text):
        """Strip labels; references are already 1-based -> FH-decode -> coords."""
        try:
            fh_lines = []
            for line in text.strip().split('\n'):
                parts = line.strip().split()
                if not parts:
                    return None
                element = _strip_label(parts[0])
                if not element:
                    return None
                # parts[1:] = [r1, dist, r2, ang, r3, dih] with 1-based refs,
                # exactly the Fenske-Hall line format.
                fh_lines.append(" ".join([element] + parts[1:]))
            return OriginalFH().parse_output("\n".join(fh_lines))
        except (ValueError, IndexError):
            return None

    def labels_for(self, smiles):
        """Scaffold labels ['C1','H2',...] (1-based) from the SMILES alone."""
        _, labels, _ = build_connectivity_prompt_h(
            smiles, ordering=self.ordering, index_base=_ZMAT_INDEX_BASE)
        return labels

    def scaffold_for(self, smiles):
        """Per-atom (label, refs) scaffold for scaffolded z-matrix inference.

        Returns list of (label, (ref1, ref2, ref3)) with 1-based labels and
        1-based references (None where not applicable), computed from the SMILES
        graph alone — the deterministic part the model does NOT generate.
        """
        from geomllama.connectivity import (
            mol_with_hs_from_smiles, get_hydrogen_atom_order, get_zmatrix_refs)
        mol = mol_with_hs_from_smiles(smiles)
        if mol is None:
            return None
        order = get_hydrogen_atom_order(mol, self.ordering)
        refs = get_zmatrix_refs(mol, order)
        refs_1b = [tuple(None if r is None else r + _ZMAT_INDEX_BASE for r in tr)
                   for tr in refs]
        labels = [f"{mol.GetAtomWithIdx(j).GetSymbol()}{i + _ZMAT_INDEX_BASE}"
                  for i, j in enumerate(order)]
        return list(zip(labels, refs_1b))


# ---------------------------------------------------------------------------
# Ablations of template_fh. These mirror, at inference time, the exact
# prompt/output transforms used to build their training data, so a benchmark
# with --format template_fh_no_{labels,graph} --ordering D reproduces the
# training distribution.
# ---------------------------------------------------------------------------

def _strip_label_suffix(token):
    """'C3.' -> 'C.', 'Cl3' -> 'Cl', 'C1' -> 'C'. Strips the trailing position
    index while preserving a trailing bond-type suffix (. = #). Matches
    scripts/create_ablation_data.py::strip_label."""
    suffix = ""
    t = token
    if t and t[-1] in ".=#":
        suffix = t[-1]
        t = t[:-1]
    i = len(t)
    while i > 0 and t[i - 1].isdigit():
        i -= 1
    return t[:i] + suffix


class TemplateFHNoGraph(TemplateFH):
    """Ablation: labeled template_fh output, but the prompt is SMILES only
    (the connectivity graph is removed). Output is identical in form to
    template_fh, so parsing is inherited unchanged."""

    instruction = (
        "You will be given the SMILES string of a molecule. Generate a realistic "
        "conformer as a Z-matrix with labeled atoms: one atom per line, each line "
        "starting with the atom label (e.g. C1, H2) followed by the reference "
        "atom numbers and the bond distance, angle, and dihedral that place the atom."
    )

    def __init__(self, ordering="D"):
        super().__init__(ordering=ordering)
        self.name = ("template_fh_no_graph" if ordering == "D"
                     else f"template_fh_no_graph_{ordering.lower()}")

    def format_prompt(self, smiles, **kwargs):
        return f"SMILES: {smiles}"


class TemplateFHNoLabels(TemplateFH):
    """Ablation: connectivity graph present but every atom label stripped, and
    output is an unlabeled FH z-matrix in template_fh's atom order. Prompt =
    label-stripped connectivity block; output parses like ori_fh (unlabeled).
    Evaluate with ordering matching this format's ordering (default 'D')."""

    instruction = (
        "You will be given the SMILES string of a molecule and a list of its atoms "
        "(including hydrogens) with their bond connectivity. Generate a realistic "
        "conformer as a Z-matrix: one atom per line in the listed order, each line "
        "giving the reference atom numbers and the bond distance, angle, and "
        "dihedral that place the atom. "
        "Bond notation: = double, # triple, . aromatic."
    )

    def __init__(self, ordering="D"):
        super().__init__(ordering=ordering)
        self.name = ("template_fh_no_labels" if ordering == "D"
                     else f"template_fh_no_labels_{ordering.lower()}")

    def format_prompt(self, smiles, **kwargs):
        prompt, _, _ = build_connectivity_prompt_h(
            smiles, ordering=self.ordering, index_base=_ZMAT_INDEX_BASE)
        if prompt is None:
            return f"SMILES: {smiles}"
        out = []
        for line in prompt.split("\n"):
            if " is connected to: " in line:
                subject, rest = line.split(" is connected to: ", 1)
                subj = _strip_label_suffix(subject)
                neigh = [_strip_label_suffix(n.strip()) for n in rest.split(", ")]
                out.append(f"{subj} is connected to: {', '.join(neigh)}")
            else:
                out.append(line)
        return "\n".join(out)

    def parse_output(self, text):
        # Unlabeled FH z-matrix -> coords (ori_fh decoder).
        return OriginalFH().parse_output(text)


# ---------------------------------------------------------------------------
# Register all built-in formats
# ---------------------------------------------------------------------------

register_format(OriginalXYZ())
register_format(RoundToXYZ(3))
register_format(OriginalFH())
for _ord in ("D", "B", "C"):
    register_format(TemplateFH(ordering=_ord))
register_format(TemplateFHNoLabels())
register_format(TemplateFHNoGraph())
