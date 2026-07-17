"""
Connectivity-based prompt generation for molecular geometry formats.

Builds numbered atom lists and bond connectivity from SMILES using RDKit,
with atoms ordered by their appearance in the canonical SMILES string.
"""

from rdkit import Chem


# Bond type suffixes for the connectivity prompt
_BOND_SUFFIX = {
    Chem.rdchem.BondType.SINGLE: "",
    Chem.rdchem.BondType.DOUBLE: "=",
    Chem.rdchem.BondType.TRIPLE: "#",
    Chem.rdchem.BondType.AROMATIC: ".",
}


def get_smiles_atom_order(mol, heavy_only=True):
    """Return the atom index mapping from canonical SMILES order to mol order.

    Args:
        mol: RDKit Mol (may still contain explicit H after RemoveHs).
        heavy_only: If True, filter out hydrogen atoms and remap indices
                    to the heavy-atom-only coordinate list.

    Returns:
        order: list of int, where order[i] is the heavy-atom coordinate
               index for the i-th heavy atom in the canonical SMILES.
    """
    # Generate canonical SMILES — this sets _smilesAtomOutputOrder on mol
    _ = Chem.MolToSmiles(mol, canonical=True)

    order_str = mol.GetProp("_smilesAtomOutputOrder")
    raw_order = list(eval(order_str))

    if not heavy_only:
        return raw_order

    # Build mapping from mol atom idx → heavy-atom coordinate index
    # (skipping H atoms)
    mol_idx_to_heavy = {}
    heavy_idx = 0
    for i in range(mol.GetNumAtoms()):
        if mol.GetAtomWithIdx(i).GetSymbol() != 'H':
            mol_idx_to_heavy[i] = heavy_idx
            heavy_idx += 1

    # Filter to heavy atoms only, remapping indices
    order = []
    for mol_idx in raw_order:
        if mol_idx in mol_idx_to_heavy:
            order.append(mol_idx_to_heavy[mol_idx])

    return order


def reorder_coordinates(coordinates, order):
    """Reorder coordinate list from mol order to SMILES order.

    Args:
        coordinates: list of [element, x, y, z] in mol atom order.
        order: list from get_smiles_atom_order — order[i] is the mol
               atom index for SMILES position i.

    Returns:
        list of [element, x, y, z] in SMILES atom order.
    """
    return [coordinates[order[i]] for i in range(len(order))]


# ---------------------------------------------------------------------------
# Hydrogen-inclusive canonical atom orderings (for labeled_xyz)
#
# The atom order MUST be a pure function of the molecular graph (canonical
# ranks + canonical-SMILES traversal), so that the order computed at training
# time on the dataset mol is reproduced exactly at inference time from
# AddHs(MolFromSmiles(smiles)) — the model only fills in coordinates for a
# scaffold whose atom identities/count come from the SMILES graph.
#
# CRITICAL: the dataset mol must be SANITIZED (aromaticity perceived) before
# computing the order, because ChEMBL3D mols are stored in Kekule form with
# aromaticity unperceived. AddHs(MolFromSmiles(smiles)) always perceives
# aromaticity, so without sanitizing the two graphs get different canonical
# ranks and the order diverges (~55% of aromatic-heterocycle molecules).
# Verified empirically: sanitize -> 100% order reproduction. See
# scripts/_linchpin_probe3.py.
#
# Three orderings, all ablatable:
#   D (default): heavy atoms in canonical-SMILES order, each immediately
#       followed by its bonded H neighbors (H sorted by canonical rank).
#   B: full canonical-SMILES output order over all atoms (H interleaved as RDKit
#       emits them).
#   C: heavy atoms in canonical-SMILES order first, then ALL H at the end
#       (H sorted by canonical rank).
# ---------------------------------------------------------------------------

ORDERINGS = ("D", "B", "C")


def _canonical_smiles_order(mol):
    """Full atom index list in canonical-SMILES output order.

    Appends any atom missing from _smilesAtomOutputOrder (defensive; should not
    happen for a sanitized connected mol) in index order so the result is always
    a complete permutation.
    """
    _ = Chem.MolToSmiles(mol, canonical=True)
    raw = list(eval(mol.GetProp("_smilesAtomOutputOrder")))
    seen = set(raw)
    for a in mol.GetAtoms():
        if a.GetIdx() not in seen:
            raw.append(a.GetIdx())
    return raw


def get_hydrogen_atom_order(mol, ordering="D"):
    """Return a full atom-index permutation of `mol` in the chosen H-inclusive
    canonical order.

    Args:
        mol: RDKit Mol that already carries explicit H atoms. For training data
             it MUST be sanitized first (see module note); pass a sanitized mol.
        ordering: "D", "B", or "C" (see module docstring).

    Returns:
        list of mol atom indices, length == mol.GetNumAtoms().
    """
    if ordering not in ORDERINGS:
        raise ValueError(f"Unknown ordering {ordering!r}; expected one of {ORDERINGS}")

    raw = _canonical_smiles_order(mol)

    if ordering == "B":
        return raw

    ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=True))
    heavy_order = [i for i in raw if mol.GetAtomWithIdx(i).GetAtomicNum() != 1]

    if ordering == "C":
        h_atoms = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 1]
        h_atoms.sort(key=lambda j: ranks[j])
        return heavy_order + h_atoms

    # ordering == "D": H immediately after its parent heavy atom
    order = []
    for h in heavy_order:
        order.append(h)
        hs = [nb.GetIdx() for nb in mol.GetAtomWithIdx(h).GetNeighbors()
              if nb.GetAtomicNum() == 1]
        hs.sort(key=lambda j: ranks[j])
        order.extend(hs)
    return order


def mol_with_hs_from_smiles(smiles):
    """Inference-side reference graph: AddHs(MolFromSmiles(smiles)).

    Returns the sanitized, H-explicit mol, or None if the SMILES can't be
    parsed. This is the canonical graph the model's scaffold is built from, and
    the same graph the training order must reproduce.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.AddHs(mol)


def sanitized_dataset_mol(mol):
    """Return a sanitized copy of a dataset mol (perceives aromaticity).

    Keeps the conformer and explicit H. Use this before get_hydrogen_atom_order
    so the dataset mol's perception matches AddHs(MolFromSmiles(smiles)).
    """
    mm = Chem.Mol(mol)
    Chem.SanitizeMol(mm)
    return mm


def reorder_mol_coords_canonical(mol, ordering="D", sanitize=True):
    """Return [(element, x, y, z), ...] for `mol`'s conformer in the chosen
    canonical H-inclusive order.

    Single source of truth for the coordinate ordering used by BOTH data prep
    (on the dataset mol) and evaluation (on the reference mol), so the training
    coordinate order and the eval reference order are identical.

    Args:
        mol: RDKit Mol with explicit H and a conformer.
        ordering: "D", "B", or "C".
        sanitize: If True (default, for raw dataset mols), perceive aromaticity
                  first so the order matches AddHs(MolFromSmiles(smiles)).

    Returns:
        List of (element, x, y, z) floats in canonical order.
    """
    m = sanitized_dataset_mol(mol) if sanitize else mol
    order = get_hydrogen_atom_order(m, ordering)
    conf = m.GetConformer()
    out = []
    for j in order:
        p = conf.GetAtomPosition(j)
        out.append((m.GetAtomWithIdx(j).GetSymbol(),
                    float(p.x), float(p.y), float(p.z)))
    return out


def get_zmatrix_refs(mol, order):
    """Graph-deterministic Z-matrix references for atoms in `order`.

    For each position i, picks (ref1, ref2, ref3) among already-placed atoms
    (positions < i), preferring bonded atoms so internal coordinates are real
    bond lengths / angles / torsions:
      - ref1: bonded neighbor of atom i with the smallest prior position
              (its parent); falls back to i-1 if none.
      - ref2: bonded neighbor of ref1 with smallest prior position (!= i);
              falls back to the smallest prior position != ref1.
      - ref3: bonded neighbor of ref2 with smallest prior position
              (not in {i, ref1, ref2}); falls back to smallest prior not in
              {ref1, ref2}.
    Pure function of (graph, order) -> reproducible at inference. In canonical-D
    order every atom i>=1 has a bonded prior atom, so the fallbacks rarely fire.

    Args:
        mol: RDKit Mol.
        order: full atom-index permutation (from get_hydrogen_atom_order).

    Returns:
        List of (ref1, ref2, ref3) of 0-based positions; leading entries are
        None where not applicable (i==0 -> (None,None,None), i==1 -> ref2/ref3
        None, i==2 -> ref3 None).
    """
    pos_of = {mol_idx: p for p, mol_idx in enumerate(order)}
    nbrs = {p: [] for p in range(len(order))}
    for p, mol_idx in enumerate(order):
        for nb in mol.GetAtomWithIdx(mol_idx).GetNeighbors():
            if nb.GetIdx() in pos_of:
                nbrs[p].append(pos_of[nb.GetIdx()])

    refs = []
    for i in range(len(order)):
        r1 = r2 = r3 = None
        prior_nbrs = sorted(p for p in nbrs[i] if p < i)
        if i >= 1:
            r1 = prior_nbrs[0] if prior_nbrs else i - 1
        if i >= 2:
            cand = sorted(p for p in nbrs[r1] if p < i and p not in (i, r1))
            r2 = cand[0] if cand else next(p for p in range(i) if p != r1)
        if i >= 3:
            cand = sorted(p for p in nbrs[r2] if p < i and p not in (i, r1, r2))
            r3 = cand[0] if cand else next(p for p in range(i)
                                           if p not in (r1, r2))
        refs.append((r1, r2, r3))
    return refs


def mol_to_zmat_rows(mol, ordering="D", sanitize=True, ref_base=1,
                     dist_dp=4, ang_dp=3):
    """Return (zmat_rows, elements) for `mol`'s conformer as a scaffolded
    Z-matrix in the chosen canonical H-inclusive order.

    Single source of truth for template_fh data prep and eval: computes the
    canonical order, graph-deterministic refs, and internal coordinates that
    round-trip exactly through converter._zmat_to_xyz.

    Args:
        mol: RDKit Mol with explicit H and a conformer.
        ordering: "D", "B", or "C".
        sanitize: perceive aromaticity first (for raw dataset mols).
        ref_base: first reference number. Default 1 (classic 1-based Z-matrix,
                  matching the template_fh 1-based atom labels); pass 0 for
                  0-based refs.

    Returns:
        (rows, elements) where rows is the list from
        converter.coordinates_to_zmat and elements is the element sequence.
    """
    from geomllama.converter import coordinates_to_zmat
    m = sanitized_dataset_mol(mol) if sanitize else mol
    order = get_hydrogen_atom_order(m, ordering)
    refs = get_zmatrix_refs(m, order)
    conf = m.GetConformer()
    positions = [[conf.GetAtomPosition(j).x, conf.GetAtomPosition(j).y,
                  conf.GetAtomPosition(j).z] for j in order]
    elements = [m.GetAtomWithIdx(j).GetSymbol() for j in order]
    rows = coordinates_to_zmat(elements, positions, refs,
                               dist_dp=dist_dp, ang_dp=ang_dp, ref_base=ref_base)
    return rows, elements


def build_connectivity_prompt_h(smiles, ordering="D", index_base=0):
    """Build an H-inclusive numbered-atom + connectivity block for `smiles`.

    Uses AddHs(MolFromSmiles(smiles)) and the chosen H-inclusive canonical
    ordering, so the atom numbering/order is exactly what the scaffolded
    inference loop and the coordinate reorder use.

    Args:
        smiles: SMILES string.
        ordering: "D", "B", or "C".
        index_base: first atom-label number (0 for labeled_xyz; 1 for
            template_fh, whose references follow the classic 1-based Z-matrix
            convention so a reference value matches the referenced atom's label).

    Returns:
        (prompt_text, labels, order) where:
          - prompt_text: connectivity prompt (atoms incl. H, numbered in order)
          - labels: list of "{El}{pos+index_base}" strings (e.g. "C0"/"C1")
          - order: full mol-index permutation (the order list)
        Returns (None, None, None) if SMILES can't be parsed.
    """
    mol = mol_with_hs_from_smiles(smiles)
    if mol is None:
        return None, None, None

    order = get_hydrogen_atom_order(mol, ordering=ordering)

    # mol atom idx -> position in the order
    mol_to_pos = {mol_idx: pos for pos, mol_idx in enumerate(order)}

    labels = [
        f"{mol.GetAtomWithIdx(mol_idx).GetSymbol()}{pos + index_base}"
        for pos, mol_idx in enumerate(order)
    ]

    conn_lines = []
    for pos, mol_idx in enumerate(order):
        atom = mol.GetAtomWithIdx(mol_idx)
        neighbors = []
        for bond in atom.GetBonds():
            other = bond.GetOtherAtomIdx(mol_idx)
            other_pos = mol_to_pos[other]
            suffix = _BOND_SUFFIX.get(bond.GetBondType(), "")
            neighbors.append((other_pos, f"{labels[other_pos]}{suffix}"))
        neighbors.sort(key=lambda x: x[0])
        neighbor_str = ", ".join(nb[1] for nb in neighbors)
        conn_lines.append(f"{labels[pos]} is connected to: {neighbor_str}")

    # Every atom appears as the subject of a connectivity line below, so an
    # explicit "Atoms to place" list would be redundant; omit it.
    prompt = (
        f"SMILES: {smiles}\n"
        f"Connectivity:\n"
        + "\n".join(conn_lines)
    )
    return prompt, labels, order


def build_connectivity_prompt(smiles, remove_hydrogens=True):
    """Build the connectivity block for a SMILES string.

    Args:
        smiles: SMILES string.
        remove_hydrogens: If True, only include heavy atoms.

    Returns:
        Tuple of (prompt_text, order) where:
        - prompt_text: the full connectivity prompt string
        - order: SMILES atom order mapping (for reordering coordinates)
        Returns (None, None) if SMILES can't be parsed.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None

    if remove_hydrogens:
        mol = Chem.RemoveHs(mol)

    order = get_smiles_atom_order(mol, heavy_only=remove_hydrogens)

    # Get the set of mol atom indices we care about (heavy only or all)
    if remove_hydrogens:
        active_mol_indices = [
            i for i in range(mol.GetNumAtoms())
            if mol.GetAtomWithIdx(i).GetSymbol() != 'H'
        ]
    else:
        active_mol_indices = list(range(mol.GetNumAtoms()))

    n = len(order)

    # Build mapping: heavy-atom coordinate index → mol atom index
    # order[smiles_pos] = heavy coord index, active_mol_indices[heavy_idx] = mol idx
    heavy_to_mol = {i: mol_idx for i, mol_idx in enumerate(active_mol_indices)}
    mol_to_heavy = {mol_idx: i for i, mol_idx in enumerate(active_mol_indices)}

    # Build reverse: heavy coord index → SMILES position
    heavy_to_smiles = {}
    for smiles_pos in range(n):
        heavy_to_smiles[order[smiles_pos]] = smiles_pos

    # Build atom labels in SMILES order
    atom_labels = []
    for smiles_pos in range(n):
        heavy_idx = order[smiles_pos]
        mol_idx = heavy_to_mol[heavy_idx]
        symbol = mol.GetAtomWithIdx(mol_idx).GetSymbol()
        atom_labels.append(f"{symbol}{smiles_pos}")

    # Build connectivity lines in SMILES order
    conn_lines = []
    for smiles_pos in range(n):
        heavy_idx = order[smiles_pos]
        mol_idx = heavy_to_mol[heavy_idx]
        atom = mol.GetAtomWithIdx(mol_idx)
        label = atom_labels[smiles_pos]

        neighbors = []
        for bond in atom.GetBonds():
            other_mol_idx = bond.GetOtherAtomIdx(mol_idx)
            if other_mol_idx not in mol_to_heavy:
                continue  # skip bonds to H when remove_hydrogens
            other_heavy_idx = mol_to_heavy[other_mol_idx]
            if other_heavy_idx not in heavy_to_smiles:
                continue
            other_smiles_pos = heavy_to_smiles[other_heavy_idx]
            suffix = _BOND_SUFFIX.get(bond.GetBondType(), "")
            neighbors.append((other_smiles_pos, f"{atom_labels[other_smiles_pos]}{suffix}"))

        # Sort neighbors by their SMILES position
        neighbors.sort(key=lambda x: x[0])
        neighbor_str = ", ".join(nb[1] for nb in neighbors)
        conn_lines.append(f"{label} is connected to: {neighbor_str}")

    atom_type = "Heavy atoms" if remove_hydrogens else "Atoms"
    atom_list_str = ", ".join(atom_labels)

    prompt = (
        f"SMILES: {smiles}\n"
        f"{atom_type} to place: {n} ({atom_list_str})\n"
        f"Connectivity:\n"
        + "\n".join(conn_lines)
    )

    return prompt, order
