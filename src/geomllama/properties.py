"""
RDKit-based molecular property extraction for property-prompted formats.
"""

from collections import Counter

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors


def _element_counts(mol):
    """Return {element_symbol: count} including implicit H."""
    mol_h = Chem.AddHs(mol)
    counts = {}
    for atom in mol_h.GetAtoms():
        sym = atom.GetSymbol()
        counts[sym] = counts.get(sym, 0) + 1
    return counts


def _hybridization_counts(mol):
    """Count atoms by hybridization state."""
    hyb = Counter()
    for atom in mol.GetAtoms():
        hyb[str(atom.GetHybridization())] += 1
    return dict(hyb)


def get_rdkit_prompt_properties(smiles):
    """Compute a human-readable block of molecular properties from SMILES.

    Returns a multi-line string suitable for injection into an LLM prompt,
    or None if the SMILES cannot be parsed.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # --- Constitution ---
    formula = rdMolDescriptors.CalcMolFormula(mol)
    mol_weight = Descriptors.ExactMolWt(mol)
    heavy_atoms = mol.GetNumHeavyAtoms()

    # --- Bonding ---
    num_rotatable = rdMolDescriptors.CalcNumRotatableBonds(mol)
    num_single = sum(
        1 for b in mol.GetBonds()
        if b.GetBondType() == Chem.rdchem.BondType.SINGLE
        and not b.GetIsAromatic()
    )
    num_double = sum(
        1 for b in mol.GetBonds()
        if b.GetBondType() == Chem.rdchem.BondType.DOUBLE
    )
    num_triple = sum(
        1 for b in mol.GetBonds()
        if b.GetBondType() == Chem.rdchem.BondType.TRIPLE
    )
    num_aromatic_bonds = sum(
        1 for b in mol.GetBonds() if b.GetIsAromatic()
    )

    # --- Ring info ---
    ring_info = mol.GetRingInfo()
    num_rings = ring_info.NumRings()
    num_aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
    num_aliphatic_rings = rdMolDescriptors.CalcNumAliphaticRings(mol)
    num_saturated_rings = rdMolDescriptors.CalcNumSaturatedRings(mol)
    num_aromatic_heterocycles = rdMolDescriptors.CalcNumAromaticHeterocycles(mol)
    num_aromatic_carbocycles = rdMolDescriptors.CalcNumAromaticCarbocycles(mol)
    num_aliphatic_heterocycles = rdMolDescriptors.CalcNumAliphaticHeterocycles(mol)
    num_spiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
    num_bridgehead = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
    ring_sizes = sorted([len(r) for r in ring_info.AtomRings()])

    # --- Stereo ---
    num_stereocenters = len(
        Chem.FindMolChiralCenters(mol, includeUnassigned=True)
    )
    num_ez = sum(
        1 for b in mol.GetBonds()
        if b.GetStereo() != Chem.rdchem.BondStereo.STEREONONE
    )

    # --- H-bond / polarity ---
    num_hba = rdMolDescriptors.CalcNumHBA(mol)
    num_hbd = rdMolDescriptors.CalcNumHBD(mol)
    tpsa = Descriptors.TPSA(mol)
    logp = Descriptors.MolLogP(mol)
    mr = Descriptors.MolMR(mol)

    # --- Degree of unsaturation ---
    atom_counts = _element_counts(mol)
    c = atom_counts.get("C", 0)
    h = atom_counts.get("H", 0)
    n = atom_counts.get("N", 0)
    halogens = sum(atom_counts.get(x, 0) for x in ("F", "Cl", "Br", "I"))
    dou = (2 * c + 2 + n - halogens - h) / 2

    # --- Complexity ---
    frac_csp3 = rdMolDescriptors.CalcFractionCSP3(mol)
    num_amide_bonds = rdMolDescriptors.CalcNumAmideBonds(mol)
    labute_asa = rdMolDescriptors.CalcLabuteASA(mol)

    # --- Hybridization ---
    hyb = _hybridization_counts(mol)

    # --- Compact strings ---
    elem_str = ", ".join(f"{k}: {v}" for k, v in sorted(atom_counts.items()))
    ring_str = ", ".join(str(s) for s in ring_sizes) if ring_sizes else "none"
    hyb_str = ", ".join(f"{k}: {v}" for k, v in sorted(hyb.items()))

    lines = [
        f"Molecular formula: {formula}",
        f"Exact molecular weight: {mol_weight:.4f}",
        f"Heavy atom count: {heavy_atoms}",
        f"Element counts: {elem_str}",
        f"Rotatable bonds: {num_rotatable}",
        f"Single bonds (non-aromatic): {num_single}",
        f"Double bonds: {num_double}",
        f"Triple bonds: {num_triple}",
        f"Aromatic bonds: {num_aromatic_bonds}",
        f"Total rings: {num_rings}",
        f"Aromatic rings: {num_aromatic_rings}",
        f"Aliphatic rings: {num_aliphatic_rings}",
        f"Saturated rings: {num_saturated_rings}",
        f"Aromatic heterocycles: {num_aromatic_heterocycles}",
        f"Aromatic carbocycles: {num_aromatic_carbocycles}",
        f"Aliphatic heterocycles: {num_aliphatic_heterocycles}",
        f"Spiro atoms: {num_spiro}",
        f"Bridgehead atoms: {num_bridgehead}",
        f"Ring sizes: {ring_str}",
        f"Stereocenters: {num_stereocenters}",
        f"E/Z stereo bonds: {num_ez}",
        f"Degree of unsaturation: {dou:.1f}",
        f"Fraction C-sp3: {frac_csp3:.3f}",
        f"Hybridization counts: {hyb_str}",
        f"H-bond acceptors: {num_hba}",
        f"H-bond donors: {num_hbd}",
        f"Topological polar surface area: {tpsa:.2f}",
        f"LogP (Wildman-Crippen): {logp:.2f}",
        f"Molar refractivity: {mr:.2f}",
        f"Labute approx. surface area: {labute_asa:.2f}",
        f"Amide bonds: {num_amide_bonds}",
    ]

    return "\n".join(lines)
