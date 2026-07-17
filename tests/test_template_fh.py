"""CPU smoke tests for the template_fh scaffolded Z-matrix format.

Covers (no dataset required):
  - registration + prompt/output/parse shape
  - exact round-trip: mol -> zmat rows -> format_output -> parse_output -> coords,
    Kabsch RMSD ~0 (the make-or-break: graph z-matrix decodes via the FH decoder)
  - reference determinism / order consistency
  - scaffolded z-matrix inference (mock LLM): exactly-N atoms with correct
    labels AND correct injected references regardless of model output
  - B/C orderings round-trip (ablation hook)
"""
import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from geomllama.data_formats import get_format, list_formats, TemplateFH
from geomllama.connectivity import (
    mol_to_zmat_rows, get_hydrogen_atom_order, get_zmatrix_refs,
    sanitized_dataset_mol, reorder_mol_coords_canonical, ORDERINGS,
)
from geomllama.evaluation import direct_positional_rmsd


SMILES_SET = [
    "O", "CCO", "c1ccccc1O",
    "CC(=O)Oc1ccccc1C(=O)O",            # aspirin
    "Cn1cnc2c1c(=O)n(C)c(=O)n2C",       # caffeine
    "[H]C1=C([H])C(=O)SS1",             # S-S aromatic heterocycle
    "C1=CC=C2C(=C1)C=CC=C2",            # naphthalene
]


def _embed(smiles):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    p = AllChem.ETKDGv3(); p.randomSeed = 0xBEEF
    assert AllChem.EmbedMolecule(mol, p) == 0, f"embed failed {smiles}"
    return mol


def _elem(label):
    i = len(label)
    while i > 0 and label[i - 1].isdigit():
        i -= 1
    return label[:i]


class TestRegistration:
    def test_registered(self):
        f = list_formats()
        assert "template_fh" in f and "template_fh_b" in f and "template_fh_c" in f

    def test_prompt_is_1based_connectivity(self):
        # z-matrix uses the H-inclusive connectivity prompt, numbered from 1
        # (classic Z-matrix convention), unlike labeled_xyz which is 0-based.
        prompt = get_format("template_fh").format_prompt("CCO")
        assert "Connectivity:" in prompt
        first_subject = prompt.split("Connectivity:\n", 1)[1].split(" is connected")[0]
        assert first_subject.endswith("1")          # first atom labelled ...1
        assert "0 is connected to:" not in prompt    # nothing labelled ...0

    def test_output_first_lines_shape(self):
        mol = _embed("CCO")
        rows, _ = mol_to_zmat_rows(mol, "D", sanitize=True, ref_base=1)
        out = get_format("template_fh").format_output(rows).strip().split("\n")
        assert len(out[0].split()) == 1            # root: label only
        assert len(out[1].split()) == 3            # label r1 dist
        assert len(out[2].split()) == 5            # label r1 dist r2 ang
        assert len(out[3].split()) == 7            # + r3 dih


class TestParse:
    def test_parse_roundtrips_simple(self):
        # water-like: build then parse
        mol = _embed("O")
        rows, _ = mol_to_zmat_rows(mol, "D", sanitize=True, ref_base=1)
        fmt = get_format("template_fh")
        parsed = fmt.parse_output(fmt.format_output(rows))
        assert parsed is not None and len(parsed) == mol.GetNumAtoms()

    def test_parse_bad(self):
        assert get_format("template_fh").parse_output("nonsense line here xx") is None


class TestRoundTrip:
    @pytest.mark.parametrize("smiles", SMILES_SET)
    @pytest.mark.parametrize("ordering", ORDERINGS)
    def test_roundtrip_rmsd_zero(self, smiles, ordering):
        mol = _embed(smiles)
        fmt = TemplateFH(ordering=ordering)
        rows, els = mol_to_zmat_rows(mol, ordering, sanitize=True, ref_base=1)
        parsed = fmt.parse_output(fmt.format_output(rows))
        assert parsed is not None
        assert len(parsed) == mol.GetNumAtoms()
        assert [p[0] for p in parsed] == els
        ref = reorder_mol_coords_canonical(mol, ordering, sanitize=True)
        rmsd = direct_positional_rmsd([(e, x, y, z) for e, x, y, z in ref], parsed)
        assert rmsd is not None
        assert rmsd < 1e-2, f"{smiles}/{ordering}: rmsd={rmsd}"


class TestReferenceDeterminism:
    @pytest.mark.parametrize("smiles", SMILES_SET)
    def test_refs_point_backward_and_consistent(self, smiles):
        mol = _embed(smiles)
        ms = sanitized_dataset_mol(mol)
        order = get_hydrogen_atom_order(ms, "D")
        refs = get_zmatrix_refs(ms, order)
        for i, (r1, r2, r3) in enumerate(refs):
            for r in (r1, r2, r3):
                if r is not None:
                    assert r < i                     # references are prior atoms
            if i >= 1: assert r1 is not None
            if i >= 2: assert r2 is not None and r2 != r1
            if i >= 3: assert r3 is not None and r3 not in (r1, r2)

    def test_labels_match_prompt_order(self):
        mol = _embed("CC(=O)Oc1ccccc1C(=O)O")
        rows, els = mol_to_zmat_rows(mol, "D", sanitize=True, ref_base=1)
        scaffold = get_format("template_fh").scaffold_for("CC(=O)Oc1ccccc1C(=O)O")
        # labels from scaffold (inference graph) match the row elements (dataset)
        assert [_elem(lbl) for lbl, _ in scaffold] == els


# ---------------------------------------------------------------------------
# Scaffolded z-matrix inference (mock LLM)
# ---------------------------------------------------------------------------

class _MockOut:
    def __init__(self, text):
        self.outputs = [type("O", (), {"text": text})()]


class _MockLLM:
    def __init__(self, mode="good"):
        self.mode = mode
        self.calls = 0

    def generate(self, prompts, params):
        self.calls += 1
        texts = {"good": "1.523", "garbage": "i refuse to comply",
                 "empty": "", "multi": "1.5 9 2.0 3 4.0"}[self.mode]
        return [_MockOut(texts) for _ in prompts]


class TestScaffoldedZmatInference:
    @pytest.mark.parametrize("mode", ["good", "garbage", "empty", "multi"])
    def test_exact_atoms_labels_and_refs(self, mode):
        from geomllama.inference import generate_scaffolded_zmat
        smiles = "CC(=O)Oc1ccccc1C(=O)O"
        fmt = get_format("template_fh")
        scaffold = fmt.scaffold_for(smiles)
        llm = _MockLLM(mode=mode)
        texts = generate_scaffolded_zmat(llm, "### Response:\n", scaffold, n=2)
        assert len(texts) == 2
        for t in texts:
            lines = t.strip().split("\n")
            # exactly N atoms, labels in order
            assert len(lines) == len(scaffold)
            assert [ln.split()[0] for ln in lines] == [lbl for lbl, _ in scaffold]
            # injected references preserved exactly (even when model emits junk)
            for i, (ln, (lbl, refs)) in enumerate(zip(lines, scaffold)):
                toks = ln.split()
                if i == 0:
                    assert toks == [lbl]
                else:
                    n_fields = 1 if i == 1 else (2 if i == 2 else 3)
                    got_refs = [int(toks[1 + 2 * k]) for k in range(n_fields)]
                    assert got_refs == [refs[k] for k in range(n_fields)]
            # and it decodes to exactly N atoms with correct elements
            parsed = fmt.parse_output(t)
            assert parsed is not None
            assert [_elem(lbl) for lbl, _ in scaffold] == [p[0] for p in parsed]

    def test_field_count_batches(self):
        from geomllama.inference import generate_scaffolded_zmat
        scaffold = get_format("template_fh").scaffold_for("CCO")  # 9 atoms
        llm = _MockLLM(mode="good")
        generate_scaffolded_zmat(llm, "p", scaffold, n=3)
        # generate calls = sum of fields = 1 + 2 + 3*(N-3) for N>=3 atoms
        N = len(scaffold)
        expected = 1 + 2 + 3 * (N - 3)
        assert llm.calls == expected
