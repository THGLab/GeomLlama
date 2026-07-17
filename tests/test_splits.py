"""Tests to verify that data splits match expected sizes and content.

These tests require the processed data files to exist in the data/ directory.
Run after prepare_qm9.py and prepare_geom.py have been executed.
"""

import json
import os
from collections import OrderedDict

import pytest

sklearn = pytest.importorskip("sklearn")
from sklearn.model_selection import train_test_split

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
QM9_TRAIN = os.path.join(DATA_DIR, 'qm9', 'train_set.json')
QM9_TEST = os.path.join(DATA_DIR, 'qm9', 'test_set.json')
GEOM_TRAIN = os.path.join(DATA_DIR, 'geomqm9', 'train_xyz_heavy.json')
GEOM_VAL = os.path.join(DATA_DIR, 'geomqm9', 'val_xyz_heavy.json')
GEOM_TEST = os.path.join(DATA_DIR, 'geomqm9', 'test_xyz_heavy.json')


def _load_json(path):
    with open(path) as f:
        return json.load(f)


# ------------------------------------------------------------------
# QM9 splits
# ------------------------------------------------------------------

@pytest.mark.skipif(not os.path.exists(QM9_TRAIN), reason="QM9 data not found")
class TestQM9Splits:
    @pytest.fixture(autouse=True)
    def load_data(self):
        self.train = _load_json(QM9_TRAIN)
        self.test = _load_json(QM9_TEST)

    def test_total_count(self):
        """QM9 has 133,885 molecules total."""
        total = len(self.train) + len(self.test)
        assert total == 133885

    def test_train_size(self):
        """99:1 split -> ~132,546 train."""
        assert len(self.train) == 132546

    def test_test_size(self):
        """99:1 split -> ~1,339 test."""
        assert len(self.test) == 1339

    def test_no_overlap(self):
        """Train and test sets should have no overlapping filenames."""
        train_fns = {m['filename'] for m in self.train}
        test_fns = {m['filename'] for m in self.test}
        assert len(train_fns & test_fns) == 0

    def test_entry_structure(self):
        """Each entry should have filename, smiles, and coordinates."""
        entry = self.train[0]
        assert 'filename' in entry
        assert 'smiles' in entry
        assert 'coordinates' in entry
        assert len(entry['coordinates']) > 0

    def test_coordinate_format(self):
        """Coordinates should be [element, x, y, z] lists."""
        entry = self.train[0]
        atom = entry['coordinates'][0]
        assert len(atom) == 4
        assert isinstance(atom[0], str)  # element symbol

    def test_split_reproducibility(self):
        """Verify that train_test_split(random_state=42) on sorted filenames
        reproduces the exact same train/test partition.

        This is the core reproducibility check: we reconstruct the input
        list (all filenames sorted), run the same split, and confirm
        every filename lands in the correct set.
        """
        all_entries = self.train + self.test
        # Reconstruct the sorted input order that scrape_qm9_dataset produces
        # (sorted glob of dsgdb9nsd_*.xyz filenames)
        all_entries_sorted = sorted(all_entries, key=lambda e: e['filename'])
        train_split, test_split = train_test_split(
            all_entries_sorted, test_size=0.01, random_state=42,
        )
        expected_test_fns = {e['filename'] for e in test_split}
        actual_test_fns = {e['filename'] for e in self.test}
        assert expected_test_fns == actual_test_fns, (
            f"Split mismatch: {len(expected_test_fns ^ actual_test_fns)} "
            f"filenames differ"
        )

    def test_split_preserves_smiles_pairs(self):
        """Verify SMILES match for a sample of test entries after re-splitting."""
        all_entries = self.train + self.test
        all_entries_sorted = sorted(all_entries, key=lambda e: e['filename'])
        _, test_split = train_test_split(
            all_entries_sorted, test_size=0.01, random_state=42,
        )
        # Build lookup from filename -> smiles for the re-split test set
        resplit_lookup = {e['filename']: e['smiles'] for e in test_split}
        # Check every entry in the actual test set
        for entry in self.test:
            fn = entry['filename']
            assert fn in resplit_lookup, f"{fn} not in re-split test set"
            assert entry['smiles'] == resplit_lookup[fn], (
                f"SMILES mismatch for {fn}: "
                f"{entry['smiles']!r} vs {resplit_lookup[fn]!r}"
            )


# ------------------------------------------------------------------
# GEOM-QM9 splits
# ------------------------------------------------------------------

@pytest.mark.skipif(not os.path.exists(GEOM_TRAIN), reason="GEOM data not found")
class TestGEOMSplits:
    @pytest.fixture(autouse=True)
    def load_data(self):
        self.train = _load_json(GEOM_TRAIN)
        self.val = _load_json(GEOM_VAL)
        self.test = _load_json(GEOM_TEST)

    def test_split_ratio(self):
        """80/10/10 split ratio."""
        total = len(self.train) + len(self.val) + len(self.test)
        train_frac = len(self.train) / total
        val_frac = len(self.val) / total
        test_frac = len(self.test) / total
        assert train_frac == pytest.approx(0.8, abs=0.01)
        assert val_frac == pytest.approx(0.1, abs=0.01)
        assert test_frac == pytest.approx(0.1, abs=0.01)

    def test_minimal_smiles_overlap(self):
        """Splits are by molecule ID (pickle path), not SMILES.

        A small number of duplicate SMILES across splits is expected
        because different pickle paths can map to the same SMILES.
        """
        train_smiles = {m['smiles'] for m in self.train}
        val_smiles = {m['smiles'] for m in self.val}
        test_smiles = {m['smiles'] for m in self.test}
        # Overlap should be small relative to total unique SMILES
        total_unique = len(train_smiles | val_smiles | test_smiles)
        overlap = len(train_smiles & val_smiles) + len(train_smiles & test_smiles) + len(val_smiles & test_smiles)
        assert overlap / total_unique < 0.01, f"Too much SMILES overlap: {overlap}/{total_unique}"

    def test_dot_smiles_are_rare(self):
        """Molecules with '.' in SMILES exist but should be uncommon.

        The '.' filter is applied during SFT data creation, not during
        the initial JSON split.
        """
        all_entries = self.train + self.val + self.test
        dot_count = sum(1 for e in all_entries if '.' in e['smiles'])
        assert dot_count / len(all_entries) < 0.01, f"Too many dot-SMILES: {dot_count}"

    def test_entry_has_conformers(self):
        """Each entry should have at least one conformer."""
        entry = self.train[0]
        assert 'smiles' in entry
        assert 'coordinates' in entry
        # coordinates can be a list of conformers or a single conformer
        assert len(entry['coordinates']) > 0

    def test_heavy_atoms_only(self):
        """Heavy-atom data should not contain hydrogen atoms."""
        entry = self.train[0]
        coords = entry['coordinates']
        # Handle both single conformer and list-of-conformers formats
        if isinstance(coords[0][0], list):
            # List of conformers
            atoms = coords[0]
        else:
            atoms = coords
        for atom in atoms:
            assert atom[0] != 'H', f"Found hydrogen in heavy-atom dataset"

    # ------------------------------------------------------------------
    # GEOM split reproducibility
    # ------------------------------------------------------------------
    # The split is by pickle path using random.seed(19970327) + shuffle,
    # then 80/10/10. We can't re-run split_indices without the raw GEOM
    # data, so we pin exact conformer counts, unique SMILES counts, and
    # sentinel SMILES that would change if the seed or ratio changed.

    def test_exact_conformer_counts(self):
        """Pin exact conformer counts per split."""
        assert len(self.train) == 1374737
        assert len(self.val) == 165204
        assert len(self.test) == 174162

    def test_exact_unique_smiles_counts(self):
        """Pin exact unique SMILES counts per split."""
        train_sm = set(e['smiles'] for e in self.train)
        val_sm = set(e['smiles'] for e in self.val)
        test_sm = set(e['smiles'] for e in self.test)
        assert len(train_sm) == 107404
        assert len(val_sm) == 13455
        assert len(test_sm) == 13451

    def test_exact_smiles_overlap_counts(self):
        """Pin exact SMILES overlap between splits.

        Small overlap is expected because different pickle paths can
        map to the same canonical SMILES.
        """
        train_sm = set(e['smiles'] for e in self.train)
        val_sm = set(e['smiles'] for e in self.val)
        test_sm = set(e['smiles'] for e in self.test)
        assert len(train_sm & val_sm) == 11
        assert len(train_sm & test_sm) == 22
        assert len(val_sm & test_sm) == 0

    def test_train_sentinel_smiles(self):
        """First and last unique SMILES in train must match.

        These are determined by the pickle-path shuffle order. If the
        seed, filtering, or split ratio changes, these will break.
        """
        smiles_order = list(OrderedDict.fromkeys(
            e['smiles'] for e in self.train
        ))
        expected_first = [
            '[H]/N=C(\\C=O)N1CC(O)C1',
            'O=C[C@@]1(O)[C@@H]2COC[C@@H]21',
            'C[C@H]1N[C@H]1C#CCO',
            'CCOC[C@@H](C)C=O',
            'N#C[C@@H]1C=C[C@@H](CO)N1',
        ]
        expected_last = [
            'C1[C@@H]2O[C@H]1[C@H]1N3CC21C3',
            'C1[C@@H]2O[C@H]1[C@@H]1N3CC21C3',
            'N#C[C@@H]1NC=N[C@@H]1C=O',
            'C[C@]12COC[C@H]1[C@H]1O[C@H]12',
            'O[C@H]1CCn2cncc21',
        ]
        assert smiles_order[:5] == expected_first
        assert smiles_order[-5:] == expected_last

    def test_val_sentinel_smiles(self):
        """First and last unique SMILES in val must match."""
        smiles_order = list(OrderedDict.fromkeys(
            e['smiles'] for e in self.val
        ))
        expected_first = [
            'CC[C@@]1(CC=O)CO1',
            'CC1=CC[C@@]23C[C@@]12CO3',
            'NCc1cc(=O)nc[nH]1',
            'CC1=C(O)CC(=O)CC1',
            'C#C[C@@H]1N[C@@H]1C(=O)CC',
        ]
        expected_last = [
            '[H]/N=C(\\N)N(C)C(=O)OC',
            'C[C@H]1C[C@@H](CC(N)=O)C1',
            'N#C[C@H]1OC[C@H]1[C@H]1CO1',
            'C[C@@H]1O[C@H](C)[C@@]1(C)C=O',
            'C[C@]12CC[C@](C#N)(C1)C2',
        ]
        assert smiles_order[:5] == expected_first
        assert smiles_order[-5:] == expected_last

    def test_test_sentinel_smiles(self):
        """First and last unique SMILES in test must match."""
        smiles_order = list(OrderedDict.fromkeys(
            e['smiles'] for e in self.test
        ))
        expected_first = [
            'CC(=O)c1cncnn1',
            'CC(=O)c1conc1O',
            'O=C[C@H]1COCC(=O)C1',
            '[H]/N=C(\\N)C(=O)[C@@H]1C[C@@H]1C',
            'CO[C@@]12[C@H](C)O[C@@H]1[C@@H]2O',
        ]
        expected_last = [
            'CC[C@]12O[C@H]1C[C@H]2OC',
            'CN1[C@@H]2[C@@H]3O[C@H]4[C@@H]([C@@H]32)[C@H]41',
            '[H]/N=C1\\O[C@@H]2[C@H]3O[C@@H]2C[C@@H]13',
            'NC(=O)OC[C@@H](O)CO',
            'C[C@H]1[C@H]2CN(C=O)[C@@H]12',
        ]
        assert smiles_order[:5] == expected_first
        assert smiles_order[-5:] == expected_last
