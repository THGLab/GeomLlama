"""Tests for the Fenske-Hall Z-matrix to Cartesian converter."""

import numpy as np
import pytest

from geomllama.converter import (
    PERIODIC_TABLE,
    fh_string_to_coordinates,
    fh_string_to_xyz_string,
    parse_fh_string,
)

# Simple water molecule in FH Z-matrix format (H-O-H)
WATER_FH = """\
O 1
H 1 0.96
H 1 0.96 2 104.5"""

# Methane: C with 4 H atoms
METHANE_FH = """\
C 1
H 1 1.089
H 1 1.089 2 109.4712
H 1 1.089 2 109.4712 3 120.0
H 1 1.089 2 109.4712 3 -120.0"""

# Ethane: 2 carbons, 6 hydrogens
ETHANE_FH = """\
C 1
C 1 1.536
H 1 1.093 2 111.17
H 1 1.093 2 111.17 3 120.0
H 1 1.093 2 111.17 3 -120.0
H 2 1.093 1 111.17 3 60.0
H 2 1.093 1 111.17 6 120.0
H 2 1.093 1 111.17 6 -120.0"""


class TestParseFHString:
    def test_water_atom_count(self):
        zmat = parse_fh_string(WATER_FH)
        assert len(zmat) == 3

    def test_water_elements(self):
        zmat = parse_fh_string(WATER_FH)
        assert [a[0] for a in zmat] == ["O", "H", "H"]

    def test_methane_atom_count(self):
        zmat = parse_fh_string(METHANE_FH)
        assert len(zmat) == 5

    def test_ethane_atom_count(self):
        zmat = parse_fh_string(ETHANE_FH)
        assert len(zmat) == 8

    def test_unknown_element(self):
        with pytest.raises(KeyError, match="Unknown element"):
            parse_fh_string("Xx 1\nC 1 1.5")

    def test_first_atom_no_references(self):
        zmat = parse_fh_string(WATER_FH)
        # First atom has empty reference lists
        assert zmat[0][1] == [[], [], []]

    def test_bond_distance_parsed(self):
        zmat = parse_fh_string(WATER_FH)
        # Second atom: bond to atom 0 with distance 0.96
        assert zmat[1][1][0] == [0, 0.96]

    def test_angle_parsed(self):
        zmat = parse_fh_string(WATER_FH)
        # Third atom: angle 104.5 degrees -> radians
        assert zmat[2][1][1][1] == pytest.approx(np.radians(104.5))


class TestFHStringToCoordinates:
    def test_water_returns_three_atoms(self):
        coords = fh_string_to_coordinates(WATER_FH)
        assert len(coords) == 3

    def test_water_elements(self):
        coords = fh_string_to_coordinates(WATER_FH)
        elements = [c[0] for c in coords]
        assert elements == ["O", "H", "H"]

    def test_water_returns_tuples(self):
        coords = fh_string_to_coordinates(WATER_FH)
        for atom in coords:
            assert len(atom) == 4
            assert isinstance(atom[0], str)
            assert all(isinstance(v, float) for v in atom[1:])

    def test_methane_bond_lengths(self):
        coords = fh_string_to_coordinates(METHANE_FH)
        c_pos = np.array(coords[0][1:])
        for h_atom in coords[1:]:
            h_pos = np.array(h_atom[1:])
            dist = np.linalg.norm(h_pos - c_pos)
            assert dist == pytest.approx(1.089, abs=0.01)

    def test_water_oh_bond_length(self):
        coords = fh_string_to_coordinates(WATER_FH)
        o_pos = np.array(coords[0][1:])
        h1_pos = np.array(coords[1][1:])
        h2_pos = np.array(coords[2][1:])
        assert np.linalg.norm(h1_pos - o_pos) == pytest.approx(0.96, abs=0.01)
        assert np.linalg.norm(h2_pos - o_pos) == pytest.approx(0.96, abs=0.01)

    def test_water_hoh_angle(self):
        coords = fh_string_to_coordinates(WATER_FH)
        o_pos = np.array(coords[0][1:])
        h1_pos = np.array(coords[1][1:])
        h2_pos = np.array(coords[2][1:])
        v1 = h1_pos - o_pos
        v2 = h2_pos - o_pos
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        angle_deg = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
        assert angle_deg == pytest.approx(104.5, abs=0.1)

    def test_ethane_cc_bond_length(self):
        coords = fh_string_to_coordinates(ETHANE_FH)
        c1_pos = np.array(coords[0][1:])
        c2_pos = np.array(coords[1][1:])
        assert np.linalg.norm(c2_pos - c1_pos) == pytest.approx(1.536, abs=0.01)

    def test_centered_on_com(self):
        """Verify the molecule is centered on its center of mass."""
        coords = fh_string_to_coordinates(METHANE_FH)
        masses = [PERIODIC_TABLE[c[0]] for c in coords]
        positions = np.array([c[1:] for c in coords])
        com = np.average(positions, axis=0, weights=masses)
        np.testing.assert_allclose(com, [0.0, 0.0, 0.0], atol=1e-8)


class TestFHStringToXYZString:
    def test_water_line_count(self):
        xyz = fh_string_to_xyz_string(WATER_FH)
        lines = xyz.strip().split('\n')
        assert len(lines) == 3

    def test_water_format(self):
        xyz = fh_string_to_xyz_string(WATER_FH)
        for line in xyz.strip().split('\n'):
            parts = line.split()
            assert len(parts) == 4
            assert parts[0] in PERIODIC_TABLE
            # Remaining should be floats
            for v in parts[1:]:
                float(v)

    def test_roundtrip_consistency(self):
        """xyz_string and coordinates should give the same positions."""
        coords = fh_string_to_coordinates(ETHANE_FH)
        xyz_str = fh_string_to_xyz_string(ETHANE_FH)
        lines = xyz_str.strip().split('\n')
        for i, line in enumerate(lines):
            parts = line.split()
            assert parts[0] == coords[i][0]
            assert float(parts[1]) == pytest.approx(coords[i][1], abs=1e-6)
            assert float(parts[2]) == pytest.approx(coords[i][2], abs=1e-6)
            assert float(parts[3]) == pytest.approx(coords[i][3], abs=1e-6)


class TestDummyAtoms:
    def test_dummy_atoms_removed(self):
        """Dummy atoms ('_') should be stripped from output."""
        fh_with_dummy = """\
_ 1
C 1 1.0
H 1 1.089 2 109.47"""
        coords = fh_string_to_coordinates(fh_with_dummy)
        elements = [c[0] for c in coords]
        assert '_' not in elements
        assert len(coords) == 2
