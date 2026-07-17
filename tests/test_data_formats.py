"""Tests for the pluggable geometry format system."""

import pytest

from geomllama.data_formats import (
    GeometryFormat,
    get_format,
    list_formats,
    register_format,
)


# Sample coordinate data (water molecule)
WATER_COORDS = [
    ["O", "0.0000000000", "0.1173646880", "0.0000000000"],
    ["H", "0.7572153434", "-0.4694587520", "0.0000000000"],
    ["H", "-0.7572153434", "-0.4694587520", "0.0000000000"],
]

# Sample FH coordinate data
WATER_FH_COORDS = [
    ["O", "1"],
    ["H", "1", "0.96"],
    ["H", "1", "0.96", "2", "104.5"],
]


class TestRegistry:
    def test_list_formats_nonempty(self):
        fmts = list_formats()
        assert len(fmts) > 0

    def test_known_formats_registered(self):
        fmts = list_formats()
        assert "ori_xyz" in fmts
        assert "roundto3_xyz" in fmts
        assert "ori_fh" in fmts
        assert "template_fh" in fmts

    def test_get_known_format(self):
        fmt = get_format("ori_xyz")
        assert isinstance(fmt, GeometryFormat)

    def test_get_unknown_format_raises(self):
        with pytest.raises(KeyError, match="Unknown format"):
            get_format("nonexistent_format_xyz")

    def test_register_custom_format(self):
        class MyFormat(GeometryFormat):
            name = "_test_custom_fmt"
            def format_prompt(self, smiles, **kw):
                return smiles
            def format_output(self, coords, **kw):
                return "custom"
            def parse_output(self, text):
                return []

        register_format(MyFormat())
        assert "_test_custom_fmt" in list_formats()
        assert get_format("_test_custom_fmt").format_output([]) == "custom"


class TestOriginalXYZ:
    def test_format_prompt(self):
        fmt = get_format("ori_xyz")
        prompt = fmt.format_prompt("O")
        assert "SMILES" in prompt
        assert "O" in prompt
        assert "xyz" in prompt

    def test_format_output(self):
        fmt = get_format("ori_xyz")
        output = fmt.format_output(WATER_COORDS)
        lines = output.strip().split('\n')
        assert len(lines) == 3
        assert lines[0].startswith("O ")
        assert lines[1].startswith("H ")

    def test_parse_output_roundtrip(self):
        fmt = get_format("ori_xyz")
        output = fmt.format_output(WATER_COORDS)
        parsed = fmt.parse_output(output)
        assert parsed is not None
        assert len(parsed) == 3
        assert parsed[0][0] == "O"
        assert parsed[1][0] == "H"

    def test_parse_bad_output(self):
        fmt = get_format("ori_xyz")
        assert fmt.parse_output("not valid xyz") is None

    def test_parse_empty(self):
        fmt = get_format("ori_xyz")
        assert fmt.parse_output("") is None


class TestRoundToXYZ:
    def test_roundto3_output(self):
        fmt = get_format("roundto3_xyz")
        coords = [["C", "1.23456789", "2.34567891", "3.45678912"]]
        output = fmt.format_output(coords)
        line = output.strip()
        parts = line.split()
        assert parts[0] == "C"
        assert parts[1] == "1.235"
        assert parts[2] == "2.346"
        assert parts[3] == "3.457"

    def test_scientific_notation(self):
        """Handles Mathematica-style *^ scientific notation."""
        fmt = get_format("roundto3_xyz")
        coords = [["C", "1.23*^-5", "0.0", "0.0"]]
        output = fmt.format_output(coords)
        parts = output.strip().split()
        assert float(parts[1]) == pytest.approx(0.0, abs=0.001)


class TestOriginalFH:
    def test_format_prompt(self):
        fmt = get_format("ori_fh")
        prompt = fmt.format_prompt("O")
        assert "Fenske-Hall" in prompt
        assert "O" in prompt

    def test_format_output(self):
        fmt = get_format("ori_fh")
        output = fmt.format_output(WATER_FH_COORDS)
        lines = output.strip().split('\n')
        assert len(lines) == 3
        assert lines[0] == "O 1"
        assert "0.96" in lines[1]
        assert "104.5" in lines[2]

    def test_parse_output_returns_coords(self):
        fmt = get_format("ori_fh")
        fh_text = "O 1\nH 1 0.96\nH 1 0.96 2 104.5"
        parsed = fmt.parse_output(fh_text)
        assert parsed is not None
        assert len(parsed) == 3
        assert parsed[0][0] == "O"

    def test_parse_bad_fh(self):
        fmt = get_format("ori_fh")
        assert fmt.parse_output("garbage text") is None
