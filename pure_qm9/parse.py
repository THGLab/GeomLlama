import re

from prompts import resolve_eval_format


def extract_xyz(raw_response: str) -> str | None:
    """Extract coordinate lines from the LLM response text.

    Handles responses wrapped in ``` fences or returned bare.
    Returns cleaned coordinate lines (element x y z), or None if empty.
    """
    # Try fenced code block first
    match = re.search(r"```(?:xyz)?\s*\n(.+?)```", raw_response, re.DOTALL)
    text = match.group(1).strip() if match else raw_response.strip()

    # Keep only lines that look like coordinate lines: element x y z
    coord_lines = []
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) == 4 and not parts[0].isdigit():
            coord_lines.append(line.strip())

    if not coord_lines:
        return None

    return "\n".join(coord_lines)


def extract_zmat(raw_response: str) -> str | None:
    """Extract a Z-matrix block from the LLM response text.

    Returns the cleaned Z-matrix string, or None if parsing fails.
    """
    match = re.search(r"```(?:zmat|z-matrix)?\s*\n(.+?)```", raw_response, re.DOTALL)
    text = match.group(1).strip() if match else raw_response.strip()

    lines = text.splitlines()
    if len(lines) < 1:
        return None

    return "\n".join(line.strip() for line in lines if line.strip())


def parse_result(result: dict) -> dict:
    """Add a 'parsed' key to a result dict from generate_geometry."""
    fmt = resolve_eval_format(result["format"])
    raw = result["raw_response"]

    if fmt == "xyz":
        parsed = extract_xyz(raw)
    else:
        parsed = extract_zmat(raw)

    return {**result, "parsed": parsed}
