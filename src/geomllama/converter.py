"""
Convert between Fenske-Hall Z-matrix and Cartesian (XYZ) coordinate formats.

Based on code by Jonathan Arnold (coauthor).
"""

import numpy as np

PERIODIC_TABLE = {
    'H': 1.00794, 'He': 4.002602, 'Li': 6.941, 'Be': 9.012182, 'B': 10.811,
    'C': 12.011, 'N': 14.00674, 'O': 15.9994, 'F': 18.9984032, 'Ne': 20.1797,
    'Na': 22.989768, 'Mg': 24.305, 'Al': 26.981539, 'Si': 28.0855,
    'P': 30.973762, 'S': 32.066, 'Cl': 35.4527, 'Ar': 39.948, 'K': 39.0983,
    'Ca': 40.078, 'Sc': 44.95591, 'Ti': 47.88, 'V': 50.9415, 'Cr': 51.9961,
    'Mn': 54.93805, 'Fe': 55.847, 'Co': 58.9332, 'Ni': 58.6934, 'Cu': 63.546,
    'Zn': 65.39, 'Ga': 69.723, 'Ge': 72.61, 'As': 74.92159, 'Se': 78.96,
    'Br': 79.904, 'Kr': 83.8, 'Rb': 85.4678, 'Sr': 87.62, 'Y': 88.90585,
    'Zr': 91.224, 'Nb': 92.90638, 'Mo': 95.94, 'Tc': 98, 'Ru': 101.07,
    'Rh': 102.9055, 'Pd': 106.42, 'Ag': 107.8682, 'Cd': 112.411,
    'In': 114.82, 'Sn': 118.71, 'Sb': 121.757, 'Te': 127.6, 'I': 126.90447,
    'Xe': 131.29, 'Cs': 132.90543, 'Ba': 137.327, 'La': 138.9055,
    'Ce': 140.115, 'Pr': 140.90765, 'Nd': 144.24, 'Pm': 145, 'Sm': 150.36,
    'Eu': 151.965, 'Gd': 157.25, 'Tb': 158.92534, 'Dy': 162.5,
    'Ho': 164.93032, 'Er': 167.26, 'Tm': 168.93421, 'Yb': 173.04,
    'Lu': 174.967, 'Hf': 178.49, 'Ta': 180.9479, 'W': 183.85, 'Re': 186.207,
    'Os': 190.2, 'Ir': 192.22, 'Pt': 195.08, 'Au': 196.96654, 'Hg': 200.59,
    'Tl': 204.3833, 'Pb': 207.2, 'Bi': 208.98037, 'Po': 209, 'At': 210,
    'Rn': 222, 'Fr': 223, 'Ra': 226.025, 'Ac': 227.028, 'Th': 232.0381,
    'Pa': 231.0359, 'U': 238.0289, 'Np': 237.048, 'Pu': 244, 'Am': 243,
    'Cm': 247, 'Bk': 247, 'Cf': 251, 'Es': 252, 'Fm': 257, 'Md': 258,
    'No': 259, 'Lr': 262, 'Rf': 261, 'Db': 262, 'Sg': 263, 'Bh': 262,
    'Hs': 265, 'Mt': 266,
    '_': 0.0,  # dummy atom
}


def _rotation_matrix(axis, angle):
    """Euler-Rodrigues rotation matrix."""
    axis = np.asarray(axis, float)
    n = np.linalg.norm(axis)
    if n < 1e-8:
        return np.eye(3)
    axis /= n
    a = np.cos(angle / 2)
    b, c, d = -axis * np.sin(angle / 2)
    return np.array([
        [a*a + b*b - c*c - d*d, 2*(b*c - a*d),       2*(b*d + a*c)],
        [2*(b*c + a*d),         a*a + c*c - b*b - d*d, 2*(c*d - a*b)],
        [2*(b*d - a*c),         2*(c*d + a*b),         a*a + d*d - b*b - c*c],
    ])


def parse_fh_string(fh_string):
    """Parse a Fenske-Hall Z-matrix string into internal representation.

    Args:
        fh_string: Multi-line string in FH Z-matrix format, e.g.:
            C 1
            C 1 1.534
            O 2 1.425 1 111.258
            C 2 1.535 1 110.858 3 121.0

    Returns:
        List of [name, [[atom_ref, value], ...], mass] entries.
    """
    zmat = []
    lines = fh_string.strip().split('\n')
    for i, line in enumerate(lines):
        items = line.split()
        name = items[0]
        if name not in PERIODIC_TABLE:
            raise KeyError(f"Unknown element '{name}'")
        mass = PERIODIC_TABLE[name]
        if i == 0:
            zmat.append([name, [[], [], []], mass])
        elif i == 1:
            _, a1, dist = items[:3]
            zmat.append([name, [[int(a1) - 1, float(dist)], [], []], mass])
        elif i == 2:
            _, a1, dist, a2, ang = items[:5]
            zmat.append([name, [
                [int(a1) - 1, float(dist)],
                [int(a2) - 1, np.radians(float(ang))],
                [],
            ], mass])
        else:
            _, a1, dist, a2, ang, a3, dih = items[:7]
            zmat.append([name, [
                [int(a1) - 1, float(dist)],
                [int(a2) - 1, np.radians(float(ang))],
                [int(a3) - 1, np.radians(float(dih))],
            ], mass])
    return zmat


def _zmat_to_xyz(zmat, center=True):
    """Convert parsed Z-matrix to Cartesian coordinates."""
    xyz = []

    # First atom at origin
    xyz.append([zmat[0][0], np.zeros(3), zmat[0][2]])

    # Second atom along x-axis
    if len(zmat) > 1:
        d = zmat[1][1][0][1]
        xyz.append([zmat[1][0], np.array([d, 0.0, 0.0]), zmat[1][2]])

    # Third atom in xy-plane
    if len(zmat) > 2:
        a1, dist = zmat[2][1][0]
        a2, ang = zmat[2][1][1]
        q = xyz[a1][1]
        r = xyz[a2][1]
        v = r - q
        v = dist * v / np.linalg.norm(v)
        v = _rotation_matrix([0, 0, 1], ang) @ v
        xyz.append([zmat[2][0], q + v, zmat[2][2]])

    # Remaining atoms
    for atom in zmat[3:]:
        name, coords, mass = atom
        a1, dist = coords[0]
        a2, ang = coords[1]
        a3, dih = coords[2]

        q = xyz[a1][1]
        r = xyz[a2][1]
        s = xyz[a3][1]

        a = r - q
        b = r - s
        n = np.cross(a, b)

        v = dist * a / np.linalg.norm(a)
        v = _rotation_matrix(n, ang) @ v
        v = _rotation_matrix(a, dih) @ v

        xyz.append([name, q + v, mass])

    # Remove dummy atoms
    xyz = [a for a in xyz if a[0] != '_']

    # Center on center of mass
    if center:
        total_mass = sum(a[2] for a in xyz)
        com = sum(a[2] * a[1] for a in xyz) / total_mass
        for a in xyz:
            a[1] = a[1] - com

    return xyz


def fh_string_to_coordinates_raw(fh_string):
    """Like fh_string_to_coordinates but without center-of-mass recentering.

    Atom 1 at origin, atom 2 along x-axis, atom 3 in xy-plane.
    Suitable for incremental feedback where the final CoM is unknown.
    """
    zmat = parse_fh_string(fh_string)
    xyz = _zmat_to_xyz(zmat, center=False)
    return [(name, float(pos[0]), float(pos[1]), float(pos[2]))
            for name, pos, _ in xyz]


def fh_string_to_xyz_string(fh_string):
    """Convert a Fenske-Hall Z-matrix string to an XYZ coordinate string.

    Args:
        fh_string: Multi-line FH Z-matrix string.

    Returns:
        Multi-line string with "Element x y z" per line.
    """
    zmat = parse_fh_string(fh_string)
    xyz = _zmat_to_xyz(zmat)
    lines = []
    for name, pos, _ in xyz:
        x, y, z = pos
        lines.append(f'{name:<2s} {x:>15.10f} {y:>15.10f} {z:>15.10f}')
    return '\n'.join(lines)


def fh_string_to_coordinates(fh_string):
    """Convert a Fenske-Hall Z-matrix string to a coordinate list.

    Args:
        fh_string: Multi-line FH Z-matrix string.

    Returns:
        List of (element, x, y, z) tuples.
    """
    zmat = parse_fh_string(fh_string)
    xyz = _zmat_to_xyz(zmat)
    return [(name, float(pos[0]), float(pos[1]), float(pos[2]))
            for name, pos, _ in xyz]


# ---------------------------------------------------------------------------
# Cartesian -> Z-matrix encoder (exact inverse of _zmat_to_xyz)
#
# Used by the scaffolded template_fh format: given a conformer and a set of
# graph-deterministic references (ref1=bonded parent, etc.), compute the
# internal coordinates (distance, angle, dihedral) for each atom. The dihedral
# sign is the inverse of the standard torsion formula so that feeding the result
# back through _zmat_to_xyz reproduces the original geometry. Verified
# empirically to ~1e-4 A round-trip RMSD (scripts/_zmat_roundtrip_probe.py).
# ---------------------------------------------------------------------------

# Sign convention that makes coordinates_to_zmat the exact inverse of
# _zmat_to_xyz's R(a, dih) dihedral rotation.
_DIHEDRAL_SIGN = -1.0


def _enc_dist(p, q):
    return float(np.linalg.norm(np.asarray(p) - np.asarray(q)))


def _enc_angle(p_new, p1, p2):
    """Angle (degrees) of the new atom about ref1: angle(new, ref1, ref2)."""
    p_new, p1, p2 = map(np.asarray, (p_new, p1, p2))
    v1 = p_new - p1
    v2 = p2 - p1
    c = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def _enc_dihedral(p_new, p1, p2, p3):
    """Torsion (degrees) new-ref1-ref2-ref3 about the ref1->ref2 axis,
    in the sign convention used by _zmat_to_xyz."""
    p_new, p1, p2, p3 = map(np.asarray, (p_new, p1, p2, p3))
    b0 = p1 - p_new
    b1 = p2 - p1
    b2 = p3 - p2
    b1n = b1 / np.linalg.norm(b1)
    n1 = np.cross(b0, b1)
    n2 = np.cross(b1, b2)
    m1 = np.cross(n1, b1n)
    x = np.dot(n1, n2)
    y = np.dot(m1, n2)
    return float(np.degrees(np.arctan2(_DIHEDRAL_SIGN * y, x)))


def coordinates_to_zmat(elements, positions, refs,
                        dist_dp=4, ang_dp=3, ref_base=0):
    """Encode Cartesian coordinates as Z-matrix rows (inverse of _zmat_to_xyz).

    Args:
        elements: list of element symbols, in placement order.
        positions: array-like (N, 3) Cartesian coords, same order.
        refs: list of (ref1, ref2, ref3) tuples of 0-based positions (entries
              may be None for the first atoms).
        dist_dp / ang_dp: decimal places for distances / angles (and dihedrals).
        ref_base: value added to every reference index in the output (0 keeps
                  0-based refs; pass 1 for classic 1-based FH refs).

    Returns:
        List of rows. Row i is a list of strings:
          i==0: [El]
          i==1: [El, ref1, dist]
          i==2: [El, ref1, dist, ref2, ang]
          i>=3: [El, ref1, dist, ref2, ang, ref3, dih]
    """
    import numpy as _np
    P = _np.asarray(positions, dtype=float)
    rows = []
    for i, el in enumerate(elements):
        if i == 0:
            rows.append([el])
            continue
        r1, r2, r3 = refs[i]
        d = _enc_dist(P[i], P[r1])
        row = [el, str(r1 + ref_base), f"{d:.{dist_dp}f}"]
        if i >= 2:
            a = _enc_angle(P[i], P[r1], P[r2])
            row += [str(r2 + ref_base), f"{a:.{ang_dp}f}"]
        if i >= 3:
            dih = _enc_dihedral(P[i], P[r1], P[r2], P[r3])
            row += [str(r3 + ref_base), f"{dih:.{ang_dp}f}"]
        rows.append(row)
    return rows
