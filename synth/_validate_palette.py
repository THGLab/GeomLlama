"""Colorblind-safety check for the categorical plot palette.

Reimplements the CVD validation checks: for every pair of series colors, simulate
deuteranopia and protanopia (Vienot 1999) and require CIEDE2000 separation >= 12.
Also checks a chroma floor and contrast against the chart surface.

    python synth/_validate_palette.py "#4682B4,#2E8B57,#FF8C00"
"""
import sys
import numpy as np

TARGET_DE = 12.0   # CVD >= 12 target; 8-12 only legal with secondary encoding
MIN_CHROMA = 20.0
SURFACE = '#ffffff'


def hex2rgb(h):
    h = h.lstrip('#')
    return np.array([int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)])


def _lin(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _unlin(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


M_RGB2XYZ = np.array([[0.4124564, 0.3575761, 0.1804375],
                      [0.2126729, 0.7151522, 0.0721750],
                      [0.0193339, 0.1191920, 0.9503041]])
WHITE = np.array([0.95047, 1.0, 1.08883])


def rgb2lab(rgb):
    xyz = M_RGB2XYZ @ _lin(rgb) / WHITE
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.array([116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])])


def simulate_cvd(rgb, kind):
    """Vienot/Brettel dichromat simulation in linear LMS."""
    lin = _lin(rgb)
    rgb2lms = np.array([[17.8824, 43.5161, 4.11935],
                        [3.45565, 27.1554, 3.86714],
                        [0.0299566, 0.184309, 1.46709]])
    lms = rgb2lms @ lin
    if kind == 'deuteranopia':
        S = np.array([[1, 0, 0], [0.494207, 0, 1.24827], [0, 0, 1]])
    elif kind == 'protanopia':
        S = np.array([[0, 2.02344, -2.52581], [0, 1, 0], [0, 0, 1]])
    else:  # tritanopia
        S = np.array([[1, 0, 0], [0, 1, 0], [-0.395913, 0.801109, 0]])
    return _unlin(np.linalg.inv(rgb2lms) @ (S @ lms))


def de2000(lab1, lab2):
    L1, a1, b1 = lab1
    L2, a2, b2 = lab2
    C1, C2 = np.hypot(a1, b1), np.hypot(a2, b2)
    Cb = (C1 + C2) / 2
    G = 0.5 * (1 - np.sqrt(Cb**7 / (Cb**7 + 25.0**7))) if Cb > 0 else 0.5
    a1p, a2p = (1 + G) * a1, (1 + G) * a2
    C1p, C2p = np.hypot(a1p, b1), np.hypot(a2p, b2)
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360
    dLp = L2 - L1
    dCp = C2p - C1p
    if C1p * C2p == 0:
        dhp = 0.0
    else:
        dh = h2p - h1p
        dhp = dh - 360 if dh > 180 else (dh + 360 if dh < -180 else dh)
    dHp = 2 * np.sqrt(C1p * C2p) * np.sin(np.radians(dhp) / 2)
    Lbp = (L1 + L2) / 2
    Cbp = (C1p + C2p) / 2
    if C1p * C2p == 0:
        hbp = h1p + h2p
    else:
        s = h1p + h2p
        if abs(h1p - h2p) > 180:
            hbp = (s + 360) / 2 if s < 360 else (s - 360) / 2
        else:
            hbp = s / 2
    T = (1 - 0.17 * np.cos(np.radians(hbp - 30)) + 0.24 * np.cos(np.radians(2 * hbp))
         + 0.32 * np.cos(np.radians(3 * hbp + 6)) - 0.20 * np.cos(np.radians(4 * hbp - 63)))
    dTh = 30 * np.exp(-(((hbp - 275) / 25) ** 2))
    Rc = 2 * np.sqrt(Cbp**7 / (Cbp**7 + 25.0**7)) if Cbp > 0 else 0
    Sl = 1 + (0.015 * (Lbp - 50) ** 2) / np.sqrt(20 + (Lbp - 50) ** 2)
    Sc = 1 + 0.045 * Cbp
    Sh = 1 + 0.015 * Cbp * T
    Rt = -np.sin(np.radians(2 * dTh)) * Rc
    return float(np.sqrt((dLp / Sl) ** 2 + (dCp / Sc) ** 2 + (dHp / Sh) ** 2
                         + Rt * (dCp / Sc) * (dHp / Sh)))


def contrast(rgb_a, rgb_b):
    def lum(c):
        return float(np.dot(_lin(c), [0.2126, 0.7152, 0.0722]))
    a, b = lum(rgb_a), lum(rgb_b)
    hi, lo = max(a, b), min(a, b)
    return (hi + 0.05) / (lo + 0.05)


def main():
    colors = [c.strip() for c in sys.argv[1].split(',')]
    rgbs = [hex2rgb(c) for c in colors]
    surf = hex2rgb(SURFACE)
    ok = True

    print(f'palette: {", ".join(colors)}   surface: {SURFACE}\n')

    print('chroma floor (>= %.0f):' % MIN_CHROMA)
    for c, rgb in zip(colors, rgbs):
        lab = rgb2lab(rgb)
        ch = float(np.hypot(lab[1], lab[2]))
        s = 'PASS' if ch >= MIN_CHROMA else 'FAIL'
        ok &= ch >= MIN_CHROMA
        print(f'  {c}  C*={ch:6.1f}  L*={lab[0]:5.1f}  {s}')

    print('\ncontrast vs surface (>= 3.0 for marks):')
    for c, rgb in zip(colors, rgbs):
        cr = contrast(rgb, surf)
        s = 'PASS' if cr >= 3.0 else 'WARN'
        print(f'  {c}  {cr:4.2f}:1  {s}')

    print(f'\npairwise separation, CIEDE2000 (target >= {TARGET_DE:.0f}):')
    for vision in ('normal', 'deuteranopia', 'protanopia', 'tritanopia'):
        print(f'  --- {vision} ---')
        for i in range(len(colors)):
            for j in range(i + 1, len(colors)):
                ri, rj = rgbs[i], rgbs[j]
                if vision != 'normal':
                    ri, rj = simulate_cvd(ri, vision), simulate_cvd(rj, vision)
                d = de2000(rgb2lab(ri), rgb2lab(rj))
                s = 'PASS' if d >= TARGET_DE else ('FLOOR' if d >= 8 else 'FAIL')
                if d < TARGET_DE:
                    ok = False
                print(f'    {colors[i]} vs {colors[j]}   dE={d:6.1f}  {s}')

    print('\n=> OVERALL:', 'PASS' if ok else 'needs attention')


if __name__ == '__main__':
    main()
