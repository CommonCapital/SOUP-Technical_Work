"""Small statistics helpers without scipy (the Kaggle image has it, the Mac env does not)."""

from __future__ import annotations

import math
from collections import Counter


def _gammainc_upper_reg(a: float, x: float) -> float:
    """Q(a, x) = Gamma(a, x) / Gamma(a), series for x < a + 1, continued fraction otherwise."""
    if x <= 0:
        return 1.0
    gln = math.lgamma(a)
    if x < a + 1:
        ap, s, d = a, 1.0 / a, 1.0 / a
        for _ in range(10000):
            ap += 1
            d *= x / ap
            s += d
            if abs(d) < abs(s) * 1e-15:
                break
        return max(0.0, 1.0 - s * math.exp(-x + a * math.log(x) - gln))
    b, c, d = x + 1 - a, 1e300, 1.0 / (x + 1 - a)
    h = d
    for i in range(1, 10000):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        d = 1e-300 if abs(d) < 1e-300 else d
        c = b + an / c
        c = 1e-300 if abs(c) < 1e-300 else c
        d = 1.0 / d
        h *= d * c
        if abs(d * c - 1) < 1e-15:
            break
    return math.exp(-x + a * math.log(x) - gln) * h


def chi2_sf(x: float, df: int) -> float:
    return _gammainc_upper_reg(df / 2.0, x / 2.0)


def chi2_homogeneity(a: list, b: list, min_expected: float = 5.0) -> dict:
    """Chi-square test that two samples of categorical values come from the same
    distribution. Categories whose expected count is < min_expected in either
    sample are pooled into one 'rare' category (standard practice)."""
    ca, cb = Counter(a), Counter(b)
    na, nb = len(a), len(b)
    n = na + nb
    keep, rare_a, rare_b = [], 0, 0
    for c in set(ca) | set(cb):
        tot = ca[c] + cb[c]
        if tot * min(na, nb) / n >= min_expected:
            keep.append((ca[c], cb[c]))
        else:
            rare_a, rare_b = rare_a + ca[c], rare_b + cb[c]
    if rare_a + rare_b > 0:
        keep.append((rare_a, rare_b))
    stat = 0.0
    for xa, xb in keep:
        tot = xa + xb
        for obs, m in ((xa, na), (xb, nb)):
            e = tot * m / n
            stat += (obs - e) ** 2 / e
    df = max(1, len(keep) - 1)
    return {"chi2": stat, "df": df, "p": chi2_sf(stat, df), "categories": len(keep)}
