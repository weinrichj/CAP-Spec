from __future__ import annotations

import numpy as np


def trapz_integral(wl: np.ndarray, y: np.ndarray, empty_value: float = float("nan")) -> float:
    if wl.size < 2:
        return float(empty_value)
    return float(np.trapz(y, wl))


def safe_ratio(a: float, b: float) -> float:
    if not np.isfinite(a) or not np.isfinite(b) or b == 0:
        return float("nan")
    return float(a / b)

