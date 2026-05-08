#!/usr/bin/env python3
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, savgol_filter
from scipy.special import wofz
from scipy.stats import spearmanr

from analysis.output_paths import (
    SCOPES,
    chemical_modeling_dir,
    ensure_all_scope_layouts,
    metadata_csv_path,
    metadata_section_dir,
)
from plots.style import apply_publication_style, get_palette, style_axes

H_PLANCK = 6.62607015e-34
C_LIGHT = 299792458.0
K_BOLTZMANN = 1.380649e-23
E_CHARGE = 1.602176634e-19

B_PRIME_CM = 1.92
ROTATIONAL_CENTER_NM = 391.44
HBETA_CENTER_NM = 486.1
ROTATIONAL_MODE = "auto"  # auto, line_fit, synthetic_band
PRESSURE_ATM_DEFAULT = 1.0
FIT_BOOTSTRAP_SAMPLES = 300
SPEARMAN_PASS_THRESHOLD = 0.25
SPEARMAN_PVALUE_THRESHOLD = 0.1
MIN_TREND_POINTS = 4
MIN_TREND_LEVELS = 3

VIB_BAND_HEADS_NM = [349.9, 353.6, 357.6, 370.9, 375.4, 380.4, 394.2, 399.7, 405.8]
VIB_BANDS_CSV = Path("configs/n2_vibrational_bands.csv")
EXCITATION_LINES_CSV = Path("configs/excitation_lines.csv")
GAS_CONDITIONS_CSV = Path("configs/gas_conditions.csv")

DISSOCIATION_LINE_NM = {"N_672": 672.0, "Ar_750": 750.4, "O_777": 777.2}
KEY_LINES_NM = {
    "N2plus_391": ROTATIONAL_CENTER_NM,
    "N2plus_428": 428.0,
    "Hbeta_486": HBETA_CENTER_NM,
    "OH_308": 308.0,
    "N2_337": 337.0,
    "N2_379": 379.0,
    "N_672": DISSOCIATION_LINE_NM["N_672"],
    "Ar_750": DISSOCIATION_LINE_NM["Ar_750"],
    "O_777": DISSOCIATION_LINE_NM["O_777"],
    **{f"N2CB_{wl:.1f}": wl for wl in VIB_BAND_HEADS_NM},
}

LINE_LABELS_LATEX = {
    "N2plus_391": r"$N_2^{+}$ (391.44 nm)",
    "N2plus_428": r"$N_2^{+}$ (428.0 nm)",
    "Hbeta_486": r"$H_{\beta}$ (486.1 nm)",
    "OH_308": r"$OH$ (308.0 nm)",
    "N2_337": r"$N_2$ (337.0 nm)",
    "N2_379": r"$N_2$ (379.0 nm)",
    "N_672": r"$N$ (672.0 nm)",
    "Ar_750": r"$Ar$ (750.4 nm)",
    "O_777": r"$O$ (777.2 nm)",
}

REACTION_TO_NODES = [
    ("e + N2 -> e + N2(C)", "N2", "N2(C)"),
    ("e + N2 -> 2e + N2+", "N2", "N2+"),
    ("e + N2 -> e + N + N", "N2", "N"),
    ("e + N2+ -> e + N2+(B)", "N2+", "N2+(B)"),
    ("Ar* + N2 -> Ar + N2(C)", "Ar*", "N2(C)"),
    ("e + O2 -> e + O + O", "O2", "O"),
    ("e + O2 -> O2-", "O2", "O2-"),
    ("N + O2 -> NO + O", "N", "NO"),
    ("O + H2O_trace -> 2OH", "O", "OH"),
]

REACTION_LABELS_LATEX = {
    "e + N2 -> e + N2(C)": r"$e + N_2 \rightarrow e + N_2(C)$",
    "e + N2 -> 2e + N2+": r"$e + N_2 \rightarrow 2e + N_2^{+}$",
    "e + N2 -> e + N + N": r"$e + N_2 \rightarrow e + N + N$",
    "e + N2+ -> e + N2+(B)": r"$e + N_2^{+} \rightarrow e + N_2^{+}(B)$",
    "Ar* + N2 -> Ar + N2(C)": r"$Ar^{*} + N_2 \rightarrow Ar + N_2(C)$",
    "e + O2 -> e + O + O": r"$e + O_2 \rightarrow e + O + O$",
    "e + O2 -> O2-": r"$e + O_2 \rightarrow O_2^{-}$",
    "N + O2 -> NO + O": r"$N + O_2 \rightarrow NO + O$",
    "O + H2O_trace -> 2OH": r"$O + H_2O_{trace} \rightarrow 2OH$",
}

NODE_LABELS_LATEX = {
    "e": r"$e$",
    "N2": r"$N_2$",
    "N2(C)": r"$N_2(C)$",
    "N2+": r"$N_2^{+}$",
    "N2+(B)": r"$N_2^{+}(B)$",
    "N": r"$N$",
    "Ar*": r"$Ar^{*}$",
    "O2": r"$O_2$",
    "O": r"$O$",
    "O2-": r"$O_2^{-}$",
    "NO": r"$NO$",
    "H2O_trace": r"$H_2O_{trace}$",
    "OH": r"$OH$",
}

PATHWAY_PEAK_EVIDENCE = {
    "e + N2 -> e + N2(C)": ["N2_379", "N2_337", "N2CB_353.6", "N2CB_357.6", "N2CB_370.9"],
    "e + N2 -> 2e + N2+": ["N2plus_428", "N2plus_391", "N2CB_394.2", "N2CB_399.7"],
    "e + N2 -> e + N + N": ["N_672", "N2plus_391"],
    "e + N2+ -> e + N2+(B)": ["N2plus_428", "N2plus_391", "N2CB_399.7", "N2CB_405.8"],
    "Ar* + N2 -> Ar + N2(C)": ["Ar_750", "N2_337", "N2CB_353.6"],
    "e + O2 -> e + O + O": ["O_777", "OH_308"],
    "e + O2 -> O2-": ["O_777"],
    "N + O2 -> NO + O": ["N_672", "O_777"],
    "O + H2O_trace -> 2OH": ["OH_308", "O_777"],
}

DEFAULT_VIB_BANDS = pd.DataFrame(
    [
        {"wavelength_nm": 349.9, "v_prime": 4, "v_double_prime": 0, "A_vv": 1.08, "E_v_eV": 1.30, "enabled": 1},
        {"wavelength_nm": 353.6, "v_prime": 3, "v_double_prime": 0, "A_vv": 1.05, "E_v_eV": 1.22, "enabled": 1},
        {"wavelength_nm": 357.6, "v_prime": 2, "v_double_prime": 0, "A_vv": 1.03, "E_v_eV": 1.15, "enabled": 1},
        {"wavelength_nm": 370.9, "v_prime": 4, "v_double_prime": 1, "A_vv": 0.99, "E_v_eV": 1.00, "enabled": 1},
        {"wavelength_nm": 375.4, "v_prime": 3, "v_double_prime": 1, "A_vv": 0.96, "E_v_eV": 0.94, "enabled": 1},
        {"wavelength_nm": 380.4, "v_prime": 2, "v_double_prime": 1, "A_vv": 0.94, "E_v_eV": 0.88, "enabled": 1},
        {"wavelength_nm": 394.2, "v_prime": 4, "v_double_prime": 2, "A_vv": 0.88, "E_v_eV": 0.74, "enabled": 1},
        {"wavelength_nm": 399.7, "v_prime": 3, "v_double_prime": 2, "A_vv": 0.85, "E_v_eV": 0.69, "enabled": 1},
        {"wavelength_nm": 405.8, "v_prime": 2, "v_double_prime": 2, "A_vv": 0.82, "E_v_eV": 0.63, "enabled": 1},
    ]
)

DEFAULT_EXCITATION_LINES = pd.DataFrame(
    [
        {"species": "Ar I", "wavelength_nm": 415.859, "A_ul": 8.03e6, "g_u": 5.0, "E_u_eV": 14.53, "enabled": 1},
        {"species": "Ar I", "wavelength_nm": 420.067, "A_ul": 7.80e6, "g_u": 3.0, "E_u_eV": 14.68, "enabled": 1},
        {"species": "Ar I", "wavelength_nm": 425.936, "A_ul": 3.94e7, "g_u": 5.0, "E_u_eV": 14.73, "enabled": 1},
        {"species": "Ar I", "wavelength_nm": 430.010, "A_ul": 3.77e7, "g_u": 3.0, "E_u_eV": 14.68, "enabled": 1},
        {"species": "O I", "wavelength_nm": 394.728, "A_ul": 9.20e5, "g_u": 5.0, "E_u_eV": 12.08, "enabled": 1},
        {"species": "O I", "wavelength_nm": 436.824, "A_ul": 5.86e5, "g_u": 5.0, "E_u_eV": 12.08, "enabled": 1},
        {"species": "O I", "wavelength_nm": 444.772, "A_ul": 8.95e5, "g_u": 3.0, "E_u_eV": 12.09, "enabled": 1},
    ]
)

GAS_CONDITION_COLS = [
    "dataset",
    "param_set",
    "channel",
    "current_a",
    "ar_to_n2",
    "o2_to_n2",
    "response_ratio_ar",
    "response_ratio_o2",
]

CURRENT_RE = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*a\b", re.IGNORECASE)
AR_RE = re.compile(r"ar[_\- ]*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
O2_RE = re.compile(r"o2[_\- ]*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)


def _safe_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _safe_numeric_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def _safe_text_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series("", index=df.index, dtype=object)
    return df[col].fillna("").astype(str)


def _relative_ci_width(value: float, ci_low: float, ci_high: float) -> float:
    if not np.isfinite(value) or value <= 0:
        return float("nan")
    if not np.isfinite(ci_low) or not np.isfinite(ci_high):
        return float("nan")
    if ci_high < ci_low:
        return float("nan")
    return float((ci_high - ci_low) / value)


def _relative_ci_width_series(value: pd.Series, ci_low: pd.Series, ci_high: pd.Series) -> pd.Series:
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = (ci_high - ci_low) / value
    rel = rel.where(np.isfinite(rel), np.nan)
    rel = rel.where((value > 0) & (ci_high >= ci_low), np.nan)
    return rel


def _fit_temperature_fields(fit: Dict[str, object]) -> Dict[str, float]:
    return {
        "temperature_ci95_low": _safe_float(fit.get("temperature_ci95_low")),
        "temperature_ci95_high": _safe_float(fit.get("temperature_ci95_high")),
        "temperature_bootstrap_ci95_low": _safe_float(fit.get("temperature_bootstrap_ci95_low")),
        "temperature_bootstrap_ci95_high": _safe_float(fit.get("temperature_bootstrap_ci95_high")),
        "rmse": _safe_float(fit.get("rmse")),
        "slope_se": _safe_float(fit.get("slope_se")),
        "bootstrap_samples_used": _safe_float(fit.get("bootstrap_samples_used")),
    }


def _rotational_result(status: str, success: bool = False, mode: str = "line_fit", **kwargs: object) -> Dict[str, object]:
    base: Dict[str, object] = {
        "success": bool(success),
        "mode": str(mode),
        "temperature": float("nan"),
        "r2": float("nan"),
        "rmse": float("nan"),
        "n_points": float("nan"),
        "status": str(status),
        "temperature_ci95_low": float("nan"),
        "temperature_ci95_high": float("nan"),
        "temperature_bootstrap_ci95_low": float("nan"),
        "temperature_bootstrap_ci95_high": float("nan"),
        "slope_se": float("nan"),
        "bootstrap_samples_used": 0.0,
    }
    for key, value in kwargs.items():
        if key in {"status", "mode", "fallback_from"}:
            base[key] = str(value)
        elif key == "success":
            base[key] = bool(value)
        else:
            base[key] = _safe_float(value)
    return base


def _boltzmann_result(status: str, **kwargs: object) -> Dict[str, object]:
    base: Dict[str, object] = {
        "temperature": float("nan"),
        "r2": float("nan"),
        "n_points": float("nan"),
        "status": str(status),
        "temperature_ci95_low": float("nan"),
        "temperature_ci95_high": float("nan"),
        "temperature_bootstrap_ci95_low": float("nan"),
        "temperature_bootstrap_ci95_high": float("nan"),
        "rmse": float("nan"),
        "slope_se": float("nan"),
        "bootstrap_samples_used": 0.0,
    }
    for key, value in kwargs.items():
        if key == "status":
            base[key] = str(value)
        else:
            base[key] = _safe_float(value)
    return base


def _norm_key(text: str) -> str:
    return str(text).strip().lower()


def _read_csv_or_default(path: Path, default_df: pd.DataFrame) -> pd.DataFrame:
    if not path.exists():
        return default_df.copy()
    try:
        df = pd.read_csv(path)
    except Exception:
        return default_df.copy()
    if df.empty:
        return default_df.copy()
    return df


def load_vibrational_metadata() -> pd.DataFrame:
    df = _read_csv_or_default(VIB_BANDS_CSV, DEFAULT_VIB_BANDS)
    needed = {"wavelength_nm", "A_vv", "E_v_eV"}
    if not needed.issubset(df.columns):
        df = DEFAULT_VIB_BANDS.copy()
    for col in ("wavelength_nm", "A_vv", "E_v_eV"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "enabled" in df.columns:
        df = df[pd.to_numeric(df["enabled"], errors="coerce").fillna(1).astype(int) != 0]
    return df.dropna(subset=["wavelength_nm", "A_vv", "E_v_eV"]).sort_values("wavelength_nm", ignore_index=True)


def load_excitation_line_metadata() -> pd.DataFrame:
    df = _read_csv_or_default(EXCITATION_LINES_CSV, DEFAULT_EXCITATION_LINES)
    needed = {"species", "wavelength_nm", "A_ul", "g_u", "E_u_eV"}
    if not needed.issubset(df.columns):
        df = DEFAULT_EXCITATION_LINES.copy()
    for col in ("wavelength_nm", "A_ul", "g_u", "E_u_eV"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "enabled" in df.columns:
        df = df[pd.to_numeric(df["enabled"], errors="coerce").fillna(1).astype(int) != 0]
    return df.dropna(subset=["wavelength_nm", "A_ul", "g_u", "E_u_eV"]).sort_values(
        ["species", "wavelength_nm"], ignore_index=True
    )


def load_gas_conditions() -> pd.DataFrame:
    if not GAS_CONDITIONS_CSV.exists():
        return pd.DataFrame(columns=GAS_CONDITION_COLS)
    try:
        df = pd.read_csv(GAS_CONDITIONS_CSV)
    except Exception:
        return pd.DataFrame(columns=GAS_CONDITION_COLS)
    for col in GAS_CONDITION_COLS:
        if col not in df.columns:
            df[col] = np.nan
    for col in ("dataset", "param_set", "channel"):
        df[col] = df[col].fillna("").astype(str).str.strip().str.lower()
    for col in ("current_a", "ar_to_n2", "o2_to_n2", "response_ratio_ar", "response_ratio_o2"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[GAS_CONDITION_COLS].copy()


def line_baseline(
    wl: np.ndarray,
    y: np.ndarray,
    center_nm: float,
    inner_half_width_nm: float = 0.7,
    outer_half_width_nm: float = 1.6,
) -> np.ndarray:
    left_mask = (wl >= center_nm - outer_half_width_nm) & (wl < center_nm - inner_half_width_nm)
    right_mask = (wl > center_nm + inner_half_width_nm) & (wl <= center_nm + outer_half_width_nm)
    if left_mask.any() and right_mask.any():
        xl = float(np.nanmean(wl[left_mask]))
        yl = float(np.nanmean(y[left_mask]))
        xr = float(np.nanmean(wl[right_mask]))
        yr = float(np.nanmean(y[right_mask]))
        if np.isfinite(xl) and np.isfinite(yl) and np.isfinite(xr) and np.isfinite(yr) and xr > xl:
            return np.interp(wl, [xl, xr], [yl, yr])
    edge = np.nanmedian(y)
    if not np.isfinite(edge):
        edge = float(np.nanmin(y)) if np.isfinite(y).any() else 0.0
    return np.full_like(wl, edge, dtype=float)


def integrated_line_signal(
    wl: np.ndarray,
    y: np.ndarray,
    center_nm: float,
    inner_half_width_nm: float = 0.7,
    outer_half_width_nm: float = 1.6,
) -> Dict[str, float]:
    mask = (wl >= center_nm - inner_half_width_nm) & (wl <= center_nm + inner_half_width_nm)
    n_points = int(mask.sum())
    if n_points == 0:
        return {"line_area": float("nan"), "line_peak": float("nan"), "line_points": 0.0}
    if n_points == 1:
        y0 = float(y[mask][0])
        side = y[(wl >= center_nm - outer_half_width_nm) & (wl <= center_nm + outer_half_width_nm) & (~mask)]
        base = float(np.nanmedian(side)) if side.size else float(np.nanmedian(y))
        if not np.isfinite(base):
            base = 0.0
        peak = max(y0 - base, 0.0)
        return {"line_area": float(peak * 1.0), "line_peak": float(peak), "line_points": 1.0}

    wl_line = wl[mask]
    y_line = y[mask]
    base = line_baseline(wl_line, y_line, center_nm, min(inner_half_width_nm, 0.35), max(outer_half_width_nm, 0.7))
    y_corr = np.clip(y_line - base, 0.0, None)
    line_area = float(np.trapz(y_corr, wl_line))
    line_peak = float(np.nanmax(y_corr)) if y_corr.size else float("nan")
    return {"line_area": line_area, "line_peak": line_peak, "line_points": float(y_corr.size)}


def fit_linear_temperature(x_vals: np.ndarray, y_vals: np.ndarray) -> Dict[str, float]:
    x = np.asarray(x_vals, dtype=float)
    y = np.asarray(y_vals, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    n_raw = int(x.size)

    out = {
        "temperature": float("nan"),
        "slope": float("nan"),
        "intercept": float("nan"),
        "r2": float("nan"),
        "rmse": float("nan"),
        "slope_se": float("nan"),
        "intercept_se": float("nan"),
        "temperature_se": float("nan"),
        "temperature_ci95_low": float("nan"),
        "temperature_ci95_high": float("nan"),
        "temperature_bootstrap_ci95_low": float("nan"),
        "temperature_bootstrap_ci95_high": float("nan"),
        "bootstrap_samples_used": 0.0,
        "n_points_raw": float(n_raw),
        "n_points_used": float(n_raw),
        "outlier_fraction": 0.0,
    }
    if n_raw < 3:
        return out

    try:
        slope0, intercept0 = np.polyfit(x, y, 1)
    except Exception:
        return out

    if n_raw >= 6:
        resid0 = y - (intercept0 + (slope0 * x))
        med = float(np.nanmedian(resid0))
        mad = float(np.nanmedian(np.abs(resid0 - med)))
        if np.isfinite(mad) and mad > 0:
            robust_sigma = 1.4826 * mad
            keep = np.abs(resid0 - med) <= (3.5 * robust_sigma)
            keep_n = int(keep.sum())
            if keep_n >= 3 and keep_n < n_raw:
                x = x[keep]
                y = y[keep]

    n = int(x.size)
    out["n_points_used"] = float(n)
    if n_raw > 0 and n <= n_raw:
        out["outlier_fraction"] = float((n_raw - n) / n_raw)
    if n < 3:
        return out

    try:
        slope, intercept = np.polyfit(x, y, 1)
    except Exception:
        return out
    pred = intercept + (slope * x)
    resid = y - pred
    ss_res = float(np.nansum(resid**2))
    ss_tot = float(np.nansum((y - np.nanmean(y)) ** 2))
    r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else float("nan")
    rmse = float(np.sqrt(ss_res / max(n, 1)))

    x_bar = float(np.nanmean(x))
    sxx = float(np.nansum((x - x_bar) ** 2))
    slope_se = float("nan")
    intercept_se = float("nan")
    if n > 2 and np.isfinite(sxx) and sxx > 0:
        sigma2 = ss_res / float(n - 2)
        if np.isfinite(sigma2) and sigma2 >= 0:
            slope_se = float(np.sqrt(sigma2 / sxx))
            intercept_se = float(np.sqrt(sigma2 * ((1.0 / n) + (x_bar * x_bar / sxx))))

    out.update(
        {
            "slope": float(slope),
            "intercept": float(intercept),
            "r2": r2,
            "rmse": rmse,
            "slope_se": slope_se,
            "intercept_se": intercept_se,
        }
    )

    if slope < 0 and np.isfinite(slope):
        temperature = -1.0 / (K_BOLTZMANN * slope)
        out["temperature"] = float(temperature)
        if np.isfinite(slope_se):
            temp_se = abs(1.0 / (K_BOLTZMANN * (slope**2))) * slope_se
            out["temperature_se"] = float(temp_se)
            out["temperature_ci95_low"] = float(max(0.0, temperature - (1.96 * temp_se)))
            out["temperature_ci95_high"] = float(temperature + (1.96 * temp_se))

    if n >= 4:
        rng_seed = int((abs(np.nanmean(x)) * 1e6) + (abs(np.nanmean(y)) * 1e6) + n) % (2**32 - 1)
        rng = np.random.default_rng(rng_seed)
        boot_t: List[float] = []
        for _ in range(FIT_BOOTSTRAP_SAMPLES):
            idx = rng.integers(0, n, size=n)
            xb = x[idx]
            yb = y[idx]
            if np.unique(xb).size < 3:
                continue
            try:
                sb, _ = np.polyfit(xb, yb, 1)
            except Exception:
                continue
            if np.isfinite(sb) and sb < 0:
                tb = -1.0 / (K_BOLTZMANN * sb)
                if np.isfinite(tb) and tb > 0:
                    boot_t.append(float(tb))
        if boot_t:
            boot_arr = np.asarray(boot_t, dtype=float)
            out["temperature_bootstrap_ci95_low"] = float(np.nanpercentile(boot_arr, 2.5))
            out["temperature_bootstrap_ci95_high"] = float(np.nanpercentile(boot_arr, 97.5))
            out["bootstrap_samples_used"] = float(boot_arr.size)
            if not np.isfinite(out["temperature_ci95_low"]) or not np.isfinite(out["temperature_ci95_high"]):
                out["temperature_ci95_low"] = out["temperature_bootstrap_ci95_low"]
                out["temperature_ci95_high"] = out["temperature_bootstrap_ci95_high"]
    return out


def rotational_line_fit(wl: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    mask = (wl >= ROTATIONAL_CENTER_NM - 2.5) & (wl <= ROTATIONAL_CENTER_NM + 2.5)
    if int(mask.sum()) < 6:
        return _rotational_result("insufficient_window_points", success=False, mode="line_fit", n_points=0.0)

    xw = wl[mask]
    yw = y[mask]
    ys = yw.copy()
    if xw.size >= 7:
        win = int(min(11, xw.size if xw.size % 2 == 1 else xw.size - 1))
        if win >= 5:
            ys = savgol_filter(yw, window_length=win, polyorder=2, mode="interp")

    prominence = float(np.nanstd(ys)) * 0.18
    if not np.isfinite(prominence) or prominence <= 0:
        prominence = float(np.nanmax(ys) - np.nanmin(ys)) * 0.1
    peak_idx, _ = find_peaks(ys, prominence=max(prominence, 1e-20))
    if peak_idx.size < 4:
        return _rotational_result(
            "insufficient_detected_rotational_peaks",
            success=False,
            mode="line_fit",
            n_points=float(peak_idx.size),
        )

    peak_int = ys[peak_idx]
    order = np.argsort(peak_int)[::-1][: min(10, peak_idx.size)]
    pick_idx = np.sort(peak_idx[order])
    peak_y = ys[pick_idx]
    baseline = float(np.nanmin(ys))
    peak_corr = np.clip(peak_y - baseline, 0.0, None)
    peak_corr = peak_corr[np.isfinite(peak_corr) & (peak_corr > 0)]
    if peak_corr.size < 4:
        return _rotational_result("insufficient_positive_peaks", success=False, mode="line_fit", n_points=float(peak_corr.size))

    k_prime = np.arange(1, peak_corr.size + 1, dtype=float)
    x_rot = k_prime * (k_prime + 1.0)
    y_rot = np.log(peak_corr / np.maximum(k_prime, 1e-12))
    fit = fit_linear_temperature(x_rot, y_rot)
    if not np.isfinite(fit["temperature"]):
        return _rotational_result(
            "nonphysical_rotational_slope",
            success=False,
            mode="line_fit",
            r2=fit["r2"],
            n_points=float(peak_corr.size),
            **_fit_temperature_fields(fit),
        )

    pred = fit["intercept"] + (fit["slope"] * x_rot)
    rmse = float(np.sqrt(np.nanmean((y_rot - pred) ** 2)))
    return _rotational_result(
        "ok",
        success=True,
        mode="line_fit",
        temperature=fit["temperature"],
        r2=fit["r2"],
        n_points=float(peak_corr.size),
        **_fit_temperature_fields(fit),
        rmse=rmse,
    )


def synthetic_band_profile(
    x_nm: np.ndarray,
    temperature_k: float,
    sigma_nm: float,
    spacing_nm: float = 0.06,
    k_max: int = 36,
) -> np.ndarray:
    k_vals = np.arange(1, k_max + 1, dtype=float)
    factor = (B_PRIME_CM * 100.0 * H_PLANCK * C_LIGHT) / (K_BOLTZMANN * max(temperature_k, 1e-6))
    intens = k_vals * np.exp(-factor * k_vals * (k_vals + 1.0))
    r_centers = ROTATIONAL_CENTER_NM + (spacing_nm * k_vals)
    p_centers = ROTATIONAL_CENTER_NM - (spacing_nm * (k_vals + 0.35))

    x_col = x_nm[:, None]
    g_r = np.exp(-0.5 * ((x_col - r_centers[None, :]) / sigma_nm) ** 2)
    g_p = np.exp(-0.5 * ((x_col - p_centers[None, :]) / sigma_nm) ** 2)
    return ((g_r @ intens) + (0.78 * (g_p @ intens))).astype(float)


def _evaluate_rotational_grid(
    xw: np.ndarray,
    y_corr: np.ndarray,
    temps: np.ndarray,
    sigmas: np.ndarray,
    best: Dict[str, float],
    accepted: List[Tuple[float, float]],
) -> Dict[str, float]:
    for temp in temps:
        for sigma in sigmas:
            p = synthetic_band_profile(xw, temperature_k=float(temp), sigma_nm=float(sigma))
            a = np.column_stack([p, np.ones_like(p)])
            try:
                coeff, *_ = np.linalg.lstsq(a, y_corr, rcond=None)
            except np.linalg.LinAlgError:
                continue
            scale = float(coeff[0])
            offset = float(coeff[1])
            if not np.isfinite(scale) or scale <= 0:
                continue
            pred = (scale * p) + offset
            rmse = float(np.sqrt(np.nanmean((y_corr - pred) ** 2)))
            ss_res = float(np.nansum((y_corr - pred) ** 2))
            ss_tot = float(np.nansum((y_corr - np.nanmean(y_corr)) ** 2))
            r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else float("nan")
            if rmse < best["rmse"]:
                best = {
                    "rmse": rmse,
                    "temperature": float(temp),
                    "r2": r2,
                    "scale": scale,
                    "offset": offset,
                    "sigma": float(sigma),
                }
            accepted.append((float(temp), rmse))
    return best


def rotational_synthetic_fit(wl: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    mask = (wl >= ROTATIONAL_CENTER_NM - 4.0) & (wl <= ROTATIONAL_CENTER_NM + 4.0)
    if int(mask.sum()) < 8:
        return _rotational_result("insufficient_window_points", success=False, mode="synthetic_band", n_points=0.0)

    xw = wl[mask]
    yw = y[mask].astype(float)
    baseline = line_baseline(xw, yw, center_nm=ROTATIONAL_CENTER_NM, inner_half_width_nm=2.0, outer_half_width_nm=3.8)
    y_corr = np.clip(yw - baseline, 0.0, None)
    if not np.isfinite(y_corr).any() or float(np.nanmax(y_corr)) <= 0:
        return _rotational_result("no_positive_signal", success=False, mode="synthetic_band", n_points=float(xw.size))

    temps = np.linspace(450.0, 6500.0, 76)
    sigmas = np.linspace(0.10, 1.20, 24)
    best = {"rmse": float("inf"), "temperature": float("nan"), "r2": float("nan")}
    accepted: List[Tuple[float, float]] = []
    best = _evaluate_rotational_grid(xw, y_corr, temps, sigmas, best, accepted)

    best_temp = _safe_float(best.get("temperature"))
    best_sigma = _safe_float(best.get("sigma"))
    if np.isfinite(best_temp) and np.isfinite(best_sigma):
        t_low = max(350.0, best_temp - 700.0)
        t_high = min(8500.0, best_temp + 700.0)
        s_low = max(0.05, best_sigma - 0.15)
        s_high = min(1.8, best_sigma + 0.15)
        refine_t = np.linspace(t_low, t_high, 55)
        refine_s = np.linspace(s_low, s_high, 24)
        best = _evaluate_rotational_grid(xw, y_corr, refine_t, refine_s, best, accepted)

    if not np.isfinite(best.get("temperature", np.nan)):
        return _rotational_result("grid_search_failed", success=False, mode="synthetic_band", n_points=float(xw.size))

    ci_low = float("nan")
    ci_high = float("nan")
    if accepted and np.isfinite(best["rmse"]) and best["rmse"] > 0:
        acc = np.asarray(accepted, dtype=float)
        threshold = float(best["rmse"]) * 1.05
        t_keep = acc[acc[:, 1] <= threshold, 0]
        if t_keep.size:
            ci_low = float(np.nanpercentile(t_keep, 2.5))
            ci_high = float(np.nanpercentile(t_keep, 97.5))

    if not np.isfinite(ci_low) or not np.isfinite(ci_high):
        ci_low = max(0.0, float(best["temperature"]) * 0.9)
        ci_high = float(best["temperature"]) * 1.1

    return _rotational_result(
        "ok",
        success=True,
        mode="synthetic_band",
        temperature=best["temperature"],
        r2=best["r2"],
        rmse=best["rmse"],
        n_points=float(xw.size),
        sigma_nm=best["sigma"],
        temperature_ci95_low=ci_low,
        temperature_ci95_high=ci_high,
    )


def estimate_rotational_temperature(wl: np.ndarray, y: np.ndarray, mode: str = ROTATIONAL_MODE) -> Dict[str, float]:
    if mode == "line_fit":
        return rotational_line_fit(wl, y)
    if mode == "synthetic_band":
        return rotational_synthetic_fit(wl, y)
    line = rotational_line_fit(wl, y)
    if line["success"]:
        return line
    synth = rotational_synthetic_fit(wl, y)
    if synth["success"]:
        synth["fallback_from"] = line.get("status", "line_fit_failed")
    return synth


def estimate_vibrational_temperature(wl: np.ndarray, y: np.ndarray, vib_meta: pd.DataFrame) -> Dict[str, float]:
    rows: List[Dict[str, float]] = []
    for _, r in vib_meta.iterrows():
        center = float(r["wavelength_nm"])
        signal = integrated_line_signal(wl, y, center_nm=center, inner_half_width_nm=0.8, outer_half_width_nm=1.8)
        intensity = signal["line_area"]
        a_vv = float(r["A_vv"])
        e_ev = float(r["E_v_eV"])
        if not np.isfinite(intensity) or intensity <= 0 or not np.isfinite(a_vv) or a_vv <= 0 or not np.isfinite(e_ev):
            continue
        nu = C_LIGHT / (center * 1e-9)
        y_log = np.log(intensity) - (4.0 * np.log(nu)) - np.log(a_vv)
        rows.append({"x": e_ev * E_CHARGE, "y": y_log})

    if len(rows) < 4:
        return _boltzmann_result("insufficient_vibrational_bands", n_points=float(len(rows)))

    fit_df = pd.DataFrame(rows)
    fit = fit_linear_temperature(fit_df["x"].to_numpy(dtype=float), fit_df["y"].to_numpy(dtype=float))
    status = "ok" if np.isfinite(fit["temperature"]) else "nonphysical_vibrational_slope"
    return _boltzmann_result(
        status,
        temperature=fit["temperature"],
        r2=fit["r2"],
        n_points=float(len(rows)),
        **_fit_temperature_fields(fit),
    )


def estimate_excitation_temperature(wl: np.ndarray, y: np.ndarray, line_meta: pd.DataFrame) -> Dict[str, float]:
    rows: List[Dict[str, float]] = []
    for _, r in line_meta.iterrows():
        center = float(r["wavelength_nm"])
        signal = integrated_line_signal(wl, y, center_nm=center, inner_half_width_nm=0.7, outer_half_width_nm=1.7)
        intensity = signal["line_area"]
        a_ul = float(r["A_ul"])
        g_u = float(r["g_u"])
        e_ev = float(r["E_u_eV"])
        if (
            not np.isfinite(intensity)
            or intensity <= 0
            or not np.isfinite(a_ul)
            or a_ul <= 0
            or not np.isfinite(g_u)
            or g_u <= 0
            or not np.isfinite(e_ev)
        ):
            continue
        wavelength_m = center * 1e-9
        y_log = np.log((intensity * wavelength_m) / (a_ul * g_u))
        rows.append({"x": e_ev * E_CHARGE, "y": y_log})

    if len(rows) < 3:
        return _boltzmann_result("insufficient_atomic_lines", n_points=float(len(rows)))

    fit_df = pd.DataFrame(rows)
    fit = fit_linear_temperature(fit_df["x"].to_numpy(dtype=float), fit_df["y"].to_numpy(dtype=float))
    status = "ok" if np.isfinite(fit["temperature"]) else "nonphysical_excitation_slope"
    return _boltzmann_result(
        status,
        temperature=fit["temperature"],
        r2=fit["r2"],
        n_points=float(len(rows)),
        **_fit_temperature_fields(fit),
    )


def voigt_profile_model(x: np.ndarray, amp: float, center: float, sigma: float, gamma: float, offset: float) -> np.ndarray:
    z = ((x - center) + (1j * gamma)) / (sigma * np.sqrt(2.0))
    profile = np.real(wofz(z)) / (sigma * np.sqrt(2.0 * np.pi))
    return (amp * profile) + offset


def _electron_density_result(status: str, **kwargs: object) -> Dict[str, float]:
    base: Dict[str, float] = {
        "estimated_electron_density": float("nan"),
        "estimated_electron_density_ci95_low": float("nan"),
        "estimated_electron_density_ci95_high": float("nan"),
        "hbeta_lorentz_fwhm_nm": float("nan"),
        "hbeta_lorentz_fwhm_ci95_low_nm": float("nan"),
        "hbeta_lorentz_fwhm_ci95_high_nm": float("nan"),
        "hbeta_stark_fwhm_nm": float("nan"),
        "hbeta_vdw_fwhm_nm": float("nan"),
        "electron_density_fit_r2": float("nan"),
        "electron_density_status": status,
    }
    for key, value in kwargs.items():
        if key == "electron_density_status":
            base[key] = str(value)
        else:
            base[key] = _safe_float(value)
    return base


def estimate_electron_density(
    wl: np.ndarray,
    y: np.ndarray,
    rotational_temperature_k: float,
    pressure_atm: float = PRESSURE_ATM_DEFAULT,
) -> Dict[str, float]:
    mask = (wl >= HBETA_CENTER_NM - 4.0) & (wl <= HBETA_CENTER_NM + 4.0)
    if int(mask.sum()) < 8:
        return _electron_density_result("insufficient_hbeta_points")

    xw = wl[mask]
    yw = y[mask]
    y_base = line_baseline(xw, yw, center_nm=HBETA_CENTER_NM, inner_half_width_nm=1.6, outer_half_width_nm=3.8)
    y_corr = np.clip(yw - y_base, 0.0, None)
    if not np.isfinite(y_corr).any() or float(np.nanmax(y_corr)) <= 0:
        return _electron_density_result("no_hbeta_signal")

    amp0 = float(np.nanmax(y_corr))
    p0 = [amp0, HBETA_CENTER_NM, 0.25, 0.25, 0.0]
    bounds = ([0.0, HBETA_CENTER_NM - 1.0, 0.02, 0.005, -np.inf], [np.inf, HBETA_CENTER_NM + 1.0, 3.0, 3.0, np.inf])

    try:
        popt, pcov = curve_fit(voigt_profile_model, xw, y_corr, p0=p0, bounds=bounds, maxfev=20000)
    except Exception:
        return _electron_density_result("voigt_fit_failed")

    pred = voigt_profile_model(xw, *popt)
    ss_res = float(np.nansum((y_corr - pred) ** 2))
    ss_tot = float(np.nansum((y_corr - np.nanmean(y_corr)) ** 2))
    r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else float("nan")

    gamma = float(popt[3])
    lorentz_fwhm_nm = 2.0 * gamma
    t_eff = rotational_temperature_k if np.isfinite(rotational_temperature_k) and rotational_temperature_k > 0 else 300.0
    delta_vdw = 3.6 * float(pressure_atm) / (float(t_eff) ** 0.7)
    delta_stark = lorentz_fwhm_nm - delta_vdw

    gamma_se = float("nan")
    if isinstance(pcov, np.ndarray) and pcov.ndim == 2 and pcov.shape[0] > 3 and pcov.shape[1] > 3:
        gamma_var = _safe_float(pcov[3, 3])
        if np.isfinite(gamma_var) and gamma_var > 0:
            gamma_se = float(np.sqrt(gamma_var))

    lorentz_ci_low = float("nan")
    lorentz_ci_high = float("nan")
    if np.isfinite(gamma_se):
        lorentz_se = 2.0 * gamma_se
        lorentz_ci_low = max(0.0, lorentz_fwhm_nm - (1.96 * lorentz_se))
        lorentz_ci_high = lorentz_fwhm_nm + (1.96 * lorentz_se)

    if not np.isfinite(delta_stark) or delta_stark <= 0:
        return _electron_density_result(
            "invalid_nonpositive_stark_width",
            hbeta_lorentz_fwhm_nm=lorentz_fwhm_nm,
            hbeta_lorentz_fwhm_ci95_low_nm=lorentz_ci_low,
            hbeta_lorentz_fwhm_ci95_high_nm=lorentz_ci_high,
            hbeta_stark_fwhm_nm=delta_stark,
            hbeta_vdw_fwhm_nm=delta_vdw,
            electron_density_fit_r2=r2,
        )

    ne_cm3 = 2e11 * (delta_stark ** 1.5)

    ne_ci_low = float("nan")
    ne_ci_high = float("nan")
    if np.isfinite(gamma_se):
        rng_seed = int(abs(np.nanmean(xw)) * 1e6 + abs(np.nanmean(y_corr)) * 1e3) % (2**32 - 1)
        rng = np.random.default_rng(rng_seed)
        gamma_draws = rng.normal(loc=gamma, scale=gamma_se, size=2000)
        gamma_draws = gamma_draws[np.isfinite(gamma_draws) & (gamma_draws > 0)]
        if gamma_draws.size:
            lorentz_draws = 2.0 * gamma_draws
            stark_draws = lorentz_draws - delta_vdw
            stark_draws = stark_draws[np.isfinite(stark_draws) & (stark_draws > 0)]
            if stark_draws.size:
                ne_draws = 2e11 * (stark_draws**1.5)
                ne_ci_low = float(np.nanpercentile(ne_draws, 2.5))
                ne_ci_high = float(np.nanpercentile(ne_draws, 97.5))

    return _electron_density_result(
        "ok",
        estimated_electron_density=ne_cm3,
        estimated_electron_density_ci95_low=ne_ci_low,
        estimated_electron_density_ci95_high=ne_ci_high,
        hbeta_lorentz_fwhm_nm=lorentz_fwhm_nm,
        hbeta_lorentz_fwhm_ci95_low_nm=lorentz_ci_low,
        hbeta_lorentz_fwhm_ci95_high_nm=lorentz_ci_high,
        hbeta_stark_fwhm_nm=delta_stark,
        hbeta_vdw_fwhm_nm=delta_vdw,
        electron_density_fit_r2=r2,
    )


def infer_gas_condition_from_param_set(param_set: str) -> Dict[str, float]:
    p = str(param_set)
    current = _safe_float(CURRENT_RE.search(p).group(1)) if CURRENT_RE.search(p) else float("nan")
    ar_ratio = _safe_float(AR_RE.search(p).group(1)) if AR_RE.search(p) else float("nan")
    o2_ratio = _safe_float(O2_RE.search(p).group(1)) if O2_RE.search(p) else float("nan")
    return {
        "current_a": current,
        "ar_to_n2": ar_ratio,
        "o2_to_n2": o2_ratio,
        "response_ratio_ar": 1.0,
        "response_ratio_o2": 1.0,
    }


def resolve_gas_condition(dataset: str, param_set: str, channel: str, gas_config: pd.DataFrame) -> Dict[str, float]:
    base = infer_gas_condition_from_param_set(param_set)
    if gas_config.empty:
        return base

    d = _norm_key(dataset)
    p = _norm_key(param_set)
    c = _norm_key(channel)
    rows = gas_config[(gas_config["dataset"] == d) & (gas_config["param_set"] == p)].copy()
    if rows.empty:
        return base
    exact = rows[rows["channel"] == c]
    wildcard = rows[rows["channel"].isin({"", "*", "all", "any"})]
    use = exact if not exact.empty else wildcard
    if use.empty:
        use = rows
    r = use.iloc[0]
    for key in ("current_a", "ar_to_n2", "o2_to_n2", "response_ratio_ar", "response_ratio_o2"):
        val = _safe_float(r.get(key))
        if np.isfinite(val):
            base[key] = val
    return base


def compute_relative_dissociation_proxy(line_lookup: Dict[str, float], gas_condition: Dict[str, float]) -> Dict[str, float]:
    i_n = _safe_float(line_lookup.get("N_672"))
    i_ar = _safe_float(line_lookup.get("Ar_750"))
    i_o = _safe_float(line_lookup.get("O_777"))
    ar_to_n2 = _safe_float(gas_condition.get("ar_to_n2"))
    o2_to_n2 = _safe_float(gas_condition.get("o2_to_n2"))
    rr_ar = _safe_float(gas_condition.get("response_ratio_ar"))
    rr_o2 = _safe_float(gas_condition.get("response_ratio_o2"))
    if not np.isfinite(rr_ar):
        rr_ar = 1.0
    if not np.isfinite(rr_o2):
        rr_o2 = 1.0

    proxy_ar = float("nan")
    if np.isfinite(ar_to_n2) and np.isfinite(i_n) and np.isfinite(i_ar) and i_ar > 0:
        proxy_ar = float(ar_to_n2 * (i_n / i_ar) * rr_ar)
    proxy_o2 = float("nan")
    if np.isfinite(o2_to_n2) and np.isfinite(i_n) and np.isfinite(i_o) and i_o > 0:
        proxy_o2 = float(o2_to_n2 * (i_n / i_o) * rr_o2)

    vals = [v for v in (proxy_ar, proxy_o2) if np.isfinite(v)]
    proxy = float(np.mean(vals)) if vals else float("nan")
    return {
        "relative_dissociation_proxy": proxy,
        "relative_dissociation_proxy_ar": proxy_ar,
        "relative_dissociation_proxy_o2": proxy_o2,
    }


def build_group_label(dataset: str, param_set: str, channel: str) -> str:
    return f"{dataset} | {param_set} | {channel}"


def collect_key_line_table(
    dataset: str,
    param_set: str,
    channel: str,
    wl: np.ndarray,
    y: np.ndarray,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    rows: List[Dict[str, object]] = []
    lookup: Dict[str, float] = {}
    for line_name, center in KEY_LINES_NM.items():
        signal = integrated_line_signal(wl, y, center_nm=center, inner_half_width_nm=0.8, outer_half_width_nm=1.8)
        rows.append(
            {
                "dataset": dataset,
                "param_set": param_set,
                "channel": channel,
                "group_label": build_group_label(dataset, param_set, channel),
                "line_name": line_name,
                "wavelength_nm": center,
                "line_area": signal["line_area"],
                "line_peak": signal["line_peak"],
                "line_points": signal["line_points"],
            }
        )
        lookup[line_name] = signal["line_area"]
    return pd.DataFrame(rows), lookup


def _rotational_estimate_columns(rot: Dict[str, object]) -> Dict[str, object]:
    temp = _safe_float(rot.get("temperature"))
    ci_low = _safe_float(rot.get("temperature_ci95_low"))
    ci_high = _safe_float(rot.get("temperature_ci95_high"))
    return {
        "estimated_rotational_temperature": temp,
        "rotational_temperature_ci95_low": ci_low,
        "rotational_temperature_ci95_high": ci_high,
        "rotational_temperature_bootstrap_ci95_low": _safe_float(rot.get("temperature_bootstrap_ci95_low")),
        "rotational_temperature_bootstrap_ci95_high": _safe_float(rot.get("temperature_bootstrap_ci95_high")),
        "rotational_temperature_rel_ci95": _relative_ci_width(temp, ci_low, ci_high),
        "rotational_fit_mode": str(rot.get("mode", "")),
        "rotational_fit_r2": _safe_float(rot.get("r2")),
        "rotational_fit_rmse": _safe_float(rot.get("rmse")),
        "rotational_fit_points": _safe_float(rot.get("n_points")),
        "rotational_fit_status": str(rot.get("status", "")),
        "rotational_fit_slope_se": _safe_float(rot.get("slope_se")),
        "rotational_fit_bootstrap_samples": _safe_float(rot.get("bootstrap_samples_used")),
    }


def _boltzmann_estimate_columns(prefix: str, fit: Dict[str, object]) -> Dict[str, object]:
    temp = _safe_float(fit.get("temperature"))
    ci_low = _safe_float(fit.get("temperature_ci95_low"))
    ci_high = _safe_float(fit.get("temperature_ci95_high"))
    return {
        f"estimated_{prefix}_temperature": temp,
        f"{prefix}_temperature_ci95_low": ci_low,
        f"{prefix}_temperature_ci95_high": ci_high,
        f"{prefix}_temperature_bootstrap_ci95_low": _safe_float(fit.get("temperature_bootstrap_ci95_low")),
        f"{prefix}_temperature_bootstrap_ci95_high": _safe_float(fit.get("temperature_bootstrap_ci95_high")),
        f"{prefix}_temperature_rel_ci95": _relative_ci_width(temp, ci_low, ci_high),
        f"{prefix}_fit_r2": _safe_float(fit.get("r2")),
        f"{prefix}_fit_rmse": _safe_float(fit.get("rmse")),
        f"{prefix}_fit_points": _safe_float(fit.get("n_points")),
        f"{prefix}_fit_status": str(fit.get("status", "")),
        f"{prefix}_fit_slope_se": _safe_float(fit.get("slope_se")),
        f"{prefix}_fit_bootstrap_samples": _safe_float(fit.get("bootstrap_samples_used")),
    }


def _electron_estimate_columns(elec: Dict[str, object]) -> Dict[str, object]:
    ne = _safe_float(elec.get("estimated_electron_density"))
    ne_low = _safe_float(elec.get("estimated_electron_density_ci95_low"))
    ne_high = _safe_float(elec.get("estimated_electron_density_ci95_high"))
    return {
        "estimated_electron_density": ne,
        "estimated_electron_density_ci95_low": ne_low,
        "estimated_electron_density_ci95_high": ne_high,
        "estimated_electron_density_rel_ci95": _relative_ci_width(ne, ne_low, ne_high),
        "hbeta_lorentz_fwhm_nm": _safe_float(elec.get("hbeta_lorentz_fwhm_nm")),
        "hbeta_lorentz_fwhm_ci95_low_nm": _safe_float(elec.get("hbeta_lorentz_fwhm_ci95_low_nm")),
        "hbeta_lorentz_fwhm_ci95_high_nm": _safe_float(elec.get("hbeta_lorentz_fwhm_ci95_high_nm")),
        "hbeta_stark_fwhm_nm": _safe_float(elec.get("hbeta_stark_fwhm_nm")),
        "hbeta_vdw_fwhm_nm": _safe_float(elec.get("hbeta_vdw_fwhm_nm")),
        "electron_density_fit_r2": _safe_float(elec.get("electron_density_fit_r2")),
        "electron_density_status": str(elec.get("electron_density_status", "")),
    }


def build_scope_estimates(
    scope: str,
    vib_meta: pd.DataFrame,
    exc_meta: pd.DataFrame,
    gas_config: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    curves_path = metadata_csv_path(scope, "spectral", "averaged_curves_long.csv")
    if not curves_path.exists():
        return pd.DataFrame(), pd.DataFrame()
    curves = pd.read_csv(curves_path)
    need = {"dataset", "param_set", "channel", "wavelength_nm", "irradiance_mean"}
    if not need.issubset(curves.columns):
        return pd.DataFrame(), pd.DataFrame()

    estimates_rows: List[Dict[str, object]] = []
    line_tables: List[pd.DataFrame] = []
    for key, g in curves.groupby(["dataset", "param_set", "channel"], dropna=False):
        dataset, param_set, channel = [str(v) for v in key]
        g = g.sort_values("wavelength_nm")
        wl = g["wavelength_nm"].to_numpy(dtype=float)
        y = g["irradiance_mean"].to_numpy(dtype=float)
        if wl.size < 8:
            continue

        line_df, line_lookup = collect_key_line_table(dataset, param_set, channel, wl, y)
        line_tables.append(line_df)
        rot = estimate_rotational_temperature(wl, y, mode=ROTATIONAL_MODE)
        vib = estimate_vibrational_temperature(wl, y, vib_meta)
        exc = estimate_excitation_temperature(wl, y, exc_meta)
        gas = resolve_gas_condition(dataset, param_set, channel, gas_config)
        elec = estimate_electron_density(wl, y, rotational_temperature_k=_safe_float(rot.get("temperature")), pressure_atm=PRESSURE_ATM_DEFAULT)
        proxy = compute_relative_dissociation_proxy(line_lookup, gas)

        estimates_rows.append(
            {
                "dataset": dataset,
                "param_set": param_set,
                "channel": channel,
                "group_label": build_group_label(dataset, param_set, channel),
                **_rotational_estimate_columns(rot),
                **_boltzmann_estimate_columns("vibrational", vib),
                **_boltzmann_estimate_columns("excitation", exc),
                **_electron_estimate_columns(elec),
                "relative_dissociation_proxy": _safe_float(proxy.get("relative_dissociation_proxy")),
                "relative_dissociation_proxy_ar": _safe_float(proxy.get("relative_dissociation_proxy_ar")),
                "relative_dissociation_proxy_o2": _safe_float(proxy.get("relative_dissociation_proxy_o2")),
                "current_a": _safe_float(gas.get("current_a")),
                "ar_to_n2": _safe_float(gas.get("ar_to_n2")),
                "o2_to_n2": _safe_float(gas.get("o2_to_n2")),
                "response_ratio_ar": _safe_float(gas.get("response_ratio_ar")),
                "response_ratio_o2": _safe_float(gas.get("response_ratio_o2")),
                "line_area_N_672": _safe_float(line_lookup.get("N_672")),
                "line_area_Ar_750": _safe_float(line_lookup.get("Ar_750")),
                "line_area_O_777": _safe_float(line_lookup.get("O_777")),
                "line_area_OH_308": _safe_float(line_lookup.get("OH_308")),
            }
        )

    estimates_df = pd.DataFrame(estimates_rows)
    lines_df = pd.concat(line_tables, ignore_index=True) if line_tables else pd.DataFrame()
    if not estimates_df.empty:
        estimates_df = estimates_df.sort_values(["dataset", "param_set", "channel"], ignore_index=True)
    if not lines_df.empty:
        lines_df = lines_df.sort_values(["dataset", "param_set", "channel", "wavelength_nm"], ignore_index=True)
    return estimates_df, lines_df


def normalize_column(values: np.ndarray) -> np.ndarray:
    out = values.astype(float).copy()
    finite = np.isfinite(out)
    if not finite.any():
        return np.full_like(out, np.nan, dtype=float)
    vmax = float(np.nanmax(out[finite]))
    if not np.isfinite(vmax) or vmax <= 0:
        return np.zeros_like(out, dtype=float)
    out[finite] = out[finite] / vmax
    out[~finite] = np.nan
    return out


def plot_estimated_state_heatmap(estimates_df: pd.DataFrame, out_path: Path) -> None:
    metrics = [
        "estimated_rotational_temperature",
        "estimated_vibrational_temperature",
        "estimated_excitation_temperature",
        "estimated_electron_density",
        "relative_dissociation_proxy",
    ]
    if estimates_df.empty:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "No chemical modeling estimates available", ha="center", va="center")
        ax.axis("off")
        fig.savefig(out_path, dpi=220)
        plt.close(fig)
        return

    arr = np.column_stack(
        [
            normalize_column(
                estimates_df[m].to_numpy(dtype=float) if m in estimates_df.columns else np.full(len(estimates_df), np.nan)
            )
            for m in metrics
        ]
    )
    metric_labels = [
        r"$\hat{T}_{rot}$",
        r"$\hat{T}_{vib}$",
        r"$\hat{T}_{exc}$",
        r"$\hat{n}_{e}$",
        r"$\hat{D}_{rel}$",
    ]
    fig, ax = plt.subplots(figsize=(10.8, max(3.8, 0.45 * len(estimates_df))))
    im = ax.imshow(np.nan_to_num(arr, nan=0.0), aspect="auto", cmap="YlOrRd")
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels(metric_labels, rotation=0, ha="center", fontsize=11.2)
    ax.set_yticks(np.arange(len(estimates_df)))
    ax.set_yticklabels(estimates_df["group_label"].tolist(), fontsize=8.8)
    ax.set_title(
        "Estimated Plasma-State Summary\n"
        r"$(\hat{T}_{rot}, \hat{T}_{vib}, \hat{T}_{exc}, \hat{n}_{e}, \hat{D}_{rel})$ normalized by metric max"
    )
    cbar = fig.colorbar(im, ax=ax, pad=0.01)
    cbar.set_label("Normalized value (0-1)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_key_peak_map(lines_df: pd.DataFrame, out_path: Path) -> None:
    if lines_df.empty:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "No key-line signals available", ha="center", va="center")
        ax.axis("off")
        fig.savefig(out_path, dpi=220)
        plt.close(fig)
        return

    focus = [
        "OH_308",
        "N2_337",
        "N2plus_391",
        "Hbeta_486",
        "N_672",
        "Ar_750",
        "O_777",
        "N2CB_349.9",
        "N2CB_353.6",
        "N2CB_357.6",
        "N2CB_370.9",
        "N2CB_375.4",
        "N2CB_380.4",
        "N2CB_394.2",
        "N2CB_399.7",
        "N2CB_405.8",
    ]
    d = lines_df[lines_df["line_name"].isin(focus)].copy()
    if d.empty:
        d = lines_df.copy()
    piv = (
        d.pivot_table(index="group_label", columns="line_name", values="line_area", aggfunc="mean", fill_value=np.nan)
        .reindex(columns=[c for c in focus if c in d["line_name"].unique()])
        .sort_index()
    )
    if piv.empty:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "No key-line pivot data available", ha="center", va="center")
        ax.axis("off")
        fig.savefig(out_path, dpi=220)
        plt.close(fig)
        return

    mat = piv.to_numpy(dtype=float)
    finite = np.isfinite(mat) & (mat > 0)
    if finite.any():
        plot_vals = np.full_like(mat, np.nan, dtype=float)
        plot_vals[finite] = np.log10(mat[finite])
        floor = float(np.nanmin(plot_vals[finite]))
        plot_vals[~np.isfinite(plot_vals)] = floor - 0.6
    else:
        plot_vals = np.zeros_like(mat, dtype=float)

    fig, ax = plt.subplots(figsize=(9.8, max(3.8, 0.45 * len(piv.index))))
    im = ax.imshow(plot_vals, aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(piv.columns)))
    ax.set_xticklabels([line_label(c) for c in piv.columns.tolist()], rotation=30, ha="right", fontsize=9.2)
    ax.set_yticks(np.arange(len(piv.index)))
    ax.set_yticklabels(piv.index.tolist(), fontsize=8.8)
    ax.set_title(r"Key-Peak Evidence Map ($\log_{10}$ integrated line intensity)")
    cbar = fig.colorbar(im, ax=ax, pad=0.01)
    cbar.set_label(r"$\log_{10}(I_{line})$")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _safe_mean(series: pd.Series) -> float:
    vals = pd.to_numeric(series, errors="coerce")
    vals = vals[np.isfinite(vals)]
    return float(vals.mean()) if not vals.empty else float("nan")


def _norm_value(value: float, ref: pd.Series) -> float:
    vals = pd.to_numeric(ref, errors="coerce")
    vals = vals[np.isfinite(vals)]
    if vals.empty or not np.isfinite(value):
        return 0.0
    vmin = float(vals.min())
    vmax = float(vals.max())
    span = vmax - vmin
    if span <= 0:
        return 0.5
    return float(np.clip((value - vmin) / span, 0.0, 1.0))


def line_label(line_name: str) -> str:
    if line_name in LINE_LABELS_LATEX:
        return LINE_LABELS_LATEX[line_name]
    if line_name.startswith("N2CB_"):
        wl = line_name.split("_", 1)[1]
        return rf"$N_2(C \rightarrow B)$ ({wl} nm)"
    return line_name


def reaction_label(reaction: str) -> str:
    return REACTION_LABELS_LATEX.get(reaction, reaction)


def alpha_id(index: int) -> str:
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    if index < 0:
        return "?"
    out = ""
    n = int(index)
    while True:
        out = alphabet[n % 26] + out
        n = (n // 26) - 1
        if n < 0:
            break
    return out


def reaction_id_map(reactions: List[str]) -> Dict[str, str]:
    uniq = sorted({str(r) for r in reactions})
    return {reaction: alpha_id(i) for i, reaction in enumerate(uniq)}


SPECIES_MATH = {
    "e": r"e",
    "N2": r"N_2",
    "N2(C)": r"N_2(C)",
    "N2+": r"N_2^{+}",
    "N2+(B)": r"N_2^{+}(B)",
    "N": r"N",
    "Ar*": r"Ar^{*}",
    "Ar": r"Ar",
    "O2": r"O_2",
    "O2-": r"O_2^{-}",
    "O": r"O",
    "NO": r"NO",
    "H2O_trace": r"H_2O_{trace}",
    "OH": r"OH",
}


def split_reaction_halves(reaction: str) -> Tuple[str, str]:
    text = str(reaction)
    if "->" in text:
        lhs, rhs = text.split("->", 1)
        return lhs.strip(), rhs.strip()
    if "→" in text:
        lhs, rhs = text.split("→", 1)
        return lhs.strip(), rhs.strip()
    return text.strip(), ""


def side_token_math(token: str) -> str:
    tok = str(token).strip()
    if not tok:
        return ""
    match = re.match(r"^([0-9]+)\s*(.+)$", tok)
    coeff = ""
    base = tok
    if match:
        coeff = match.group(1).strip()
        base = match.group(2).strip()
    base_math = SPECIES_MATH.get(base, base.replace("_", r"\_"))
    return f"{coeff}{base_math}" if coeff else base_math


def reaction_side_label(side: str) -> str:
    parts = [side_token_math(part) for part in re.split(r"\s+\+\s+", str(side).strip())]
    parts = [p for p in parts if p]
    if not parts:
        return str(side)
    return "$" + r" + ".join(parts) + "$"


def build_reaction_story_layout(story_df: pd.DataFrame, max_reactions: int = 6) -> pd.DataFrame:
    cols = [
        "reaction",
        "reaction_latex",
        "reaction_weight",
        "reaction_id",
        "lhs",
        "rhs",
        "lhs_label",
        "rhs_label",
    ]
    if story_df.empty:
        return pd.DataFrame(columns=cols)

    base = (
        story_df.groupby(["reaction", "reaction_latex"], dropna=False)["sum_link_weight"]
        .sum()
        .sort_values(ascending=False)
        .head(max_reactions)
        .reset_index(name="reaction_weight")
    )
    if base.empty:
        return pd.DataFrame(columns=cols)

    base["reaction"] = base["reaction"].astype(str)
    base["reaction_latex"] = base.apply(
        lambda r: r["reaction_latex"] if str(r["reaction_latex"]).strip() and str(r["reaction_latex"]).lower() != "nan" else reaction_label(str(r["reaction"])),
        axis=1,
    )
    base["reaction_id"] = [alpha_id(i) for i in range(len(base))]
    lhs_rhs = base["reaction"].map(split_reaction_halves)
    base["lhs"] = lhs_rhs.map(lambda x: x[0])
    base["rhs"] = lhs_rhs.map(lambda x: x[1])
    base["lhs_label"] = base["lhs"].map(reaction_side_label)
    base["rhs_label"] = base["rhs"].map(reaction_side_label)
    return base[cols].copy()


def build_pathway_edges(estimates_df: pd.DataFrame) -> pd.DataFrame:
    if estimates_df.empty:
        return pd.DataFrame(columns=["reaction", "reaction_latex", "weight"])
    rot_n = _norm_value(_safe_mean(estimates_df["estimated_rotational_temperature"]), estimates_df["estimated_rotational_temperature"])
    vib_n = _norm_value(_safe_mean(estimates_df["estimated_vibrational_temperature"]), estimates_df["estimated_vibrational_temperature"])
    exc_n = _norm_value(_safe_mean(estimates_df["estimated_excitation_temperature"]), estimates_df["estimated_excitation_temperature"])
    ne_n = _norm_value(_safe_mean(estimates_df["estimated_electron_density"]), estimates_df["estimated_electron_density"])
    proxy_n = _norm_value(_safe_mean(estimates_df["relative_dissociation_proxy"]), estimates_df["relative_dissociation_proxy"])
    ar_n = _norm_value(_safe_mean(estimates_df["ar_to_n2"]), estimates_df["ar_to_n2"])
    o2_n = _norm_value(_safe_mean(estimates_df["o2_to_n2"]), estimates_df["o2_to_n2"])
    oh_n = _norm_value(_safe_mean(estimates_df["line_area_OH_308"]), estimates_df["line_area_OH_308"])
    edges = [
        {"reaction": "e + N2 -> e + N2(C)", "weight": np.mean([vib_n, exc_n])},
        {"reaction": "e + N2 -> 2e + N2+", "weight": np.mean([ne_n, rot_n])},
        {"reaction": "e + N2 -> e + N + N", "weight": np.mean([proxy_n, ne_n])},
        {"reaction": "e + N2+ -> e + N2+(B)", "weight": np.mean([rot_n, ne_n])},
        {"reaction": "Ar* + N2 -> Ar + N2(C)", "weight": np.mean([ar_n, vib_n])},
        {"reaction": "e + O2 -> e + O + O", "weight": np.mean([o2_n, exc_n])},
        {"reaction": "e + O2 -> O2-", "weight": np.mean([o2_n, 1.0 - ne_n])},
        {"reaction": "N + O2 -> NO + O", "weight": np.mean([proxy_n, o2_n])},
        {"reaction": "O + H2O_trace -> 2OH", "weight": np.mean([oh_n, o2_n])},
    ]
    out = pd.DataFrame(edges)
    out["weight"] = out["weight"].astype(float).clip(lower=0.0, upper=1.0)
    out["reaction_latex"] = out["reaction"].map(reaction_label)
    return out


def reaction_wavelength_notes(story_df: pd.DataFrame, max_per_reaction: int = 3) -> Dict[str, str]:
    if story_df.empty:
        return {}
    d = story_df.copy()
    d["wavelength_nm"] = pd.to_numeric(d["wavelength_nm"], errors="coerce")
    d["sum_link_weight"] = pd.to_numeric(d["sum_link_weight"], errors="coerce")
    d = d[np.isfinite(d["wavelength_nm"]) & np.isfinite(d["sum_link_weight"])].copy()
    if d.empty:
        return {}

    agg = (
        d.groupby(["reaction", "wavelength_nm"], dropna=False)["sum_link_weight"]
        .sum()
        .reset_index()
        .sort_values(["reaction", "sum_link_weight"], ascending=[True, False])
    )
    out: Dict[str, str] = {}
    for reaction, g in agg.groupby("reaction", dropna=False):
        top = g.head(max_per_reaction)
        vals = ", ".join(f"{float(v):.1f}" for v in top["wavelength_nm"].tolist())
        out[str(reaction)] = f"lambda: {vals} nm"
    return out


def plot_reduced_pathways(estimates_df: pd.DataFrame, story_df: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    edges_df = build_pathway_edges(estimates_df)
    if edges_df.empty:
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.text(0.5, 0.5, "No pathway weights available", ha="center", va="center")
        ax.axis("off")
        fig.savefig(out_path, dpi=220)
        plt.close(fig)
        return edges_df

    fig, ax = plt.subplots(figsize=(14.2, 8.8))
    ax.axis("off")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    wv_map = reaction_wavelength_notes(story_df, max_per_reaction=5)
    nodes = {
        "e": (0.08, 0.84),
        "N2": (0.23, 0.84),
        "N2(C)": (0.48, 0.92),
        "N2+": (0.48, 0.73),
        "N2+(B)": (0.76, 0.83),
        "N": (0.48, 0.56),
        "Ar*": (0.23, 0.95),
        "O2": (0.23, 0.29),
        "O": (0.48, 0.29),
        "O2-": (0.48, 0.12),
        "NO": (0.76, 0.35),
        "H2O_trace": (0.23, 0.10),
        "OH": (0.48, 0.03),
    }
    node_groups = {
        "electron": {"e"},
        "nitrogen": {"N2", "N2(C)", "N2+", "N2+(B)", "N"},
        "oxygen": {"O2", "O", "O2-", "NO", "OH", "H2O_trace"},
        "argon": {"Ar*"},
    }
    group_color = {
        "electron": "#f1f5f9",
        "nitrogen": "#eaf4ff",
        "oxygen": "#eefbf3",
        "argon": "#fff6ea",
    }
    node_face: Dict[str, str] = {}
    for group_name, members in node_groups.items():
        for member in members:
            node_face[member] = group_color[group_name]

    for name, (x, y) in nodes.items():
        ax.scatter([x], [y], s=980, color=node_face.get(name, "#fafafa"), edgecolors="#2f2f2f", linewidths=1.1, zorder=4)
        ax.text(x, y, NODE_LABELS_LATEX.get(name, name), ha="center", va="center", fontsize=11.0, zorder=5)

    edge_palette = get_palette(max(1, len(REACTION_TO_NODES)), name="tab20")
    edge_colors = {reaction: edge_palette[i % len(edge_palette)] for i, (reaction, _, _) in enumerate(REACTION_TO_NODES)}
    wmap = {r["reaction"]: float(r["weight"]) for _, r in edges_df.iterrows()}
    for idx, (reaction, src, dst) in enumerate(REACTION_TO_NODES):
        sx, sy = nodes[src]
        dx, dy = nodes[dst]
        w = float(np.clip(wmap.get(reaction, 0.0), 0.0, 1.0))
        color = edge_colors.get(reaction, "#64748b")
        lw = 1.1 + (4.8 * w)
        rad = -0.16 + (0.04 * (idx % 9))
        arrow = FancyArrowPatch(
            (sx, sy),
            (dx, dy),
            arrowstyle="-|>",
            mutation_scale=13,
            connectionstyle=f"arc3,rad={rad}",
            lw=lw,
            color=color,
            alpha=0.95,
            zorder=2,
        )
        ax.add_patch(arrow)
        mx = 0.5 * (sx + dx) + (0.07 * rad)
        my = 0.5 * (sy + dy) + (0.18 * rad)
        label = reaction_label(reaction)
        wl_text = wv_map.get(reaction, "lambda: n/a")
        ax.text(
            mx,
            my + 0.017,
            label,
            fontsize=8.6,
            color="#111827",
            ha="center",
            va="center",
            zorder=6,
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "#ffffff", "edgecolor": "#d1d5db", "alpha": 0.92},
        )
        ax.text(
            mx,
            my - 0.012,
            wl_text,
            fontsize=8.0,
            color="#334155",
            ha="center",
            va="center",
            zorder=6,
            bbox={"boxstyle": "round,pad=0.14", "facecolor": "#ffffff", "edgecolor": "#e5e7eb", "alpha": 0.88},
        )

    ax.plot([0.02, 0.10], [0.04, 0.04], color="#475569", lw=1.2, alpha=0.7)
    ax.plot([0.02, 0.10], [0.01, 0.01], color="#475569", lw=4.4, alpha=0.8)
    ax.text(0.11, 0.04, "lower pathway strength", fontsize=8.6, ha="left", va="center", color="#334155")
    ax.text(0.11, 0.01, "higher pathway strength", fontsize=8.6, ha="left", va="center", color="#334155")
    ax.set_title(r"Fig1. Reduced Chemical Pathways With Peak-Wavelength Evidence")
    fig.tight_layout()
    fig.savefig(out_path, dpi=240)
    plt.close(fig)
    return edges_df


def compute_pathway_weights_per_group(estimates_df: pd.DataFrame) -> pd.DataFrame:
    if estimates_df.empty:
        return pd.DataFrame(columns=["group_label", "reaction", "pathway_weight"])

    d = estimates_df.copy()
    metric_cols = [
        "estimated_rotational_temperature",
        "estimated_vibrational_temperature",
        "estimated_excitation_temperature",
        "estimated_electron_density",
        "relative_dissociation_proxy",
        "ar_to_n2",
        "o2_to_n2",
        "line_area_OH_308",
    ]
    for col in metric_cols:
        vals = pd.to_numeric(d[col], errors="coerce") if col in d.columns else pd.Series(np.nan, index=d.index)
        finite = vals[np.isfinite(vals)]
        if finite.empty:
            d[f"{col}__norm"] = 0.0
            continue
        vmin = float(finite.min())
        vmax = float(finite.max())
        span = vmax - vmin
        if span <= 0:
            d[f"{col}__norm"] = 0.5
        else:
            d[f"{col}__norm"] = ((vals - vmin) / span).clip(lower=0.0, upper=1.0).fillna(0.0)

    rows: List[Dict[str, object]] = []
    for _, r in d.iterrows():
        pathway_weight = {
            "e + N2 -> e + N2(C)": float(np.mean([r["estimated_vibrational_temperature__norm"], r["estimated_excitation_temperature__norm"]])),
            "e + N2 -> 2e + N2+": float(np.mean([r["estimated_electron_density__norm"], r["estimated_rotational_temperature__norm"]])),
            "e + N2 -> e + N + N": float(np.mean([r["relative_dissociation_proxy__norm"], r["estimated_electron_density__norm"]])),
            "e + N2+ -> e + N2+(B)": float(np.mean([r["estimated_rotational_temperature__norm"], r["estimated_electron_density__norm"]])),
            "Ar* + N2 -> Ar + N2(C)": float(np.mean([r["ar_to_n2__norm"], r["estimated_vibrational_temperature__norm"]])),
            "e + O2 -> e + O + O": float(np.mean([r["o2_to_n2__norm"], r["estimated_excitation_temperature__norm"]])),
            "e + O2 -> O2-": float(np.mean([r["o2_to_n2__norm"], 1.0 - r["estimated_electron_density__norm"]])),
            "N + O2 -> NO + O": float(np.mean([r["relative_dissociation_proxy__norm"], r["o2_to_n2__norm"]])),
            "O + H2O_trace -> 2OH": float(np.mean([r["line_area_OH_308__norm"], r["o2_to_n2__norm"]])),
        }
        for reaction, weight in pathway_weight.items():
            rows.append(
                {
                    "dataset": r["dataset"],
                    "param_set": r["param_set"],
                    "channel": r["channel"],
                    "group_label": r["group_label"],
                    "reaction": reaction,
                    "reaction_latex": reaction_label(reaction),
                    "pathway_weight": float(np.clip(weight, 0.0, 1.0)),
                }
            )
    return pd.DataFrame(rows)


def build_peak_to_pathway_links(estimates_df: pd.DataFrame, lines_df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "dataset",
        "param_set",
        "channel",
        "group_label",
        "reaction",
        "reaction_latex",
        "pathway_weight",
        "line_name",
        "line_label",
        "wavelength_nm",
        "line_area",
        "line_strength_norm",
        "link_weight",
    ]
    if estimates_df.empty or lines_df.empty:
        return pd.DataFrame(columns=cols)

    pathways = compute_pathway_weights_per_group(estimates_df)
    if pathways.empty:
        return pd.DataFrame(columns=cols)

    line_table = lines_df.copy()
    line_table["line_area"] = pd.to_numeric(line_table["line_area"], errors="coerce")
    line_table["wavelength_nm"] = pd.to_numeric(line_table["wavelength_nm"], errors="coerce")
    line_table["line_strength_norm"] = 0.0
    for group_label, g in line_table.groupby("group_label", dropna=False):
        finite = g["line_area"][np.isfinite(g["line_area"]) & (g["line_area"] > 0)]
        if finite.empty:
            continue
        vmax = float(finite.max())
        idx = g.index
        line_table.loc[idx, "line_strength_norm"] = (line_table.loc[idx, "line_area"] / vmax).clip(lower=0.0, upper=1.0).fillna(0.0)
    line_table["line_label"] = line_table["line_name"].map(line_label)

    rows: List[Dict[str, object]] = []
    for _, pw in pathways.iterrows():
        group_label = str(pw["group_label"])
        reaction = str(pw["reaction"])
        evidence_lines = PATHWAY_PEAK_EVIDENCE.get(reaction, [])
        if not evidence_lines:
            continue
        g_lines = line_table[(line_table["group_label"].astype(str) == group_label) & (line_table["line_name"].isin(evidence_lines))].copy()
        if g_lines.empty:
            continue
        evidence_norm_sum = float(g_lines["line_strength_norm"].sum())
        if evidence_norm_sum <= 0:
            continue
        for _, lr in g_lines.iterrows():
            evidence_share = float(lr["line_strength_norm"]) / evidence_norm_sum
            link_weight = float(pw["pathway_weight"]) * evidence_share
            rows.append(
                {
                    "dataset": pw["dataset"],
                    "param_set": pw["param_set"],
                    "channel": pw["channel"],
                    "group_label": group_label,
                    "reaction": reaction,
                    "reaction_latex": pw["reaction_latex"],
                    "pathway_weight": float(pw["pathway_weight"]),
                    "line_name": lr["line_name"],
                    "line_label": lr["line_label"],
                    "wavelength_nm": float(lr["wavelength_nm"]),
                    "line_area": float(lr["line_area"]) if np.isfinite(lr["line_area"]) else float("nan"),
                    "line_strength_norm": float(lr["line_strength_norm"]),
                    "link_weight": float(link_weight),
                }
            )

    if not rows:
        return pd.DataFrame(columns=cols)
    out = pd.DataFrame(rows)
    return out[cols].sort_values(["group_label", "reaction", "link_weight"], ascending=[True, True, False], ignore_index=True)


def summarize_peak_pathway_story(link_df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "reaction",
        "reaction_latex",
        "line_name",
        "line_label",
        "wavelength_nm",
        "mean_link_weight",
        "sum_link_weight",
        "mean_line_strength_norm",
        "n_groups",
    ]
    if link_df.empty:
        return pd.DataFrame(columns=cols)

    out = (
        link_df.groupby(["reaction", "reaction_latex", "line_name", "line_label", "wavelength_nm"], dropna=False)
        .agg(
            mean_link_weight=("link_weight", "mean"),
            sum_link_weight=("link_weight", "sum"),
            mean_line_strength_norm=("line_strength_norm", "mean"),
            n_groups=("group_label", "nunique"),
        )
        .reset_index()
    )
    return out[cols].sort_values(["reaction", "sum_link_weight"], ascending=[True, False], ignore_index=True)


def plot_peak_to_pathway_network(
    story_df: pd.DataFrame,
    out_path: Path,
    figure_label: str = "Fig2",
    context_label: str = "",
    footer_note: str = "IDs map to reaction equations in Fig3.",
) -> None:
    if story_df.empty:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "No peak-to-pathway links available", ha="center", va="center")
        ax.axis("off")
        fig.savefig(out_path, dpi=220)
        plt.close(fig)
        return

    d = story_df.copy()
    d["sum_link_weight"] = pd.to_numeric(d["sum_link_weight"], errors="coerce")
    d["wavelength_nm"] = pd.to_numeric(d["wavelength_nm"], errors="coerce")
    layout = build_reaction_story_layout(d, max_reactions=6)
    if layout.empty:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "No reaction layout available for peak-to-pathway map", ha="center", va="center")
        ax.axis("off")
        fig.savefig(out_path, dpi=220)
        plt.close(fig)
        return

    selected = layout["reaction"].astype(str).tolist()
    d = d[d["reaction"].astype(str).isin(selected)].copy()
    d = (
        d.sort_values(["reaction", "sum_link_weight"], ascending=[True, False])
        .groupby("reaction", dropna=False)
        .head(3)
        .reset_index(drop=True)
    )
    if d.empty:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "No pathway evidence lines available after filtering", ha="center", va="center")
        ax.axis("off")
        fig.savefig(out_path, dpi=220)
        plt.close(fig)
        return

    right_reactions = layout["reaction"].astype(str).tolist()
    right_rank = {reaction: i for i, reaction in enumerate(right_reactions)}
    lhs_rank: Dict[str, List[float]] = {}
    for _, row in layout.iterrows():
        lhs = str(row["lhs"])
        reaction = str(row["reaction"])
        lhs_rank.setdefault(lhs, []).append(float(right_rank.get(reaction, 0)))
    left_nodes = sorted(lhs_rank.keys(), key=lambda lhs: float(np.mean(lhs_rank.get(lhs, [0.0]))))

    left_x = 0.24
    left_w = 0.30
    right_species_w = 0.20
    right_id_w = 0.10
    right_gap = 0.014
    right_species_x = 0.73
    right_id_x = right_species_x + (0.5 * right_species_w) + right_gap + (0.5 * right_id_w)
    node_h = 0.10
    left_y_vals = np.linspace(0.88, 0.12, len(left_nodes))
    right_y_vals = np.linspace(0.88, 0.12, len(layout))
    left_pos = {lhs: (left_x, y) for lhs, y in zip(left_nodes, left_y_vals)}
    right_pos = {str(r["reaction"]): (right_species_x, right_id_x, y) for y, (_, r) in zip(right_y_vals, layout.iterrows())}

    id_map = {str(r["reaction"]): str(r["reaction_id"]) for _, r in layout.iterrows()}
    lhs_label_map = {str(r["lhs"]): str(r["lhs_label"]) for _, r in layout.iterrows()}
    rhs_label_map = {str(r["reaction"]): str(r["rhs_label"]) for _, r in layout.iterrows()}

    fig, ax = plt.subplots(figsize=(13.0, 7.2))
    ax.axis("off")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.text(0.24, 0.965, "Left Half", ha="center", va="bottom", fontsize=10.8, weight="semibold")
    ax.text(right_species_x, 0.965, "Species (Other Half)", ha="center", va="bottom", fontsize=10.8, weight="semibold")
    ax.text(right_id_x, 0.965, "Reaction ID", ha="center", va="bottom", fontsize=10.8, weight="semibold")

    for lhs in left_nodes:
        x, y = left_pos[lhs]
        card = FancyBboxPatch(
            (x - (left_w * 0.5), y - (node_h * 0.5)),
            left_w,
            node_h,
            boxstyle="round,pad=0.010,rounding_size=0.016",
            facecolor="#ffffff",
            edgecolor="#374151",
            linewidth=1.0,
            zorder=4,
        )
        ax.add_patch(card)
        ax.text(
            x,
            y,
            lhs_label_map.get(lhs, reaction_side_label(lhs)),
            ha="center",
            va="center",
            fontsize=12.0,
            color="#111827",
            zorder=5,
        )

    for _, row in layout.iterrows():
        reaction = str(row["reaction"])
        species_x, id_x, y = right_pos[reaction]
        species_card = FancyBboxPatch(
            (species_x - (right_species_w * 0.5), y - (node_h * 0.5)),
            right_species_w,
            node_h,
            boxstyle="round,pad=0.010,rounding_size=0.016",
            facecolor="#ffffff",
            edgecolor="#111827",
            linewidth=1.1,
            zorder=4,
        )
        id_card = FancyBboxPatch(
            (id_x - (right_id_w * 0.5), y - (node_h * 0.5)),
            right_id_w,
            node_h,
            boxstyle="round,pad=0.010,rounding_size=0.016",
            facecolor="#ffffff",
            edgecolor="#111827",
            linewidth=1.1,
            zorder=4,
        )
        ax.add_patch(species_card)
        ax.add_patch(id_card)
        ax.text(
            species_x,
            y,
            rhs_label_map.get(reaction, reaction_side_label(str(row["rhs"]))),
            ha="center",
            va="center",
            fontsize=11.8,
            color="#111827",
            zorder=6,
        )
        rid = id_map.get(reaction, "?")
        ax.text(id_x, y, rid, ha="center", va="center", fontsize=15.2, color="#111827", weight="bold", zorder=6)

    max_w = float(np.nanmax(d["sum_link_weight"].to_numpy(dtype=float))) if np.isfinite(d["sum_link_weight"]).any() else 1.0
    if max_w <= 0:
        max_w = 1.0

    for _, row in layout.iterrows():
        reaction = str(row["reaction"])
        lhs = str(row["lhs"])
        if lhs not in left_pos or reaction not in right_pos:
            continue
        sx, sy = left_pos[lhs]
        species_x, _, dy = right_pos[reaction]
        x0 = sx + (left_w * 0.5) - 0.004
        x1 = species_x - (right_species_w * 0.5) + 0.004
        y0 = sy
        y1 = dy

        edges = d[d["reaction"].astype(str) == reaction].sort_values("sum_link_weight", ascending=False)
        if edges.empty:
            continue
        primary = edges.iloc[0]
        preferred_lines = PATHWAY_PEAK_EVIDENCE.get(reaction, [])
        for ln in preferred_lines:
            cand = edges[edges["line_name"].astype(str) == str(ln)]
            if not cand.empty:
                primary = cand.iloc[0]
                break
        primary_w = float(primary["sum_link_weight"]) if np.isfinite(primary["sum_link_weight"]) else 0.0
        w = max(0.0, min(1.0, primary_w / max_w))
        line_width = 1.4 + (3.6 * w)
        ax.plot([x0, x1], [y0, y1], color="#374151", lw=line_width, alpha=0.85, zorder=2)

        if np.isfinite(float(primary["wavelength_nm"])):
            lx = x0 + ((x1 - x0) * 0.56)
            ly = y0 + ((y1 - y0) * 0.56)
            ax.text(
                lx,
                ly,
                f"{float(primary['wavelength_nm']):.1f} nm",
                ha="center",
                va="center",
                fontsize=8.6,
                color="#111827",
                zorder=6,
                bbox={"boxstyle": "round,pad=0.12", "facecolor": "#ffffff", "edgecolor": "#d1d5db", "alpha": 0.95},
            )

    ax.text(0.02, 0.02, footer_note, ha="left", va="bottom", fontsize=9.0, color="#334155")
    title = f"{figure_label}. Peak-Wavelength Links Across Reaction Halves"
    if context_label:
        title += f" | {context_label}"
    ax.set_title(title)
    fig.subplots_adjust(left=0.03, right=0.99, top=0.94, bottom=0.05)
    fig.savefig(out_path, dpi=260)
    plt.close(fig)


def plot_reaction_pathway_key(story_df: pd.DataFrame, out_path: Path, figure_label: str = "Fig3") -> None:
    if story_df.empty:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "No pathway reaction key available", ha="center", va="center")
        ax.axis("off")
        fig.savefig(out_path, dpi=220)
        plt.close(fig)
        return

    d = story_df.copy()
    d["sum_link_weight"] = pd.to_numeric(d["sum_link_weight"], errors="coerce")
    d["wavelength_nm"] = pd.to_numeric(d["wavelength_nm"], errors="coerce")
    layout = build_reaction_story_layout(d, max_reactions=6)
    if layout.empty:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "No reaction key rows available", ha="center", va="center")
        ax.axis("off")
        fig.savefig(out_path, dpi=220)
        plt.close(fig)
        return

    reactions = layout["reaction"].astype(str).tolist()
    palette = get_palette(max(1, len(reactions)), name="Set2")
    color_map = {reaction: palette[i % len(palette)] for i, reaction in enumerate(reactions)}

    fig_h = max(7.2, (1.7 * len(layout)) + 1.6)
    fig, axes = plt.subplots(len(layout), 1, figsize=(15.0, fig_h))
    if len(layout) == 1:
        axes = [axes]

    for ax, (_, row) in zip(axes, layout.iterrows()):
        reaction = str(row["reaction"])
        rid = str(row["reaction_id"])
        ax.axis("off")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)

        card = FancyBboxPatch(
            (0.015, 0.10),
            0.97,
            0.80,
            boxstyle="round,pad=0.015,rounding_size=0.022",
            facecolor="#f8fafc",
            edgecolor="#cbd5e1",
            linewidth=1.0,
            zorder=1,
        )
        ax.add_patch(card)

        ax.scatter([0.05], [0.50], s=720, color=color_map.get(reaction, "#cbd5e1"), edgecolors="#334155", linewidths=1.0, zorder=3)
        ax.text(0.05, 0.50, rid, ha="center", va="center", fontsize=14.0, weight="bold", color="#0f172a", zorder=4)

        eq_text = reaction_label(reaction)
        ax.text(0.105, 0.64, eq_text, ha="left", va="center", fontsize=13.0, color="#111827")

        ax.text(
            0.105,
            0.42,
            f"Half mapping: {row['lhs_label']}  $\\rightarrow$  {row['rhs_label']}",
            ha="left",
            va="center",
            fontsize=10.6,
            color="#334155",
        )

        top_wl = (
            d[d["reaction"].astype(str) == reaction]
            .sort_values("sum_link_weight", ascending=False)["wavelength_nm"]
            .dropna()
            .head(6)
            .tolist()
        )
        if top_wl:
            wl_text = ", ".join(f"{float(w):.1f}" for w in top_wl) + " nm"
        else:
            wl_text = "n/a"
        ax.text(
            0.105,
            0.22,
            rf"Peak wavelengths shown on Fig2 links: {wl_text}",
            ha="left",
            va="center",
            fontsize=10.6,
            color="#0f172a",
        )

    fig.suptitle(f"{figure_label}. Reaction Equation Key (letters match pathway-story IDs)", fontsize=14.2, y=0.995)
    fig.subplots_adjust(top=0.93, bottom=0.04, left=0.03, right=0.99, hspace=0.16)
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


def plot_pathway_evidence_matrix(story_df: pd.DataFrame, out_path: Path) -> None:
    if story_df.empty:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "No pathway evidence matrix available", ha="center", va="center")
        ax.axis("off")
        fig.savefig(out_path, dpi=220)
        plt.close(fig)
        return

    d = story_df.copy()
    top_lines = (
        d.groupby("line_label", dropna=False)["sum_link_weight"]
        .sum()
        .sort_values(ascending=False)
        .head(12)
        .index.tolist()
    )
    mat = (
        d[d["line_label"].isin(top_lines)]
        .pivot_table(index="reaction_latex", columns="line_label", values="sum_link_weight", aggfunc="sum", fill_value=0.0)
        .reindex(columns=top_lines)
    )
    if mat.empty:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "No matrix rows after top-line filter", ha="center", va="center")
        ax.axis("off")
        fig.savefig(out_path, dpi=220)
        plt.close(fig)
        return

    arr = mat.to_numpy(dtype=float)
    if np.isfinite(arr).any() and float(np.nanmax(arr)) > 0:
        arr = arr / float(np.nanmax(arr))
    fig, ax = plt.subplots(figsize=(12.0, max(4.6, 0.48 * len(mat.index))))
    im = ax.imshow(arr, aspect="auto", cmap="cividis")
    ax.set_xticks(np.arange(len(mat.columns)))
    ax.set_xticklabels(mat.columns.tolist(), rotation=35, ha="right", fontsize=8.8)
    ax.set_yticks(np.arange(len(mat.index)))
    ax.set_yticklabels(mat.index.tolist(), fontsize=8.8)
    ax.set_xlabel(r"Peak Evidence Line ($\lambda$, nm)")
    ax.set_ylabel("Reduced Reaction Pathway")
    ax.set_title(r"Pathway-Peak Evidence Matrix (normalized within scope)")
    cbar = fig.colorbar(im, ax=ax, pad=0.01)
    cbar.set_label("Normalized evidence weight")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def spearman_rho(x: pd.Series, y: pd.Series) -> float:
    rho, _ = spearman_stats(x, y)
    return rho


def spearman_stats(x: pd.Series, y: pd.Series) -> Tuple[float, float]:
    x_num = pd.to_numeric(pd.Series(x), errors="coerce")
    y_num = pd.to_numeric(pd.Series(y), errors="coerce")
    valid = np.isfinite(x_num.to_numpy(dtype=float)) & np.isfinite(y_num.to_numpy(dtype=float))
    if int(valid.sum()) < 2:
        return float("nan"), float("nan")
    rho, pvalue = spearmanr(x_num[valid], y_num[valid])
    rho_f = _safe_float(rho)
    p_f = _safe_float(pvalue)
    if not np.isfinite(rho_f):
        return float("nan"), float("nan")
    return rho_f, p_f


def _metric_quality_mask(estimates_df: pd.DataFrame, metric: str) -> pd.Series:
    metric_vals = _safe_numeric_series(estimates_df, metric)
    mask = np.isfinite(metric_vals)

    if metric == "estimated_rotational_temperature":
        status_ok = _safe_text_series(estimates_df, "rotational_fit_status").str.lower() == "ok"
        r2_ok = _safe_numeric_series(estimates_df, "rotational_fit_r2") >= 0.30
        rel_ci = _relative_ci_width_series(
            metric_vals,
            _safe_numeric_series(estimates_df, "rotational_temperature_ci95_low"),
            _safe_numeric_series(estimates_df, "rotational_temperature_ci95_high"),
        )
        ci_ok = (~np.isfinite(rel_ci)) | (rel_ci <= 1.50)
        return mask & (metric_vals > 0) & status_ok & r2_ok & ci_ok

    if metric == "estimated_vibrational_temperature":
        status_ok = _safe_text_series(estimates_df, "vibrational_fit_status").str.lower() == "ok"
        r2_ok = _safe_numeric_series(estimates_df, "vibrational_fit_r2") >= 0.20
        rel_ci = _relative_ci_width_series(
            metric_vals,
            _safe_numeric_series(estimates_df, "vibrational_temperature_ci95_low"),
            _safe_numeric_series(estimates_df, "vibrational_temperature_ci95_high"),
        )
        ci_ok = (~np.isfinite(rel_ci)) | (rel_ci <= 2.00)
        return mask & (metric_vals > 0) & status_ok & r2_ok & ci_ok

    if metric == "estimated_excitation_temperature":
        status_ok = _safe_text_series(estimates_df, "excitation_fit_status").str.lower() == "ok"
        r2_ok = _safe_numeric_series(estimates_df, "excitation_fit_r2") >= 0.20
        rel_ci = _relative_ci_width_series(
            metric_vals,
            _safe_numeric_series(estimates_df, "excitation_temperature_ci95_low"),
            _safe_numeric_series(estimates_df, "excitation_temperature_ci95_high"),
        )
        ci_ok = (~np.isfinite(rel_ci)) | (rel_ci <= 2.00)
        return mask & (metric_vals > 0) & status_ok & r2_ok & ci_ok

    if metric == "estimated_electron_density":
        status_ok = _safe_text_series(estimates_df, "electron_density_status").str.lower() == "ok"
        r2_ok = _safe_numeric_series(estimates_df, "electron_density_fit_r2") >= 0.20
        rel_ci = _relative_ci_width_series(
            metric_vals,
            _safe_numeric_series(estimates_df, "estimated_electron_density_ci95_low"),
            _safe_numeric_series(estimates_df, "estimated_electron_density_ci95_high"),
        )
        ci_ok = (~np.isfinite(rel_ci)) | (rel_ci <= 3.00)
        return mask & (metric_vals > 0) & status_ok & r2_ok & ci_ok

    if metric == "relative_dissociation_proxy":
        return mask & (metric_vals >= 0)

    return mask


def build_trend_checks(estimates_df: pd.DataFrame) -> pd.DataFrame:
    checks = [
        ("current_a", "estimated_rotational_temperature", "positive"),
        ("current_a", "estimated_vibrational_temperature", "positive"),
        ("current_a", "estimated_excitation_temperature", "positive"),
        ("current_a", "estimated_electron_density", "positive"),
        ("current_a", "relative_dissociation_proxy", "positive"),
        ("ar_to_n2", "estimated_electron_density", "positive"),
        ("ar_to_n2", "relative_dissociation_proxy", "positive"),
        ("o2_to_n2", "estimated_electron_density", "negative"),
    ]
    rows: List[Dict[str, object]] = []
    for variable, metric, expected in checks:
        if variable not in estimates_df.columns or metric not in estimates_df.columns:
            rows.append(
                {
                    "variable": variable,
                    "metric": metric,
                    "expected_direction": expected,
                    "spearman_rho": float("nan"),
                    "spearman_pvalue": float("nan"),
                    "n_points": 0,
                    "n_points_raw": 0,
                    "n_points_filtered_out": 0,
                    "n_levels": 0,
                    "trend_status": "insufficient_data",
                    "note": "missing_required_columns",
                    "rho_threshold": SPEARMAN_PASS_THRESHOLD,
                    "pvalue_threshold": SPEARMAN_PVALUE_THRESHOLD,
                }
            )
            continue

        x_all = _safe_numeric_series(estimates_df, variable)
        y_all = _safe_numeric_series(estimates_df, metric)
        raw_mask = np.isfinite(x_all) & np.isfinite(y_all)
        quality_mask = _metric_quality_mask(estimates_df, metric)
        keep = raw_mask & quality_mask

        n_points_raw = int(raw_mask.sum())
        n_points = int(keep.sum())
        n_levels = int(x_all[keep].nunique())
        filtered = int(n_points_raw - n_points)
        if n_points < MIN_TREND_POINTS or n_levels < MIN_TREND_LEVELS:
            rows.append(
                {
                    "variable": variable,
                    "metric": metric,
                    "expected_direction": expected,
                    "spearman_rho": float("nan"),
                    "spearman_pvalue": float("nan"),
                    "n_points": n_points,
                    "n_points_raw": n_points_raw,
                    "n_points_filtered_out": filtered,
                    "n_levels": n_levels,
                    "trend_status": "insufficient_data",
                    "note": f"need_at_least_{MIN_TREND_POINTS}_points_and_{MIN_TREND_LEVELS}_levels_after_quality_filter",
                    "rho_threshold": SPEARMAN_PASS_THRESHOLD,
                    "pvalue_threshold": SPEARMAN_PVALUE_THRESHOLD,
                }
            )
            continue

        rho, pvalue = spearman_stats(x_all[keep], y_all[keep])
        if not np.isfinite(rho):
            status = "insufficient_data"
            note = "non_finite_correlation"
        else:
            sign = 1.0 if expected == "positive" else -1.0
            aligned_rho = sign * rho
            direction_ok = aligned_rho >= SPEARMAN_PASS_THRESHOLD
            pvalue_ok = np.isfinite(pvalue) and pvalue <= SPEARMAN_PVALUE_THRESHOLD
            if direction_ok and pvalue_ok:
                status = "pass"
                note = ""
            elif direction_ok:
                status = "weak_pass"
                note = "direction_matches_but_not_statistically_significant"
            else:
                status = "fail"
                note = ""
        rows.append(
            {
                "variable": variable,
                "metric": metric,
                "expected_direction": expected,
                "spearman_rho": rho,
                "spearman_pvalue": pvalue,
                "n_points": n_points,
                "n_points_raw": n_points_raw,
                "n_points_filtered_out": filtered,
                "n_levels": n_levels,
                "trend_status": status,
                "note": note,
                "rho_threshold": SPEARMAN_PASS_THRESHOLD,
                "pvalue_threshold": SPEARMAN_PVALUE_THRESHOLD,
            }
        )
    return pd.DataFrame(rows)


def plot_trend_checks(trend_df: pd.DataFrame, out_path: Path) -> None:
    if trend_df.empty:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "No trend checks available", ha="center", va="center")
        ax.axis("off")
        fig.savefig(out_path, dpi=220)
        plt.close(fig)
        return

    label_map = {
        "current_a": r"$I_{current}$",
        "ar_to_n2": r"$[Ar]/[N_2]$",
        "o2_to_n2": r"$[O_2]/[N_2]$",
        "estimated_rotational_temperature": r"$\hat{T}_{rot}$",
        "estimated_vibrational_temperature": r"$\hat{T}_{vib}$",
        "estimated_excitation_temperature": r"$\hat{T}_{exc}$",
        "estimated_electron_density": r"$\hat{n}_{e}$",
        "relative_dissociation_proxy": r"$\hat{D}_{rel}$",
    }
    labels = []
    for _, r in trend_df.iterrows():
        src = label_map.get(str(r["variable"]), str(r["variable"]))
        dst = label_map.get(str(r["metric"]), str(r["metric"]))
        labels.append(f"{src} -> {dst}")
    vals = trend_df["spearman_rho"].to_numpy(dtype=float)
    vals = np.where(np.isfinite(vals), vals, 0.0)
    colors = []
    for status in trend_df["trend_status"].astype(str).tolist():
        if status == "pass":
            colors.append("#2a9d8f")
        elif status == "weak_pass":
            colors.append("#e9a03b")
        elif status == "fail":
            colors.append("#d1495b")
        else:
            colors.append("#9ca3af")

    fig, ax = plt.subplots(figsize=(11.0, max(4.0, 0.5 * len(labels))))
    y = np.arange(len(labels))
    ax.barh(y, vals, color=colors, alpha=0.9)
    ax.axvline(0.0, color="#333333", linewidth=1.0)
    ax.set_xlim(-1.0, 1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.8)
    ax.set_xlabel(r"Spearman $\rho$")
    ax.set_title("Model Validation Trend Checks")
    style_axes(ax, grid_axis="x")

    for i, (_, r) in enumerate(trend_df.iterrows()):
        n_used = int(_safe_float(r.get("n_points")))
        n_raw = int(_safe_float(r.get("n_points_raw")))
        pvalue = _safe_float(r.get("spearman_pvalue"))
        p_text = f", p={pvalue:.2g}" if np.isfinite(pvalue) else ""
        ax.text(
            0.98,
            i,
            f"{r['trend_status']} (n={n_used}/{n_raw}{p_text})",
            ha="right",
            va="center",
            transform=ax.get_yaxis_transform(),
            fontsize=8.4,
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_fit_diagnostics(scope: str, curves_df: pd.DataFrame, out_path: Path) -> None:
    if curves_df.empty:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "No spectra available for fit diagnostics", ha="center", va="center")
        ax.axis("off")
        fig.savefig(out_path, dpi=220)
        plt.close(fig)
        return

    groups = curves_df.groupby(["dataset", "param_set", "channel"], dropna=False)
    best_key = None
    best_signal = -np.inf
    for key, g in groups:
        wl = g["wavelength_nm"].to_numpy(dtype=float)
        y = g["irradiance_mean"].to_numpy(dtype=float)
        sig = integrated_line_signal(wl, y, center_nm=ROTATIONAL_CENTER_NM)["line_area"]
        if np.isfinite(sig) and sig > best_signal:
            best_signal = sig
            best_key = key
    if best_key is None:
        best_key = next(iter(groups.groups.keys()))

    g = groups.get_group(best_key).sort_values("wavelength_nm")
    wl = g["wavelength_nm"].to_numpy(dtype=float)
    y = g["irradiance_mean"].to_numpy(dtype=float)
    group_label = build_group_label(str(best_key[0]), str(best_key[1]), str(best_key[2]))
    rot = rotational_synthetic_fit(wl, y)
    hbeta = estimate_electron_density(wl, y, rotational_temperature_k=_safe_float(rot.get("temperature")))

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.0))
    ax = axes[0]
    m = (wl >= ROTATIONAL_CENTER_NM - 4.0) & (wl <= ROTATIONAL_CENTER_NM + 4.0)
    xw = wl[m]
    yw = y[m]
    if xw.size >= 2:
        baseline = line_baseline(xw, yw, center_nm=ROTATIONAL_CENTER_NM, inner_half_width_nm=2.0, outer_half_width_nm=3.8)
        y_corr = np.clip(yw - baseline, 0.0, None)
        ax.plot(xw, y_corr, label="Observed", color="#1f4e79")
        if np.isfinite(_safe_float(rot.get("temperature"))):
            sigma = _safe_float(rot.get("sigma_nm"))
            if not np.isfinite(sigma):
                sigma = 0.35
            profile = synthetic_band_profile(xw, _safe_float(rot.get("temperature")), sigma_nm=sigma)
            a = np.column_stack([profile, np.ones_like(profile)])
            coeff, *_ = np.linalg.lstsq(a, y_corr, rcond=None)
            pred = (float(coeff[0]) * profile) + float(coeff[1])
            ax.plot(xw, pred, "--", label="Synthetic-band fit", color="#d1495b")
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Intensity (a.u.)")
        ax.set_title(
            r"$N_2^{+}$ 391.44 nm Rotational Fit"
            + "\n"
            + rf"$\hat{{T}}_{{rot}}={_safe_float(rot.get('temperature')):.1f}\,\mathrm{{K}}$ | mode={rot.get('mode')}"
        )
        style_axes(ax, grid_axis="both")
        ax.legend(loc="best", fontsize=8)
    else:
        ax.text(0.5, 0.5, "Insufficient rotational-window points", ha="center", va="center")
        ax.axis("off")

    ax = axes[1]
    m = (wl >= HBETA_CENTER_NM - 4.0) & (wl <= HBETA_CENTER_NM + 4.0)
    xh = wl[m]
    yh = y[m]
    if xh.size >= 2:
        baseline = line_baseline(xh, yh, center_nm=HBETA_CENTER_NM, inner_half_width_nm=1.6, outer_half_width_nm=3.8)
        y_corr = np.clip(yh - baseline, 0.0, None)
        ax.plot(xh, y_corr, label="Observed", color="#2a9d8f")
        try:
            p0 = [float(np.nanmax(y_corr)), HBETA_CENTER_NM, 0.25, 0.25, 0.0]
            bounds = ([0.0, HBETA_CENTER_NM - 1.0, 0.02, 0.005, -np.inf], [np.inf, HBETA_CENTER_NM + 1.0, 3.0, 3.0, np.inf])
            popt, _ = curve_fit(voigt_profile_model, xh, y_corr, p0=p0, bounds=bounds, maxfev=20000)
            ax.plot(xh, voigt_profile_model(xh, *popt), "--", label="Voigt fit", color="#e76f51")
        except Exception:
            pass
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Intensity (a.u.)")
        ax.set_title(
            r"$H_{\beta}$ 486.1 nm Voigt Fit"
            + "\n"
            + rf"$\hat{{n}}_e={_safe_float(hbeta.get('estimated_electron_density')):.3e}\,\mathrm{{cm^{{-3}}}}$"
            + f" | status={hbeta.get('electron_density_status')}"
        )
        style_axes(ax, grid_axis="both")
        ax.legend(loc="best", fontsize=8)
    else:
        ax.text(0.5, 0.5, r"Insufficient $H_{\beta}$-window points", ha="center", va="center")
        ax.axis("off")

    fig.suptitle(f"Chemical Modeling Fit Diagnostics | {scope} | {group_label}", fontsize=11.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def write_scope_outputs(
    scope: str,
    estimates_df: pd.DataFrame,
    lines_df: pd.DataFrame,
    trend_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    link_df: pd.DataFrame,
    story_df: pd.DataFrame,
) -> List[Path]:
    out_paths: List[Path] = []
    meta_dir = metadata_section_dir(scope, "chemical_modeling")
    fig_dir = chemical_modeling_dir(scope)
    meta_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    estimates_csv = meta_dir / "chemical_modeling_estimates.csv"
    lines_csv = meta_dir / "key_line_intensity_table.csv"
    trend_csv = meta_dir / "trend_validation_checks.csv"
    edges_csv = meta_dir / "reduced_pathway_edge_weights.csv"
    links_csv = meta_dir / "peak_to_pathway_links.csv"
    story_csv = meta_dir / "pathway_peak_story_summary.csv"
    estimates_df.to_csv(estimates_csv, index=False)
    lines_df.to_csv(lines_csv, index=False)
    trend_df.to_csv(trend_csv, index=False)
    edges_df.to_csv(edges_csv, index=False)
    link_df.to_csv(links_csv, index=False)
    story_df.to_csv(story_csv, index=False)
    out_paths.extend([estimates_csv, lines_csv, trend_csv, edges_csv, links_csv, story_csv])

    fig1 = fig_dir / "Fig1.png"
    fig2 = fig_dir / "Fig2.png"
    fig3 = fig_dir / "Fig3.png"

    group_specs: List[Tuple[Path, pd.DataFrame, str, str]] = []
    next_fig_num = 4
    if not link_df.empty:
        group_rows = (
            link_df[["dataset", "param_set", "channel", "group_label"]]
            .drop_duplicates()
            .sort_values(["dataset", "param_set", "channel", "group_label"], ignore_index=True)
        )
        for _, g in group_rows.iterrows():
            g_label = str(g["group_label"])
            g_links = link_df[link_df["group_label"].astype(str) == g_label].copy()
            g_story = summarize_peak_pathway_story(g_links) if not g_links.empty else pd.DataFrame()
            if g_story.empty:
                continue
            fig_path = fig_dir / f"Fig{next_fig_num}.png"
            group_specs.append((fig_path, g_story, g_label, f"Fig{next_fig_num}"))
            next_fig_num += 1

    for old_fig in fig_dir.glob("*.png"):
        old_fig.unlink()

    plot_reduced_pathways(estimates_df, story_df, fig1)
    plot_peak_to_pathway_network(story_df, fig2, figure_label="Fig2", context_label=f"{scope} summary")
    plot_reaction_pathway_key(story_df, fig3, figure_label="Fig3")
    out_paths.extend([fig1, fig2, fig3])

    for fig_path, g_story, g_label, fig_label in group_specs:
        plot_peak_to_pathway_network(
            g_story,
            fig_path,
            figure_label=fig_label,
            context_label=g_label,
            footer_note="IDs map to reaction equations in Fig3.",
        )
        out_paths.append(fig_path)
    return out_paths


def process_scope(scope: str, vib_meta: pd.DataFrame, exc_meta: pd.DataFrame, gas_config: pd.DataFrame) -> List[Path]:
    estimates_df, lines_df = build_scope_estimates(scope, vib_meta, exc_meta, gas_config)
    trend_df = build_trend_checks(estimates_df) if not estimates_df.empty else pd.DataFrame()
    edges_df = build_pathway_edges(estimates_df) if not estimates_df.empty else pd.DataFrame()
    link_df = build_peak_to_pathway_links(estimates_df, lines_df) if (not estimates_df.empty and not lines_df.empty) else pd.DataFrame()
    story_df = summarize_peak_pathway_story(link_df) if not link_df.empty else pd.DataFrame()
    return write_scope_outputs(scope, estimates_df, lines_df, trend_df, edges_df, link_df, story_df)


def main() -> int:
    apply_publication_style()
    ensure_all_scope_layouts()

    vib_meta = load_vibrational_metadata()
    exc_meta = load_excitation_line_metadata()
    gas_config = load_gas_conditions()

    written: List[Path] = []
    for scope in SCOPES:
        written.extend(process_scope(scope, vib_meta, exc_meta, gas_config))

    print("Wrote chemical modeling outputs:")
    for path in sorted(set(written)):
        print(f"  {path}")
    print(
        "Done. Conservative labels used: estimated_rotational_temperature, "
        "estimated_vibrational_temperature, estimated_excitation_temperature, "
        "estimated_electron_density, relative_dissociation_proxy"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

