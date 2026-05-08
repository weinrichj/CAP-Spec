#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from analysis.numeric_utils import safe_ratio, trapz_integral
from analysis.output_paths import ensure_all_scope_layouts, metadata_csv_path
from data_ingestion.scoped_writes import write_scoped_csv

LONG_CSV = metadata_csv_path("meta", "spectral", "spectra_long.csv")
BANDS_CSV = Path("configs/wavelengths.csv")
NORMALIZE = "none"
DEFAULT_BANDS: List[Tuple[str, float, float]] = [
    ("UVC", 200.0, 280.0),
    ("UVB", 280.0, 315.0),
    ("UVA", 315.0, 400.0),
]


def load_bands(config_path: Path) -> List[Tuple[str, float, float]]:
    if not config_path.exists():
        return DEFAULT_BANDS

    df = pd.read_csv(config_path)
    needed = {"band", "start_nm", "end_nm"}
    if not needed.issubset(set(c.lower() for c in df.columns)):
        raise ValueError(f"{config_path} must have columns: band,start_nm,end_nm")

    df = df.rename(columns={c: c.lower() for c in df.columns})
    bands: List[Tuple[str, float, float]] = []
    for _, r in df.iterrows():
        band = str(r["band"]).strip()
        start = float(r["start_nm"])
        end = float(r["end_nm"])
        if end <= start:
            raise ValueError(f"Band {band} has end_nm <= start_nm ({start}, {end})")
        bands.append((band, start, end))
    return bands


def centroid(wl: np.ndarray, y: np.ndarray) -> float:
    area = float(np.trapz(y, wl))
    if not np.isfinite(area) or area <= 0:
        return float("nan")
    return float(np.trapz(wl * y, wl) / area)


def band_integral(wl: np.ndarray, y: np.ndarray, start: float, end: float) -> float:
    mask = (wl >= start) & (wl <= end)
    if mask.sum() < 2:
        return 0.0
    return trapz_integral(wl[mask], y[mask])


def normalize_curve(wl: np.ndarray, y: np.ndarray) -> np.ndarray:
    mode = NORMALIZE.lower()
    if mode == "none":
        return y
    if mode == "area":
        area = trapz_integral(wl, y)
        return y / area if np.isfinite(area) and area != 0 else y
    if mode == "max":
        mx = float(np.nanmax(y)) if y.size else float("nan")
        return y / mx if np.isfinite(mx) and mx != 0 else y
    raise ValueError(f"Unknown NORMALIZE mode: {NORMALIZE}")


def write_scoped_features(features_df: pd.DataFrame) -> List[Path]:
    return write_scoped_csv(features_df, section="features", filename="features.csv")


def main() -> int:
    ensure_all_scope_layouts()

    if not LONG_CSV.exists():
        print(f"Missing {LONG_CSV}. Run preprocess.py first.")
        return 1

    bands = load_bands(BANDS_CSV)
    df = pd.read_csv(LONG_CSV)

    required = {"dataset", "sample_id", "param_set", "channel", "wavelength_nm", "irradiance_W_m2_nm"}
    if not required.issubset(df.columns):
        print(f"{LONG_CSV} must have columns: {sorted(required)}")
        return 2

    group_cols = ["sample_id"] + [c for c in ["dataset", "param_set", "trial", "channel"] if c in df.columns]
    rows: List[Dict[str, object]] = []

    for key, g in df.groupby(group_cols, dropna=False):
        g = g.sort_values("wavelength_nm")
        wl = g["wavelength_nm"].to_numpy(dtype=float)
        y = normalize_curve(wl, g["irradiance_W_m2_nm"].to_numpy(dtype=float))

        total = trapz_integral(wl, y)
        c_nm = centroid(wl, y)
        if y.size:
            i_peak = int(np.nanargmax(y))
            peak_wl = float(wl[i_peak])
            peak_y = float(y[i_peak])
        else:
            peak_wl = float("nan")
            peak_y = float("nan")

        band_vals = {band: band_integral(wl, y, start, end) for band, start, end in bands}
        bmap = {b.upper(): v for b, v in band_vals.items()}

        out: Dict[str, object] = {}
        if isinstance(key, tuple):
            for col, val in zip(group_cols, key):
                out[col] = val
        else:
            out[group_cols[0]] = key

        out.update(
            {
                "normalize": NORMALIZE,
                "total_irradiance": total,
                "centroid_nm": c_nm,
                "peak_wavelength_nm": peak_wl,
                "peak_irradiance": peak_y,
            }
        )

        for band, _, _ in bands:
            out[f"{band}_integral"] = band_vals[band]
            out[f"{band}_frac"] = safe_ratio(band_vals[band], total)

        out["uva_uvb_ratio"] = safe_ratio(bmap.get("UVA", float("nan")), bmap.get("UVB", float("nan")))
        out["uvb_uvc_ratio"] = safe_ratio(bmap.get("UVB", float("nan")), bmap.get("UVC", float("nan")))
        out["uva_uvc_ratio"] = safe_ratio(bmap.get("UVA", float("nan")), bmap.get("UVC", float("nan")))
        rows.append(out)

    out_df = pd.DataFrame(rows).sort_values(["dataset", "sample_id"], ignore_index=True)
    written = write_scoped_features(out_df)

    print(f"Wrote {metadata_csv_path('meta', 'features', 'features.csv')} ({len(out_df)} rows)")
    for path in sorted(written):
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
