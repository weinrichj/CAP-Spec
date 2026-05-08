from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

DEFAULT_PEAK_KEY_COLS = ["dataset", "param_set", "channel", "peak_rank", "peak_wavelength_nm_0p1"]


def require_columns(df: pd.DataFrame, cols: Sequence[str], source_name: str) -> bool:
    missing = sorted(set(cols) - set(df.columns))
    if missing:
        print(f"{source_name} missing columns: {missing}")
        return False
    return True


def average_curves(df: pd.DataFrame, wavelength_round: int = 3) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    grouped = df.groupby(["dataset", "param_set", "channel"], dropna=False)
    for (dataset, param_set, channel), group in grouped:
        series_list: List[pd.Series] = []
        for sample_id, g in group.groupby("sample_id", dropna=False):
            wl = np.round(g["wavelength_nm"].to_numpy(dtype=float), wavelength_round)
            y = g["irradiance_W_m2_nm"].to_numpy(dtype=float)
            s = pd.Series(y, index=wl, name=str(sample_id))
            series_list.append(s[~pd.Index(s.index).duplicated(keep="first")])

        if not series_list:
            continue

        aligned = pd.concat(series_list, axis=1, sort=True)
        mean = aligned.mean(axis=1, skipna=True)
        std = aligned.std(axis=1, ddof=1, skipna=True)
        n_curves = aligned.count(axis=1)
        rows.append(
            pd.DataFrame(
                {
                    "dataset": str(dataset),
                    "param_set": str(param_set),
                    "channel": str(channel),
                    "wavelength_nm": mean.index.to_numpy(dtype=float),
                    "irradiance_mean": mean.to_numpy(dtype=float),
                    "irradiance_std": std.to_numpy(dtype=float),
                    "n_curves": n_curves.to_numpy(dtype=int),
                }
            ).sort_values("wavelength_nm", ignore_index=True)
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "dataset",
                "param_set",
                "channel",
                "wavelength_nm",
                "irradiance_mean",
                "irradiance_std",
                "n_curves",
            ]
        )

    return pd.concat(rows, ignore_index=True).sort_values(
        ["dataset", "param_set", "channel", "wavelength_nm"], ignore_index=True
    )


def local_maxima_indices(y: np.ndarray) -> List[int]:
    if y.size < 3:
        return []
    out: List[int] = []
    for i in range(1, y.size - 1):
        left = y[i - 1]
        mid = y[i]
        right = y[i + 1]
        if np.isfinite(left) and np.isfinite(mid) and np.isfinite(right) and mid > left and mid > right:
            out.append(i)
    return out


def refine_peak_quadratic(
    x_left: float, y_left: float, x_mid: float, y_mid: float, x_right: float, y_right: float
) -> Dict[str, float]:
    denom = y_left - (2.0 * y_mid) + y_right
    if not np.isfinite(denom) or denom == 0:
        return {"refined_wavelength_nm": float(x_mid), "refined_intensity": float(y_mid)}

    offset = 0.5 * (y_left - y_right) / denom
    if not np.isfinite(offset):
        return {"refined_wavelength_nm": float(x_mid), "refined_intensity": float(y_mid)}

    offset = float(np.clip(offset, -1.0, 1.0))
    step = 0.5 * (x_right - x_left)
    if not np.isfinite(step) or step <= 0:
        return {"refined_wavelength_nm": float(x_mid), "refined_intensity": float(y_mid)}

    refined_wl = float(np.clip(x_mid + (offset * step), min(x_left, x_right), max(x_left, x_right)))
    refined_y = float(y_mid - (0.25 * (y_left - y_right) * offset))
    return {"refined_wavelength_nm": refined_wl, "refined_intensity": refined_y}


def detect_top_peaks(
    wl: np.ndarray,
    y: np.ndarray,
    top_n: int,
    intensity_col_name: str,
    peak_decimals: int = 1,
) -> List[Dict[str, float]]:
    indices = local_maxima_indices(y)
    peaks: List[Dict[str, float]] = []
    for idx in indices:
        refined = refine_peak_quadratic(wl[idx - 1], y[idx - 1], wl[idx], y[idx], wl[idx + 1], y[idx + 1])
        peaks.append(
            {
                "peak_wavelength_nm_grid": float(wl[idx]),
                "peak_wavelength_nm_refined": float(refined["refined_wavelength_nm"]),
                "peak_wavelength_nm_0p1": round(float(refined["refined_wavelength_nm"]), peak_decimals),
                intensity_col_name: float(y[idx]),
                "peak_intensity_refined": float(refined["refined_intensity"]),
            }
        )

    peaks.sort(key=lambda row: (-row[intensity_col_name], row["peak_wavelength_nm_grid"]))
    return peaks[:top_n]


def build_peak_table(
    spectra_df: pd.DataFrame,
    group_cols: Sequence[str],
    value_col: str,
    intensity_label: str,
    top_n: int,
    extra_cols: Sequence[str] | None = None,
    peak_decimals: int = 1,
) -> pd.DataFrame:
    extra_cols = list(extra_cols or [])
    rows: List[Dict[str, object]] = []
    for key, g in spectra_df.groupby(list(group_cols), dropna=False):
        g = g.sort_values("wavelength_nm", ignore_index=True)
        wl = g["wavelength_nm"].to_numpy(dtype=float)
        y = g[value_col].to_numpy(dtype=float)

        peaks = detect_top_peaks(wl, y, top_n=top_n, intensity_col_name=intensity_label, peak_decimals=peak_decimals)
        if not peaks:
            continue

        if not isinstance(key, tuple):
            key = (key,)
        key_map = dict(zip(group_cols, key))

        extra_map: Dict[str, object] = {}
        for col in extra_cols:
            if col in g.columns:
                val = g[col].iloc[0]
                if pd.isna(val):
                    continue
                extra_map[col] = val

        for rank, peak in enumerate(peaks, start=1):
            row: Dict[str, object] = {"peak_rank": rank}
            row.update(key_map)
            row.update(extra_map)
            row.update(peak)
            rows.append(row)

    if not rows:
        cols = list(group_cols) + list(extra_cols) + [
            "peak_rank",
            "peak_wavelength_nm_grid",
            "peak_wavelength_nm_refined",
            "peak_wavelength_nm_0p1",
            intensity_label,
            "peak_intensity_refined",
        ]
        return pd.DataFrame(columns=cols)

    out = pd.DataFrame(rows)
    return out.sort_values(list(group_cols) + ["peak_rank"], ignore_index=True)


def match_peaks_to_target_species(
    averaged_peaks: pd.DataFrame,
    targets: pd.DataFrame,
    tolerance_nm: float,
) -> pd.DataFrame:
    cols = [
        "dataset",
        "param_set",
        "channel",
        "species",
        "target_wavelength_nm",
        "target_rank",
        "matched",
        "matched_peak_rank",
        "matched_peak_wavelength_nm_0p1",
        "matched_peak_intensity",
        "delta_nm",
    ]
    if averaged_peaks.empty or targets.empty:
        return pd.DataFrame(columns=cols)

    rows: List[Dict[str, object]] = []
    group_cols = ["dataset", "param_set", "channel"]
    required_cols = set(group_cols + ["peak_rank", "peak_wavelength_nm_0p1"])
    if not required_cols.issubset(averaged_peaks.columns):
        return pd.DataFrame(columns=cols)

    for key, gp in averaged_peaks.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        group_map = dict(zip(group_cols, key))
        peak_wl = pd.to_numeric(gp["peak_wavelength_nm_0p1"], errors="coerce")
        peak_rank = pd.to_numeric(gp["peak_rank"], errors="coerce")
        peak_int = pd.to_numeric(
            gp.get("peak_intensity_refined", gp.get("peak_intensity", np.nan)), errors="coerce"
        )

        work = gp.copy()
        work["peak_wl"] = peak_wl
        work["peak_rank_num"] = peak_rank
        work["peak_int_num"] = peak_int
        work = work[np.isfinite(work["peak_wl"])].copy()
        if work.empty:
            continue

        for t_idx, t in targets.reset_index(drop=True).iterrows():
            target_wl = float(t["wavelength_nm"])
            species = str(t["species"])
            delta = np.abs(work["peak_wl"].to_numpy(dtype=float) - target_wl)
            best_pos = int(np.argmin(delta))
            best_row = work.iloc[best_pos]
            best_delta = float(delta[best_pos])
            matched = bool(np.isfinite(best_delta) and best_delta <= float(tolerance_nm))
            rows.append(
                {
                    **group_map,
                    "species": species,
                    "target_wavelength_nm": target_wl,
                    "target_rank": int(t_idx + 1),
                    "matched": matched,
                    "matched_peak_rank": int(best_row["peak_rank_num"]) if matched else np.nan,
                    "matched_peak_wavelength_nm_0p1": float(best_row["peak_wl"]) if matched else np.nan,
                    "matched_peak_intensity": float(best_row["peak_int_num"]) if matched else np.nan,
                    "delta_nm": best_delta,
                }
            )

    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows).sort_values(
        ["dataset", "param_set", "channel", "target_rank"], ignore_index=True
    )


def build_target_match_summary(target_matches: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "dataset",
        "param_set",
        "channel",
        "species",
        "targets_total",
        "targets_matched",
        "match_rate",
        "mean_delta_nm",
        "mean_matched_peak_intensity",
    ]
    if target_matches.empty:
        return pd.DataFrame(columns=cols)

    d = target_matches.copy()
    d["matched_num"] = d["matched"].astype(bool).astype(int)
    d["delta_for_match"] = pd.to_numeric(d["delta_nm"], errors="coerce").where(d["matched"].astype(bool), np.nan)
    d["int_for_match"] = pd.to_numeric(d["matched_peak_intensity"], errors="coerce").where(
        d["matched"].astype(bool), np.nan
    )

    out = (
        d.groupby(["dataset", "param_set", "channel", "species"], dropna=False)
        .agg(
            targets_total=("target_wavelength_nm", "size"),
            targets_matched=("matched_num", "sum"),
            mean_delta_nm=("delta_for_match", "mean"),
            mean_matched_peak_intensity=("int_for_match", "mean"),
        )
        .reset_index()
    )
    out["match_rate"] = (
        pd.to_numeric(out["targets_matched"], errors="coerce")
        / pd.to_numeric(out["targets_total"], errors="coerce").replace(0, np.nan)
    ).fillna(0.0)
    return out[cols].sort_values(["dataset", "param_set", "channel", "species"], ignore_index=True)


def match_peaks_to_nist(
    averaged_peaks: pd.DataFrame,
    nist_df: pd.DataFrame | None,
    nist_tolerance_nm: float = 0.5,
    top_n_nist_candidates: int = 3,
) -> pd.DataFrame:
    if averaged_peaks.empty or nist_df is None or nist_df.empty:
        base_cols = list(averaged_peaks.columns)
        return pd.DataFrame(columns=base_cols + ["candidate_rank", "delta_nm", "nist_wavelength_nm"])

    rows: List[Dict[str, object]] = []
    has_rel_intensity = "rel_intensity" in nist_df.columns
    pass_through_cols = [c for c in nist_df.columns if c != "wavelength_nm"]

    for peak in averaged_peaks.to_dict(orient="records"):
        peak_wl = float(peak["peak_wavelength_nm_0p1"])
        candidates = nist_df[np.abs(nist_df["wavelength_nm"] - peak_wl) <= nist_tolerance_nm].copy()
        if candidates.empty:
            continue

        candidates["delta_nm"] = np.abs(candidates["wavelength_nm"] - peak_wl)
        sort_cols = ["delta_nm"]
        asc = [True]
        if has_rel_intensity:
            sort_cols.append("rel_intensity")
            asc.append(False)
        candidates = candidates.sort_values(sort_cols, ascending=asc).head(top_n_nist_candidates)

        for rank, (_, cand) in enumerate(candidates.iterrows(), start=1):
            row = dict(peak)
            row["candidate_rank"] = rank
            row["delta_nm"] = float(cand["delta_nm"])
            row["nist_wavelength_nm"] = float(cand["wavelength_nm"])
            for col in pass_through_cols:
                row[f"nist_{col}"] = cand[col]
            rows.append(row)

    if not rows:
        base_cols = list(averaged_peaks.columns)
        return pd.DataFrame(columns=base_cols + ["candidate_rank", "delta_nm", "nist_wavelength_nm"])

    out = pd.DataFrame(rows)
    return out.sort_values(
        ["dataset", "param_set", "channel", "peak_rank", "candidate_rank"], ignore_index=True
    )


def build_nist_match_summary(matches: pd.DataFrame) -> pd.DataFrame:
    cols = ["dataset", "param_set", "channel", "candidate_label", "matched_peak_count", "candidate_rows", "score"]
    if matches.empty:
        return pd.DataFrame(columns=cols)

    label_col = "nist_species" if "nist_species" in matches.columns else None
    if label_col is None and "nist_spectra_query" in matches.columns:
        label_col = "nist_spectra_query"
    if label_col is None and "nist_element" in matches.columns:
        label_col = "nist_element"

    df = matches.copy()
    if label_col is None:
        df["candidate_label"] = "unknown"
    else:
        df["candidate_label"] = df[label_col].fillna("unknown").astype(str)

    df["match_score"] = (1.0 / df["candidate_rank"].clip(lower=1).astype(float)) * (
        1.0 / (1.0 + df["delta_nm"].astype(float))
    )
    key_cols = ["dataset", "param_set", "channel", "candidate_label"]
    grouped = df.groupby(key_cols, dropna=False).agg(
        candidate_rows=("candidate_rank", "size"),
        matched_peak_count=("peak_rank", "nunique"),
        score=("match_score", "sum"),
    )
    out = grouped.reset_index().sort_values(
        ["dataset", "param_set", "channel", "score", "matched_peak_count"],
        ascending=[True, True, True, False, False],
        ignore_index=True,
    )
    return out


def unmatched_averaged_peaks(
    averaged_peaks: pd.DataFrame,
    matches: pd.DataFrame,
    peak_key_cols: Sequence[str] = DEFAULT_PEAK_KEY_COLS,
) -> pd.DataFrame:
    if averaged_peaks.empty:
        return averaged_peaks.copy()
    if matches.empty:
        return averaged_peaks.copy()

    matched_keys = matches[list(peak_key_cols)].drop_duplicates().assign(_matched=1)
    tagged = averaged_peaks.merge(matched_keys, how="left", on=list(peak_key_cols))
    return tagged[tagged["_matched"].isna()].drop(columns=["_matched"]).reset_index(drop=True)


def build_summary(
    averaged_peaks: pd.DataFrame,
    trial_peaks: pd.DataFrame,
    matches: pd.DataFrame,
    peak_key_cols: Sequence[str] = DEFAULT_PEAK_KEY_COLS,
) -> pd.DataFrame:
    group_cols = ["dataset", "param_set", "channel"]

    avg_counts = (
        averaged_peaks.groupby(group_cols, dropna=False).size().rename("n_averaged_peaks").reset_index()
        if not averaged_peaks.empty
        else pd.DataFrame(columns=group_cols + ["n_averaged_peaks"])
    )
    matched_counts = (
        matches[group_cols + [peak_key_cols[-2], peak_key_cols[-1]]]
        .drop_duplicates()
        .groupby(group_cols, dropna=False)
        .size()
        .rename("n_averaged_peaks_matched")
        .reset_index()
        if not matches.empty
        else pd.DataFrame(columns=group_cols + ["n_averaged_peaks_matched"])
    )
    trial_counts = (
        trial_peaks.groupby(group_cols, dropna=False).size().rename("n_trial_peaks").reset_index()
        if not trial_peaks.empty
        else pd.DataFrame(columns=group_cols + ["n_trial_peaks"])
    )
    candidate_counts = (
        matches.groupby(group_cols, dropna=False).size().rename("n_nist_candidates").reset_index()
        if not matches.empty
        else pd.DataFrame(columns=group_cols + ["n_nist_candidates"])
    )

    summary = avg_counts.merge(matched_counts, on=group_cols, how="outer")
    summary = summary.merge(trial_counts, on=group_cols, how="outer")
    summary = summary.merge(candidate_counts, on=group_cols, how="outer")

    for col in ["n_averaged_peaks", "n_averaged_peaks_matched", "n_trial_peaks", "n_nist_candidates"]:
        if col not in summary.columns:
            summary[col] = 0
        summary[col] = pd.to_numeric(summary[col], errors="coerce").fillna(0).astype(int)
    summary["n_averaged_peaks_unmatched"] = summary["n_averaged_peaks"] - summary["n_averaged_peaks_matched"]

    return summary.sort_values(group_cols, ignore_index=True)

