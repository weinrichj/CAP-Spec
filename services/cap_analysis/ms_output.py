#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

import pandas as pd

from analysis.ms_core import (
    average_curves,
    build_nist_match_summary,
    build_peak_table,
    build_summary,
    build_target_match_summary,
    match_peaks_to_nist,
    match_peaks_to_target_species,
    require_columns,
    unmatched_averaged_peaks,
)
from analysis.output_paths import SCOPES, ensure_all_scope_layouts, metadata_csv_path, metadata_section_dir
from data_ingestion.nist_wire import (
    get_nist_lines_for_range as fetch_nist_lines_for_range,
    load_target_species_lines as load_target_species_lines_ingestion,
)
from data_ingestion.scoped_writes import write_scoped_csv as write_scoped_section_csv

IN_LONG = metadata_csv_path("meta", "spectral", "spectra_long.csv")
NIST_CSV = Path("configs/nist_lines.csv")
NIST_FETCH_SPECIES_CSV = Path("configs/nist_fetch_species.csv")
TARGET_SPECIES_CSV = Path("configs/target_species_lines.csv")

AVERAGED_CURVES_NAME = "averaged_curves_long.csv"
AVERAGED_PEAKS_NAME = "averaged_peaks_top10.csv"
TRIAL_PEAKS_NAME = "trial_peaks_top10.csv"
NIST_MATCHES_NAME = "nist_matches_top3.csv"
NIST_MATCH_SUMMARY_NAME = "nist_match_summary.csv"
TARGET_MATCHES_NAME = "target_species_peak_matches.csv"
TARGET_MATCH_SUMMARY_NAME = "target_species_match_summary.csv"
NIST_SOURCE_LINES_NAME = "nist_lines_live.csv"
NIST_FETCH_STATUS_NAME = "nist_fetch_status.csv"
WORKBOOK_NAME = "peak_species_review.xlsx"

TOP_N_PEAKS = 10
TOP_NIST_CANDIDATES = 3
PEAK_DECIMALS = 1
NIST_TOLERANCE_NM = 0.5
TARGET_MATCH_TOLERANCE_NM = 2.0
WAVELENGTH_ROUND = 3

REQUIRED_LONG_COLS = {
    "dataset",
    "sample_id",
    "param_set",
    "channel",
    "wavelength_nm",
    "irradiance_W_m2_nm",
}

PEAK_KEY_COLS = ["dataset", "param_set", "channel", "peak_rank", "peak_wavelength_nm_0p1"]


def scope_raw_dir(scope: str) -> Path:
    return metadata_section_dir(scope, "spectral")


def write_scoped_csv(df: pd.DataFrame, file_name: str, allow_global: bool = False) -> List[Path]:
    return write_scoped_section_csv(
        df,
        section="spectral",
        filename=file_name,
        allow_global=allow_global,
        scopes=SCOPES,
    )


def meta_raw_path(file_name: str) -> Path:
    out = scope_raw_dir("meta") / file_name
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def write_excel(
    summary: pd.DataFrame,
    averaged_peaks: pd.DataFrame,
    trial_peaks: pd.DataFrame,
    matches: pd.DataFrame,
    nist_match_summary: pd.DataFrame,
    target_matches: pd.DataFrame,
    target_match_summary: pd.DataFrame,
    nist_lines: pd.DataFrame,
    fetch_status: pd.DataFrame,
    unmatched: pd.DataFrame,
    out_path: Path,
) -> None:
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="Summary", index=False)
        averaged_peaks.to_excel(writer, sheet_name="AveragedPeaks", index=False)
        trial_peaks.to_excel(writer, sheet_name="TrialPeaks", index=False)
        matches.to_excel(writer, sheet_name="NISTMatches", index=False)
        nist_match_summary.to_excel(writer, sheet_name="NISTMatchSummary", index=False)
        target_matches.to_excel(writer, sheet_name="TargetSpeciesMatches", index=False)
        target_match_summary.to_excel(writer, sheet_name="TargetSpeciesSummary", index=False)
        nist_lines.to_excel(writer, sheet_name="NISTSourceLines", index=False)
        fetch_status.to_excel(writer, sheet_name="NISTFetchStatus", index=False)
        unmatched.to_excel(writer, sheet_name="UnmatchedAveragedPeaks", index=False)


def main() -> int:
    ensure_all_scope_layouts()

    if not IN_LONG.exists():
        print(f"Missing {IN_LONG}. Run preprocess.py first.")
        return 1

    spectra_long = pd.read_csv(IN_LONG)
    if not require_columns(spectra_long, REQUIRED_LONG_COLS, str(IN_LONG)):
        return 2

    averaged = average_curves(spectra_long, wavelength_round=WAVELENGTH_ROUND)
    if averaged.empty:
        print("No averaged curves could be built from input.")
        return 2

    written_paths: List[Path] = []
    written_paths.extend(write_scoped_csv(averaged, AVERAGED_CURVES_NAME))

    averaged_peaks = build_peak_table(
        averaged,
        group_cols=["dataset", "param_set", "channel"],
        value_col="irradiance_mean",
        intensity_label="peak_intensity",
        top_n=TOP_N_PEAKS,
        extra_cols=["n_curves"],
        peak_decimals=PEAK_DECIMALS,
    )
    written_paths.extend(write_scoped_csv(averaged_peaks, AVERAGED_PEAKS_NAME))

    trial_group_cols = ["dataset", "param_set", "channel", "sample_id"]
    if "trial" in spectra_long.columns:
        trial_group_cols.append("trial")
    trial_peaks = build_peak_table(
        spectra_long,
        group_cols=trial_group_cols,
        value_col="irradiance_W_m2_nm",
        intensity_label="peak_intensity",
        top_n=TOP_N_PEAKS,
        peak_decimals=PEAK_DECIMALS,
    )
    written_paths.extend(write_scoped_csv(trial_peaks, TRIAL_PEAKS_NAME))

    try:
        targets = load_target_species_lines_ingestion(TARGET_SPECIES_CSV)
    except Exception as e:
        print(f"[FAIL] Could not load target species file {TARGET_SPECIES_CSV}: {e}")
        return 2
    target_matches = match_peaks_to_target_species(
        averaged_peaks=averaged_peaks,
        targets=targets,
        tolerance_nm=TARGET_MATCH_TOLERANCE_NM,
    )
    written_paths.extend(write_scoped_csv(target_matches, TARGET_MATCHES_NAME))
    target_match_summary = build_target_match_summary(target_matches)
    written_paths.extend(write_scoped_csv(target_match_summary, TARGET_MATCH_SUMMARY_NAME))

    range_low = float(averaged["wavelength_nm"].min())
    range_high = float(averaged["wavelength_nm"].max())
    try:
        nist_df, fetch_status = fetch_nist_lines_for_range(
            low_nm=range_low,
            high_nm=range_high,
            fetch_species_csv=NIST_FETCH_SPECIES_CSV,
            fallback_csv=NIST_CSV,
        )
    except Exception as e:
        print(f"[FAIL] Could not retrieve NIST lines: {e}")
        return 2

    matches = match_peaks_to_nist(
        averaged_peaks=averaged_peaks,
        nist_df=nist_df,
        nist_tolerance_nm=NIST_TOLERANCE_NM,
        top_n_nist_candidates=TOP_NIST_CANDIDATES,
    )
    written_paths.extend(write_scoped_csv(matches, NIST_MATCHES_NAME))
    nist_match_summary = build_nist_match_summary(matches)
    written_paths.extend(write_scoped_csv(nist_match_summary, NIST_MATCH_SUMMARY_NAME))
    written_paths.extend(write_scoped_csv(fetch_status, NIST_FETCH_STATUS_NAME, allow_global=True))
    if nist_df is not None and not nist_df.empty:
        nist_live_path = meta_raw_path(NIST_SOURCE_LINES_NAME)
        nist_df.to_csv(nist_live_path, index=False)
        written_paths.append(nist_live_path)

    unmatched = unmatched_averaged_peaks(averaged_peaks, matches, peak_key_cols=PEAK_KEY_COLS)
    summary = build_summary(averaged_peaks, trial_peaks, matches, peak_key_cols=PEAK_KEY_COLS)

    excel_error: Exception | None = None
    try:
        write_excel(
            summary,
            averaged_peaks,
            trial_peaks,
            matches,
            nist_match_summary,
            target_matches,
            target_match_summary,
            nist_df if nist_df is not None else pd.DataFrame(),
            fetch_status,
            unmatched,
            meta_raw_path(WORKBOOK_NAME),
        )
    except Exception as e:
        excel_error = e

    print("\nWrote:")
    for path in sorted(set(written_paths)):
        print(f"  {path}")
    print(f"  {meta_raw_path(WORKBOOK_NAME)}")

    if excel_error is not None:
        print(f"[FAIL] Excel workbook write failed: {excel_error}")
        return 2

    if nist_df is None:
        print(f"[WARN] NIST matches are empty because live fetch failed and {NIST_CSV} was unavailable/invalid.")
    print(
        f"Done. averaged_curves={len(averaged)} averaged_peaks={len(averaged_peaks)} "
        f"trial_peaks={len(trial_peaks)} nist_matches={len(matches)} "
        f"target_species_matches={len(target_matches)} "
        f"nist_source_lines={0 if nist_df is None else len(nist_df)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

