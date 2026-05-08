#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.output_paths import SCOPES, chemspecies_figures_dir, ensure_all_scope_layouts, metadata_csv_path, metadata_section_dir
from plots.style import apply_publication_style, get_palette, style_axes, to_species_label

TARGET_MATCHES_CSV = metadata_csv_path("meta", "spectral", "target_species_peak_matches.csv")
DPI = 220
TOP_SPECIES = 8


def scope_csv_dir(scope: str) -> Path:
    return metadata_section_dir(scope, "chemspecies")


def scope_fig_dir(scope: str) -> Path:
    return chemspecies_figures_dir(scope)


def safe_ratio(a: pd.Series, b: pd.Series) -> pd.Series:
    out = a / b.replace(0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan)


def group_label_frame(df: pd.DataFrame) -> pd.Series:
    return (
        df["dataset"].astype(str)
        + " | "
        + df["param_set"].astype(str)
        + " | "
        + df["channel"].astype(str)
    )


def load_target_matches(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    m = pd.read_csv(path)
    if m.empty:
        return pd.DataFrame()

    required = {"dataset", "param_set", "channel", "species", "target_wavelength_nm", "delta_nm"}
    missing = required - set(m.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")

    m = m.copy()
    if "matched" in m.columns:
        m = m[m["matched"].astype(bool)].copy()
    if m.empty:
        return pd.DataFrame()

    m["species"] = m["species"].fillna("").astype(str).str.strip()
    m = m[m["species"] != ""].copy()
    if m.empty:
        return pd.DataFrame()

    m["target_wavelength_nm"] = pd.to_numeric(m["target_wavelength_nm"], errors="coerce")
    m["delta_nm"] = pd.to_numeric(m["delta_nm"], errors="coerce")
    if "matched_peak_intensity" in m.columns:
        m["intensity"] = pd.to_numeric(m["matched_peak_intensity"], errors="coerce")
    elif "peak_intensity_refined" in m.columns:
        m["intensity"] = pd.to_numeric(m["peak_intensity_refined"], errors="coerce")
    elif "peak_intensity" in m.columns:
        m["intensity"] = pd.to_numeric(m["peak_intensity"], errors="coerce")
    else:
        m["intensity"] = np.nan
    m["intensity"] = m["intensity"].fillna(0.0)
    m = m.dropna(subset=["target_wavelength_nm", "delta_nm"]).reset_index(drop=True)
    return m


def build_group_concentration_tables(matches: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = [
        "dataset",
        "param_set",
        "channel",
        "species",
        "peak_assignments",
        "concentration_score",
        "relative_concentration",
        "mean_delta_nm",
        "group_total_signal",
        "group_label",
    ]
    if matches.empty:
        return pd.DataFrame(columns=cols), pd.DataFrame(columns=["dataset", "param_set", "channel", "group_label"])

    m = matches.copy()
    m["delta_weight"] = 1.0 / (1.0 + m["delta_nm"].abs().fillna(0.0))
    m["contribution"] = m["intensity"] * m["delta_weight"]

    group_cols = ["dataset", "param_set", "channel", "species"]
    long = (
        m.groupby(group_cols, dropna=False)
        .agg(
            peak_assignments=("target_wavelength_nm", "size"),
            concentration_score=("contribution", "sum"),
            mean_delta_nm=("delta_nm", "mean"),
        )
        .reset_index()
    )
    long["peak_assignments"] = pd.to_numeric(long["peak_assignments"], errors="coerce").fillna(0).astype(int)

    key_cols = ["dataset", "param_set", "channel"]
    group_totals = (
        long.groupby(key_cols, dropna=False)["concentration_score"]
        .sum()
        .rename("group_total_signal")
        .reset_index()
    )
    long = long.merge(group_totals, on=key_cols, how="left")
    long["relative_concentration"] = safe_ratio(long["concentration_score"], long["group_total_signal"]).fillna(0.0)
    long["group_label"] = group_label_frame(long)
    long = long.sort_values(
        ["dataset", "param_set", "channel", "relative_concentration"],
        ascending=[True, True, True, False],
        ignore_index=True,
    )

    wide = (
        long.pivot_table(
            index=key_cols,
            columns="species",
            values="relative_concentration",
            aggfunc="sum",
            fill_value=0.0,
        )
        .reset_index()
    )
    wide.columns = [str(c) for c in wide.columns]
    wide["group_label"] = group_label_frame(wide)
    wide = wide.sort_values(key_cols, ignore_index=True)
    return long, wide


def summarize_dataset_species(group_long: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "dataset",
        "species",
        "mean_relative_concentration",
        "std_relative_concentration",
        "mean_concentration_score",
        "mean_peak_assignments",
        "n_groups",
    ]
    if group_long.empty:
        return pd.DataFrame(columns=cols)

    base = (
        group_long.groupby(["dataset", "species"], dropna=False)
        .agg(
            mean_relative_concentration=("relative_concentration", "mean"),
            std_relative_concentration=("relative_concentration", "std"),
            mean_concentration_score=("concentration_score", "mean"),
            mean_peak_assignments=("peak_assignments", "mean"),
            n_groups=("group_label", "nunique"),
        )
        .reset_index()
    )
    base["std_relative_concentration"] = pd.to_numeric(
        base["std_relative_concentration"], errors="coerce"
    ).fillna(0.0)

    combined = (
        group_long.groupby("species", dropna=False)
        .agg(
            mean_relative_concentration=("relative_concentration", "mean"),
            std_relative_concentration=("relative_concentration", "std"),
            mean_concentration_score=("concentration_score", "mean"),
            mean_peak_assignments=("peak_assignments", "mean"),
            n_groups=("group_label", "nunique"),
        )
        .reset_index()
    )
    combined["dataset"] = "combined"
    combined["std_relative_concentration"] = pd.to_numeric(
        combined["std_relative_concentration"], errors="coerce"
    ).fillna(0.0)

    out = pd.concat([base, combined], ignore_index=True)
    return out[cols].sort_values(
        ["dataset", "mean_relative_concentration"], ascending=[True, False], ignore_index=True
    )


def air_vs_diameter_species_delta(summary: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "species",
        "air_mean_relative_concentration",
        "diameter_mean_relative_concentration",
        "air_minus_diameter",
        "abs_delta",
    ]
    if summary.empty:
        return pd.DataFrame(columns=cols)

    d = summary[summary["dataset"].isin(["air", "diameter"])].copy()
    if d.empty:
        return pd.DataFrame(columns=cols)
    piv = d.pivot_table(
        index="species",
        columns="dataset",
        values="mean_relative_concentration",
        aggfunc="mean",
        fill_value=0.0,
    ).reset_index()
    if "air" not in piv.columns:
        piv["air"] = 0.0
    if "diameter" not in piv.columns:
        piv["diameter"] = 0.0
    piv["air_minus_diameter"] = piv["air"] - piv["diameter"]
    piv["abs_delta"] = piv["air_minus_diameter"].abs()
    piv = piv.rename(
        columns={
            "air": "air_mean_relative_concentration",
            "diameter": "diameter_mean_relative_concentration",
        }
    )
    return piv[cols].sort_values("abs_delta", ascending=False, ignore_index=True)


def build_key_group_findings(group_long: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "dataset",
        "param_set",
        "channel",
        "group_label",
        "species",
        "relative_concentration",
        "concentration_score",
    ]
    if group_long.empty:
        return pd.DataFrame(columns=cols)

    rows: List[pd.DataFrame] = []
    for _, g in group_long.groupby(["dataset", "param_set", "channel"], dropna=False):
        rows.append(g.nlargest(3, "relative_concentration")[cols])
    out = pd.concat(rows, ignore_index=True)
    return out.sort_values(
        ["dataset", "param_set", "channel", "relative_concentration"],
        ascending=[True, True, True, False],
        ignore_index=True,
    )


def scope_rows(group_long: pd.DataFrame, scope: str) -> pd.DataFrame:
    if group_long.empty:
        return pd.DataFrame(columns=group_long.columns)
    d = group_long[group_long["dataset"].astype(str).str.lower() == scope.lower()].copy()
    if d.empty:
        return d
    d["scope_group"] = d["param_set"].astype(str) + " | " + d["channel"].astype(str)
    return d


def write_empty_figure(out_path: Path, message: str, figsize: tuple[float, float] = (8.0, 4.0)) -> None:
    fig, ax = plt.subplots(figsize=figsize)
    ax.text(0.5, 0.5, message, ha="center", va="center")
    ax.axis("off")
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)


def plot_scope_group_heatmap(group_long: pd.DataFrame, scope: str, out_path: Path) -> None:
    d = scope_rows(group_long, scope)
    if d.empty:
        write_empty_figure(out_path, f"No {scope} group concentration data")
        return

    top_species = (
        d.groupby("species", dropna=False)["relative_concentration"]
        .mean()
        .sort_values(ascending=False)
        .head(TOP_SPECIES)
        .index.tolist()
    )
    mat = (
        d[d["species"].isin(top_species)]
        .pivot_table(
            index="scope_group",
            columns="species",
            values="relative_concentration",
            aggfunc="sum",
            fill_value=0.0,
        )
        .sort_index()
    )
    if mat.empty:
        write_empty_figure(out_path, f"No {scope} species rows for heatmap")
        return

    fig, ax = plt.subplots(figsize=(10.8, max(4.2, 0.55 * len(mat.index))))
    im = ax.imshow(mat.to_numpy(dtype=float), aspect="auto", cmap="YlGnBu")
    ax.set_yticks(np.arange(len(mat.index)))
    ax.set_yticklabels(mat.index.tolist(), fontsize=9.6)
    ax.set_xticks(np.arange(len(mat.columns)))
    ax.set_xticklabels([to_species_label(v) for v in mat.columns.tolist()], rotation=25, ha="right", fontsize=10.8)
    ax.set_xlabel("Species")
    ax.set_ylabel(f"{scope.capitalize()} Group (param_set | channel)")
    ax.set_title(f"{scope.capitalize()} Relative Target-Species Concentration Heatmap")
    cbar = fig.colorbar(im, ax=ax, pad=0.01)
    cbar.set_label("Relative concentration")
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)


def plot_scope_species_mix(group_long: pd.DataFrame, scope: str, out_path: Path) -> None:
    d = scope_rows(group_long, scope)
    if d.empty:
        write_empty_figure(out_path, f"No {scope} species concentration mix")
        return

    top_species = (
        d.groupby("species", dropna=False)["relative_concentration"]
        .mean()
        .sort_values(ascending=False)
        .head(TOP_SPECIES)
        .index.tolist()
    )
    p = (
        d[d["species"].isin(top_species)]
        .pivot_table(
            index="scope_group",
            columns="species",
            values="relative_concentration",
            aggfunc="sum",
            fill_value=0.0,
        )
        .sort_index()
    )
    if p.empty:
        write_empty_figure(out_path, f"No {scope} species selected for mix")
        return

    p = p.div(p.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    fig, ax = plt.subplots(figsize=(11.0, 5.4))
    bottoms = np.zeros(len(p.index))
    palette = get_palette(len(p.columns))
    x = np.arange(len(p.index))
    for color, species in zip(palette, p.columns):
        vals = p[species].to_numpy(dtype=float)
        ax.bar(x, vals, bottom=bottoms, color=color, alpha=0.92, label=to_species_label(species))
        bottoms += vals
    ax.set_xticks(x)
    ax.set_xticklabels(p.index.tolist(), rotation=28, ha="right", fontsize=9.8)
    ax.set_xlabel(f"{scope.capitalize()} Group (param_set | channel)")
    ax.set_ylabel("Normalized relative concentration")
    ax.set_title(f"{scope.capitalize()} Group-Level Species Concentration Mix")
    ax.legend(loc="upper right", ncol=2, fontsize=10.2, title_fontsize=10.2)
    style_axes(ax, grid_axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)


def plot_scope_species_rank(summary: pd.DataFrame, scope: str, out_path: Path) -> None:
    if summary.empty:
        write_empty_figure(out_path, f"No {scope} species summary data")
        return
    d = summary[summary["dataset"].astype(str).str.lower() == scope.lower()].copy()
    if d.empty:
        write_empty_figure(out_path, f"No {scope} rows in species summary")
        return

    d["mean_relative_concentration"] = pd.to_numeric(d["mean_relative_concentration"], errors="coerce").fillna(0.0)
    d["std_relative_concentration"] = pd.to_numeric(d["std_relative_concentration"], errors="coerce").fillna(0.0)
    d = d.nlargest(TOP_SPECIES, "mean_relative_concentration").sort_values("mean_relative_concentration", ascending=True)
    if d.empty:
        write_empty_figure(out_path, f"No ranked species for {scope}")
        return

    fig, ax = plt.subplots(figsize=(10.2, 5.4))
    y = np.arange(len(d))
    ax.barh(
        y,
        d["mean_relative_concentration"].to_numpy(dtype=float),
        xerr=d["std_relative_concentration"].to_numpy(dtype=float),
        color="#2a9d8f",
        alpha=0.92,
        edgecolor="#2f2f2f",
        linewidth=0.35,
    )
    ax.set_yticks(y)
    ax.set_yticklabels([to_species_label(v) for v in d["species"].astype(str).tolist()], fontsize=10.8)
    ax.set_xlabel("Mean relative concentration across groups")
    ax.set_ylabel("Species")
    ax.set_title(f"{scope.capitalize()} Top Species with Between-Group Variability")
    style_axes(ax, grid_axis="x")
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)


def plot_scope_param_heatmap(group_long: pd.DataFrame, scope: str, out_path: Path) -> None:
    d = scope_rows(group_long, scope)
    if d.empty:
        write_empty_figure(out_path, f"No {scope} parameter-species data")
        return

    top_species = (
        d.groupby("species", dropna=False)["relative_concentration"]
        .mean()
        .sort_values(ascending=False)
        .head(TOP_SPECIES)
        .index.tolist()
    )
    mat = (
        d[d["species"].isin(top_species)]
        .pivot_table(
            index="param_set",
            columns="species",
            values="relative_concentration",
            aggfunc="mean",
            fill_value=0.0,
        )
        .sort_index()
    )
    if mat.empty:
        write_empty_figure(out_path, f"No {scope} parameter-level heatmap rows")
        return

    fig, ax = plt.subplots(figsize=(10.0, max(4.0, 0.6 * len(mat.index))))
    im = ax.imshow(mat.to_numpy(dtype=float), aspect="auto", cmap="YlOrBr")
    ax.set_yticks(np.arange(len(mat.index)))
    ax.set_yticklabels(mat.index.astype(str).tolist(), fontsize=9.8)
    ax.set_xticks(np.arange(len(mat.columns)))
    ax.set_xticklabels([to_species_label(v) for v in mat.columns.astype(str).tolist()], rotation=25, ha="right", fontsize=10.8)
    ax.set_xlabel("Species")
    ax.set_ylabel(f"{scope.capitalize()} Parameter Set")
    ax.set_title(f"{scope.capitalize()} Parameter-Level Species Heatmap")
    cbar = fig.colorbar(im, ax=ax, pad=0.01)
    cbar.set_label("Mean relative concentration")
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)


def plot_group_heatmap(group_long: pd.DataFrame, out_path: Path) -> None:
    if group_long.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No target species concentration data", ha="center", va="center")
        ax.axis("off")
        fig.savefig(out_path, dpi=DPI)
        plt.close(fig)
        return

    top_species = (
        group_long.groupby("species", dropna=False)["relative_concentration"]
        .mean()
        .sort_values(ascending=False)
        .head(TOP_SPECIES)
        .index.tolist()
    )
    d = group_long[group_long["species"].isin(top_species)].copy()
    mat = (
        d.pivot_table(
            index="group_label",
            columns="species",
            values="relative_concentration",
            aggfunc="sum",
            fill_value=0.0,
        )
        .sort_index()
    )

    fig, ax = plt.subplots(figsize=(11.2, max(4.6, 0.35 * len(mat.index))))
    im = ax.imshow(mat.to_numpy(dtype=float), aspect="auto", cmap="YlGnBu")
    ax.set_yticks(np.arange(len(mat.index)))
    ax.set_yticklabels(mat.index.tolist(), fontsize=8)
    ax.set_xticks(np.arange(len(mat.columns)))
    ax.set_xticklabels([to_species_label(v) for v in mat.columns.tolist()], rotation=30, ha="right", fontsize=10.8)
    ax.set_xlabel("Species")
    ax.set_ylabel("Experiment Group")
    ax.set_title("Relative Target-Species Concentration by Group")
    cbar = fig.colorbar(im, ax=ax, pad=0.01)
    cbar.set_label("Relative concentration")
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)


def plot_dataset_stacked(summary: pd.DataFrame, out_path: Path) -> None:
    d = summary[summary["dataset"].isin(["air", "diameter", "combined"])].copy()
    if d.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No dataset concentration summary", ha="center", va="center")
        ax.axis("off")
        fig.savefig(out_path, dpi=DPI)
        plt.close(fig)
        return

    top_species = (
        d.groupby("species", dropna=False)["mean_relative_concentration"]
        .mean()
        .sort_values(ascending=False)
        .head(TOP_SPECIES)
        .index.tolist()
    )
    p = (
        d[d["species"].isin(top_species)]
        .pivot_table(
            index="dataset",
            columns="species",
            values="mean_relative_concentration",
            aggfunc="mean",
            fill_value=0.0,
        )
        .reindex(index=[x for x in ["air", "diameter", "combined"] if x in d["dataset"].unique()])
    )
    if p.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No species selected for stacked view", ha="center", va="center")
        ax.axis("off")
        fig.savefig(out_path, dpi=DPI)
        plt.close(fig)
        return

    p = p.div(p.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    fig, ax = plt.subplots(figsize=(10, 5.4))
    bottoms = np.zeros(len(p.index))
    palette = get_palette(len(p.columns))
    for color, species in zip(palette, p.columns):
        vals = p[species].to_numpy(dtype=float)
        ax.bar(p.index, vals, bottom=bottoms, color=color, alpha=0.92, label=to_species_label(species))
        bottoms += vals
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Normalized relative concentration")
    ax.set_title("Dataset Target-Species Concentration Mix")
    ax.legend(loc="upper right", ncol=2, fontsize=10.2, title_fontsize=10.2)
    style_axes(ax, grid_axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)


def plot_air_vs_diameter_delta(delta_df: pd.DataFrame, out_path: Path) -> None:
    if delta_df.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No air vs diameter species delta data", ha="center", va="center")
        ax.axis("off")
        fig.savefig(out_path, dpi=DPI)
        plt.close(fig)
        return

    d = delta_df.head(TOP_SPECIES).sort_values("air_minus_diameter", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 5.6))
    colors = ["#2a9d8f" if v >= 0 else "#d1495b" for v in d["air_minus_diameter"]]
    species_labels = [to_species_label(v) for v in d["species"].tolist()]
    ax.barh(species_labels, d["air_minus_diameter"], color=colors, alpha=0.92)
    ax.axvline(0, color="#333333", linewidth=1.0)
    ax.set_xlabel("Air - Diameter mean relative concentration")
    ax.set_ylabel("Species")
    ax.set_title("Air vs Diameter Target-Species Concentration Delta")
    ax.tick_params(axis="y", labelsize=10.8)
    style_axes(ax, grid_axis="x")
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)


def plot_group_total_signal(group_long: pd.DataFrame, out_path: Path) -> None:
    if group_long.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No group signal data", ha="center", va="center")
        ax.axis("off")
        fig.savefig(out_path, dpi=DPI)
        plt.close(fig)
        return

    d = (
        group_long[["group_label", "group_total_signal"]]
        .drop_duplicates()
        .sort_values("group_total_signal", ascending=False)
        .head(12)
    )
    fig, ax = plt.subplots(figsize=(11, 5.8))
    palette = get_palette(len(d))
    ax.bar(np.arange(len(d)), d["group_total_signal"].to_numpy(dtype=float), color=palette, alpha=0.9)
    ax.set_xticks(np.arange(len(d)))
    ax.set_xticklabels(d["group_label"], rotation=28, ha="right", fontsize=8)
    ax.set_xlabel("Experiment Group")
    ax.set_ylabel("Total matched signal")
    ax.set_title("Highest Target-Species Signal Groups")
    style_axes(ax, grid_axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)


def scope_csv_slice(df: pd.DataFrame, scope: str, allow_global: bool = False) -> pd.DataFrame:
    if scope == "meta":
        return df.copy()
    if "dataset" in df.columns:
        return df[df["dataset"].astype(str).str.lower() == scope].copy()
    return df.copy() if allow_global else pd.DataFrame(columns=df.columns)


def clear_root_pngs(fig_dir: Path) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    for old in fig_dir.glob("*.png"):
        old.unlink()


def main() -> int:
    apply_publication_style()
    ensure_all_scope_layouts()
    for scope in SCOPES:
        scope_csv_dir(scope).mkdir(parents=True, exist_ok=True)
        scope_fig_dir(scope).mkdir(parents=True, exist_ok=True)

    matches = load_target_matches(TARGET_MATCHES_CSV)
    group_long, group_wide = build_group_concentration_tables(matches)
    summary = summarize_dataset_species(group_long)
    delta = air_vs_diameter_species_delta(summary)
    findings = build_key_group_findings(group_long)

    csv_outputs = {
        "group_species_concentration_long.csv": group_long,
        "group_species_concentration_wide.csv": group_wide,
        "dataset_species_concentration_summary.csv": summary,
        "air_vs_diameter_species_delta.csv": delta,
        "key_species_group_findings.csv": findings,
    }
    written_paths: List[Path] = []
    for scope in SCOPES:
        for name, df in csv_outputs.items():
            allow_global = name == "air_vs_diameter_species_delta.csv"
            part = scope_csv_slice(df, scope, allow_global=allow_global)
            if part.empty:
                continue
            out_path = scope_csv_dir(scope) / name
            part.to_csv(out_path, index=False)
            written_paths.append(out_path)

    meta_fig_dir = scope_fig_dir("meta")
    clear_root_pngs(meta_fig_dir)
    meta_figs = [meta_fig_dir / f"Fig{i}.png" for i in range(1, 5)]
    plot_group_heatmap(group_long, meta_figs[0])
    plot_dataset_stacked(summary, meta_figs[1])
    plot_air_vs_diameter_delta(delta, meta_figs[2])
    plot_group_total_signal(group_long, meta_figs[3])
    written_paths.extend(meta_figs)
    for scope in [s for s in SCOPES if s in {"air", "diameter"}]:
        fig_dir = scope_fig_dir(scope)
        clear_root_pngs(fig_dir)
        scope_figs = [fig_dir / f"Fig{i}.png" for i in range(1, 5)]
        plot_scope_group_heatmap(group_long, scope, scope_figs[0])
        plot_scope_species_mix(group_long, scope, scope_figs[1])
        plot_scope_species_rank(summary, scope, scope_figs[2])
        plot_scope_param_heatmap(group_long, scope, scope_figs[3])
        written_paths.extend(scope_figs)

    print("Wrote chemSpecies outputs:")
    for path in sorted(set(written_paths)):
        print(f"  {path}")
    print(
        f"Done. groups={len(group_wide)} species_rows={len(group_long)} "
        f"dataset_rows={len(summary)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
