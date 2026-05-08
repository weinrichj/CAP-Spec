#!/usr/bin/env python3
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from analysis.output_paths import SCOPES, ensure_all_scope_layouts, metadata_csv_path, metadata_section_dir, pca_dir
from plots.style import apply_publication_style, get_palette, style_axes

IN_CSV = metadata_csv_path("meta", "features", "features.csv")
N_COMPONENTS = 5
SCALE = False
INCLUDE_COLS: Optional[List[str]] = None
EXCLUDE_NUMERIC_COLS = {"trial"}


def select_variable_columns(df: pd.DataFrame) -> List[str]:
    if INCLUDE_COLS:
        return [c for c in INCLUDE_COLS if c in df.columns]
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in numeric if c not in EXCLUDE_NUMERIC_COLS]


def impute_nan_matrix(x: np.ndarray) -> np.ndarray:
    x = np.where(np.isfinite(x), x, np.nan)
    col_means = np.zeros(x.shape[1], dtype=float)
    valid_cols = np.any(np.isfinite(x), axis=0)
    if np.any(valid_cols):
        col_means[valid_cols] = np.nanmean(x[:, valid_cols], axis=0)
    mask = np.isnan(x)
    if mask.any():
        x[mask] = np.take(col_means, np.where(mask)[1])
    return x


def run_pca_block(df: pd.DataFrame, scope: str, color_col: Optional[str] = None) -> int:
    if df.empty:
        print(f"Skipping {scope}: no rows")
        return 0

    var_cols = select_variable_columns(df)
    if not var_cols:
        print(f"Skipping {scope}: no numeric variables")
        return 0

    x = df[var_cols].to_numpy(dtype=float)
    x = impute_nan_matrix(x)

    if SCALE:
        means = np.nanmean(x, axis=0)
        stds = np.nanstd(x, axis=0, ddof=1)
        stds[stds == 0] = 1.0
        x = (x - means) / stds

    n_comp = min(N_COMPONENTS, x.shape[0], x.shape[1])
    if n_comp < 1:
        print(f"Skipping {scope}: insufficient shape for PCA ({x.shape[0]}x{x.shape[1]})")
        return 0

    pca = PCA(n_components=n_comp)
    try:
        scores = pca.fit_transform(x)
    except Exception as e:
        print(f"PCA failed for {scope}: {e}")
        return 1

    png_dir = pca_dir(scope)
    csv_dir = metadata_section_dir(scope, "pca")
    png_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)

    score_cols = [f"PC{i + 1}" for i in range(scores.shape[1])]
    meta_cols = [c for c in df.columns if c not in var_cols]

    scores_df = pd.DataFrame(scores, columns=score_cols)
    for c in meta_cols:
        scores_df[c] = df[c].values
    scores_df.to_csv(csv_dir / "pca_scores.csv", index=False)

    pd.DataFrame(pca.components_.T, index=var_cols, columns=score_cols).to_csv(csv_dir / "pca_loadings.csv")
    pd.DataFrame(
        {
            "explained_variance": pca.explained_variance_,
            "explained_variance_ratio": pca.explained_variance_ratio_,
        },
        index=score_cols,
    ).to_csv(csv_dir / "pca_explained_variance.csv")

    if scores.shape[1] >= 2:
        try:
            import matplotlib.pyplot as plt

            apply_publication_style()
            fig, ax = plt.subplots(figsize=(6.5, 6))
            if color_col and color_col in scores_df.columns:
                groups = list(scores_df.groupby(color_col, dropna=False))
                palette = get_palette(len(groups))
                for color, (label, grp) in zip(palette, groups):
                    ax.scatter(
                        grp["PC1"],
                        grp["PC2"],
                        label=str(label),
                        s=34,
                        color=color,
                        alpha=0.9,
                        edgecolors="#222222",
                        linewidths=0.35,
                    )
                ax.legend(loc="best", fontsize=8)
            else:
                ax.scatter(
                    scores_df["PC1"],
                    scores_df["PC2"],
                    s=34,
                    color=get_palette(1)[0],
                    alpha=0.9,
                    edgecolors="#222222",
                    linewidths=0.35,
                )

            evr = pca.explained_variance_ratio_
            pc1_var = 100.0 * float(evr[0]) if evr.size >= 1 else 0.0
            pc2_var = 100.0 * float(evr[1]) if evr.size >= 2 else 0.0
            ax.axhline(0, color="#888888", linewidth=0.8, alpha=0.8, zorder=0)
            ax.axvline(0, color="#888888", linewidth=0.8, alpha=0.8, zorder=0)
            ax.set_xlabel(f"PC1 ({pc1_var:.1f}%)")
            ax.set_ylabel(f"PC2 ({pc2_var:.1f}%)")
            ax.set_title(f"PCA scores ({scope})")
            style_axes(ax, grid_axis="both")
            fig.tight_layout()
            fig.savefig(png_dir / "Fig1.png", dpi=170)
            plt.close(fig)
        except ImportError:
            print("matplotlib not available, skipping score scatter plot")

    print(f"Wrote PCA outputs for scope={scope}")
    return 0


def main() -> int:
    ensure_all_scope_layouts()

    if not IN_CSV.exists():
        print(f"Input file not found: {IN_CSV}. Run features.py first.")
        return 1

    df = pd.read_csv(IN_CSV)
    if df.empty:
        print(f"Input CSV is empty: {IN_CSV}")
        return 2

    status = 0
    status = max(status, run_pca_block(df, "meta", color_col="dataset"))

    if "dataset" in df.columns:
        for scope in [s for s in SCOPES if s in {"air", "diameter"}]:
            part = df[df["dataset"].astype(str).str.lower() == scope].reset_index(drop=True)
            status = max(status, run_pca_block(part, scope, color_col="param_set"))

    return status


if __name__ == "__main__":
    raise SystemExit(main())
