#!/usr/bin/env python3
from __future__ import annotations

import re
import textwrap
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple
from xml.sax.saxutils import escape

import pandas as pd

from analysis.output_paths import active_scopes

OUTPUT_ROOT = Path("output")
NARRATIVE_MD_PATH = Path("reaction wavelength narrative.md")
NARRATIVE_PDF_PATH = Path("reaction wavelength narrative.pdf")
NARRATIVE_MD_FALLBACK = Path("reaction_wavelength_narrative.generated.md")
NARRATIVE_PDF_FALLBACK = Path("reaction_wavelength_narrative.generated.pdf")


def _safe_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _safe_int(value: object, default: int = -1) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


def _load_csv(path: Path, required_cols: Sequence[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    df = pd.read_csv(path)
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {', '.join(missing)}")
    return df


def _resolve_narrative_source(scopes: Sequence[str]) -> Tuple[str, List[str]]:
    selected = [str(s).strip().lower() for s in scopes]
    dataset_scopes = [s for s in selected if s in {"air", "diameter"}]
    if len(dataset_scopes) == 1:
        return dataset_scopes[0], dataset_scopes
    return "meta", dataset_scopes


def _filter_by_datasets(df: pd.DataFrame, datasets: Sequence[str]) -> pd.DataFrame:
    if not datasets or "dataset" not in df.columns:
        return df
    allowed = {str(s).strip().lower() for s in datasets}
    out = df.copy()
    out["dataset"] = out["dataset"].astype(str)
    out = out[out["dataset"].str.lower().isin(allowed)].copy()
    return out.reset_index(drop=True)


def _equation_catalog() -> List[Tuple[str, str, str]]:
    return [
        (
            "R1",
            r"e^- + N_2(X\,^1\Sigma_g^+) \rightarrow e^- + N_2(C\,^3\Pi_u)",
            "Electron-impact excitation of molecular nitrogen into the N2(C) radiative manifold.",
        ),
        (
            "R2",
            r"N_2(C\,^3\Pi_u, v') \rightarrow N_2(B\,^3\Pi_g, v'') + h\nu",
            "Second Positive System radiative decay pathway used to interpret N2-band emission.",
        ),
        ("R3", r"e^- + N_2 \rightarrow 2e^- + N_2^+", "Electron-impact ionization of molecular nitrogen."),
        (
            "R4",
            r"e^- + N_2^+(X\,^2\Sigma_g^+) \rightarrow e^- + N_2^+(B\,^2\Sigma_u^+)",
            "Stepwise excitation of N2+ into the B-state prior to first-negative emission.",
        ),
        (
            "R5",
            r"N_2^+(B\,^2\Sigma_u^+) \rightarrow N_2^+(X\,^2\Sigma_g^+) + h\nu",
            "First Negative System radiative decay for ionic molecular nitrogen.",
        ),
        ("R6", r"e^- + Ar \rightarrow e^- + Ar^*", "Electron-impact excitation of argon metastable/upper states."),
        ("R7", r"Ar^* + N_2 \rightarrow Ar + N_2(C)", "Argon-assisted transfer pathway feeding N2(C) emission."),
        ("R8", r"e^- + O_2 \rightarrow e^- + O(^3P) + O(^1D)", "Electron-impact dissociation channel for oxygen."),
        ("R9", r"O(^1D) + H_2O_{trace} \rightarrow 2OH(X\,^2\Pi)", "Trace-water assisted OH production route."),
        ("R10", r"H(n \ge 4) \rightarrow H(n=2) + h\nu_{H\beta}", "Balmer-beta emission channel at 486.1 nm."),
        ("R11", r"e^- + O_2 + M \rightarrow O_2^- + M", "Three-body dissociative electron attachment pathway."),
        ("R12", r"N + O_2 \rightarrow NO + O", "Active-nitrogen NO formation channel."),
        ("R13", r"e^- + He \rightarrow e^- + He^*", "Electron-impact helium excitation."),
        ("R14", r"He^* + N_2 \rightarrow He + N_2(C)", "Helium-assisted transfer into N2(C)."),
        ("R15", r"e^- + N_2 \rightarrow e^- + N + N", "Electron-impact dissociation of molecular nitrogen."),
        ("R16", r"e^- + C \rightarrow e^- + C^*;\; C^* \rightarrow C + h\nu", "Atomic carbon excitation-emission surrogate pathway."),
    ]


def _line_to_equations(line_names: Iterable[str]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {
        "N2_337": ["R1", "R2"],
        "N2_379": ["R1", "R2"],
        "N2plus_391": ["R3", "R4", "R5"],
        "N2plus_428": ["R3", "R4", "R5"],
        "OH_308": ["R8", "R9"],
        "Ar_750": ["R6", "R7"],
        "O_777": ["R8", "R12"],
        "Hbeta_486": ["R10"],
        "N_672": ["R15", "R12"],
    }
    for line_name in line_names:
        if str(line_name).startswith("N2CB_"):
            out[str(line_name)] = ["R1", "R2"]
    return out


def _reaction_to_equations() -> Dict[str, List[str]]:
    return {
        "e + N2 -> e + N2(C)": ["R1", "R2"],
        "e + N2 -> 2e + N2+": ["R3"],
        "e + N2+ -> e + N2+(B)": ["R4", "R5"],
        "Ar* + N2 -> Ar + N2(C)": ["R6", "R7", "R2"],
        "e + O2 -> e + O + O": ["R8"],
        "O + H2O_trace -> 2OH": ["R9"],
        "e + O2 -> O2-": ["R11"],
        "N + O2 -> NO + O": ["R12"],
        "e + N2 -> e + N + N": ["R15"],
    }


def _species_to_equations(species: str) -> List[str]:
    s = str(species).strip().lower()
    if not s:
        return []
    if "n2+" in s:
        return ["R3", "R4", "R5"]
    if s == "n2" or "n2 " in s or " n2" in s:
        return ["R1", "R2"]
    if "oh" in s:
        return ["R8", "R9"]
    if s.startswith("o ii") or s.startswith("o i") or s == "o" or s.startswith("o "):
        return ["R8", "R12"]
    if "o2-" in s:
        return ["R11"]
    if s.startswith("o2"):
        return ["R8", "R11"]
    if s.startswith("ar"):
        return ["R6", "R7"]
    if s.startswith("h i") or s.startswith("h ii") or s == "h":
        return ["R10"]
    if s.startswith("he"):
        return ["R13", "R14"]
    if s.startswith("n i") or s.startswith("n ii") or s == "n":
        return ["R15", "R12"]
    if s.startswith("no"):
        return ["R12"]
    if s.startswith("c i") or s.startswith("c ii") or s == "c":
        return ["R16"]
    return []


def _sort_equation_ids(eq_ids: Iterable[str]) -> List[str]:
    unique = {str(x).strip() for x in eq_ids if str(x).strip()}

    def _key(eq_id: str) -> Tuple[int, str]:
        if eq_id.startswith("R") and eq_id[1:].isdigit():
            return (int(eq_id[1:]), eq_id)
        return (9999, eq_id)

    return sorted(unique, key=_key)


def _build_peak_assignment_table(
    peaks: pd.DataFrame,
    nist_matches: pd.DataFrame,
    target_matches: pd.DataFrame,
    key_lines: pd.DataFrame,
) -> pd.DataFrame:
    for col in ("dataset", "param_set", "channel"):
        peaks[col] = peaks[col].astype(str)
        nist_matches[col] = nist_matches[col].astype(str)
        target_matches[col] = target_matches[col].astype(str)
        key_lines[col] = key_lines[col].astype(str)

    peaks["peak_rank"] = pd.to_numeric(peaks["peak_rank"], errors="coerce")
    nist_matches["peak_rank"] = pd.to_numeric(nist_matches["peak_rank"], errors="coerce")
    target_matches["matched_peak_rank"] = pd.to_numeric(target_matches["matched_peak_rank"], errors="coerce")
    nist_matches["candidate_rank"] = pd.to_numeric(nist_matches["candidate_rank"], errors="coerce")

    nist_top = nist_matches[nist_matches["candidate_rank"] == 1].copy()
    nist_top = nist_top.sort_values(["dataset", "param_set", "channel", "peak_rank", "delta_nm"]).drop_duplicates(
        subset=["dataset", "param_set", "channel", "peak_rank"],
        keep="first",
    )

    matched_flag = target_matches["matched"].astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})
    targets = target_matches[matched_flag].copy()
    targets = targets.rename(columns={"matched_peak_rank": "peak_rank", "species": "target_species"})

    key_cols = ["dataset", "param_set", "channel", "line_name", "wavelength_nm"]
    key_lookup: Dict[Tuple[str, str, str], List[Tuple[str, float]]] = defaultdict(list)
    for _, row in key_lines[key_cols].dropna().iterrows():
        key = (str(row["dataset"]), str(row["param_set"]), str(row["channel"]))
        key_lookup[key].append((str(row["line_name"]), _safe_float(row["wavelength_nm"])))

    line_map = _line_to_equations(key_lines["line_name"].dropna().astype(str).unique().tolist())

    rows: List[Dict[str, object]] = []
    peaks = peaks.sort_values(["dataset", "param_set", "channel", "peak_rank"], ignore_index=True)
    for _, peak in peaks.iterrows():
        key = (str(peak["dataset"]), str(peak["param_set"]), str(peak["channel"]))
        peak_rank = _safe_int(peak["peak_rank"], default=-1)
        peak_wl = _safe_float(peak.get("peak_wavelength_nm_0p1"))

        nearest_line = ""
        nearest_delta = float("nan")
        if key in key_lookup:
            best_name = ""
            best_delta = float("inf")
            for line_name, line_wl in key_lookup[key]:
                d = abs(peak_wl - line_wl)
                if d < best_delta:
                    best_delta = d
                    best_name = line_name
            if best_name and best_delta <= 1.2:
                nearest_line = best_name
                nearest_delta = best_delta

        nmask = (
            (nist_top["dataset"] == key[0])
            & (nist_top["param_set"] == key[1])
            & (nist_top["channel"] == key[2])
            & (nist_top["peak_rank"] == peak_rank)
        )
        nist_species = ""
        if int(nmask.sum()) > 0:
            nist_species = str(nist_top.loc[nmask, "nist_species"].iloc[0]).strip()

        tmask = (
            (targets["dataset"] == key[0])
            & (targets["param_set"] == key[1])
            & (targets["channel"] == key[2])
            & (targets["peak_rank"] == peak_rank)
        )
        target_species = sorted(
            {
                str(x).strip()
                for x in targets.loc[tmask, "target_species"].dropna().astype(str).tolist()
                if str(x).strip()
            }
        )

        eq_ids: List[str] = []
        if nearest_line:
            eq_ids.extend(line_map.get(nearest_line, []))
        for sp in target_species:
            eq_ids.extend(_species_to_equations(sp))
        eq_ids.extend(_species_to_equations(nist_species))
        eq_ids = _sort_equation_ids(eq_ids)

        evidence: List[str] = []
        if nearest_line:
            evidence.append(f"line={nearest_line} (abs_d={nearest_delta:.2f} nm)")
        if target_species:
            evidence.append(f"target={'/'.join(target_species)}")
        if nist_species:
            evidence.append(f"nist1={nist_species}")

        if nearest_line or target_species:
            confidence = "high"
        elif nist_species:
            confidence = "medium"
        else:
            confidence = "low"

        rows.append(
            {
                "dataset": key[0],
                "param_set": key[1],
                "channel": key[2],
                "peak_rank": peak_rank,
                "wavelength_nm": peak_wl,
                "peak_intensity": _safe_float(peak.get("peak_intensity_refined")),
                "equation_ids": ", ".join(eq_ids) if eq_ids else "Unassigned",
                "confidence": confidence,
                "evidence": "; ".join(evidence) if evidence else "none",
            }
        )
    return pd.DataFrame(rows)


def _build_wavelength_summary(peak_map: pd.DataFrame) -> pd.DataFrame:
    def _collect_ids(series: pd.Series) -> str:
        ids: List[str] = []
        for value in series.astype(str).tolist():
            for token in value.split(","):
                token = token.strip()
                if token and token != "Unassigned":
                    ids.append(token)
        out = _sort_equation_ids(ids)
        return ", ".join(out) if out else "Unassigned"

    def _collect_evidence(series: pd.Series) -> str:
        vals = sorted({str(v) for v in series.astype(str).tolist() if str(v) and str(v) != "none"})
        return "; ".join(vals)

    return (
        peak_map.groupby("wavelength_nm", as_index=False)
        .agg(
            occurrences=("wavelength_nm", "size"),
            max_intensity=("peak_intensity", "max"),
            equation_ids=("equation_ids", _collect_ids),
            evidence=("evidence", _collect_evidence),
        )
        .sort_values("wavelength_nm", ignore_index=True)
    )


def _build_story_summary(story: pd.DataFrame) -> pd.DataFrame:
    mapping = _reaction_to_equations()
    out = story.copy()
    out["reaction"] = out["reaction"].astype(str)
    out["eq_ids"] = out["reaction"].map(lambda r: ", ".join(mapping.get(r, [])) if mapping.get(r) else "N/A")
    return out.sort_values(["reaction", "sum_link_weight"], ascending=[True, False], ignore_index=True)


def _build_pathway_ranking(story_summary: pd.DataFrame) -> pd.DataFrame:
    if story_summary.empty:
        return pd.DataFrame(columns=["reaction", "eq_ids", "total_link_weight", "mean_link_weight", "n_lines"])
    ranked = (
        story_summary.groupby(["reaction", "eq_ids"], as_index=False)
        .agg(
            total_link_weight=("sum_link_weight", "sum"),
            mean_link_weight=("mean_link_weight", "mean"),
            n_lines=("wavelength_nm", "count"),
        )
        .sort_values(["total_link_weight", "mean_link_weight"], ascending=[False, False], ignore_index=True)
    )
    return ranked


def _build_top_peaks_table(peak_map: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    if peak_map.empty:
        return pd.DataFrame(columns=["wavelength_nm", "occurrences", "mean_intensity", "max_intensity", "equation_ids"])

    def _collect_ids(series: pd.Series) -> str:
        ids: List[str] = []
        for value in series.astype(str).tolist():
            for token in value.split(","):
                token = token.strip()
                if token and token != "Unassigned":
                    ids.append(token)
        out = _sort_equation_ids(ids)
        return ", ".join(out) if out else "Unassigned"

    top = (
        peak_map.groupby("wavelength_nm", as_index=False)
        .agg(
            occurrences=("wavelength_nm", "size"),
            mean_intensity=("peak_intensity", "mean"),
            max_intensity=("peak_intensity", "max"),
            equation_ids=("equation_ids", _collect_ids),
        )
        .sort_values(["mean_intensity", "occurrences"], ascending=[False, False], ignore_index=True)
        .head(top_n)
    )
    return top


def _build_confidence_summary(peak_map: pd.DataFrame) -> pd.DataFrame:
    if peak_map.empty:
        return pd.DataFrame(columns=["confidence", "count", "fraction"])
    counts = peak_map["confidence"].astype(str).str.lower().value_counts(dropna=False).rename_axis("confidence").reset_index(name="count")
    total = int(counts["count"].sum())
    counts["fraction"] = counts["count"] / max(total, 1)
    order = {"high": 0, "medium": 1, "low": 2}
    counts["sort_key"] = counts["confidence"].map(order).fillna(99)
    counts = counts.sort_values("sort_key", ignore_index=True).drop(columns=["sort_key"])
    return counts


def _reaction_to_markdown_notation(reaction: str) -> str:
    text = str(reaction)
    replacements = {
        "H2O_trace": "H<sub>2</sub>O<sub>trace</sub>",
        "N2+(B)": "N<sub>2</sub><sup>+</sup>(B)",
        "N2+(X)": "N<sub>2</sub><sup>+</sup>(X)",
        "N2(C)": "N<sub>2</sub>(C)",
        "N2+": "N<sub>2</sub><sup>+</sup>",
        "N2": "N<sub>2</sub>",
        "O2-": "O<sub>2</sub><sup>-</sup>",
        "O2": "O<sub>2</sub>",
        "Ar*": "Ar<sup>*</sup>",
    }
    for raw in sorted(replacements.keys(), key=len, reverse=True):
        pattern = rf"(?<![A-Za-z0-9_]){re.escape(raw)}(?![A-Za-z0-9_])"
        text = re.sub(pattern, replacements[raw], text)
    text = re.sub(r"(?<![A-Za-z0-9_])([0-9]+)e(?![A-Za-z0-9_])", r"\1e<sup>-</sup>", text)
    text = re.sub(r"(?<![A-Za-z0-9_])e(?![A-Za-z0-9_])", "e<sup>-</sup>", text)
    text = text.replace("->", "&rarr;")
    return text


def _render_markdown(
    peak_map: pd.DataFrame,
    wavelength_summary: pd.DataFrame,
    story_summary: pd.DataFrame,
    source_scope: str,
    dataset_scopes: Sequence[str],
    active_scope_values: Sequence[str],
) -> str:
    pathway_rank = _build_pathway_ranking(story_summary)
    top_peaks = _build_top_peaks_table(peak_map, top_n=15)
    confidence_summary = _build_confidence_summary(peak_map)

    mode_label = ", ".join(active_scope_values) if active_scope_values else "all"
    dataset_label = ", ".join(dataset_scopes) if dataset_scopes else "all available datasets"
    source_prefix = f"output/{source_scope}/metadata"
    dominant_pathway = "n/a"
    if not pathway_rank.empty:
        dominant_pathway = _reaction_to_markdown_notation(str(pathway_rank.iloc[0]["reaction"]))
    top_peak_label = "n/a"
    if not top_peaks.empty:
        top_peak_label = f"{_safe_float(top_peaks.iloc[0]['wavelength_nm']):.1f} nm"
    conf_text = "n/a"
    if not confidence_summary.empty:
        conf_text = ", ".join(
            f"{str(r['confidence']).upper()}: {int(r['count'])} ({float(r['fraction']):.0%})"
            for _, r in confidence_summary.iterrows()
        )

    lines: List[str] = []
    lines.append("# Reaction-Chemical Pathway Narrative")
    lines.append("")
    lines.append("_Mode-scoped spectral interpretation with equation-traceable chemical pathway assignments._")
    lines.append("")
    lines.append("## Executive Summary")
    lines.append("")
    lines.append("This report links observed spectral peaks to reaction pathways using key-line proximity, target-species support, and rank-1 NIST assignments.")
    lines.append("Pathways are ranked by aggregate link weight, with complete peak-level evidence preserved for reproducibility.")
    lines.append("")
    lines.append(f"- Group-resolved peaks analyzed: **{len(peak_map)}**")
    lines.append(f"- Distinct rounded wavelengths: **{peak_map['wavelength_nm'].nunique()}**")
    lines.append(f"- Dominant pathway by total link weight: **{dominant_pathway}**")
    lines.append(f"- Highest mean-intensity wavelength: **{top_peak_label}**")
    lines.append(f"- Assignment confidence distribution: **{conf_text}**")
    lines.append("")
    lines.append("## Scope and Source Data")
    lines.append("")
    lines.append(f"- Run mode scopes: **{mode_label}**")
    lines.append(f"- Dataset coverage in this narrative: **{dataset_label}**")
    lines.append("")
    lines.append("Primary compiled inputs:")
    lines.append("")
    lines.append(f"- `{source_prefix}/spectral/averaged_peaks_top10.csv`")
    lines.append(f"- `{source_prefix}/spectral/nist_matches_top3.csv` (`candidate_rank = 1`)")
    lines.append(f"- `{source_prefix}/spectral/target_species_peak_matches.csv`")
    lines.append(f"- `{source_prefix}/chemical_modeling/key_line_intensity_table.csv`")
    lines.append(f"- `{source_prefix}/chemical_modeling/pathway_peak_story_summary.csv`")
    lines.append("")
    lines.append("## 1. Governing Reaction Equation Set")
    lines.append("")
    lines.append("Model equations used for assignment and pathway interpretation:")
    lines.append("")
    for eq_id, equation, description in _equation_catalog():
        lines.append(f"- **{eq_id}**: $${equation}$$")
        lines.append(f"  {description}")
    lines.append("")
    lines.append("## 2. Derived Analytical Relations")
    lines.append("")
    lines.append("- Emission energy by wavelength: $$E = \\frac{hc}{\\lambda}$$")
    lines.append("- Pathway total link weight: $$W_r = \\sum_i w_{r,i}$$")
    lines.append("- Pathway mean link weight: $$\\bar{W}_r = \\frac{1}{n_r}\\sum_i w_{r,i}$$")
    lines.append("- Ranking criterion: descending $W_r$, then descending $\\bar{W}_r$.")
    lines.append("")
    lines.append("## 3. Ranked Pathway Findings")
    lines.append("")
    lines.append("| Rank | Reaction | Eq IDs | Total Link Weight | Mean Link Weight | Evidence Lines |")
    lines.append("|---:|---|---|---:|---:|---:|")
    if pathway_rank.empty:
        lines.append("| 1 | n/a | n/a | 0.000000 | 0.000000 | 0 |")
    else:
        for idx, row in pathway_rank.head(10).iterrows():
            lines.append(
                "| "
                + f"{idx + 1} | {_reaction_to_markdown_notation(row['reaction'])} | {row['eq_ids']} | "
                + f"{_safe_float(row['total_link_weight']):.6f} | {_safe_float(row['mean_link_weight']):.6f} | "
                + f"{_safe_int(row['n_lines'], default=0)} |"
            )
    lines.append("")
    lines.append("## 4. Dominant Peak Features (Top 15 by Mean Intensity)")
    lines.append("")
    lines.append("| Rank | Wavelength (nm) | Occurrences | Mean Intensity | Max Intensity | Equation IDs |")
    lines.append("|---:|---:|---:|---:|---:|---|")
    if top_peaks.empty:
        lines.append("| 1 | n/a | 0 | 0.000000e+00 | 0.000000e+00 | n/a |")
    else:
        for idx, row in top_peaks.iterrows():
            lines.append(
                "| "
                + f"{idx + 1} | {_safe_float(row['wavelength_nm']):.1f} | {_safe_int(row['occurrences'], default=0)} | "
                + f"{_safe_float(row['mean_intensity']):.6e} | {_safe_float(row['max_intensity']):.6e} | {row['equation_ids']} |"
            )
    lines.append("")
    lines.append("## 5. Confidence Summary")
    lines.append("")
    lines.append("| Confidence | Count | Fraction |")
    lines.append("|---|---:|---:|")
    if confidence_summary.empty:
        lines.append("| N/A | 0 | 0.0% |")
    else:
        for _, row in confidence_summary.iterrows():
            lines.append(
                "| "
                + f"{str(row['confidence']).upper()} | {_safe_int(row['count'], default=0)} | {float(row['fraction']):.1%} |"
            )
    lines.append("")
    lines.append("Interpretation of confidence tiers:")
    lines.append("- `HIGH`: nearest key-line and/or target-species support present")
    lines.append("- `MEDIUM`: NIST-only support")
    lines.append("- `LOW`: no supporting assignment evidence")
    lines.append("")
    lines.append("## 6. Pathway-to-Wavelength Evidence (Full)")
    lines.append("")
    lines.append("| Reaction | Eq IDs | Line (nm) | Sum Link Weight | Mean Link Weight | n Groups |")
    lines.append("|---|---|---:|---:|---:|---:|")
    if story_summary.empty:
        lines.append("| n/a | n/a | 0.00 | 0.000000 | 0.000000 | 0 |")
    else:
        for _, row in story_summary.iterrows():
            lines.append(
                "| "
                + f"{_reaction_to_markdown_notation(row['reaction'])} | {row['eq_ids']} | {float(row['wavelength_nm']):.2f} | "
                + f"{_safe_float(row['sum_link_weight']):.6f} | {_safe_float(row['mean_link_weight']):.6f} | "
                + f"{_safe_int(row['n_groups'], default=0)} |"
            )
    lines.append("")
    lines.append("## Appendix A. Unique Wavelength-to-Equation Mapping (All Peaks)")
    lines.append("")
    lines.append("| Wavelength (nm) | Occurrences | Max Intensity | Equation IDs | Evidence Snapshot |")
    lines.append("|---:|---:|---:|---|---|")
    if wavelength_summary.empty:
        lines.append("| n/a | 0 | 0.000000e+00 | n/a | n/a |")
    else:
        for _, row in wavelength_summary.iterrows():
            evidence = str(row["evidence"]).replace("|", "/").replace("\n", " ")
            if len(evidence) > 180:
                evidence = evidence[:177] + "..."
            lines.append(
                "| "
                + f"{_safe_float(row['wavelength_nm']):.1f} | {_safe_int(row['occurrences'], default=0)} | "
                + f"{_safe_float(row['max_intensity']):.6e} | {row['equation_ids']} | {evidence or 'n/a'} |"
            )
    lines.append("")
    lines.append("## Appendix B. Group-Resolved Peak Assignment Table (Complete)")
    lines.append("")
    lines.append("| Dataset | Param Set | Channel | Peak Rank | Wavelength (nm) | Peak Intensity | Equation IDs | Confidence | Evidence |")
    lines.append("|---|---|---|---:|---:|---:|---|---|---|")
    if peak_map.empty:
        lines.append("| n/a | n/a | n/a | -1 | 0.0 | 0.000000e+00 | n/a | n/a | n/a |")
    else:
        for _, row in peak_map.iterrows():
            evidence = str(row["evidence"]).replace("|", "/").replace("\n", " ")
            lines.append(
                "| "
                + f"{row['dataset']} | {row['param_set']} | {row['channel']} | {_safe_int(row['peak_rank'], default=-1)} | "
                + f"{_safe_float(row['wavelength_nm']):.1f} | {_safe_float(row['peak_intensity']):.6e} | "
                + f"{row['equation_ids']} | {row['confidence']} | {evidence} |"
            )
    lines.append("")
    lines.append("## Appendix C. Assignment Method")
    lines.append("")
    lines.append("Equation assignment sequence:")
    lines.append("")
    lines.append("1. Nearest chemical-model key line (threshold: 1.2 nm).")
    lines.append("2. Target species match from `target_species_peak_matches.csv`.")
    lines.append("3. Rank-1 NIST species candidate from `nist_matches_top3.csv`.")
    lines.append("")
    lines.append("Confidence labeling:")
    lines.append("")
    lines.append("- `high`: key-line and/or target-species support present")
    lines.append("- `medium`: NIST-only support")
    lines.append("- `low`: no supporting assignment evidence")
    lines.append("")
    lines.append("Quality note: complete peak-level evidence and equation IDs are preserved in appendices for reproducibility.")
    lines.append("")
    return "\n".join(lines)


def _truncate_text(value: object, max_len: int = 140) -> str:
    text = str(value or "").strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 3].rstrip() + "..."


def _latex_to_pdf_markup(expr: str) -> str:
    text = escape(str(expr))
    text = text.replace(r"\rightarrow", "&#8594;")
    text = text.replace(r"\leftrightarrow", "&#8646;")
    text = text.replace(r"\ge", "&ge;")
    text = text.replace(r"\le", "&le;")
    text = text.replace(r"\times", "&#215;")
    text = text.replace(r"\cdot", "&#183;")
    text = text.replace(r"\nu", "&#957;")
    text = text.replace(r"\beta", "&#946;")
    text = text.replace(r"\alpha", "&#945;")
    text = text.replace(r"\Delta", "&#916;")
    text = text.replace(r"\Pi", "&#928;")
    text = text.replace(r"\Sigma", "&#931;")
    text = text.replace(r"\;", " ")
    text = text.replace(r"\,", " ")
    text = text.replace("\\", "")
    text = re.sub(r"_\{([^{}]+)\}", lambda m: f"<sub>{m.group(1)}</sub>", text)
    text = re.sub(r"\^\{([^{}]+)\}", lambda m: f"<super>{m.group(1)}</super>", text)
    text = re.sub(r"_([A-Za-z0-9+\-*])", lambda m: f"<sub>{m.group(1)}</sub>", text)
    text = re.sub(r"\^([A-Za-z0-9+\-*])", lambda m: f"<super>{m.group(1)}</super>", text)
    return text


def _reaction_to_pdf_markup(reaction: str) -> str:
    text = escape(str(reaction))
    text = text.replace("->", "&#8594;")
    replacements = {
        "H2O_trace": "H<sub>2</sub>O<sub>trace</sub>",
        "N2+(B)": "N<sub>2</sub><super>+</super>(B)",
        "N2+(X)": "N<sub>2</sub><super>+</super>(X)",
        "N2(C)": "N<sub>2</sub>(C)",
        "N2+": "N<sub>2</sub><super>+</super>",
        "N2": "N<sub>2</sub>",
        "O2-": "O<sub>2</sub><super>-</super>",
        "O2": "O<sub>2</sub>",
        "Ar*": "Ar<super>*</super>",
    }
    for raw in sorted(replacements.keys(), key=len, reverse=True):
        pattern = rf"(?<![A-Za-z0-9_]){re.escape(raw)}(?![A-Za-z0-9_])"
        text = re.sub(pattern, replacements[raw], text)
    text = re.sub(r"(?<![A-Za-z0-9_])([0-9]+)e(?![A-Za-z0-9_])", r"\1e<super>-</super>", text)
    text = re.sub(r"(?<![A-Za-z0-9_])e(?![A-Za-z0-9_])", "e<super>-</super>", text)
    return text


def _register_pdf_fonts() -> Tuple[str, str]:
    try:
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
    except Exception:
        return "Helvetica", "Helvetica-Bold"

    font_candidates = [
        ("NarrativeRegular", Path(r"C:\Windows\Fonts\arial.ttf")),
        ("NarrativeBold", Path(r"C:\Windows\Fonts\arialbd.ttf")),
    ]
    registered: Dict[str, str] = {}
    for font_name, font_path in font_candidates:
        if font_path.exists():
            try:
                pdfmetrics.registerFont(TTFont(font_name, str(font_path)))
                registered[font_name] = font_name
            except Exception:
                continue

    if "NarrativeRegular" in registered and "NarrativeBold" in registered:
        return registered["NarrativeRegular"], registered["NarrativeBold"]
    return "Helvetica", "Helvetica-Bold"


def _build_styled_pdf(
    path: Path,
    source_scope: str,
    active_scope_values: Sequence[str],
    dataset_scopes: Sequence[str],
    peak_map: pd.DataFrame,
    wavelength_summary: pd.DataFrame,
    story_summary: pd.DataFrame,
) -> None:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import landscape, letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import LongTable, PageBreak, Paragraph, SimpleDocTemplate, Spacer, TableStyle

    pathway_rank = _build_pathway_ranking(story_summary)
    top_peaks = _build_top_peaks_table(peak_map, top_n=15)
    confidence_summary = _build_confidence_summary(peak_map)

    body_font, bold_font = _register_pdf_fonts()
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "NarrTitle",
        parent=styles["Title"],
        fontName=bold_font,
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#0f172a"),
        alignment=1,
        spaceAfter=6,
    )
    subtitle_style = ParagraphStyle(
        "NarrSubTitle",
        parent=styles["Normal"],
        fontName=body_font,
        fontSize=9.4,
        leading=12,
        textColor=colors.HexColor("#1f2937"),
        spaceAfter=4,
    )
    heading_style = ParagraphStyle(
        "NarrHeading",
        parent=styles["Heading2"],
        fontName=bold_font,
        fontSize=12.6,
        leading=16,
        textColor=colors.HexColor("#0b3a53"),
        spaceBefore=8,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "NarrBody",
        parent=styles["Normal"],
        fontName=body_font,
        fontSize=9.2,
        leading=12,
        textColor=colors.HexColor("#111827"),
    )
    body_small_style = ParagraphStyle(
        "NarrBodySmall",
        parent=body_style,
        fontSize=8.2,
        leading=10.6,
    )
    body_small_right_style = ParagraphStyle(
        "NarrBodySmallRight",
        parent=body_small_style,
        alignment=2,
    )
    equation_style = ParagraphStyle(
        "NarrEquation",
        parent=body_style,
        fontName=body_font,
        fontSize=9.7,
        leading=12.5,
    )
    table_header_style = ParagraphStyle(
        "NarrTableHeader",
        parent=body_small_style,
        fontName=bold_font,
        textColor=colors.white,
        alignment=1,
    )
    caption_style = ParagraphStyle(
        "NarrCaption",
        parent=body_small_style,
        textColor=colors.HexColor("#475569"),
        spaceAfter=4,
    )

    def _p(text: object, style: ParagraphStyle, markup: bool = False) -> Paragraph:
        if markup:
            return Paragraph(str(text), style)
        return Paragraph(escape(str(text)), style)

    def _styled_long_table(
        headers: Sequence[str],
        rows: Sequence[Sequence[Paragraph]],
        col_widths: Sequence[float],
        header_bg: str = "#0b3a53",
    ) -> LongTable:
        data: List[List[Paragraph]] = [[_p(h, table_header_style, markup=False) for h in headers]]
        data.extend([list(r) for r in rows])
        table = LongTable(data, colWidths=list(col_widths), repeatRows=1, hAlign="LEFT")
        style_cmds: List[Tuple[object, ...]] = [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(header_bg)),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("ALIGN", (0, 0), (-1, 0), "CENTER"),
            ("LINEBELOW", (0, 0), (-1, 0), 0.8, colors.HexColor(header_bg)),
            ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#d1d5db")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]
        for row_idx in range(1, len(data)):
            if row_idx % 2 == 0:
                style_cmds.append(("BACKGROUND", (0, row_idx), (-1, row_idx), colors.HexColor("#f8fafc")))
        table.setStyle(TableStyle(style_cmds))
        return table

    def _placeholder_row(n_cols: int, text: str = "No rows available for this section.") -> List[Paragraph]:
        row = [_p(text, body_small_style)]
        for _ in range(n_cols - 1):
            row.append(_p("", body_small_style))
        return row

    doc = SimpleDocTemplate(
        str(path),
        pagesize=landscape(letter),
        leftMargin=0.45 * inch,
        rightMargin=0.45 * inch,
        topMargin=0.5 * inch,
        bottomMargin=0.42 * inch,
        title="Reaction-Chemical Pathway Narrative",
        author="plasmaData_analysis",
    )

    mode_label = ", ".join(active_scope_values) if active_scope_values else "all"
    dataset_label = ", ".join(dataset_scopes) if dataset_scopes else "all available datasets"
    source_prefix = f"output/{source_scope}/metadata"
    generated_on = datetime.now().strftime("%Y-%m-%d %H:%M")
    dominant_pathway = "n/a"
    if not pathway_rank.empty:
        dominant_pathway = _reaction_to_pdf_markup(str(pathway_rank.iloc[0]["reaction"]))
    top_peak_label = "n/a"
    if not top_peaks.empty:
        top_peak_label = f"{_safe_float(top_peaks.iloc[0]['wavelength_nm']):.1f} nm"
    confidence_label = "n/a"
    if not confidence_summary.empty:
        confidence_label = ", ".join(
            f"{str(r['confidence']).upper()}: {int(r['count'])} ({float(r['fraction']):.0%})"
            for _, r in confidence_summary.iterrows()
        )

    story: List[object] = []
    story.append(_p("Reaction-Chemical Pathway Narrative", title_style))
    story.append(_p("Publication-style mode-scoped spectral pathway report", subtitle_style))
    story.append(_p(f"Generated: {generated_on} local time", subtitle_style))
    story.append(_p(f"Run mode scopes: {mode_label}", subtitle_style))
    story.append(_p(f"Dataset coverage: {dataset_label}", subtitle_style))
    story.append(Spacer(1, 4))

    story.append(_p("Executive Summary", heading_style))
    story.append(
        _p(
            "Observed peaks are mapped to reaction equations using key-line proximity, target-species support, and rank-1 NIST candidates.",
            body_style,
        )
    )
    summary_rows: List[List[Paragraph]] = [
        [_p("Group-resolved peaks analyzed", body_small_style), _p(str(len(peak_map)), body_small_right_style), _p("Total rows in the assignment table.", body_small_style)],
        [_p("Distinct rounded wavelengths", body_small_style), _p(str(peak_map["wavelength_nm"].nunique()), body_small_right_style), _p("Unique 0.1 nm bins represented.", body_small_style)],
        [_p("Dominant pathway", body_small_style), _p(dominant_pathway, body_small_style, markup=True), _p("Highest pathway total link-weight.", body_small_style)],
        [_p("Top mean-intensity wavelength", body_small_style), _p(top_peak_label, body_small_right_style), _p("First row of the top-peak ranking.", body_small_style)],
        [_p("Confidence distribution", body_small_style), _p(confidence_label, body_small_style), _p("HIGH / MEDIUM / LOW assignment support tiers.", body_small_style)],
    ]
    story.append(
        _styled_long_table(
            headers=["Metric", "Value", "Interpretation"],
            rows=summary_rows,
            col_widths=[2.4 * inch, 2.4 * inch, 4.9 * inch],
            header_bg="#1d4ed8",
        )
    )
    story.append(Spacer(1, 6))

    story.append(_p("Scope and Source Data", heading_style))
    story.append(_p("Compiled input artifacts for this run mode:", caption_style))
    source_rows: List[List[Paragraph]] = [
        [_p(f"{source_prefix}/spectral/averaged_peaks_top10.csv", body_small_style), _p("Detected peak list per dataset / channel.", body_small_style)],
        [_p(f"{source_prefix}/spectral/nist_matches_top3.csv", body_small_style), _p("Ranked NIST candidate species (rank-1 consumed).", body_small_style)],
        [_p(f"{source_prefix}/spectral/target_species_peak_matches.csv", body_small_style), _p("Target-species based peak associations.", body_small_style)],
        [_p(f"{source_prefix}/chemical_modeling/key_line_intensity_table.csv", body_small_style), _p("Reference key lines for nearest-line assignment.", body_small_style)],
        [_p(f"{source_prefix}/chemical_modeling/pathway_peak_story_summary.csv", body_small_style), _p("Pathway to wavelength linkage metrics.", body_small_style)],
    ]
    story.append(
        _styled_long_table(
            headers=["Input Artifact", "Role in Narrative"],
            rows=source_rows,
            col_widths=[5.4 * inch, 4.3 * inch],
            header_bg="#0f766e",
        )
    )
    story.append(PageBreak())

    story.append(_p("1. Governing Reaction Equation Set", heading_style))
    story.append(_p("Equations are written in publication notation with explicit states and charges.", caption_style))
    eq_rows: List[List[Paragraph]] = []
    for eq_id, equation, description in _equation_catalog():
        eq_rows.append(
            [
                _p(eq_id, body_style),
                _p(_latex_to_pdf_markup(equation), equation_style, markup=True),
                _p(description, body_style),
            ]
        )
    if not eq_rows:
        eq_rows.append(_placeholder_row(3))
    story.append(
        _styled_long_table(
            headers=["ID", "Equation", "Interpretation"],
            rows=eq_rows,
            col_widths=[0.6 * inch, 3.9 * inch, 5.2 * inch],
            header_bg="#0b3a53",
        )
    )
    story.append(Spacer(1, 8))

    story.append(_p("2. Derived Analytical Relations", heading_style))
    story.append(_p("Core relations used to interpret and rank pathway evidence:", caption_style))
    derived_rows: List[List[Paragraph]] = [
        [_p("Emission energy", body_small_style), _p("E = hc / lambda", equation_style), _p("Converts wavelength-scale observations to photon energy scale.", body_small_style)],
        [_p("Pathway total link weight", body_small_style), _p("W_r = sum_i w_(r,i)", equation_style), _p("Aggregated evidence score for pathway r.", body_small_style)],
        [_p("Pathway mean link weight", body_small_style), _p("Wbar_r = (1/n_r) sum_i w_(r,i)", equation_style), _p("Average per-line support for pathway r.", body_small_style)],
        [_p("Ranking criterion", body_small_style), _p("Sort by W_r then Wbar_r", equation_style), _p("Primary ranking used in the findings table.", body_small_style)],
    ]
    story.append(
        _styled_long_table(
            headers=["Relation", "Expression", "Use in Report"],
            rows=derived_rows,
            col_widths=[2.2 * inch, 2.8 * inch, 4.7 * inch],
            header_bg="#6d28d9",
        )
    )
    story.append(Spacer(1, 8))

    story.append(_p("3. Ranked Pathway Findings (Top 10)", heading_style))
    rp_rows: List[List[Paragraph]] = []
    for idx, row in pathway_rank.head(10).iterrows():
        rp_rows.append(
            [
                _p(str(idx + 1), body_small_right_style),
                _p(_reaction_to_pdf_markup(row["reaction"]), body_small_style, markup=True),
                _p(row["eq_ids"], body_small_style),
                _p(f"{_safe_float(row['total_link_weight']):.6f}", body_small_right_style),
                _p(f"{_safe_float(row['mean_link_weight']):.6f}", body_small_right_style),
                _p(f"{_safe_int(row['n_lines'], default=0)}", body_small_right_style),
            ]
        )
    if not rp_rows:
        rp_rows.append(_placeholder_row(6))
    story.append(
        _styled_long_table(
            headers=["Rank", "Reaction", "Eq IDs", "Total Link", "Mean Link", "n Lines"],
            rows=rp_rows,
            col_widths=[0.55 * inch, 3.35 * inch, 1.0 * inch, 1.15 * inch, 1.15 * inch, 0.9 * inch],
            header_bg="#334155",
        )
    )
    story.append(Spacer(1, 8))

    story.append(_p("4. Dominant Peak Features (Top 15 by Mean Intensity)", heading_style))
    tp_rows: List[List[Paragraph]] = []
    for idx, row in top_peaks.iterrows():
        tp_rows.append(
            [
                _p(str(idx + 1), body_small_right_style),
                _p(f"{_safe_float(row['wavelength_nm']):.1f}", body_small_right_style),
                _p(f"{_safe_int(row['occurrences'], default=0)}", body_small_right_style),
                _p(f"{_safe_float(row['mean_intensity']):.6e}", body_small_right_style),
                _p(f"{_safe_float(row['max_intensity']):.6e}", body_small_right_style),
                _p(row["equation_ids"], body_small_style),
            ]
        )
    if not tp_rows:
        tp_rows.append(_placeholder_row(6))
    story.append(
        _styled_long_table(
            headers=["Rank", "Wavelength (nm)", "Occurrences", "Mean Intensity", "Max Intensity", "Eq IDs"],
            rows=tp_rows,
            col_widths=[0.55 * inch, 1.1 * inch, 0.9 * inch, 1.3 * inch, 1.3 * inch, 3.1 * inch],
            header_bg="#7c2d12",
        )
    )
    story.append(Spacer(1, 8))

    story.append(_p("5. Confidence Summary", heading_style))
    conf_rows: List[List[Paragraph]] = []
    for _, row in confidence_summary.iterrows():
        conf_rows.append(
            [
                _p(str(row["confidence"]).upper(), body_small_style),
                _p(str(_safe_int(row["count"], default=0)), body_small_right_style),
                _p(f"{float(row['fraction']):.1%}", body_small_right_style),
                _p(
                    "Key-line and/or target support"
                    if str(row["confidence"]).lower() == "high"
                    else ("NIST-only support" if str(row["confidence"]).lower() == "medium" else "No supporting species evidence"),
                    body_small_style,
                ),
            ]
        )
    if not conf_rows:
        conf_rows.append(_placeholder_row(4))
    story.append(
        _styled_long_table(
            headers=["Confidence", "Count", "Fraction", "Interpretation"],
            rows=conf_rows,
            col_widths=[1.5 * inch, 0.9 * inch, 1.0 * inch, 6.0 * inch],
            header_bg="#14532d",
        )
    )
    story.append(PageBreak())

    story.append(_p("6. Pathway-to-Wavelength Evidence (Full)", heading_style))
    story.append(_p("Complete pathway table ordered by reaction and wavelength.", caption_style))
    full_rows: List[List[Paragraph]] = []
    for _, row in story_summary.iterrows():
        full_rows.append(
            [
                _p(_reaction_to_pdf_markup(row["reaction"]), body_small_style, markup=True),
                _p(row["eq_ids"], body_small_style),
                _p(f"{_safe_float(row['wavelength_nm']):.2f}", body_small_right_style),
                _p(f"{_safe_float(row['sum_link_weight']):.6f}", body_small_right_style),
                _p(f"{_safe_float(row['mean_link_weight']):.6f}", body_small_right_style),
                _p(f"{_safe_int(row['n_groups'], default=0)}", body_small_right_style),
            ]
        )
    if not full_rows:
        full_rows.append(_placeholder_row(6))
    story.append(
        _styled_long_table(
            headers=["Reaction", "Eq IDs", "Line (nm)", "Sum Link", "Mean Link", "n Groups"],
            rows=full_rows,
            col_widths=[3.3 * inch, 1.0 * inch, 0.9 * inch, 1.1 * inch, 1.1 * inch, 0.9 * inch],
            header_bg="#0b3a53",
        )
    )
    story.append(Spacer(1, 8))

    story.append(_p("Appendix A. Unique Wavelength-to-Equation Mapping", heading_style))
    ws_rows: List[List[Paragraph]] = []
    for _, row in wavelength_summary.iterrows():
        ws_rows.append(
            [
                _p(f"{_safe_float(row['wavelength_nm']):.1f}", body_small_right_style),
                _p(f"{_safe_int(row['occurrences'], default=0)}", body_small_right_style),
                _p(f"{_safe_float(row['max_intensity']):.6e}", body_small_right_style),
                _p(row["equation_ids"], body_small_style),
                _p(_truncate_text(row["evidence"], max_len=170) or "n/a", body_small_style),
            ]
        )
    if not ws_rows:
        ws_rows.append(_placeholder_row(5))
    story.append(
        _styled_long_table(
            headers=["Wavelength (nm)", "Occurrences", "Max Intensity", "Eq IDs", "Evidence Snapshot"],
            rows=ws_rows,
            col_widths=[1.0 * inch, 0.9 * inch, 1.4 * inch, 1.1 * inch, 5.2 * inch],
            header_bg="#1e40af",
        )
    )
    story.append(Spacer(1, 8))

    story.append(_p("Appendix B. Group-Resolved Peak Assignments (Complete)", heading_style))
    gp_rows: List[List[Paragraph]] = []
    for _, row in peak_map.iterrows():
        gp_rows.append(
            [
                _p(row["dataset"], body_small_style),
                _p(row["param_set"], body_small_style),
                _p(row["channel"], body_small_style),
                _p(str(_safe_int(row["peak_rank"], default=-1)), body_small_right_style),
                _p(f"{_safe_float(row['wavelength_nm']):.1f}", body_small_right_style),
                _p(f"{_safe_float(row['peak_intensity']):.6e}", body_small_right_style),
                _p(row["equation_ids"], body_small_style),
                _p(str(row["confidence"]).upper(), body_small_style),
                _p(_truncate_text(row["evidence"], max_len=120), body_small_style),
            ]
        )
    if not gp_rows:
        gp_rows.append(_placeholder_row(9))
    story.append(
        _styled_long_table(
            headers=["Dataset", "Param Set", "Channel", "Rank", "lambda (nm)", "Intensity", "Eq IDs", "Confidence", "Evidence"],
            rows=gp_rows,
            col_widths=[0.8 * inch, 1.25 * inch, 0.85 * inch, 0.5 * inch, 0.8 * inch, 1.2 * inch, 1.0 * inch, 0.8 * inch, 2.4 * inch],
            header_bg="#0f766e",
        )
    )
    story.append(Spacer(1, 8))

    story.append(_p("Appendix C. Assignment Method", heading_style))
    story.append(
        _p(
            "Assignment sequence: (1) nearest chemical-model key line within 1.2 nm, (2) target-species match, (3) rank-1 NIST candidate.",
            body_style,
        )
    )
    story.append(
        _p(
            "Confidence classes: HIGH (key-line and/or target support), MEDIUM (NIST-only), LOW (no direct species evidence).",
            body_style,
        )
    )

    def _page_footer(canvas, doc_obj) -> None:
        canvas.saveState()
        canvas.setFont(body_font, 8)
        footer = f"Reaction-Chemical Pathway Narrative | Generated {generated_on} | Page {canvas.getPageNumber()}"
        canvas.setFillColor(colors.HexColor("#334155"))
        canvas.drawRightString(doc_obj.pagesize[0] - 24, 14, footer)
        canvas.restoreState()

    path.parent.mkdir(parents=True, exist_ok=True)
    doc.build(story, onFirstPage=_page_footer, onLaterPages=_page_footer)
    story.append(
        _p(
            (
                f"Source files: {source_prefix}/spectral/averaged_peaks_top10.csv, "
                f"{source_prefix}/spectral/nist_matches_top3.csv, "
                f"{source_prefix}/spectral/target_species_peak_matches.csv, "
                f"{source_prefix}/chemical_modeling/key_line_intensity_table.csv, "
                f"{source_prefix}/chemical_modeling/pathway_peak_story_summary.csv"
            ),
            subtitle_style,
        )
    )
    story.append(
        _p(
            (
                f"Total group-resolved peaks: {len(peak_map)}. "
                f"Distinct rounded wavelengths: {peak_map['wavelength_nm'].nunique()}."
            ),
            subtitle_style,
        )
    )
    story.append(Spacer(1, 5))

    story.append(_p("1. Governing Equation Set", heading_style))
    eq_rows: List[List[Paragraph]] = []
    for eq_id, equation, description in _equation_catalog():
        eq_rows.append(
            [
                _p(eq_id, body_style),
                _p(_latex_to_pdf_markup(equation), equation_style, markup=True),
                _p(description, body_style),
            ]
        )
    story.append(
        _styled_long_table(
            headers=["ID", "Equation", "Interpretation"],
            rows=eq_rows,
            col_widths=[0.6 * inch, 3.5 * inch, 5.5 * inch],
        )
    )
    story.append(Spacer(1, 8))

    story.append(_p("2. Reaction-Pathway Evidence by Wavelength", heading_style))
    rp_rows: List[List[Paragraph]] = []
    for _, row in story_summary.iterrows():
        rp_rows.append(
            [
                _p(_reaction_to_pdf_markup(row["reaction"]), body_small_style, markup=True),
                _p(row["eq_ids"], body_small_style),
                _p(f"{_safe_float(row['wavelength_nm']):.2f}", body_small_right_style),
                _p(f"{_safe_float(row['sum_link_weight']):.6f}", body_small_right_style),
                _p(f"{_safe_float(row['mean_link_weight']):.6f}", body_small_right_style),
                _p(f"{_safe_int(row['n_groups'], default=0)}", body_small_right_style),
            ]
        )
    story.append(
        _styled_long_table(
            headers=["Reaction", "Eq IDs", "Line (nm)", "Sum Link", "Mean Link", "n"],
            rows=rp_rows,
            col_widths=[3.6 * inch, 1.1 * inch, 0.9 * inch, 1.1 * inch, 1.1 * inch, 0.7 * inch],
        )
    )
    story.append(Spacer(1, 8))

    story.append(_p("3. Unique Peak-Wavelength to Equation Mapping", heading_style))
    ws_rows: List[List[Paragraph]] = []
    for _, row in wavelength_summary.iterrows():
        ws_rows.append(
            [
                _p(f"{_safe_float(row['wavelength_nm']):.1f}", body_small_right_style),
                _p(f"{_safe_int(row['occurrences'], default=0)}", body_small_right_style),
                _p(f"{_safe_float(row['max_intensity']):.6e}", body_small_right_style),
                _p(row["equation_ids"], body_small_style),
                _p(_truncate_text(row["evidence"], max_len=170) or "n/a", body_small_style),
            ]
        )
    story.append(
        _styled_long_table(
            headers=["Wavelength (nm)", "Occurrences", "Max Intensity", "Eq IDs", "Evidence Snapshot"],
            rows=ws_rows,
            col_widths=[1.0 * inch, 0.8 * inch, 1.45 * inch, 1.1 * inch, 5.4 * inch],
        )
    )
    story.append(Spacer(1, 8))

    story.append(_p("4. Group-Resolved Peak Assignments", heading_style))
    gp_rows: List[List[Paragraph]] = []
    for _, row in peak_map.iterrows():
        gp_rows.append(
            [
                _p(row["dataset"], body_small_style),
                _p(row["param_set"], body_small_style),
                _p(row["channel"], body_small_style),
                _p(str(_safe_int(row["peak_rank"], default=-1)), body_small_right_style),
                _p(f"{_safe_float(row['wavelength_nm']):.1f}", body_small_right_style),
                _p(f"{_safe_float(row['peak_intensity']):.6e}", body_small_right_style),
                _p(row["equation_ids"], body_small_style),
                _p(str(row["confidence"]).upper(), body_small_style),
                _p(_truncate_text(row["evidence"], max_len=120), body_small_style),
            ]
        )
    story.append(
        _styled_long_table(
            headers=["Dataset", "Param Set", "Channel", "Rank", "lambda (nm)", "Intensity", "Eq IDs", "Confidence", "Evidence"],
            rows=gp_rows,
            col_widths=[0.8 * inch, 1.4 * inch, 0.9 * inch, 0.5 * inch, 0.8 * inch, 1.25 * inch, 1.1 * inch, 0.8 * inch, 2.2 * inch],
        )
    )
    story.append(Spacer(1, 8))

    story.append(_p("5. Assignment Logic", heading_style))
    story.append(
        _p(
            (
                "Equation assignment sequence: (1) nearest chemical-model key line within 1.2 nm, "
                "(2) target species match, (3) rank-1 NIST species candidate. "
                "Confidence tiers: HIGH (key-line and/or target-supported), MEDIUM (NIST-only), LOW (no species evidence)."
            ),
            body_style,
        )
    )

    def _page_footer(canvas, doc_obj) -> None:
        canvas.saveState()
        canvas.setFont(body_font, 8)
        footer = f"Reaction-Chemical Pathway Narrative | Page {canvas.getPageNumber()}"
        canvas.setFillColor(colors.HexColor("#334155"))
        canvas.drawRightString(doc_obj.pagesize[0] - 24, 14, footer)
        canvas.restoreState()

    path.parent.mkdir(parents=True, exist_ok=True)
    doc.build(story, onFirstPage=_page_footer, onLaterPages=_page_footer)


def _strip_style_block(markdown: str) -> str:
    lines = markdown.splitlines()
    out: List[str] = []
    inside = False
    for line in lines:
        if "<style>" in line:
            inside = True
            continue
        if "</style>" in line:
            inside = False
            continue
        if not inside:
            out.append(line)
    return "\n".join(out)


def _markdown_to_plain_lines(markdown: str, width: int = 100) -> List[str]:
    text = _strip_style_block(markdown)
    lines: List[str] = []
    link_re = re.compile(r"\[([^\]]+)\]\([^)]+\)")
    for raw in text.splitlines():
        line = raw.rstrip()
        line = link_re.sub(r"\1", line)
        line = line.replace("&rarr;", "->")
        line = re.sub(r"</?(?:sub|sup)>", "", line, flags=re.IGNORECASE)
        line = line.replace("`", "")
        line = line.replace("$$", "")
        line = line.replace("\\rightarrow", "->")
        line = line.replace("\\ge", ">=")
        line = line.replace("\\nu", "nu")
        line = line.replace("\\beta", "beta")
        line = line.replace("**", "")
        if line.startswith("#"):
            line = line.lstrip("#").strip()
            if line:
                line = line.upper()
        if not line:
            lines.append("")
            continue
        wrapped = textwrap.wrap(
            line,
            width=width,
            break_long_words=True,
            break_on_hyphens=False,
            replace_whitespace=False,
            drop_whitespace=False,
        )
        if wrapped:
            lines.extend(wrapped)
        else:
            lines.append(line)
    return lines


def _pdf_escape(text: str) -> str:
    safe = text.encode("latin-1", "replace").decode("latin-1")
    return safe.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _write_simple_text_pdf(path: Path, lines: Sequence[str]) -> None:
    lines = list(lines) if lines else ["Reaction-Chemical Pathway Narrative"]
    line_height = 12
    font_size = 10
    text_x = 40
    text_y = 760
    body_lines_per_page = 56

    pages: List[List[str]] = [lines[i : i + body_lines_per_page] for i in range(0, len(lines), body_lines_per_page)]
    total_pages = len(pages)
    paged_lines: List[List[str]] = []
    for idx, page in enumerate(pages, start=1):
        out_page = list(page)
        out_page.append("")
        out_page.append(f"Page {idx} of {total_pages}")
        paged_lines.append(out_page)

    objects: List[bytes | None] = [None]
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")  # 1
    objects.append(b"<< /Type /Pages /Kids [] /Count 0 >>")  # 2
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")  # 3

    page_ids: List[int] = []
    for page in paged_lines:
        ops: List[str] = ["BT", f"/F1 {font_size} Tf", f"{line_height} TL", f"{text_x} {text_y} Td"]
        for line in page:
            ops.append(f"({_pdf_escape(line)}) Tj")
            ops.append("T*")
        ops.append("ET")
        stream = "\n".join(ops).encode("latin-1", "replace")
        content_obj = b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"\nendstream"
        objects.append(content_obj)
        content_id = len(objects) - 1

        page_obj = (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 3 0 R >> >> /Contents "
            + str(content_id).encode("ascii")
            + b" 0 R >>"
        )
        objects.append(page_obj)
        page_ids.append(len(objects) - 1)

    kids = " ".join(f"{pid} 0 R" for pid in page_ids).encode("ascii")
    objects[2] = b"<< /Type /Pages /Kids [" + kids + b"] /Count " + str(len(page_ids)).encode("ascii") + b" >>"

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
        offsets = [0] * len(objects)
        for idx in range(1, len(objects)):
            offsets[idx] = f.tell()
            f.write(f"{idx} 0 obj\n".encode("ascii"))
            f.write(objects[idx] or b"")
            f.write(b"\nendobj\n")
        xref_pos = f.tell()
        f.write(f"xref\n0 {len(objects)}\n".encode("ascii"))
        f.write(b"0000000000 65535 f \n")
        for idx in range(1, len(objects)):
            f.write(f"{offsets[idx]:010d} 00000 n \n".encode("ascii"))
        f.write(b"trailer\n")
        f.write(f"<< /Size {len(objects)} /Root 1 0 R >>\n".encode("ascii"))
        f.write(b"startxref\n")
        f.write(f"{xref_pos}\n".encode("ascii"))
        f.write(b"%%EOF\n")


def _write_text_with_fallback(path: Path, text: str, fallback: Path) -> Path:
    try:
        path.write_text(text, encoding="utf-8")
        return path
    except PermissionError:
        fallback.write_text(text, encoding="utf-8")
        return fallback


def _write_pdf_with_fallback(
    path: Path,
    fallback: Path,
    markdown_text: str,
    source_scope: str,
    active_scope_values: Sequence[str],
    dataset_scopes: Sequence[str],
    peak_map: pd.DataFrame,
    wavelength_summary: pd.DataFrame,
    story_summary: pd.DataFrame,
) -> Path:
    try:
        _build_styled_pdf(
            path=path,
            source_scope=source_scope,
            active_scope_values=active_scope_values,
            dataset_scopes=dataset_scopes,
            peak_map=peak_map,
            wavelength_summary=wavelength_summary,
            story_summary=story_summary,
        )
        return path
    except PermissionError:
        _build_styled_pdf(
            path=fallback,
            source_scope=source_scope,
            active_scope_values=active_scope_values,
            dataset_scopes=dataset_scopes,
            peak_map=peak_map,
            wavelength_summary=wavelength_summary,
            story_summary=story_summary,
        )
        return fallback
    except Exception as exc:
        print(f"[reaction_narrative] Styled PDF renderer failed; using plain fallback renderer. Reason: {exc}")
        lines = _markdown_to_plain_lines(markdown_text, width=100)
        try:
            _write_simple_text_pdf(path, lines)
            return path
        except PermissionError:
            _write_simple_text_pdf(fallback, lines)
            return fallback


def main() -> int:
    current_scopes = list(active_scopes())
    source_scope, dataset_scopes = _resolve_narrative_source(current_scopes)
    base_root = OUTPUT_ROOT / source_scope / "metadata"

    try:
        peaks = _load_csv(
            base_root / "spectral" / "averaged_peaks_top10.csv",
            ["dataset", "param_set", "channel", "peak_rank", "peak_wavelength_nm_0p1", "peak_intensity_refined"],
        )
        nist = _load_csv(
            base_root / "spectral" / "nist_matches_top3.csv",
            ["dataset", "param_set", "channel", "peak_rank", "candidate_rank", "nist_species", "delta_nm"],
        )
        target = _load_csv(
            base_root / "spectral" / "target_species_peak_matches.csv",
            ["dataset", "param_set", "channel", "matched", "matched_peak_rank", "species"],
        )
        key_lines = _load_csv(
            base_root / "chemical_modeling" / "key_line_intensity_table.csv",
            ["dataset", "param_set", "channel", "line_name", "wavelength_nm"],
        )
        story = _load_csv(
            base_root / "chemical_modeling" / "pathway_peak_story_summary.csv",
            ["reaction", "wavelength_nm", "sum_link_weight", "mean_link_weight", "n_groups"],
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"[reaction_narrative] {exc}")
        return 1

    peaks = _filter_by_datasets(peaks, dataset_scopes)
    nist = _filter_by_datasets(nist, dataset_scopes)
    target = _filter_by_datasets(target, dataset_scopes)
    key_lines = _filter_by_datasets(key_lines, dataset_scopes)

    if peaks.empty:
        if dataset_scopes:
            print(
                "[reaction_narrative] No peaks available after mode filter "
                f"for dataset scope(s): {', '.join(dataset_scopes)}."
            )
        else:
            print("[reaction_narrative] No peaks available for narrative generation.")
        return 1

    peak_map = _build_peak_assignment_table(peaks, nist, target, key_lines)
    wavelength_summary = _build_wavelength_summary(peak_map)
    story_summary = _build_story_summary(story)
    markdown_text = _render_markdown(
        peak_map,
        wavelength_summary,
        story_summary,
        source_scope=source_scope,
        dataset_scopes=dataset_scopes,
        active_scope_values=current_scopes,
    )

    md_written = _write_text_with_fallback(NARRATIVE_MD_PATH, markdown_text, NARRATIVE_MD_FALLBACK)
    pdf_written = _write_pdf_with_fallback(
        path=NARRATIVE_PDF_PATH,
        fallback=NARRATIVE_PDF_FALLBACK,
        markdown_text=markdown_text,
        source_scope=source_scope,
        active_scope_values=current_scopes,
        dataset_scopes=dataset_scopes,
        peak_map=peak_map,
        wavelength_summary=wavelength_summary,
        story_summary=story_summary,
    )

    dataset_label = ", ".join(dataset_scopes) if dataset_scopes else "all"
    print(
        "[reaction_narrative] Context: "
        f"active_scopes={', '.join(current_scopes) if current_scopes else 'all'} | "
        f"source_scope={source_scope} | datasets={dataset_label}"
    )
    print("[reaction_narrative] Wrote:")
    print(f"  {md_written}")
    print(f"  {pdf_written}")
    if md_written != NARRATIVE_MD_PATH:
        print(f"[reaction_narrative] Note: target markdown was locked, wrote fallback {md_written}.")
    if pdf_written != NARRATIVE_PDF_PATH:
        print(f"[reaction_narrative] Note: target PDF was locked, wrote fallback {pdf_written}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
