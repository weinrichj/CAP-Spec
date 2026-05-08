"""
CAP-Spec Reporting Service
===========================
Assembles features + model results into a JSON dashboard payload with
base64-encoded charts. All charting is done here — we do NOT call the
plot_* functions from chemical_modeling.py (those write to disk).
"""
from __future__ import annotations

import base64
import io
import json
import logging
import os
from datetime import datetime

import boto3
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import psycopg2
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from psycopg2.extras import RealDictCursor
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATABASE_URL  = os.getenv("DATABASE_URL", "postgresql://capspec:capspec@postgres:5432/capspec")
S3_ENDPOINT   = os.getenv("S3_ENDPOINT",  "http://minio:9000")
S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY", "capspec_admin")
S3_SECRET_KEY = os.getenv("S3_SECRET_KEY", "capspec_secret")
S3_BUCKET     = os.getenv("S3_BUCKET",    "capspec")

THEME = {
    "bg":      "#1a0a20", "surface": "#2d1640",
    "accent":  "#9b59b6", "text":    "#f0e6ff", "grid":  "#3d2050",
    "species": {
        "OH":  "#3498db", "N2":  "#e67e22", "N2+": "#e74c3c",
        "O":   "#2ecc71", "He":  "#9b59b6", "Ha":  "#f39c12",
        "NO":  "#1abc9c", "Ar":  "#e91e63",
    },
}


def get_s3():
    return boto3.client("s3", endpoint_url=S3_ENDPOINT,
                        aws_access_key_id=S3_ACCESS_KEY,
                        aws_secret_access_key=S3_SECRET_KEY)

def get_db():
    return psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)

def load_json_s3(key: str) -> dict:
    obj = get_s3().get_object(Bucket=S3_BUCKET, Key=key)
    return json.loads(obj["Body"].read())

def load_spectrum(key: str):
    obj  = get_s3().get_object(Bucket=S3_BUCKET, Key=key)
    rows = []
    for line in obj["Body"].read().decode().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.strip().split(",")
        if len(parts) >= 2:
            try:
                rows.append((float(parts[0]), float(parts[1])))
            except ValueError:
                continue
    arr = np.array(rows)
    return arr[:, 0], arr[:, 1]

def update_job(job_id, status, report_path=None, error=None):
    conn = get_db()
    try:
        with conn.cursor() as cur:
            if error:
                cur.execute("UPDATE jobs SET status=%s, error_message=%s, updated_at=NOW() WHERE id=%s",
                            (status, error, job_id))
            else:
                cur.execute("UPDATE jobs SET status=%s, report_path=%s, updated_at=NOW() WHERE id=%s",
                            (status, report_path, job_id))
        conn.commit()
    finally:
        conn.close()

def fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

def _ax_theme(ax, title=None, xlabel=None, ylabel=None):
    ax.set_facecolor(THEME["surface"])
    ax.tick_params(colors=THEME["text"])
    ax.spines[:].set_color(THEME["grid"])
    for lbl, txt in [(title, ax.set_title), (xlabel, ax.set_xlabel), (ylabel, ax.set_ylabel)]:
        if lbl:
            txt(lbl, color=THEME["text"], fontsize=10)

def _f(v):
    try:
        f = float(v)
        return f if np.isfinite(f) else None
    except Exception:
        return None


# ─── Chart generators ─────────────────────────────────────────────────────────

def chart_spectrum(wl: np.ndarray, y: np.ndarray, all_peaks: list) -> str:
    fig, ax = plt.subplots(figsize=(13, 4), facecolor=THEME["bg"])
    _ax_theme(ax, "CAP Emission Spectrum (Preprocessed & Normalized)",
              "Wavelength (nm)", "Norm. Irradiance (a.u.)")
    ax.plot(wl, y, color=THEME["accent"], linewidth=0.8, alpha=0.9)
    ax.fill_between(wl, y, alpha=0.12, color=THEME["accent"])
    for p in sorted(all_peaks, key=lambda p: p.get("peak_intensity") or 0, reverse=True)[:12]:
        wl_p = p.get("peak_wavelength_nm_refined") or p.get("peak_wavelength_nm_grid")
        y_p  = p.get("peak_intensity") or 0
        if wl_p and y_p > 0.08:
            ax.axvline(wl_p, color="#ffffff", alpha=0.2, linewidth=0.6, linestyle=":")
            ax.text(wl_p + 0.5, min(y_p + 0.04, 0.95), f"{wl_p:.1f}",
                    color="#cccccc", fontsize=6, rotation=90, va="bottom")
    ax.set_xlim(wl.min(), wl.max()); ax.set_ylim(-0.02, 1.05)
    ax.grid(True, color=THEME["grid"], alpha=0.4, linewidth=0.5)
    plt.tight_layout()
    b64 = fig_to_b64(fig); plt.close(fig); return b64


def chart_species_bars(sp_windows: dict) -> str:
    LABELS = {
        "oh_309":        "OH (309)",     "n2_337":       "N₂ (337)",
        "n2_315":        "N₂ (315)",     "n2plus_391":   "N₂⁺ (391)",
        "n2plus_427":    "N₂⁺ (427)",   "no_gamma":     "NO γ",
        "hbeta_486":     "Hβ (486)",     "uvc_continuum":"UVC cont.",
        "n2_357":        "N₂ (357)",     "n2_380":       "N₂ (380)",
    }
    COLORS = ["#3498db","#e67e22","#e74c3c","#2ecc71","#9b59b6",
              "#f39c12","#1abc9c","#e91e63","#aaaaaa","#5dade2"]
    items = [(k.replace("_area",""), v) for k, v in sp_windows.items()
             if k.endswith("_area") and v and _f(v) and _f(v) > 0]
    items.sort(key=lambda x: x[1], reverse=True); items = items[:10]
    if not items:
        return ""
    labels = [LABELS.get(k, k) for k, _ in items]
    values = [float(v) for _, v in items]
    total  = max(sum(values), 1e-12)
    pcts   = [v / total * 100 for v in values]

    fig, ax = plt.subplots(figsize=(9, 4), facecolor=THEME["bg"])
    _ax_theme(ax, "Species Window Integrals", "Species", "Relative Contribution (%)")
    bars = ax.bar(labels, pcts, color=COLORS[:len(labels)], edgecolor=THEME["grid"], linewidth=0.5)
    for bar, pct in zip(bars, pcts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{pct:.1f}%", ha="center", va="bottom", color=THEME["text"], fontsize=8)
    ax.tick_params(axis="x", rotation=20)
    ax.grid(True, axis="y", color=THEME["grid"], alpha=0.4)
    plt.tight_layout(); b64 = fig_to_b64(fig); plt.close(fig); return b64


def chart_temperatures(model_results: dict) -> str:
    temps = model_results.get("temperatures", {})
    rows  = [
        ("T_rot (K)",  temps.get("rotational",  {}).get("temperature"),
                       temps.get("rotational",  {}).get("temperature_ci95_low"),
                       temps.get("rotational",  {}).get("temperature_ci95_high")),
        ("T_vib (K)",  temps.get("vibrational", {}).get("temperature"), None, None),
        ("T_exc (K)",  temps.get("excitation",  {}).get("temperature"), None, None),
    ]
    valid = [(lbl, v, lo, hi) for lbl, v, lo, hi in rows
             if v is not None and np.isfinite(float(v))]
    if not valid:
        return ""
    labels = [r[0] for r in valid]
    values = [float(r[1]) for r in valid]
    errs   = [
        (float(hi) - float(v)) if (lo is not None and hi is not None
                                   and np.isfinite(float(lo)) and np.isfinite(float(hi))) else 0
        for _, v, lo, hi in valid
    ]
    fig, ax = plt.subplots(figsize=(7, 4), facecolor=THEME["bg"])
    _ax_theme(ax, "Estimated Plasma Temperatures", None, "Temperature (K)")
    colors = ["#9b59b6", "#3498db", "#e67e22"]
    ax.bar(labels, values, yerr=errs if any(e > 0 for e in errs) else None,
           color=colors[:len(labels)], edgecolor=THEME["grid"],
           capsize=5, error_kw={"ecolor": "#ffffff", "alpha": 0.6})
    for i, v in enumerate(values):
        ax.text(i, v + max(errs[i] if i < len(errs) else 0, v * 0.02),
                f"{v:.0f} K", ha="center", va="bottom", color=THEME["text"], fontsize=9)
    ax.grid(True, axis="y", color=THEME["grid"], alpha=0.4)
    plt.tight_layout(); b64 = fig_to_b64(fig); plt.close(fig); return b64


def chart_biomedical_radar(scores: dict) -> str:
    if not scores:
        return ""
    labels = list(scores.keys())
    values = [scores[k] for k in labels]
    N      = len(labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    vplot  = values + [values[0]]
    angles = angles + [angles[0]]
    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw={"polar": True}, facecolor=THEME["bg"])
    ax.set_facecolor(THEME["surface"])
    ax.plot(angles, vplot, color=THEME["accent"], linewidth=2)
    ax.fill(angles, vplot, alpha=0.25, color=THEME["accent"])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([l.replace("_", "\n") for l in labels], color=THEME["text"], fontsize=8)
    ax.set_ylim(0, 1); ax.grid(color=THEME["grid"], alpha=0.5)
    ax.tick_params(colors=THEME["text"])
    ax.set_title("Biomedical Suitability", color=THEME["text"], fontsize=11, pad=20)
    plt.tight_layout(); b64 = fig_to_b64(fig); plt.close(fig); return b64


def chart_pathway_edges(edges: list) -> str:
    if not edges:
        return ""
    df = [e for e in edges if _f(e.get("weight")) is not None]
    if not df:
        return ""
    df.sort(key=lambda e: e["weight"])
    df = df[-9:]
    labels = [e.get("reaction", "?") for e in df]
    values = [float(e["weight"]) for e in df]
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(labels)))
    fig, ax = plt.subplots(figsize=(10, max(3, len(df)*0.5)), facecolor=THEME["bg"])
    _ax_theme(ax, "Reaction Pathway Strengths",
              "Pathway Weight (evidence-weighted)", None)
    ax.barh(labels, values, color=colors, edgecolor=THEME["grid"], linewidth=0.4)
    ax.set_xlim(0, 1.05)
    ax.grid(True, axis="x", color=THEME["grid"], alpha=0.4)
    plt.tight_layout(); b64 = fig_to_b64(fig); plt.close(fig); return b64


def chart_peaks(peaks: list) -> str:
    if not peaks:
        return ""
    top = sorted(peaks, key=lambda p: p.get("peak_intensity") or 0, reverse=True)[:15]
    labels = [f"{p.get('peak_wavelength_nm_refined') or p.get('peak_wavelength_nm_grid', '?'):.1f} nm"
              for p in top]
    values = [p.get("peak_intensity") or 0 for p in top]
    colors = [THEME["accent"] if v == max(values) else "#6c3483" for v in values]
    fig, ax = plt.subplots(figsize=(8, max(3, len(top)*0.38)), facecolor=THEME["bg"])
    _ax_theme(ax, "Top Detected Emission Peaks", "Normalised Intensity", None)
    ax.barh(labels[::-1], values[::-1], color=colors[::-1],
            edgecolor=THEME["grid"], linewidth=0.4)
    ax.grid(True, axis="x", color=THEME["grid"], alpha=0.4)
    plt.tight_layout(); b64 = fig_to_b64(fig); plt.close(fig); return b64


# ─── Report assembly ──────────────────────────────────────────────────────────

def assemble_report(job: dict, features: dict, model_results: dict,
                    wl: np.ndarray, y: np.ndarray) -> dict:
    meta     = job.get("metadata") or {}
    if isinstance(meta, str):
        meta = json.loads(meta)
    sp_wins  = features.get("species_windows", {})
    bio      = model_results.get("biomedical_assessment", {})
    temps    = model_results.get("temperatures", {})
    elec     = model_results.get("electron_density", {})
    edges    = model_results.get("pathway_edges", [])
    peaks    = features.get("all_peaks", [])
    matches  = features.get("species_matches", [])
    band_f   = features.get("band_features", {})

    charts = {
        "spectrum":     chart_spectrum(wl, y, peaks),
        "species_bars": chart_species_bars(sp_wins),
        "temperatures": chart_temperatures(model_results),
        "biomedical":   chart_biomedical_radar(bio.get("scores", {})),
        "pathways":     chart_pathway_edges(edges),
        "peaks":        chart_peaks(peaks),
    }

    return {
        "job_id":       str(job["id"]),
        "generated_at": datetime.utcnow().isoformat(),
        "version":      "1.0.0",
        "metadata":     {
            "file_name": job.get("file_name"),
            "gas":       meta.get("gas"),
            "device_id": meta.get("device_id"),
            "notes":     meta.get("notes"),
        },
        "spectrum_stats":  band_f,
        "species_windows": sp_wins,
        "species_matches": matches,
        "temperatures": {
            "rotational_K":  _f(temps.get("rotational",  {}).get("temperature")),
            "vibrational_K": _f(temps.get("vibrational", {}).get("temperature")),
            "excitation_K":  _f(temps.get("excitation",  {}).get("temperature")),
            "rot_ci95": [
                _f(temps.get("rotational", {}).get("temperature_ci95_low")),
                _f(temps.get("rotational", {}).get("temperature_ci95_high")),
            ],
            "rot_fit_mode":   temps.get("rotational", {}).get("mode"),
            "rot_fit_status": temps.get("rotational", {}).get("status"),
            "rot_r2":         _f(temps.get("rotational", {}).get("r2")),
        },
        "electron_density": {
            "ne_cm3":    _f(elec.get("estimated_electron_density")),
            "ne_ci95":   [
                _f(elec.get("estimated_electron_density_ci95_low")),
                _f(elec.get("estimated_electron_density_ci95_high")),
            ],
            "status":                elec.get("electron_density_status"),
            "hbeta_lorentz_fwhm_nm": _f(elec.get("hbeta_lorentz_fwhm_nm")),
            "hbeta_stark_fwhm_nm":   _f(elec.get("hbeta_stark_fwhm_nm")),
        },
        "pathway_edges": edges,
        "pathway_story": model_results.get("pathway_story", []),
        "all_peaks":     peaks,
        "biomedical":    bio,
        "charts":        charts,
    }


# ─── App ─────────────────────────────────────────────────────────────────────

app = FastAPI(title="CAP-Spec Reporting Service", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class ReportRequest(BaseModel):
    job_id: str


@app.post("/report")
async def generate_report(req: ReportRequest):
    conn = get_db()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM jobs WHERE id = %s", (req.job_id,))
            job = cur.fetchone()
    finally:
        conn.close()

    if not job:
        raise HTTPException(404, f"Job {req.job_id} not found")
    if job["status"] not in ("modeled", "reporting"):
        raise HTTPException(409, f"Job status '{job['status']}' — expected 'modeled'")

    update_job(req.job_id, "reporting")

    try:
        features      = load_json_s3(job["features_file_path"])
        model_results = load_json_s3(job["model_results_path"])
        wl, y         = load_spectrum(job["preprocessed_file_path"])
        report        = assemble_report(dict(job), features, model_results, wl, y)
    except Exception as e:
        update_job(req.job_id, "failed", error=str(e))
        logger.exception("Reporting failed")
        raise HTTPException(500, f"Reporting failed: {e}")

    report_key = f"reports/{req.job_id}/report.json"
    get_s3().put_object(Bucket=S3_BUCKET, Key=report_key,
                        Body=json.dumps(report, indent=2, default=str).encode())
    update_job(req.job_id, "complete", report_path=report_key)
    logger.info(f"Report complete for job {req.job_id}")

    summary = {k: v for k, v in report.items() if k != "charts"}
    return {**summary, "status": "complete",
            "charts_available": list(report["charts"].keys()),
            "report_path": report_key}


@app.get("/report/{job_id}")
async def get_report(job_id: str, include_charts: bool = True):
    conn = get_db()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT report_path FROM jobs WHERE id = %s", (job_id,))
            row = cur.fetchone()
    finally:
        conn.close()
    if not row or not row["report_path"]:
        raise HTTPException(404, "Report not yet available")
    obj    = get_s3().get_object(Bucket=S3_BUCKET, Key=row["report_path"])
    report = json.loads(obj["Body"].read())
    if not include_charts:
        report.pop("charts", None)
    return report


@app.get("/report/{job_id}/chart/{chart_name}")
async def get_chart(job_id: str, chart_name: str):
    conn = get_db()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT report_path FROM jobs WHERE id = %s", (job_id,))
            row = cur.fetchone()
    finally:
        conn.close()
    if not row or not row["report_path"]:
        raise HTTPException(404, "Report not available")
    obj    = get_s3().get_object(Bucket=S3_BUCKET, Key=row["report_path"])
    charts = json.loads(obj["Body"].read()).get("charts", {})
    if chart_name not in charts or not charts[chart_name]:
        raise HTTPException(404, f"Chart '{chart_name}' not found. Available: {list(charts)}")
    return Response(content=base64.b64decode(charts[chart_name]),
                    media_type="image/png",
                    headers={"Content-Disposition": f'inline; filename="{chart_name}.png"'})


@app.get("/report/{job_id}/summary")
async def get_summary(job_id: str):
    conn = get_db()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT report_path, status FROM jobs WHERE id = %s", (job_id,))
            row = cur.fetchone()
    finally:
        conn.close()
    if not row:
        raise HTTPException(404, "Job not found")
    if row["status"] != "complete":
        return {"status": row["status"], "message": "Report not yet complete"}
    obj = get_s3().get_object(Bucket=S3_BUCKET, Key=row["report_path"])
    r   = json.loads(obj["Body"].read())
    return {
        "job_id":            job_id,
        "temperatures":      r.get("temperatures", {}),
        "electron_density":  r.get("electron_density", {}),
        "top_species":       sorted(
            [m for m in r.get("species_matches", []) if m.get("detected")],
            key=lambda x: x.get("match_rate") or 0, reverse=True
        )[:5],
        "biomedical_scores": r.get("biomedical", {}).get("scores", {}),
        "overall_score":     r.get("biomedical", {}).get("overall_suitability"),
        "recommendations":   r.get("biomedical", {}).get("recommendations", []),
        "charts_available":  list(r.get("charts", {}).keys()),
    }


@app.get("/health")
async def health():
    return {"service": "reporting", "status": "healthy",
            "timestamp": datetime.utcnow().isoformat()}
