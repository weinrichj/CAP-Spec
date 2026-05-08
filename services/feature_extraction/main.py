"""
CAP-Spec Feature Extraction Service
=====================================
Uses your analysis code directly:
  - analysis.numeric_utils  → safe_ratio, trapz_integral
  - analysis.species        → load_windows, extract_window_metrics, add_grouped_species_features
  - analysis.ms_core        → detect_top_peaks, match_peaks_to_target_species, build_target_match_summary
  - analysis.features       → centroid, band_integral, DEFAULT_BANDS

All imports use the 'analysis' package prefix so they resolve correctly
when PYTHONPATH=/app and analysis code lives at /app/analysis/.
"""
from __future__ import annotations
import os, json, logging, sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import boto3
import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# analysis package is at /app/analysis; PYTHONPATH=/app so 'analysis' resolves
from analysis.numeric_utils import safe_ratio, trapz_integral
from analysis.species import load_windows, extract_window_metrics, add_grouped_species_features
from analysis.ms_core import detect_top_peaks, match_peaks_to_target_species, build_target_match_summary
from analysis.features import centroid, band_integral, DEFAULT_BANDS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATABASE_URL  = os.getenv("DATABASE_URL", "postgresql://capspec:capspec@postgres:5432/capspec")
S3_ENDPOINT   = os.getenv("S3_ENDPOINT",  "http://minio:9000")
S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY", "capspec_admin")
S3_SECRET_KEY = os.getenv("S3_SECRET_KEY", "capspec_secret")
S3_BUCKET     = os.getenv("S3_BUCKET",    "capspec")

# Configs are copied to /app/configs/ in the Dockerfile
WINDOWS_CSV        = Path("/app/configs/species_windows.csv")
TARGET_SPECIES_CSV = Path("/app/configs/target_species_lines.csv")


def get_s3():
    return boto3.client("s3", endpoint_url=S3_ENDPOINT,
                        aws_access_key_id=S3_ACCESS_KEY,
                        aws_secret_access_key=S3_SECRET_KEY)

def get_db():
    return psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)

def load_spectrum(s3_key: str) -> Tuple[np.ndarray, np.ndarray]:
    obj = get_s3().get_object(Bucket=S3_BUCKET, Key=s3_key)
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

def save_features(features: dict, job_id: str) -> str:
    key = f"features/{job_id}/features.json"
    get_s3().put_object(Bucket=S3_BUCKET, Key=key,
                        Body=json.dumps(features, indent=2, default=str).encode())
    return key

def update_job(job_id, status, features_path=None, error=None):
    conn = get_db()
    try:
        with conn.cursor() as cur:
            if error:
                cur.execute("UPDATE jobs SET status=%s, error_message=%s, updated_at=NOW() WHERE id=%s",
                            (status, error, job_id))
            else:
                cur.execute("UPDATE jobs SET status=%s, features_file_path=%s, updated_at=NOW() WHERE id=%s",
                            (status, features_path, job_id))
        conn.commit()
    finally:
        conn.close()

def _f(v):
    try:
        f = float(v)
        return f if np.isfinite(f) else None
    except Exception:
        return None


# ─── Feature functions (call your real code) ──────────────────────────────────

def compute_band_features(wl: np.ndarray, y: np.ndarray) -> dict:
    total  = trapz_integral(wl, y)
    c_nm   = centroid(wl, y)
    peak_i = int(np.nanargmax(y)) if y.size else 0
    result = {
        "total_irradiance":   _f(total),
        "centroid_nm":        _f(c_nm),
        "peak_wavelength_nm": _f(wl[peak_i]),
        "peak_irradiance":    _f(y[peak_i]),
        "n_points":           int(len(wl)),
        "wl_min_nm":          _f(wl.min()),
        "wl_max_nm":          _f(wl.max()),
        "wl_resolution_nm":   _f(np.median(np.diff(wl))),
        "snr":                _f(float(y[peak_i]) / (float(np.std(y[:20])) + 1e-12)) if y.size else None,
    }
    band_vals = {b: band_integral(wl, y, s, e) for b, s, e in DEFAULT_BANDS}
    for b, s, e in DEFAULT_BANDS:
        result[f"{b}_integral"] = _f(band_vals[b])
        result[f"{b}_frac"]     = _f(safe_ratio(band_vals[b], total))
    bmap = {b.upper(): v for b, v in band_vals.items()}
    result["uva_uvb_ratio"] = _f(safe_ratio(bmap.get("UVA", float("nan")), bmap.get("UVB", float("nan"))))
    result["uvb_uvc_ratio"] = _f(safe_ratio(bmap.get("UVB", float("nan")), bmap.get("UVC", float("nan"))))
    result["uva_uvc_ratio"] = _f(safe_ratio(bmap.get("UVA", float("nan")), bmap.get("UVC", float("nan"))))
    return result


def compute_species_windows(wl: np.ndarray, y: np.ndarray) -> dict:
    """Uses your load_windows + extract_window_metrics + add_grouped_species_features."""
    windows = load_windows(WINDOWS_CSV)
    total   = trapz_integral(wl, y, empty_value=0.0)
    row: Dict = {"total_integral": _f(total), "n_windows": len(windows)}
    for w in windows:
        stem    = str(w["species_slug"])
        metrics = extract_window_metrics(wl, y, float(w["start_nm"]), float(w["end_nm"]))
        row[f"{stem}_area"]       = _f(metrics["area"])
        row[f"{stem}_peak"]       = _f(metrics["peak"])
        row[f"{stem}_peak_nm"]    = _f(metrics["peak_nm"])
        row[f"{stem}_frac_total"] = _f(safe_ratio(metrics["area"], total))
        row[f"{stem}_points"]     = int(metrics["n_points"])
    add_grouped_species_features(row)
    return {k: (_f(v) if isinstance(v, (float, np.floating)) else v) for k, v in row.items()}


def compute_peaks(wl: np.ndarray, y: np.ndarray, top_n: int = 30) -> List[dict]:
    """Uses your detect_top_peaks with quadratic refinement."""
    peaks = detect_top_peaks(wl, y, top_n=top_n, intensity_col_name="peak_intensity")
    return [{k: (_f(v) if isinstance(v, float) else v) for k, v in p.items()} for p in peaks]


def compute_species_matches(peaks: List[dict], tolerance_nm: float = 2.0) -> List[dict]:
    """Uses your match_peaks_to_target_species + build_target_match_summary."""
    if not TARGET_SPECIES_CSV.exists() or not peaks:
        return []
    targets = pd.read_csv(TARGET_SPECIES_CSV)
    if targets.empty or "wavelength_nm" not in targets.columns or "species" not in targets.columns:
        return []
    peaks_df = pd.DataFrame(peaks)
    if "peak_wavelength_nm_0p1" not in peaks_df.columns:
        return []
    # match_peaks_to_target_species expects dataset/param_set/channel group columns
    peaks_df["dataset"]   = "upload"
    peaks_df["param_set"] = "upload"
    peaks_df["channel"]   = "ch0"
    peaks_df["peak_rank"] = range(1, len(peaks_df) + 1)
    matches = match_peaks_to_target_species(peaks_df, targets, tolerance_nm=tolerance_nm)
    summary = build_target_match_summary(matches)
    return [
        {
            "species":         str(r["species"]),
            "targets_total":   int(r["targets_total"]),
            "targets_matched": int(r["targets_matched"]),
            "match_rate":      _f(r["match_rate"]),
            "mean_delta_nm":   _f(r.get("mean_delta_nm")),
            "detected":        bool(int(r["targets_matched"]) > 0),
        }
        for _, r in summary.iterrows()
    ]


# ─── App ─────────────────────────────────────────────────────────────────────

app = FastAPI(title="CAP-Spec Feature Extraction Service", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class ExtractRequest(BaseModel):
    job_id: str
    top_n_peaks: int = 30
    species_tolerance_nm: float = 2.0


@app.post("/extract")
async def extract_features(req: ExtractRequest):
    conn = get_db()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM jobs WHERE id = %s", (req.job_id,))
            job = cur.fetchone()
    finally:
        conn.close()
    if not job:
        raise HTTPException(404, f"Job {req.job_id} not found")
    if job["status"] not in ("preprocessed", "features"):
        raise HTTPException(409, f"Job status '{job['status']}' — expected 'preprocessed'")

    update_job(req.job_id, "features")
    try:
        wl, y          = load_spectrum(job["preprocessed_file_path"])
        band_feats     = compute_band_features(wl, y)
        species_feats  = compute_species_windows(wl, y)
        all_peaks      = compute_peaks(wl, y, top_n=req.top_n_peaks)
        sp_matches     = compute_species_matches(all_peaks, tolerance_nm=req.species_tolerance_nm)
        features = {
            "job_id":           req.job_id,
            "extracted_at":     datetime.utcnow().isoformat(),
            "band_features":    band_feats,
            "species_windows":  species_feats,
            "all_peaks":        all_peaks,
            "species_matches":  sp_matches,
            "n_peaks_detected": len(all_peaks),
            "species_detected": [s["species"] for s in sp_matches if s["detected"]],
        }
    except Exception as e:
        update_job(req.job_id, "failed", error=str(e))
        logger.exception("Feature extraction failed")
        raise HTTPException(500, f"Feature extraction failed: {e}")

    out_key = save_features(features, req.job_id)
    update_job(req.job_id, "features_extracted", features_path=out_key)
    logger.info(f"Job {req.job_id}: {len(all_peaks)} peaks, {features['species_detected']}")
    return {"job_id": req.job_id, "status": "features_extracted",
            "n_peaks": len(all_peaks), "species_detected": features["species_detected"],
            "output_path": out_key}


@app.get("/features/{job_id}")
async def get_features(job_id: str):
    conn = get_db()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT features_file_path FROM jobs WHERE id = %s", (job_id,))
            row = cur.fetchone()
    finally:
        conn.close()
    if not row or not row["features_file_path"]:
        raise HTTPException(404, "Features not yet computed")
    obj = get_s3().get_object(Bucket=S3_BUCKET, Key=row["features_file_path"])
    return json.loads(obj["Body"].read())


@app.get("/health")
async def health():
    return {"service": "feature_extraction", "status": "healthy",
            "timestamp": datetime.utcnow().isoformat()}
