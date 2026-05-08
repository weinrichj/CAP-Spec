"""
CAP-Spec Preprocessing Service
===============================
Pipeline:
  1. Load raw spectrum from object storage
  2. Clip negative values
  3. Savitzky-Golay smoothing
  4. Asymmetric Least Squares (ALS) baseline correction
  5. Min-max normalisation
  6. Persist preprocessed spectrum back to MinIO
  7. Update job status
"""
import io
import os
import json
import logging
from datetime import datetime
from typing import Tuple

import boto3
import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor
from scipy.signal import savgol_filter
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── Config ───────────────────────────────────────────────────────────────────

DATABASE_URL  = os.getenv("DATABASE_URL", "postgresql://capspec:capspec@postgres:5432/capspec")
S3_ENDPOINT   = os.getenv("S3_ENDPOINT", "http://minio:9000")
S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY", "capspec_admin")
S3_SECRET_KEY = os.getenv("S3_SECRET_KEY", "capspec_secret")
S3_BUCKET     = os.getenv("S3_BUCKET", "capspec")

# ─── Helpers ──────────────────────────────────────────────────────────────────

def get_s3():
    return boto3.client("s3", endpoint_url=S3_ENDPOINT,
                        aws_access_key_id=S3_ACCESS_KEY,
                        aws_secret_access_key=S3_SECRET_KEY)

def get_db():
    return psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)

def load_spectrum_from_s3(s3_key: str) -> np.ndarray:
    s3 = get_s3()
    obj = s3.get_object(Bucket=S3_BUCKET, Key=s3_key)
    content = obj["Body"].read().decode("utf-8", errors="replace")
    rows = []
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith(("#", ";", "!")):
            continue
        for delim in [",", "\t", " ", ";"]:
            parts = [p.strip() for p in line.split(delim) if p.strip()]
            if len(parts) >= 2:
                try:
                    rows.append((float(parts[0]), float(parts[1])))
                    break
                except ValueError:
                    continue
    arr = np.array(rows)
    return arr[arr[:, 0].argsort()]

def save_spectrum_to_s3(arr: np.ndarray, key: str):
    buf = io.StringIO()
    buf.write("# CAP-Spec preprocessed spectrum\n")
    buf.write("# wavelength_nm,intensity_normalized\n")
    for wl, intensity in arr:
        buf.write(f"{wl:.4f},{intensity:.8f}\n")
    s3 = get_s3()
    s3.put_object(Bucket=S3_BUCKET, Key=key, Body=buf.getvalue().encode())

def update_job(job_id: str, status: str, preprocessed_path: str = None, error: str = None):
    conn = get_db()
    try:
        with conn.cursor() as cur:
            if error:
                cur.execute("UPDATE jobs SET status=%s, error_message=%s, updated_at=NOW() WHERE id=%s",
                            (status, error, job_id))
            else:
                cur.execute("UPDATE jobs SET status=%s, preprocessed_file_path=%s, updated_at=NOW() WHERE id=%s",
                            (status, preprocessed_path, job_id))
        conn.commit()
    finally:
        conn.close()


# ─── Signal Processing ────────────────────────────────────────────────────────

def clip_negatives(intensity: np.ndarray) -> np.ndarray:
    """Replace negative intensities with zero."""
    return np.clip(intensity, 0, None)


def smooth_spectrum(intensity: np.ndarray, window: int = 11, poly: int = 3) -> np.ndarray:
    """
    Apply Savitzky-Golay filter for noise reduction while preserving peak shapes.
    Automatically reduces window size if spectrum is short.
    """
    window = min(window, len(intensity) - 2)
    if window % 2 == 0:
        window -= 1
    window = max(window, poly + 2)
    return savgol_filter(intensity, window_length=window, polyorder=poly)


def als_baseline(intensity: np.ndarray, lam: float = 1e5, p: float = 0.01,
                 n_iter: int = 10) -> np.ndarray:
    """
    Asymmetric Least Squares baseline correction (Eilers & Boelens 2005).
    lam – smoothness (larger = smoother baseline)
    p   – asymmetry (smaller = more asymmetric, suitable for emission spectra)
    Returns estimated baseline.
    """
    L = len(intensity)
    D = diags([1, -2, 1], [0, 1, 2], shape=(L - 2, L))
    H = lam * D.T @ D
    w = np.ones(L)
    for _ in range(n_iter):
        W   = diags(w, 0)
        Z   = W + H
        z   = spsolve(Z, w * intensity)
        w   = p * (intensity > z) + (1 - p) * (intensity <= z)
    return z


def normalize(intensity: np.ndarray) -> np.ndarray:
    """Min-max normalise to [0, 1]."""
    mn, mx = intensity.min(), intensity.max()
    if mx - mn < 1e-12:
        return np.zeros_like(intensity)
    return (intensity - mn) / (mx - mn)


def full_preprocess(raw: np.ndarray, smooth_window: int = 11,
                    als_lam: float = 1e5, als_p: float = 0.01) -> Tuple[np.ndarray, dict]:
    """
    Run the full preprocessing pipeline.
    Returns (processed_spectrum, stats_dict).
    """
    wl        = raw[:, 0]
    intensity = raw[:, 1].copy()

    intensity = clip_negatives(intensity)
    intensity = smooth_spectrum(intensity, window=smooth_window)
    baseline  = als_baseline(intensity, lam=als_lam, p=als_p)
    intensity = intensity - baseline
    intensity = clip_negatives(intensity)  # remove residual negatives after subtraction
    intensity = normalize(intensity)

    stats = {
        "n_points":       len(wl),
        "wl_min_nm":      float(wl.min()),
        "wl_max_nm":      float(wl.max()),
        "wl_resolution":  float(np.median(np.diff(wl))),
        "peak_intensity":  float(intensity.max()),
        "snr_estimate":    float(intensity.max() / (np.std(intensity[:20]) + 1e-12)),
        "smooth_window":  smooth_window,
        "als_lambda":     als_lam,
        "als_p":          als_p,
    }
    return np.column_stack([wl, intensity]), stats


# ─── App ─────────────────────────────────────────────────────────────────────

app = FastAPI(title="CAP-Spec Preprocessing Service", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class PreprocessRequest(BaseModel):
    job_id: str
    smooth_window: int = 11        # Savitzky-Golay window (odd integer)
    als_lambda: float  = 1e5       # ALS smoothness
    als_p: float       = 0.01      # ALS asymmetry


@app.post("/preprocess", summary="Run preprocessing pipeline on a job's raw spectrum")
async def preprocess(req: PreprocessRequest):
    # Load job
    conn = get_db()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM jobs WHERE id = %s", (req.job_id,))
            job = cur.fetchone()
    finally:
        conn.close()

    if not job:
        raise HTTPException(404, f"Job {req.job_id} not found")
    if job["status"] not in ("ingested", "preprocessing"):
        raise HTTPException(409, f"Job is in status '{job['status']}', expected 'ingested'")

    update_job(req.job_id, "preprocessing")

    try:
        raw = load_spectrum_from_s3(job["raw_file_path"])
        processed, stats = full_preprocess(
            raw,
            smooth_window=req.smooth_window,
            als_lam=req.als_lambda,
            als_p=req.als_p,
        )
    except Exception as e:
        update_job(req.job_id, "failed", error=str(e))
        raise HTTPException(500, f"Preprocessing failed: {e}")

    # Store result
    out_key = f"preprocessed/{req.job_id}/spectrum_preprocessed.csv"
    try:
        save_spectrum_to_s3(processed, out_key)
    except Exception as e:
        update_job(req.job_id, "failed", error=str(e))
        raise HTTPException(502, f"Storage error: {e}")

    update_job(req.job_id, "preprocessed", preprocessed_path=out_key)
    logger.info(f"Job {req.job_id} preprocessed: {stats}")

    return {"job_id": req.job_id, "status": "preprocessed",
            "output_path": out_key, "stats": stats}


@app.get("/preview/{job_id}", summary="Return raw & preprocessed spectrum points for plotting")
async def preview(job_id: str, n_points: int = 500):
    """Returns downsampled spectrum before/after preprocessing for the UI."""
    conn = get_db()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT raw_file_path, preprocessed_file_path FROM jobs WHERE id = %s", (job_id,))
            job = cur.fetchone()
    finally:
        conn.close()

    if not job:
        raise HTTPException(404, "Job not found")

    raw = load_spectrum_from_s3(job["raw_file_path"])
    step = max(1, len(raw) // n_points)
    raw_ds = raw[::step]

    result = {"raw": {"wavelength": raw_ds[:, 0].tolist(),
                      "intensity": raw_ds[:, 1].tolist()}}

    if job["preprocessed_file_path"]:
        proc = load_spectrum_from_s3(job["preprocessed_file_path"])
        proc_ds = proc[::step]
        result["preprocessed"] = {"wavelength": proc_ds[:, 0].tolist(),
                                   "intensity": proc_ds[:, 1].tolist()}
    return result


@app.get("/health")
async def health():
    return {"service": "preprocessing", "status": "healthy",
            "timestamp": datetime.utcnow().isoformat()}
