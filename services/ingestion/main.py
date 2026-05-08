"""
CAP-Spec Ingestion Service
==========================
Handles spectrum file upload, validation, and job creation.
Supports CSV, TSV, and plain-text two-column (wavelength, intensity) files.
"""
import os
import io
import uuid
import logging
import hashlib
from datetime import datetime
from typing import Optional

import boto3
import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── Config ───────────────────────────────────────────────────────────────────

DATABASE_URL   = os.getenv("DATABASE_URL", "postgresql://capspec:capspec@postgres:5432/capspec")
S3_ENDPOINT    = os.getenv("S3_ENDPOINT",  "http://minio:9000")
S3_ACCESS_KEY  = os.getenv("S3_ACCESS_KEY", "capspec_admin")
S3_SECRET_KEY  = os.getenv("S3_SECRET_KEY", "capspec_secret")
S3_BUCKET      = os.getenv("S3_BUCKET",    "capspec")
MAX_FILE_MB    = int(os.getenv("MAX_FILE_MB", 50))

SUPPORTED_EXTENSIONS = {".csv", ".tsv", ".txt", ".dat"}

# ─── Clients ─────────────────────────────────────────────────────────────────

def get_s3():
    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=S3_ACCESS_KEY,
        aws_secret_access_key=S3_SECRET_KEY,
    )

def get_db():
    return psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)

def ensure_bucket():
    s3 = get_s3()
    try:
        s3.head_bucket(Bucket=S3_BUCKET)
    except Exception:
        s3.create_bucket(Bucket=S3_BUCKET)


# ─── Validation ──────────────────────────────────────────────────────────────

def parse_spectrum(content: bytes, filename: str) -> np.ndarray:
    """
    Parse a spectrum file into a (N, 2) array of [wavelength, intensity].
    Supports CSV, TSV, space-delimited.  Skips comment lines (# ; !).
    Returns array sorted by wavelength.
    """
    text = content.decode("utf-8", errors="replace")
    lines = []
    delimiters_tried = [",", "\t", " ", ";"]

    data_lines = [
        ln.strip() for ln in text.splitlines()
        if ln.strip() and not ln.strip().startswith(("#", ";", "!", "W", "w"))
    ]
    if not data_lines:
        raise ValueError("File contains no data lines")

    parsed = []
    for line in data_lines:
        for delim in delimiters_tried:
            parts = [p.strip() for p in line.split(delim) if p.strip()]
            if len(parts) >= 2:
                try:
                    wl = float(parts[0])
                    intensity = float(parts[1])
                    parsed.append((wl, intensity))
                    break
                except ValueError:
                    continue

    if len(parsed) < 10:
        raise ValueError(f"Could not parse spectrum – only {len(parsed)} data points found")

    arr = np.array(parsed)
    arr = arr[arr[:, 0].argsort()]  # sort by wavelength
    return arr


def validate_spectrum(arr: np.ndarray):
    """Check that the spectrum is physically plausible for CAP."""
    wl = arr[:, 0]
    intensity = arr[:, 1]

    if wl.min() < 150 or wl.max() > 1100:
        raise ValueError(f"Wavelength range {wl.min():.1f}–{wl.max():.1f} nm is outside expected UV-NIR range")
    if (intensity < 0).any():
        logger.warning("Spectrum contains negative intensities – will be clipped during preprocessing")
    if wl.max() - wl.min() < 50:
        raise ValueError("Wavelength span is too narrow (< 50 nm) for meaningful CAP analysis")
    if len(arr) < 50:
        raise ValueError("Spectrum has too few data points (< 50)")


# ─── App ─────────────────────────────────────────────────────────────────────

app = FastAPI(title="CAP-Spec Ingestion Service", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.on_event("startup")
def startup():
    ensure_bucket()


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.post("/upload", summary="Upload a spectrum file and create an analysis job")
async def upload_spectrum(
    file: UploadFile = File(...),
    gas: str = Query("helium", description="Carrier gas: helium | argon | nitrogen | air"),
    device_id: Optional[str] = Query(None, description="Optional device identifier"),
    notes: Optional[str] = Query(None, description="Free-text notes"),
):
    # Size check
    content = await file.read()
    if len(content) > MAX_FILE_MB * 1024 * 1024:
        raise HTTPException(413, f"File exceeds {MAX_FILE_MB} MB limit")

    # Extension check
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(415, f"Unsupported file type '{ext}'. Use: {SUPPORTED_EXTENSIONS}")

    # Parse & validate
    try:
        spectrum = parse_spectrum(content, file.filename)
        validate_spectrum(spectrum)
    except ValueError as e:
        raise HTTPException(422, str(e))

    job_id   = str(uuid.uuid4())
    checksum = hashlib.sha256(content).hexdigest()
    s3_key   = f"raw/{job_id}/{file.filename}"

    # Upload to MinIO
    try:
        s3 = get_s3()
        s3.put_object(Bucket=S3_BUCKET, Key=s3_key, Body=content,
                      Metadata={"job_id": job_id, "sha256": checksum})
    except Exception as e:
        logger.error(f"S3 upload failed: {e}")
        raise HTTPException(502, "Object storage unavailable")

    # Create job record
    metadata = {
        "gas": gas,
        "device_id": device_id,
        "notes": notes,
        "file_sha256": checksum,
        "n_points": len(spectrum),
        "wl_min": float(spectrum[:, 0].min()),
        "wl_max": float(spectrum[:, 0].max()),
        "original_filename": file.filename,
    }
    conn = get_db()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO jobs (id, status, file_name, raw_file_path, metadata, created_at, updated_at)
                   VALUES (%s, 'ingested', %s, %s, %s::jsonb, NOW(), NOW()) RETURNING *""",
                (job_id, file.filename, s3_key, __import__("json").dumps(metadata)),
            )
            job = dict(cur.fetchone())
        conn.commit()
    finally:
        conn.close()

    logger.info(f"Job {job_id} created for {file.filename} ({len(spectrum)} pts)")

    return {
        "job_id": job_id,
        "status": "ingested",
        "file_name": file.filename,
        "n_points": len(spectrum),
        "wavelength_range_nm": [float(spectrum[:, 0].min()), float(spectrum[:, 0].max())],
        "metadata": metadata,
        "message": "File validated and stored. Call /analyze/{job_id} on the gateway to run the full pipeline.",
    }


@app.get("/job/{job_id}", summary="Get job status and metadata")
async def get_job_status(job_id: str):
    conn = get_db()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM jobs WHERE id = %s", (job_id,))
            row = cur.fetchone()
    finally:
        conn.close()

    if not row:
        raise HTTPException(404, f"Job {job_id} not found")
    return {k: str(v) if k in ("id", "user_id") else v for k, v in dict(row).items()}


@app.get("/jobs", summary="List recent jobs")
async def list_jobs(limit: int = Query(20, le=100), status: Optional[str] = None):
    conn = get_db()
    try:
        with conn.cursor() as cur:
            if status:
                cur.execute("SELECT * FROM jobs WHERE status = %s ORDER BY created_at DESC LIMIT %s",
                            (status, limit))
            else:
                cur.execute("SELECT * FROM jobs ORDER BY created_at DESC LIMIT %s", (limit,))
            return [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()


@app.get("/raw/{job_id}", summary="Download the raw spectrum file")
async def download_raw(job_id: str):
    """Return the raw uploaded file from object storage."""
    conn = get_db()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT raw_file_path, file_name FROM jobs WHERE id = %s", (job_id,))
            row = cur.fetchone()
    finally:
        conn.close()

    if not row:
        raise HTTPException(404, "Job not found")

    s3 = get_s3()
    try:
        obj = s3.get_object(Bucket=S3_BUCKET, Key=row["raw_file_path"])
        body = obj["Body"].read()
    except Exception as e:
        raise HTTPException(502, f"Could not fetch from storage: {e}")

    from fastapi.responses import Response
    return Response(content=body, media_type="text/plain",
                    headers={"Content-Disposition": f'attachment; filename="{row["file_name"]}"'})


@app.delete("/job/{job_id}", summary="Delete a job and its stored files")
async def delete_job(job_id: str):
    conn = get_db()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT raw_file_path, preprocessed_file_path, features_file_path, "
                        "model_results_path, report_path FROM jobs WHERE id = %s", (job_id,))
            row = cur.fetchone()
        if not row:
            raise HTTPException(404, "Job not found")

        s3 = get_s3()
        for key in row.values():
            if key:
                try:
                    s3.delete_object(Bucket=S3_BUCKET, Key=key)
                except Exception:
                    pass

        with conn.cursor() as cur:
            cur.execute("DELETE FROM usage_stats WHERE job_id = %s", (job_id,))
            cur.execute("DELETE FROM jobs WHERE id = %s", (job_id,))
        conn.commit()
    finally:
        conn.close()

    return {"deleted": job_id}


@app.get("/health")
async def health():
    return {"service": "ingestion", "status": "healthy", "timestamp": datetime.utcnow().isoformat()}
