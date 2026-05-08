"""
CAP-Spec Modeling Service
==========================
Uses analysis.chemical_modeling for all physics calculations.
All imports use the 'analysis' package prefix — resolved via PYTHONPATH=/app.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime

import boto3
import numpy as np
import pandas as pd
import psycopg2
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from psycopg2.extras import RealDictCursor
from pydantic import BaseModel

# analysis package lives at /app/analysis/, importable as 'analysis.xxx' via PYTHONPATH=/app
from analysis.chemical_modeling import (
    _safe_float,
    build_group_label,
    build_pathway_edges,
    build_peak_to_pathway_links,
    collect_key_line_table,
    compute_relative_dissociation_proxy,
    estimate_electron_density,
    estimate_excitation_temperature,
    estimate_rotational_temperature,
    estimate_vibrational_temperature,
    load_excitation_line_metadata,
    load_gas_conditions,
    load_vibrational_metadata,
    summarize_peak_pathway_story,
    PRESSURE_ATM_DEFAULT,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATABASE_URL  = os.getenv("DATABASE_URL", "postgresql://capspec:capspec@postgres:5432/capspec")
S3_ENDPOINT   = os.getenv("S3_ENDPOINT",  "http://minio:9000")
S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY", "capspec_admin")
S3_SECRET_KEY = os.getenv("S3_SECRET_KEY", "capspec_secret")
S3_BUCKET     = os.getenv("S3_BUCKET",    "capspec")


def get_s3():
    return boto3.client("s3", endpoint_url=S3_ENDPOINT,
                        aws_access_key_id=S3_ACCESS_KEY,
                        aws_secret_access_key=S3_SECRET_KEY)

def get_db():
    return psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)

def load_spectrum(s3_key: str):
    obj  = get_s3().get_object(Bucket=S3_BUCKET, Key=s3_key)
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

def load_json_s3(key: str) -> dict:
    obj = get_s3().get_object(Bucket=S3_BUCKET, Key=key)
    return json.loads(obj["Body"].read())

def save_results(data: dict, job_id: str) -> str:
    key = f"models/{job_id}/model_results.json"
    get_s3().put_object(Bucket=S3_BUCKET, Key=key,
                        Body=json.dumps(data, indent=2, default=str).encode())
    return key

def update_job(job_id, status, model_path=None, error=None):
    conn = get_db()
    try:
        with conn.cursor() as cur:
            if error:
                cur.execute("UPDATE jobs SET status=%s, error_message=%s, updated_at=NOW() WHERE id=%s",
                            (status, error, job_id))
            else:
                cur.execute("UPDATE jobs SET status=%s, model_results_path=%s, updated_at=NOW() WHERE id=%s",
                            (status, model_path, job_id))
        conn.commit()
    finally:
        conn.close()

def _f(v):
    try:
        f = float(v)
        return f if np.isfinite(f) else None
    except Exception:
        return None

def _ser(obj):
    """Recursively JSON-serialise numpy types."""
    if isinstance(obj, dict):
        return {k: _ser(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_ser(i) for i in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return _f(float(obj))
    if isinstance(obj, float):
        return _f(obj)
    return obj


# ─── Biomedical scoring ───────────────────────────────────────────────────────

def biomedical_assessment(sp_windows: dict, T_rot: float | None) -> dict:
    oh   = _safe_float(sp_windows.get("oh_309_area",     0.0)) or 0.0
    n2p  = _safe_float(sp_windows.get("n2plus_391_area", 0.0)) or 0.0
    no   = _safe_float(sp_windows.get("no_gamma_area",   0.0)) or 0.0
    total = oh + n2p + no + (_safe_float(sp_windows.get("n2_337_area", 0.0)) or 0.0) + 1e-12

    c_oh  = oh  / total
    c_n2p = n2p / total
    c_no  = no  / total

    sterilization = min(1.0, c_oh * 3.0 + c_n2p * 0.5)
    wound_healing = min(1.0, c_no * 4.0 + c_oh * 1.0)
    anti_cancer   = min(1.0, c_oh * 2.0 + c_n2p * 1.5)

    if T_rot and np.isfinite(T_rot) and T_rot > 0:
        thermal_safety = max(0.0, 1.0 - (T_rot - 300) / 600)
        thermal_note   = f"T_rot ≈ {T_rot:.0f} K ({T_rot - 273:.0f} °C)"
    else:
        thermal_safety = 0.75
        thermal_note   = "Rotational temperature unavailable"

    uv_risk = min(1.0, c_n2p * 6.0)
    scores  = {
        "sterilization":  round(sterilization, 3),
        "wound_healing":  round(wound_healing, 3),
        "anti_cancer":    round(anti_cancer, 3),
        "thermal_safety": round(thermal_safety, 3),
        "uv_risk":        round(uv_risk, 3),
    }
    overall = round(
        (sterilization + wound_healing + anti_cancer + thermal_safety - uv_risk * 0.5) / 4, 3
    )
    recs = []
    if sterilization > 0.5:
        recs.append("✓ Strong OH signal — suitable for sterilization")
    if wound_healing > 0.4:
        recs.append("✓ NO/OH balance favorable for wound healing")
    if anti_cancer > 0.4:
        recs.append("✓ RONS profile supports selective cancer cell targeting")
    if uv_risk > 0.5:
        recs.append("⚠ Elevated N₂⁺ — consider UV shielding for tissue")
    if T_rot and T_rot > 400:
        recs.append(f"⚠ Elevated gas temperature ({T_rot:.0f} K) — limit treatment duration")
    if not recs:
        recs.append("Optimize source parameters to enhance RONS production")

    return {"scores": scores, "overall_suitability": overall,
            "thermal_note": thermal_note, "recommendations": recs}


# ─── App ─────────────────────────────────────────────────────────────────────

app = FastAPI(title="CAP-Spec Modeling Service", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Load metadata once at startup (these fall back to built-in defaults if CSVs missing)
VIB_META = load_vibrational_metadata()
EXC_META = load_excitation_line_metadata()
GAS_CFG  = load_gas_conditions()


class AnalyzeRequest(BaseModel):
    job_id: str
    pressure_atm: float = 1.0
    rotational_mode: str = "auto"  # auto | line_fit | synthetic_band


@app.post("/analyze")
async def analyze(req: AnalyzeRequest):
    conn = get_db()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM jobs WHERE id = %s", (req.job_id,))
            job = cur.fetchone()
    finally:
        conn.close()

    if not job:
        raise HTTPException(404, f"Job {req.job_id} not found")
    if job["status"] not in ("features_extracted", "modeling"):
        raise HTTPException(409, f"Job status '{job['status']}' — expected 'features_extracted'")

    update_job(req.job_id, "modeling")

    try:
        wl, y      = load_spectrum(job["preprocessed_file_path"])
        features   = load_json_s3(job["features_file_path"])
        sp_windows = features.get("species_windows", {})

        # ── Temperatures ──────────────────────────────────────────────────────
        rot  = estimate_rotational_temperature(wl, y, mode=req.rotational_mode)
        vib  = estimate_vibrational_temperature(wl, y, VIB_META)
        exc  = estimate_excitation_temperature(wl, y, EXC_META)
        T_rot = _safe_float(rot.get("temperature"))

        # ── Electron density ──────────────────────────────────────────────────
        elec = estimate_electron_density(
            wl, y,
            rotational_temperature_k=T_rot,
            pressure_atm=req.pressure_atm,
        )

        # ── Key line table ────────────────────────────────────────────────────
        meta      = job.get("metadata") or {}
        if isinstance(meta, str):
            meta = json.loads(meta)
        dataset   = meta.get("device_id") or "upload"
        param_set = meta.get("gas")       or "unknown"
        channel   = meta.get("notes")     or "ch0"

        line_df, line_lookup = collect_key_line_table(dataset, param_set, channel, wl, y)

        gas_info = {"current_a": float("nan"), "ar_to_n2": float("nan"),
                    "o2_to_n2": float("nan"), "response_ratio_ar": 1.0,
                    "response_ratio_o2": 1.0}
        proxy = compute_relative_dissociation_proxy(line_lookup, gas_info)

        # ── Pathway analysis ──────────────────────────────────────────────────
        estimates_df = pd.DataFrame([{
            "dataset":   dataset,
            "param_set": param_set,
            "channel":   channel,
            "group_label": build_group_label(dataset, param_set, channel),
            "estimated_rotational_temperature":  T_rot,
            "estimated_vibrational_temperature": _safe_float(vib.get("temperature")),
            "estimated_excitation_temperature":  _safe_float(exc.get("temperature")),
            "estimated_electron_density":        _safe_float(elec.get("estimated_electron_density")),
            "relative_dissociation_proxy":       _safe_float(proxy.get("relative_dissociation_proxy")),
            "ar_to_n2":          float("nan"),
            "o2_to_n2":          float("nan"),
            "line_area_OH_308":  _safe_float(line_lookup.get("OH_308")),
        }])

        edges_df = build_pathway_edges(estimates_df)
        link_df  = build_peak_to_pathway_links(estimates_df, line_df)
        story_df = summarize_peak_pathway_story(link_df)

        bio = biomedical_assessment(sp_windows, T_rot)

        results = {
            "job_id":      req.job_id,
            "analyzed_at": datetime.utcnow().isoformat(),
            "temperatures": {
                "rotational":  _ser(rot),
                "vibrational": _ser(vib),
                "excitation":  _ser(exc),
            },
            "electron_density":    _ser(elec),
            "dissociation_proxy":  _ser(proxy),
            "key_line_intensities": {
                row["line_name"]: _f(row["line_area"])
                for _, row in line_df.iterrows()
            },
            "pathway_edges": edges_df.to_dict(orient="records"),
            "pathway_story": story_df.to_dict(orient="records") if not story_df.empty else [],
            "biomedical_assessment": bio,
            "summary": {
                "T_rot_K":        _f(T_rot),
                "T_vib_K":        _f(vib.get("temperature")),
                "T_exc_K":        _f(exc.get("temperature")),
                "n_e_cm3":        _f(elec.get("estimated_electron_density")),
                "overall_score":  bio["overall_suitability"],
                "rot_fit_status": rot.get("status"),
                "ne_status":      elec.get("electron_density_status"),
            },
        }

    except Exception as e:
        update_job(req.job_id, "failed", error=str(e))
        logger.exception("Modeling failed")
        raise HTTPException(500, f"Modeling failed: {e}")

    out_key = save_results(results, req.job_id)
    update_job(req.job_id, "modeled", model_path=out_key)
    logger.info(f"Job {req.job_id}: T_rot={results['summary']['T_rot_K']}")

    return {"job_id": req.job_id, "status": "modeled",
            "summary": results["summary"], "output_path": out_key}


@app.get("/results/{job_id}")
async def get_results(job_id: str):
    conn = get_db()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT model_results_path FROM jobs WHERE id = %s", (job_id,))
            row = cur.fetchone()
    finally:
        conn.close()
    if not row or not row["model_results_path"]:
        raise HTTPException(404, "Results not yet available")
    obj = get_s3().get_object(Bucket=S3_BUCKET, Key=row["model_results_path"])
    return json.loads(obj["Body"].read())


@app.get("/health")
async def health():
    return {"service": "modeling", "status": "healthy",
            "timestamp": datetime.utcnow().isoformat()}
