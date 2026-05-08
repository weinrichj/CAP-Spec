#!/usr/bin/env python3
"""
CAP-Spec Load Testing Script
=============================
Measures response latency under increasing concurrent load.
Tests each endpoint individually and the full pipeline end-to-end.

Usage:
    python scripts/load_test.py --url http://localhost:8000 --users 1 5 10 20
    python scripts/load_test.py --endpoint /health --requests 100
    python scripts/load_test.py --full-pipeline --spectrum data/air/1H.9A.csv
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import io
import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import httpx

BASE_URL = "http://localhost:8000"
DEFAULT_USERNAME = "admin"
DEFAULT_PASSWORD = "secret"


# ─── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class RequestResult:
    endpoint: str
    method: str
    status_code: int
    latency_ms: float
    error: Optional[str] = None

@dataclass
class LoadTestResult:
    endpoint: str
    n_requests: int
    concurrency: int
    results: List[RequestResult] = field(default_factory=list)

    @property
    def successes(self):
        return [r for r in self.results if r.status_code < 400]

    @property
    def failures(self):
        return [r for r in self.results if r.status_code >= 400 or r.error]

    @property
    def latencies(self):
        return [r.latency_ms for r in self.successes]

    def summary(self) -> dict:
        lats = self.latencies
        if not lats:
            return {"error": "no successful requests"}
        return {
            "endpoint":        self.endpoint,
            "n_requests":      self.n_requests,
            "concurrency":     self.concurrency,
            "success_rate":    round(len(self.successes) / self.n_requests * 100, 1),
            "p50_ms":          round(statistics.median(lats), 2),
            "p95_ms":          round(sorted(lats)[int(len(lats) * 0.95)], 2),
            "p99_ms":          round(sorted(lats)[int(len(lats) * 0.99)], 2),
            "mean_ms":         round(statistics.mean(lats), 2),
            "min_ms":          round(min(lats), 2),
            "max_ms":          round(max(lats), 2),
            "std_ms":          round(statistics.stdev(lats) if len(lats) > 1 else 0, 2),
            "throughput_rps":  round(self.n_requests / (sum(lats) / 1000 / self.concurrency), 2),
            "error_count":     len(self.failures),
        }


# ─── Auth ─────────────────────────────────────────────────────────────────────

async def get_token(base_url: str, username: str, password: str) -> str:
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(f"{base_url}/auth/token",
                              data={"username": username, "password": password})
        r.raise_for_status()
        return r.json()["access_token"]


# ─── Single request ───────────────────────────────────────────────────────────

async def single_request(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    headers: dict,
    content=None,
    files=None,
) -> RequestResult:
    endpoint = url.split("//", 1)[-1].split("/", 1)[-1]
    start = time.perf_counter()
    try:
        if files:
            r = await client.request(method, url, headers=headers, files=files)
        elif content:
            r = await client.request(method, url, headers=headers, content=content)
        else:
            r = await client.request(method, url, headers=headers)
        latency = (time.perf_counter() - start) * 1000
        return RequestResult(endpoint=endpoint, method=method,
                             status_code=r.status_code, latency_ms=latency)
    except Exception as e:
        latency = (time.perf_counter() - start) * 1000
        return RequestResult(endpoint=endpoint, method=method,
                             status_code=0, latency_ms=latency, error=str(e))


# ─── Load test a single endpoint ──────────────────────────────────────────────

async def load_test_endpoint(
    base_url: str,
    token: str,
    endpoint: str,
    method: str = "GET",
    n_requests: int = 50,
    concurrency: int = 10,
    body: Optional[dict] = None,
) -> LoadTestResult:
    url     = f"{base_url}{endpoint}"
    headers = {"Authorization": f"Bearer {token}"}
    if body:
        headers["Content-Type"] = "application/json"
    content = json.dumps(body).encode() if body else None

    result = LoadTestResult(endpoint=endpoint, n_requests=n_requests, concurrency=concurrency)

    sem = asyncio.Semaphore(concurrency)
    async def bounded(client):
        async with sem:
            return await single_request(client, method, url, headers, content)

    async with httpx.AsyncClient(timeout=60, limits=httpx.Limits(max_connections=concurrency + 5)) as client:
        tasks   = [bounded(client) for _ in range(n_requests)]
        results = await asyncio.gather(*tasks, return_exceptions=False)
        result.results.extend(results)

    return result


# ─── Full pipeline test ───────────────────────────────────────────────────────

async def test_full_pipeline(base_url: str, token: str, spectrum_path: Path) -> dict:
    """Upload a spectrum and run the full pipeline. Measure each stage."""
    headers  = {"Authorization": f"Bearer {token}"}
    timings  = {}

    async with httpx.AsyncClient(timeout=300, follow_redirects=True) as client:
        # Upload
        t0 = time.perf_counter()
        with open(spectrum_path, "rb") as f:
            r = await client.post(
                f"{base_url}/ingest/upload?gas=air",
                headers={"Authorization": f"Bearer {token}"},
                files={"file": (spectrum_path.name, f, "text/plain")},
            )
        timings["upload_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        if r.status_code != 200:
            return {"error": f"Upload failed: {r.status_code} {r.text}"}
        job_id = r.json()["job_id"]

        # Full pipeline via gateway
        t0 = time.perf_counter()
        r  = await client.post(f"{base_url}/analyze/{job_id}",
                               headers={**headers, "Content-Type": "application/json"})
        timings["full_pipeline_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        if r.status_code != 200:
            return {"error": f"Pipeline failed: {r.status_code} {r.text}",
                    "timings": timings}

        result = r.json()
        timings["job_id"]   = job_id
        timings["summary"]  = result.get("summary", {})
        timings["status"]   = "complete"

    return timings


# ─── Concurrency sweep ────────────────────────────────────────────────────────

async def concurrency_sweep(
    base_url: str,
    token: str,
    endpoint: str,
    concurrency_levels: List[int],
    requests_per_level: int = 50,
) -> List[dict]:
    summaries = []
    for c in concurrency_levels:
        print(f"  Testing {endpoint} at concurrency={c} ...", end="", flush=True)
        result = await load_test_endpoint(
            base_url, token, endpoint,
            n_requests=requests_per_level,
            concurrency=c,
        )
        s = result.summary()
        summaries.append(s)
        print(f"  p50={s.get('p50_ms')}ms  p95={s.get('p95_ms')}ms  "
              f"success={s.get('success_rate')}%")
    return summaries


# ─── Report ───────────────────────────────────────────────────────────────────

def print_table(rows: List[dict]):
    if not rows:
        return
    cols = list(rows[0].keys())
    widths = {c: max(len(c), max(len(str(r.get(c, ""))) for r in rows)) for c in cols}
    header = "  ".join(c.ljust(widths[c]) for c in cols)
    print(header)
    print("-" * len(header))
    for row in rows:
        print("  ".join(str(row.get(c, "")).ljust(widths[c]) for c in cols))


def save_csv(rows: List[dict], path: Path):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved CSV: {path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser(description="CAP-Spec Load Tester")
    parser.add_argument("--url",         default=BASE_URL)
    parser.add_argument("--username",    default=DEFAULT_USERNAME)
    parser.add_argument("--password",    default=DEFAULT_PASSWORD)
    parser.add_argument("--users",       nargs="+", type=int, default=[1, 5, 10, 20],
                        help="Concurrency levels to test")
    parser.add_argument("--requests",    type=int, default=50,
                        help="Requests per concurrency level")
    parser.add_argument("--endpoint",    default=None,
                        help="Single endpoint to test (e.g. /health)")
    parser.add_argument("--full-pipeline", action="store_true")
    parser.add_argument("--spectrum",    type=Path, default=None,
                        help="Spectrum CSV for full pipeline test")
    parser.add_argument("--output",      type=Path, default=Path("load_test_results.csv"))
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  CAP-Spec Load Test  |  {args.url}")
    print(f"{'='*60}\n")

    print("Authenticating...")
    try:
        token = await get_token(args.url, args.username, args.password)
        print("  ✓ Token obtained\n")
    except Exception as e:
        print(f"  ✗ Auth failed: {e}")
        sys.exit(1)

    all_summaries = []

    if args.full_pipeline:
        spectrum = args.spectrum or Path("data/air/1H.9A.csv")
        if not spectrum.exists():
            print(f"Spectrum file not found: {spectrum}")
            sys.exit(1)
        print(f"Full pipeline test with {spectrum.name}...")
        result = await test_full_pipeline(args.url, token, spectrum)
        print(json.dumps(result, indent=2))
        return

    endpoints = (
        [args.endpoint] if args.endpoint else [
            "/health",
            "/ingest/jobs",
            "/admin/stats",
        ]
    )

    for ep in endpoints:
        print(f"\n[ {ep} ] — concurrency sweep {args.users}")
        summaries = await concurrency_sweep(
            args.url, token, ep, args.users, args.requests
        )
        all_summaries.extend(summaries)
        print()
        print_table(summaries)

    if all_summaries:
        save_csv(all_summaries, args.output)


if __name__ == "__main__":
    asyncio.run(main())
