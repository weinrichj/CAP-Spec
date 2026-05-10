# CAP-Spec

**Cold Atmospheric Plasma Spectral Analysis — Microservices Application**  
CSC5201 Final Project | John Weinrich

CAP-Spec is a microservices application for automated analysis of optical emission spectroscopy (OES) data from Cold Atmospheric Plasma (CAP) experiments. Users upload raw spectrum files and receive a complete analysis report: baseline-corrected spectra, identified chemical species, estimated plasma temperatures (rotational and vibrational), electron density, and reaction pathway summaries; all orchestrated through a REST API pipeline.

---

## Architecture

```
                        ┌─────────────┐
                        │   Frontend  │  nginx / port 3000
                        │  (index.html│
                        └──────┬──────┘
                               │ HTTP
                        ┌──────▼──────┐
                        │   Gateway   │  JWT auth · routing · admin stats
                        │  port 8000  │
                        └──┬──┬──┬──┬─┘
                ┌──────────┘  │  │  └──────────┐
         ┌──────▼──────┐      │  │      ┌──────▼──────┐
         │  Ingestion  │      │  │      │  Reporting  │
         │  port 8001  │      │  │      │  port 8005  │
         └─────────────┘      │  │      └─────────────┘
                       ┌──────▼──┐
                       │Preproc. │
                       │port 8002│
                       └────┬────┘
                       ┌────▼────┐
                       │Features │
                       │port 8003│
                       └────┬────┘
                       ┌────▼────┐
                       │Modeling │
                       │port 8004│
                       └─────────┘

    Infrastructure: PostgreSQL (jobs · users · stats) + MinIO (spectrum files)
```

### Services

| Service | Port | Responsibilities |
|---------|------|-----------------|
| **Gateway** | 8000 | JWT auth, role-based access control, request proxying, admin usage statistics |
| **Ingestion** | 8001 | Spectrum file upload and validation, job creation, MinIO storage |
| **Preprocessing** | 8002 | ALS baseline correction, Savitzky-Golay smoothing, normalization |
| **Feature Extraction** | 8003 | Species window integration, peak detection, target species matching |
| **Modeling** | 8004 | Rotational/vibrational temperature, electron density, reaction pathways |
| **Reporting** | 8005 | Chart generation, JSON report assembly, executive summary |

---

## Requirements

### Local (Docker Compose)

| Tool | Minimum version |
|------|-----------------|
| Docker Desktop | 24+ |
| Docker Compose | v2 (bundled) |
| Python | 3.11+ (for load testing only) |

### Cloud / Kubernetes

- `kubectl` configured for your cluster
- A container registry (Docker Hub, ECR, GCR, or GHCR)
- Cloud account with EKS, GKE, or AKS

---

## Installation

### 1 — Clone the repository

```bash
git clone https://github.com/<your-username>/cap-spec.git
cd cap-spec
```

### 2 — Set a secret key

Open `docker-compose.yml` and replace the placeholder on the `SECRET_KEY` line:

```yaml
SECRET_KEY: CHANGE_THIS_TO_A_RANDOM_SECRET_IN_PRODUCTION
```

Generate a secure key:

```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

### 3 — Build and start

```bash
docker compose up --build
```

The first build takes 10–15 minutes (scipy/matplotlib install across containers). All subsequent starts take ~30 seconds.

### 4 — Verify

```bash
# All services healthy?
curl http://localhost:8000/health

# Open the web UI
open http://localhost:3000
```

Default credentials: `admin` / `secret` — **change these before any real deployment.**

---

## Running an Analysis

### Via the Web UI

1. Open `http://localhost:3000`
2. Log in with your credentials
3. Click **Upload Spectrum** and select a `.csv`, `.tsv`, `.txt`, or `.dat` file  
   (two-column format: wavelength, intensity)
4. Click **Run Analysis** — the pipeline runs automatically
5. View results and download the report from the job detail page

### Via the API

```bash
# 1. Authenticate
TOKEN=$(curl -s -X POST http://localhost:8000/auth/token \
  -d "username=admin&password=secret" | jq -r .access_token)

# 2. Upload a spectrum file
JOB=$(curl -s -X POST http://localhost:8000/ingest/upload \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@spectrum.csv" | jq -r .job_id)

# 3. Run the full pipeline
curl -X POST http://localhost:8000/analyze/$JOB \
  -H "Authorization: Bearer $TOKEN"

# 4. Poll for completion
curl http://localhost:8000/ingest/jobs/$JOB \
  -H "Authorization: Bearer $TOKEN"

# 5. Retrieve results
curl http://localhost:8000/report/$JOB \
  -H "Authorization: Bearer $TOKEN"
```

---

## API Reference

All routes go through the gateway at `http://localhost:8000`. Requests require a `Bearer` token except where noted.

### Authentication

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/auth/token` | Obtain JWT (form: `username`, `password`) |
| `POST` | `/auth/register` | Create a new user account |
| `GET`  | `/auth/me` | Return current user info |

### Ingestion

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/ingest/upload` | Upload a spectrum file; returns `job_id` |
| `GET`  | `/ingest/jobs/{job_id}` | Get job status and metadata |
| `DELETE` | `/ingest/jobs/{job_id}` | Delete a job and its stored files |

### Analysis Pipeline

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/analyze/{job_id}` | Run the full pipeline (ingest → preprocess → features → model → report) |
| `POST` | `/preprocess/preprocess` | Run preprocessing step only |
| `GET`  | `/preprocess/preview/{job_id}` | Raw vs. preprocessed spectrum for plotting |
| `POST` | `/features/extract` | Run feature extraction step only |
| `GET`  | `/features/features/{job_id}` | Retrieve extracted features |
| `POST` | `/model/analyze` | Run modeling step only |
| `GET`  | `/model/results/{job_id}` | Retrieve modeling results |
| `POST` | `/report/report` | Generate report for a job |
| `GET`  | `/report/{job_id}` | Retrieve full report JSON |
| `GET`  | `/report/{job_id}/chart/{chart_name}` | Retrieve a specific chart (PNG) |
| `GET`  | `/report/{job_id}/summary` | Retrieve executive summary |

### Admin *(admin role required)*

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/admin/stats` | Per-endpoint usage statistics (call count, avg latency, error rate) |
| `GET` | `/admin/stats/timeseries` | Time-bucketed stats for charting |
| `GET` | `/admin/jobs` | All jobs across all users |

### Health

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Gateway + all downstream health check |

---

## Input File Format

Spectrum files must be plain-text with two columns — wavelength (nm) and intensity (counts). Headers, comments (`#`, `;`, `!`), and most common delimiters (comma, tab, space, semicolon) are handled automatically.

```
# Example: air plasma spectrum
wavelength,intensity
200.1,312.5
200.3,318.2
...
```

Accepted extensions: `.csv`, `.tsv`, `.txt`, `.dat`. Maximum file size: 50 MB.

---

---

## Project Structure

```
cap-spec/
├── docker-compose.yml          ← Runs the entire stack locally
├── DEPLOYMENT.md               ← Full deployment guide
├── db/
│   └── init.sql                ← Schema + seed data (auto-runs on first boot)
├── frontend/
│   └── index.html              ← Web UI (served by nginx on port 3000)
├── k8s/
│   └── manifests.yaml          ← Kubernetes deployment
├── scripts/
│   └── load_test.py            ← Latency benchmarking
└── services/
    ├── gateway/                ← Auth, routing, admin stats      (port 8000)
    ├── ingestion/              ← File upload, job creation       (port 8001)
    ├── preprocessing/          ← Baseline correction, smoothing  (port 8002)
    ├── feature_extraction/     ← Species windows, peak detection (port 8003)
    ├── modeling/               ← Temperatures, electron density  (port 8004)
    ├── reporting/              ← Charts, final report JSON       (port 8005)
    ├── cap_analysis/           ← Core plasma analysis library
    ├── cap_analysis_configs/   ← Species windows, excitation lines, etc.
    └── shims/                  ← No-op stubs for legacy import compatibility
```
