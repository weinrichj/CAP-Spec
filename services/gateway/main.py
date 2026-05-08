"""
CAP-Spec Gateway Service
========================
Handles:
  - User registration / login (JWT)
  - Role-based access control
  - Request proxying to downstream services
  - Admin usage-statistics endpoint
  - Health aggregation
"""
import os
import time
import logging
from datetime import datetime, timedelta
from typing import Optional, List

import httpx
import psycopg2
from psycopg2.extras import RealDictCursor

from fastapi import FastAPI, Depends, HTTPException, status, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm

from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── Config ───────────────────────────────────────────────────────────────────

SECRET_KEY      = os.getenv("SECRET_KEY", "CHANGE_ME_IN_PRODUCTION_32_CHARS_MIN")
ALGORITHM       = "HS256"
TOKEN_EXPIRE_MIN = int(os.getenv("TOKEN_EXPIRE_MIN", 60 * 8))
DATABASE_URL    = os.getenv("DATABASE_URL", "postgresql://capspec:capspec@postgres:5432/capspec")

SERVICES = {
    "ingestion":    os.getenv("INGESTION_URL",    "http://ingestion:8001"),
    "preprocessing":os.getenv("PREPROCESS_URL",   "http://preprocessing:8002"),
    "features":     os.getenv("FEATURES_URL",     "http://feature_extraction:8003"),
    "modeling":     os.getenv("MODELING_URL",     "http://modeling:8004"),
    "reporting":    os.getenv("REPORTING_URL",    "http://reporting:8005"),
}

# ─── Auth helpers ─────────────────────────────────────────────────────────────

pwd_ctx   = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2    = OAuth2PasswordBearer(tokenUrl="/auth/token")


def get_db():
    conn = psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
    try:
        yield conn
    finally:
        conn.close()


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_ctx.verify(plain, hashed)


def hash_password(plain: str) -> str:
    return pwd_ctx.hash(plain)


def create_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    payload = data.copy()
    expire  = datetime.utcnow() + (expires_delta or timedelta(minutes=TOKEN_EXPIRE_MIN))
    payload["exp"] = expire
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")


def get_user(conn, username: str) -> Optional[dict]:
    with conn.cursor() as cur:
        cur.execute("SELECT * FROM users WHERE username = %s AND is_active = TRUE", (username,))
        row = cur.fetchone()
        return dict(row) if row else None


def authenticate_user(conn, username: str, password: str) -> Optional[dict]:
    user = get_user(conn, username)
    if not user or not verify_password(password, user["hashed_password"]):
        return None
    return user


async def current_user(token: str = Depends(oauth2), conn=Depends(get_db)):
    payload = decode_token(token)
    username = payload.get("sub")
    if not username:
        raise HTTPException(status_code=401, detail="Invalid token payload")
    user = get_user(conn, username)
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user


def require_role(*roles):
    async def checker(user=Depends(current_user)):
        if user["role"] not in roles:
            raise HTTPException(status_code=403, detail=f"Requires role: {roles}")
        return user
    return checker


# ─── Schemas ─────────────────────────────────────────────────────────────────

class UserCreate(BaseModel):
    username: str
    email: str
    password: str
    role: str = "researcher"

class UserOut(BaseModel):
    id: str
    username: str
    email: str
    role: str
    created_at: datetime

class Token(BaseModel):
    access_token: str
    token_type: str
    role: str
    username: str

class StatsSummary(BaseModel):
    service: str
    endpoint: str
    method: str
    total_calls: int
    avg_response_ms: float
    error_rate: float

# ─── App ─────────────────────────────────────────────────────────────────────

app = FastAPI(title="CAP-Spec Gateway", version="1.0.0", description="Auth & routing gateway for CAP-Spec")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Usage stat recording middleware ─────────────────────────────────────────

@app.middleware("http")
async def record_usage(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    elapsed = (time.time() - start) * 1000

    # Skip internal paths
    if not request.url.path.startswith("/internal"):
        try:
            conn = psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
            with conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO usage_stats
                       (service, endpoint, method, status_code, response_time_ms, ip_address, created_at)
                       VALUES ('gateway', %s, %s, %s, %s, %s, NOW())""",
                    (request.url.path, request.method, response.status_code,
                     round(elapsed, 2), request.client.host if request.client else None),
                )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning(f"Stat recording failed: {e}")

    response.headers["X-Response-Time-Ms"] = str(round(elapsed, 2))
    return response


# ─── Auth Routes ──────────────────────────────────────────────────────────────

@app.post("/auth/token", response_model=Token, tags=["auth"])
async def login(form: OAuth2PasswordRequestForm = Depends(), conn=Depends(get_db)):
    """Obtain a JWT access token."""
    user = authenticate_user(conn, form.username, form.password)
    if not user:
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    token = create_token({"sub": user["username"], "role": user["role"]})
    return {"access_token": token, "token_type": "bearer",
            "role": user["role"], "username": user["username"]}


@app.post("/auth/register", response_model=UserOut, tags=["auth"])
async def register(body: UserCreate, conn=Depends(get_db),
                   _=Depends(require_role("admin"))):
    """Register a new user (admin only)."""
    with conn.cursor() as cur:
        cur.execute(
            """INSERT INTO users (username, email, hashed_password, role)
               VALUES (%s, %s, %s, %s) RETURNING *""",
            (body.username, body.email, hash_password(body.password), body.role),
        )
        user = dict(cur.fetchone())
    conn.commit()
    return {**user, "id": str(user["id"])}


@app.get("/auth/me", tags=["auth"])
async def me(user=Depends(current_user)):
    """Get current user info."""
    return {k: str(v) if k == "id" else v
            for k, v in user.items() if k != "hashed_password"}


# ─── Admin Stats Routes ───────────────────────────────────────────────────────

@app.get("/admin/stats", tags=["admin"])
async def get_stats(
    service: Optional[str] = None,
    since_hours: int = 24,
    conn=Depends(get_db),
    _=Depends(require_role("admin")),
):
    """
    Admin endpoint: aggregated usage statistics per service/endpoint.
    """
    since = datetime.utcnow() - timedelta(hours=since_hours)
    with conn.cursor() as cur:
        query = """
            SELECT
                service,
                endpoint,
                method,
                COUNT(*)                                          AS total_calls,
                ROUND(AVG(response_time_ms)::numeric, 2)         AS avg_response_ms,
                ROUND(AVG(response_time_ms)::numeric, 2)         AS p50_ms,
                ROUND(PERCENTILE_CONT(0.95) WITHIN GROUP
                      (ORDER BY response_time_ms)::numeric, 2)   AS p95_ms,
                ROUND(100.0 * SUM(CASE WHEN status_code >= 500 THEN 1 ELSE 0 END)
                      / COUNT(*)::numeric, 2)                    AS error_rate_pct
            FROM usage_stats
            WHERE created_at >= %s
            {svc_filter}
            GROUP BY service, endpoint, method
            ORDER BY total_calls DESC
        """
        if service:
            query = query.format(svc_filter="AND service = %s")
            cur.execute(query, (since, service))
        else:
            query = query.format(svc_filter="")
            cur.execute(query, (since,))

        rows = [dict(r) for r in cur.fetchall()]

    return {
        "window_hours": since_hours,
        "generated_at": datetime.utcnow().isoformat(),
        "endpoints": rows,
    }


@app.get("/admin/stats/timeseries", tags=["admin"])
async def get_stats_timeseries(
    service: Optional[str] = None,
    bucket_minutes: int = 60,
    since_hours: int = 24,
    conn=Depends(get_db),
    _=Depends(require_role("admin")),
):
    """Per-bucket call counts and latency over time."""
    since = datetime.utcnow() - timedelta(hours=since_hours)
    with conn.cursor() as cur:
        svc_filter = "AND service = %s" if service else ""
        params = [since, service] if service else [since]
        cur.execute(
            f"""
            SELECT
                date_trunc('hour', created_at)
                    + (EXTRACT(MINUTE FROM created_at)::int / {bucket_minutes}) * interval '{bucket_minutes} minutes' AS bucket,
                service,
                COUNT(*) AS calls,
                ROUND(AVG(response_time_ms)::numeric, 2) AS avg_ms
            FROM usage_stats
            WHERE created_at >= %s {svc_filter}
            GROUP BY bucket, service
            ORDER BY bucket
            """,
            params,
        )
        return {"buckets": [dict(r) for r in cur.fetchall()]}


@app.get("/admin/jobs", tags=["admin"])
async def list_jobs(
    limit: int = 50,
    status: Optional[str] = None,
    conn=Depends(get_db),
    _=Depends(require_role("admin")),
):
    """List all analysis jobs."""
    with conn.cursor() as cur:
        if status:
            cur.execute("SELECT * FROM jobs WHERE status = %s ORDER BY created_at DESC LIMIT %s",
                        (status, limit))
        else:
            cur.execute("SELECT * FROM jobs ORDER BY created_at DESC LIMIT %s", (limit,))
        return [dict(r) for r in cur.fetchall()]


# ─── Proxy Routes ─────────────────────────────────────────────────────────────

async def _proxy(request: Request, service_url: str, path: str):
    """Forward a request to a downstream service, forwarding auth header."""
    url = f"{service_url}{path}"
    headers = dict(request.headers)
    headers.pop("host", None)

    body = await request.body()

    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.request(
            method=request.method,
            url=url,
            headers=headers,
            content=body,
            params=dict(request.query_params),
        )

    return Response(
        content=resp.content,
        status_code=resp.status_code,
        headers=dict(resp.headers),
        media_type=resp.headers.get("content-type"),
    )


@app.api_route("/ingest/{path:path}", methods=["GET", "POST", "DELETE"], tags=["proxy"])
async def proxy_ingestion(path: str, request: Request, _=Depends(current_user)):
    return await _proxy(request, SERVICES["ingestion"], f"/{path}")


@app.api_route("/preprocess/{path:path}", methods=["GET", "POST"], tags=["proxy"])
async def proxy_preprocessing(path: str, request: Request, _=Depends(current_user)):
    return await _proxy(request, SERVICES["preprocessing"], f"/{path}")


@app.api_route("/features/{path:path}", methods=["GET", "POST"], tags=["proxy"])
async def proxy_features(path: str, request: Request, _=Depends(current_user)):
    return await _proxy(request, SERVICES["features"], f"/{path}")


@app.api_route("/model/{path:path}", methods=["GET", "POST"], tags=["proxy"])
async def proxy_modeling(path: str, request: Request, _=Depends(current_user)):
    return await _proxy(request, SERVICES["modeling"], f"/{path}")


@app.api_route("/report/{path:path}", methods=["GET", "POST"], tags=["proxy"])
async def proxy_reporting(path: str, request: Request, _=Depends(current_user)):
    return await _proxy(request, SERVICES["reporting"], f"/{path}")


# ─── Full pipeline trigger ────────────────────────────────────────────────────

@app.post("/analyze/{job_id}", tags=["pipeline"])
async def run_full_pipeline(job_id: str, request: Request, user=Depends(current_user)):
    """
    Trigger the full analysis pipeline for a job sequentially:
    preprocess → feature extraction → modeling → reporting.
    Returns the final report.
    """
    auth_header = request.headers.get("authorization", "")
    headers = {"Authorization": auth_header, "Content-Type": "application/json"}

    async with httpx.AsyncClient(timeout=300.0) as client:
        # Step 1 – Preprocess
        r = await client.post(f"{SERVICES['preprocessing']}/preprocess",
                              json={"job_id": job_id}, headers=headers)
        if r.status_code != 200:
            raise HTTPException(status_code=502, detail=f"Preprocessing failed: {r.text}")

        # Step 2 – Feature extraction
        r = await client.post(f"{SERVICES['features']}/extract",
                              json={"job_id": job_id}, headers=headers)
        if r.status_code != 200:
            raise HTTPException(status_code=502, detail=f"Feature extraction failed: {r.text}")

        # Step 3 – Modeling
        r = await client.post(f"{SERVICES['modeling']}/analyze",
                              json={"job_id": job_id}, headers=headers)
        if r.status_code != 200:
            raise HTTPException(status_code=502, detail=f"Modeling failed: {r.text}")

        # Step 4 – Reporting
        r = await client.post(f"{SERVICES['reporting']}/report",
                              json={"job_id": job_id}, headers=headers)
        if r.status_code != 200:
            raise HTTPException(status_code=502, detail=f"Reporting failed: {r.text}")

    return r.json()


# ─── Health ───────────────────────────────────────────────────────────────────

@app.get("/health", tags=["health"])
async def health():
    """Aggregate health check across all services."""
    results = {"gateway": "healthy"}
    async with httpx.AsyncClient(timeout=5.0) as client:
        for name, url in SERVICES.items():
            try:
                r = await client.get(f"{url}/health")
                results[name] = "healthy" if r.status_code == 200 else "degraded"
            except Exception:
                results[name] = "unreachable"
    overall = "healthy" if all(v == "healthy" for v in results.values()) else "degraded"
    return {"status": overall, "services": results, "timestamp": datetime.utcnow().isoformat()}
