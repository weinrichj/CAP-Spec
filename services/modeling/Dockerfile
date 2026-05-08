FROM python:3.11-slim
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends gcc libpq-dev curl     && rm -rf /var/lib/apt/lists/*

COPY modeling/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# analysis package → /app/analysis/  (importable as 'analysis.xxx')
COPY cap_analysis/          /app/analysis/
# configs (species_windows.csv etc) → /app/configs/
COPY cap_analysis_configs/  /app/configs/
# shims for plots.style and data_ingestion.scoped_writes
# these live at /app/plots/ and /app/data_ingestion/ so Python
# finds them when analysis code does 'from plots.style import ...'
COPY shims/                 /app/

COPY modeling/main.py main.py

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3     CMD curl -f http://localhost:${PORT:-8000}/health || exit 1

CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 2
