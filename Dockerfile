# Runtime image. Training stays in the pipeline extras; SHAP/UMAP/Evidently
# are not installed here. Pass --build-arg RUN_PIPELINE=1 only when you want
# the builder to train (not in CI).
FROM python:3.11-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
RUN useradd --create-home --uid 10001 appuser

FROM base AS deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM deps AS runtime
COPY --chown=appuser:appuser src ./src
COPY --chown=appuser:appuser app ./app
COPY --chown=appuser:appuser config ./config
COPY --chown=appuser:appuser pyproject.toml ./
COPY --chown=appuser:appuser docs ./docs
COPY --chown=appuser:appuser registry ./registry
COPY --chown=appuser:appuser artifacts ./artifacts
COPY --chown=appuser:appuser data/features ./data/features
COPY --chown=appuser:appuser reports ./reports
ENV PYTHONPATH=/app PORT=8000
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
