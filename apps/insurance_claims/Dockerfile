# syntax=docker/dockerfile:1.7
# SOP-guided insurance claims agent: FastAPI server + static UI + fixtures + prompts.
# Secrets are NEVER baked into the image: pass OPENAI_API_KEY (and optionally STATE_ENCRYPTION_KEY)
# at runtime (compose.yaml does). Without STATE_ENCRYPTION_KEY the app generates /data/state.key.

FROM python:3.12-slim AS build
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /src
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip wheel --wheel-dir /wheels .

FROM python:3.12-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HOST=0.0.0.0 \
    PORT=8000 \
    APP_ENV=development \
    FIXTURES_DIR=/app/fixtures \
    PROMPTS_PATH=/app/prompts.toml \
    DATA_DIR=/data \
    TRACE_DIR=/data/traces
RUN groupadd --system app && useradd --system --gid app --home /app --shell /usr/sbin/nologin app
COPY --from=build /wheels /wheels
RUN pip install /wheels/*.whl && rm -rf /wheels
WORKDIR /app
COPY fixtures ./fixtures
COPY prompts.toml ./prompts.toml
RUN mkdir -p /data && chown app:app /data && chmod 700 /data
USER app
VOLUME ["/data"]
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"
CMD ["python", "-m", "insurance_claims"]
