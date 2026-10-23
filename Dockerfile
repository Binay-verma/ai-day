# Edge runtime image: CPU-only, no PyTorch (~400 MB). Multi-arch: linux/amd64, linux/arm64 (Raspberry Pi 5, Jetson CPU).
# Build:  docker build -t edge-copilot:1.1.0 .
FROM python:3.11.11-slim-bookworm AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system app && useradd --system --gid app --home /app --shell /usr/sbin/nologin app
WORKDIR /app
COPY pyproject.toml README.md ./
COPY edgecopilot ./edgecopilot
RUN pip install --no-cache-dir ".[mqtt]"
COPY configs ./configs
# models are mounted at runtime (./models:/app/models:ro) so images stay small and models can be updated without a rebuild
RUN mkdir -p /app/models /app/data /app/calibration /app/logs && chown -R app:app /app/data /app/calibration /app/logs
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=2).status == 200 else 1)"
ENTRYPOINT ["python", "-m", "edgecopilot.server.app"]
CMD ["--config", "configs/campus.yaml", "--host", "0.0.0.0", "--port", "8000"]
