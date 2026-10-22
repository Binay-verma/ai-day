# Edge runtime image (CPU only, no PyTorch): ~350 MB. Works on x86_64 and arm64 (Raspberry Pi 5 / Jetson CPU).
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml README.md ./
COPY edgecopilot ./edgecopilot
RUN pip install --no-cache-dir .
COPY configs ./configs
COPY models/*.onnx ./models/
COPY demo ./demo
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/api/status')" || exit 1
CMD ["python", "-m", "edgecopilot.server.app", "--host", "0.0.0.0", "--port", "8000", "--source", "demo/demo_campus.mp4"]
