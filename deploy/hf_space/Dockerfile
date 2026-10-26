# Hugging Face Space (Docker SDK) / any container host: the public demo on recorded footage.
# Assembled and uploaded by scripts/deploy_hf_space.py; the Space's README.md sets app_port: 7860.
FROM python:3.11.11-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 PIP_NO_CACHE_DIR=1 HOME=/home/user
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 ca-certificates \
    && rm -rf /var/lib/apt/lists/* && useradd --create-home --uid 1000 user
WORKDIR /home/user/app
COPY --chown=user pyproject.toml README.md ./
COPY --chown=user edgecopilot ./edgecopilot
RUN pip install --no-cache-dir --upgrade pip "setuptools>=83" && pip install --no-cache-dir .
COPY --chown=user configs ./configs
COPY --chown=user models ./models
COPY --chown=user demo ./demo
COPY --chown=user deploy/hf_space/start.py ./deploy/hf_space/start.py
RUN mkdir -p data calibration logs && chown -R user:user /home/user/app
USER user
EXPOSE 7860
CMD ["python", "deploy/hf_space/start.py"]
