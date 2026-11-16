# ---- AI Meter Reading API: CPU inference container (App Runner / ECS / any Docker host) ----
# Base image is a build arg: ECR Public by default (works from AWS / GitHub Actions);
# on Google Cloud Build, ECR Public rate-limits anonymous pulls, so pass
# --build-arg BASE_IMAGE=mirror.gcr.io/library/python:3.12-slim (see cloudbuild.yaml).
ARG BASE_IMAGE=public.ecr.aws/docker/library/python:3.12-slim
FROM ${BASE_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    YOLO_OFFLINE=1 YOLO_CONFIG_DIR=/tmp/ultralytics \
    OMP_NUM_THREADS=2

RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only PyTorch keeps the image ~1 GB instead of ~6 GB with CUDA
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir torch torchvision --index-url ${TORCH_INDEX}

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY models/ models/
COPY src/ src/
COPY samples/ samples/

# warm-up: loads the 3 models once at build time so a broken image fails here, not in prod
RUN python -c "import sys; sys.path.insert(0,'src'); from meter_reader import MeterReader; MeterReader()"

RUN useradd -m app && mkdir -p /app/data && chown -R app /app /tmp
USER app

EXPOSE 8080
CMD ["uvicorn", "meter_reader.api:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
