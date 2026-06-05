# ── Base image ────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# ── System dependencies for OpenCV headless + ffmpeg (RTSP) ──────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgomp1 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Python dependencies (layer-cached separately from source) ─────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Pre-download the YOLOv8 nano weights so the container works offline ────────
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# ── Application source ────────────────────────────────────────────────────────
COPY src/ ./src/

# ── Runtime ───────────────────────────────────────────────────────────────────
WORKDIR /app/src
EXPOSE 5000

CMD ["python", "main.py"]
