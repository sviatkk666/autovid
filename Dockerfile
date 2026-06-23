# autovid — FastAPI dashboard + pipeline. Needs ffmpeg (montage/audio),
# headless Chromium (HTML->PNG charts/thumbnails/photo_edit) and Piper (local TTS).
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
      ffmpeg chromium fonts-liberation fonts-noto-color-emoji \
      curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Piper (local, free TTS) + two voices (auto-discovered for per-scene casting).
RUN curl -fsSL -o /tmp/piper.tar.gz \
      https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_linux_x86_64.tar.gz \
    && tar -xzf /tmp/piper.tar.gz -C /opt && rm /tmp/piper.tar.gz \
    && mkdir -p /opt/voices \
    && for v in ryan amy; do \
        curl -fsSL -o /opt/voices/en_US-$v-medium.onnx \
          "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/$v/medium/en_US-$v-medium.onnx"; \
        curl -fsSL -o /opt/voices/en_US-$v-medium.onnx.json \
          "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/$v/medium/en_US-$v-medium.onnx.json"; \
      done

ENV PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    FFMPEG_BINARY=ffmpeg \
    PIPER_BINARY=/opt/piper/piper \
    PIPER_MODEL=/opt/voices/en_US-ryan-medium.onnx

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

EXPOSE 8000
CMD ["sh", "-c", "python -m uvicorn autovid.server:app --host 0.0.0.0 --port ${PORT:-8000}"]
