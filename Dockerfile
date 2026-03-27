# Podcast Generator API (FastAPI + Chatterbox TTS). CPU PyTorch by default; use NVIDIA runtime for GPU.
FROM python:3.12-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md ./
COPY podcast_generator ./podcast_generator/

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir ".[tts]"

EXPOSE 8000

CMD ["uvicorn", "podcast_generator.main:app", "--host", "0.0.0.0", "--port", "8000"]
