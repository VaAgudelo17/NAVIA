FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-spa \
    tesseract-ocr-eng \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1 \
    git \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Instalar torch CPU-only primero (evita descargar la versión GPU de ~3GB)
RUN pip install --no-cache-dir \
    torch==2.5.1 \
    torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cpu

# Instalar CLIP de ultralytics antes que ultralytics (dependencia implícita de YOLO-World)
RUN pip install --no-cache-dir "git+https://github.com/ultralytics/CLIP.git"

# Instalar el resto de dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código (excluye lo que está en .dockerignore)
COPY . .

# Descargar modelo Piper TTS (es_MX-claude-high) desde HuggingFace
RUN mkdir -p models && \
    curl -L "https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_MX/claude/high/es_MX-claude-high.onnx" \
         -o models/es_MX-claude-high.onnx && \
    curl -L "https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_MX/claude/high/es_MX-claude-high.onnx.json" \
         -o models/es_MX-claude-high.onnx.json

ENV TESSERACT_CMD=/usr/bin/tesseract
ENV TTS_BACKEND=piper
ENV TTS_MODEL_NAME=es_MX-claude-high
ENV DEBUG_MODE=False
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface

EXPOSE 8000

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
