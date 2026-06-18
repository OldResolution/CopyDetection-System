FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    NLTK_DATA=/opt/nltk_data \
    HF_HOME=/opt/hf_cache \
    SENTENCE_TRANSFORMERS_HOME=/opt/hf_cache

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN python - <<'PY'
import nltk
from sentence_transformers import SentenceTransformer

for package in ("punkt", "punkt_tab", "stopwords"):
    nltk.download(package, download_dir="/opt/nltk_data", quiet=True)

SentenceTransformer("all-MiniLM-L6-v2")
PY

COPY . .

EXPOSE 5001

CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5001", "src.main:app"]
