FROM python:3.12-slim

WORKDIR /service

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

EXPOSE 8000

# Lightweight liveness check — просто убеждается что uvicorn жив.
# LLM probe выполняется один раз при старте через lifespan (см. main.py).
HEALTHCHECK --start-period=60s --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request, sys; \
r = urllib.request.urlopen('http://localhost:8000/health', timeout=4); \
sys.exit(0 if r.status == 200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
