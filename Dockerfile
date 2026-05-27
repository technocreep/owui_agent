FROM python:3.12-slim

WORKDIR /service

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

EXPOSE 8000

# Проверяет что сервис запустился И vLLM отвечает.
# --start-period=30s — даёт время на загрузку модели при первом старте.
# --interval=30s     — частота проверок после прогрева.
# --timeout=15s      — лимит на ответ vLLM (micro-запрос, должен быть быстрым).
# --retries=3        — контейнер unhealthy только после 3 неудач подряд.
HEALTHCHECK --start-period=30s --interval=30s --timeout=15s --retries=3 \
    CMD python -c "import urllib.request, sys; \
r = urllib.request.urlopen('http://localhost:8000/health', timeout=14); \
sys.exit(0 if r.status == 200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
