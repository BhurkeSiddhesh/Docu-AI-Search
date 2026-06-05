FROM python:3.11-slim

WORKDIR /app

RUN adduser --disabled-password --gecos "" --uid 1000 appuser && \
    mkdir -p data models && \
    chown -R appuser:appuser /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=appuser:appuser . .

USER appuser

EXPOSE 8000

CMD ["uvicorn", "backend.api:app", "--host", "0.0.0.0", "--port", "8000"]
