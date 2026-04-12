FROM python:3.11-slim

# Optimized for fast startup and minimal image size (~150MB)
# Designed for vCPU=2, memory=8GB constraints

WORKDIR /app

COPY requirements.txt /app/
RUN pip install --no-cache-dir --disable-pip-version-check -r requirements.txt \
    && rm -rf /root/.cache/pip

COPY . /app/

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')" || exit 1

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
