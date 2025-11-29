# --- builder ---
FROM python:3.11-slim as builder
WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./

# create and use a venv
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY main.py ./
COPY .env /
# --- runtime ---
FROM python:3.11-slim
WORKDIR /app

# create non-root user
RUN useradd --create-home --shell /bin/bash appuser


COPY --from=builder --chown=appuser:appuser /opt/venv /opt/venv
COPY --from=builder --chown=appuser:appuser /app/src ./src
COPY --from=builder --chown=appuser:appuser /app/main.py ./main.py
RUN mkdir -p /app/hf_cache && chown -R appuser:appuser /app/hf_cache
RUN mkdir -p /app/chroma_db && chown -R appuser:appuser /app/chroma_db
# set environment
ENV PATH="/opt/venv/bin:${PATH}"
ENV PYTHONUNBUFFERED=1

USER appuser

EXPOSE 8000

CMD ["python","main.py"]