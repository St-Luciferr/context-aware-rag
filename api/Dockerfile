# Stage 1: Builder - Install dependencies
FROM python:3.11-slim AS builder

ENV PYTHONUNBUFFERED=1
ENV VENV_PATH=/opt/venv
ENV PATH="${VENV_PATH}/bin:${PATH}"

# Install build dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv ${VENV_PATH}

# Copy requirements
COPY requirements.txt ./

# Install CPU-only PyTorch first (much smaller than CUDA version)
# Then install other dependencies
RUN ${VENV_PATH}/bin/pip install --no-cache-dir --upgrade pip setuptools wheel \
    && ${VENV_PATH}/bin/pip install --no-cache-dir \
       torch --index-url https://download.pytorch.org/whl/cpu \
    && ${VENV_PATH}/bin/pip install --no-cache-dir -r requirements.txt

# Stage 2: Runtime - Minimal production image
FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1
ENV VENV_PATH=/opt/venv
ENV PATH="${VENV_PATH}/bin:${PATH}"
ENV PYTHONDONTWRITEBYTECODE=1

# Install minimal runtime dependencies only
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create unprivileged user
RUN useradd --create-home --shell /bin/bash appuser

# Copy virtual environment from builder
COPY --from=builder ${VENV_PATH} ${VENV_PATH}

WORKDIR /home/appuser/app

# Create necessary directories
RUN mkdir -p hf_cache chroma_db data \
    && chown -R appuser:appuser /home/appuser

# Copy application files
COPY --chown=appuser:appuser src ./src
COPY --chown=appuser:appuser main.py ./
COPY --chown=appuser:appuser data ./data

# Copy .env if exists
COPY --chown=appuser:appuser .env* ./

USER appuser

EXPOSE 8000

CMD ["python", "main.py"]
