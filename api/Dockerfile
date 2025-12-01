FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV VENV_PATH=/home/appuser/venv
ENV PATH="${VENV_PATH}/bin:${PATH}"

# Install system deps needed for building wheels and venv
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential python3-venv ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

# Create an unprivileged user (home: /home/appuser)
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /home/appuser/app

# Copy requirements first (cacheable)
COPY requirements.txt ./

# Copy application files and .env; set ownership at copy-time (fast)
# NOTE: COPY --chown requires a relatively recent Docker; it's faster than chown -R
COPY --chown=appuser:appuser src ./src
COPY --chown=appuser:appuser main.py ./
COPY --chown=appuser:appuser .env ./


RUN mkdir -p /home/appuser/app/hf_cache && chown -R appuser:appuser /home/appuser/app/hf_cache
RUN mkdir -p /home/appuser/app/chroma_db && chown -R appuser:appuser /home/appuser/app/chroma_db

USER appuser

# Create venv and install Python deps
RUN python -m venv ${VENV_PATH} \
 && ${VENV_PATH}/bin/pip install --upgrade pip setuptools wheel \
 && ${VENV_PATH}/bin/pip install --no-cache-dir -r requirements.txt


# Expose port and run
EXPOSE 8000

# Default command (keeps your existing main.py behavior)
CMD ["python", "main.py"]