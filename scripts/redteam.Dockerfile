# Dependencies only. Application snapshots are mounted read-only at runtime.
FROM mcr.microsoft.com/devcontainers/python:2-3.12-trixie
COPY requirements-redteam.txt /tmp/requirements-redteam.txt
RUN python -m pip install --no-cache-dir --upgrade "pip>=26.2,<27"
RUN python -m pip install --no-cache-dir -r /tmp/requirements-redteam.txt
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HOME=/tmp
WORKDIR /app
USER 65534:65534
