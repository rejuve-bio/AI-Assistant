# Use the official Python 3.12 slim image as the base image
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1
ENV POETRY_HTTP_TIMEOUT=600

# patch zlib1g
RUN apt-get update && apt-get upgrade -y zlib1g && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /AI-Assistant


# Create log directory here
RUN mkdir -p /AI-Assistant/logfiles

# Install Poetry (explicitly upgrade wheel to fix CVE-2026-24049)
RUN pip install --upgrade pip "wheel>=0.46.2" poetry

# Copy the application code
COPY . /AI-Assistant

RUN poetry config virtualenvs.create false && \
    poetry install --no-root --no-interaction && \
    pip uninstall -y poetry && \
    python -m pip uninstall -y pip && \
    rm -rf /root/.cache/pypoetry /root/.cache/pip
CMD gunicorn -k uvicorn.workers.UvicornWorker -w 4 --bind 0.0.0.0:${FASTAPI_PORT:-5053} --timeout 600 --log-level info run:asgi_app
