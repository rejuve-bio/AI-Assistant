# Use the official Python 3.11 slim image as the base image
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV POETRY_HTTP_TIMEOUT=600

# Set the working directory
WORKDIR /AI-Assistant


# Create log directory here
RUN mkdir -p /AI-Assistant/logfiles

# Install Poetry (explicitly upgrade wheel to fix CVE-2026-24049)
RUN pip install --upgrade pip "wheel>=0.46.2" poetry

# Copy the application code
COPY . /AI-Assistant

# Install dependencies 
RUN poetry config virtualenvs.create false && \
    poetry install --no-root --no-interaction

# Run the application
CMD ["gunicorn", "-w", "4", "--bind", "0.0.0.0:$FLASK_PORT", "run:app"]
