# Use the official Python 3.10 slim image as the base image
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1
ENV POETRY_HTTP_TIMEOUT=1200
ENV PIP_DEFAULT_TIMEOUT=1200
ENV PIP_READ_TIMEOUT=1200

# Set the working directory
WORKDIR /AI-Assistant

# Install Poetry
RUN pip install poetry

# Copy the application code
COPY . /AI-Assistant

# Install dependencies with retry logic
RUN poetry config virtualenvs.create false && \
    poetry config installer.max-workers 1 && \
    poetry install --no-root --no-interaction || \
    (echo "First attempt failed, retrying..." && sleep 10 && poetry install --no-root --no-interaction)

# Run the application
CMD ["gunicorn", "-w", "4", "--bind", "0.0.0.0:$FLASK_PORT", "run:app"]
# CMD ["poetry", "run", "gunicorn", "-w", "4", "--bind", "0.0.0.0:$FLASK_PORT", "run:app"]
