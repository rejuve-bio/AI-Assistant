# Use the official Python 3.10 slim image as the base image
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1

# Set the working directory
WORKDIR /AI-Assistant

# Install Poetry
RUN pip install poetry

# Copy only the dependency files first
COPY pyproject.toml poetry.lock /AI-Assistant/

# Install dependencies
RUN poetry config virtualenvs.create false && poetry install --no-root

# Copy the application code
COPY . /AI-Assistant

# Run the application
CMD ["gunicorn", "-w", "4", "--bind", "0.0.0.0:5001", "run:app"]
