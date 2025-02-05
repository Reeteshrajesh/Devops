# Use an official Python runtime as a base image
FROM python:3.9-slim-buster

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV APP_HOME=/app

# Set work directory
WORKDIR $APP_HOME

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project requirements files
COPY requirements/base.txt requirements/base.txt
COPY requirements/prod.txt requirements/prod.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements/prod.txt

# Copy project files
COPY src/ $APP_HOME/src/
COPY scripts/train_model.py $APP_HOME/scripts/train_model.py

# Train the model during image build
RUN python scripts/train_model.py

# Copy the rest of the application
COPY . $APP_HOME

# Create a non-root user and switch to it
RUN addgroup --system app && adduser --system --ingroup app app
USER app

# Expose the port the app runs on
EXPOSE 5000

# Health check to verify container is working
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s \
  CMD curl -f http://localhost:5000/health || exit 1

# Use gunicorn as production WSGI server
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "3", "src.flask_api:app"]