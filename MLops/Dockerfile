# Use an official Python runtime as a parent image
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

# Copy entire project
COPY . $APP_HOME/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements/prod.txt

# Create a non-root user
RUN addgroup --system app && adduser --system --ingroup app app
USER app

# Expose the port the app runs on
EXPOSE 5000

# Use gunicorn as production WSGI server
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "3", "src.flask_api:app"]
