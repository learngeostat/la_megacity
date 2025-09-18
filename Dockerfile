# Use an official Python runtime as a parent image

FROM python:3.11-slim
# Set environment variables

ENV APP_HOME /app

ENV PYTHONUNBUFFERED True

ENV PORT 8080
# Create and set the working directory
WORKDIR $APP_HOME
# Install system dependencies required for geospatial libraries like geopandas
RUN apt-get update && apt-get install -y \
    build-essential \
    libgdal-dev \
    gdal-bin \
    libproj-dev \
    libgeos-dev \
    libspatialite-dev \
    --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*
# Copy requirements file first (for better Docker layer caching)
COPY requirements.txt .
# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
# Copy the application code into the container
COPY . .
# Expose the port
EXPOSE $PORT
# Set the command to run the application using Gunicorn
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app:app.server
