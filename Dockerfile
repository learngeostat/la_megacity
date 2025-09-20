FROM python:3.11-slim

# Install system dependencies for GDAL before installing Python packages
# - apt-get update: Refreshes the package list
# - libgdal-dev: Contains the development headers needed to build fiona
# - gdal-bin: Provides the `gdal-config` command-line tool
# - rm -rf ...: Cleans up the apt cache to keep the image size small
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgdal-dev gdal-bin && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
# Now that GDAL is installed, this pip install will succeed
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080

# This is commented out for local development using the script,
# but can be un-commented for deployment.
# ENV GS_NO_SIGN_REQUEST=YES

# Fixed: Use app:server and increase timeout
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--timeout", "120", "--workers", "1", "app:server"]
