FROM python:3.11-slim

# Set the working directory early
WORKDIR /app

# --- Dependency Layers ---
# These layers will be cached unless system dependencies or requirements.txt change.

# 1. Install system dependencies for GDAL
# This layer is very stable and will almost always be cached.
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgdal-dev gdal-bin && \
    rm -rf /var/lib/apt/lists/*

# 2. Copy only the requirements file first
COPY requirements.txt .

# 3. Install Python packages
# This layer will only be rebuilt if requirements.txt changes.
RUN pip install --no-cache-dir -r requirements.txt


# --- Application Layer ---
# This is the only layer that will be rebuilt when you change your code.
# Since it's just a copy operation, it's extremely fast.
COPY . .


# --- Final Configuration ---
EXPOSE 8080

# Commented out for local dev, un-comment for deployment
# ENV GS_NO_SIGN_REQUEST=YES

CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--timeout", "120", "--workers", "1", "app:server"]

