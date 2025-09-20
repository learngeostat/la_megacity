#!/bin/bash

# ==============================================================================
# Local Development Script for Dockerized GCS Application
#
# This script automates the process of building and running the Docker container
# locally for development. It handles:
#   1. Setting application-specific names.
#   2. Cleaning up old containers.
#   3. Building a fresh Docker image.
#   4. Running the new container with local Google Cloud credentials mounted
#      for live GCS access.
#
# PRE-REQUISITE:
# You must run this command ONCE in your terminal before using this script:
#   gcloud auth application-default login
# ==============================================================================

# --- Configuration ---
# Set the name for your Docker image and the running container.
IMAGE_NAME="la-megacity-phase4d"
CONTAINER_NAME="la-megacity-phase4d-dev"

# --- Main Script Logic ---

# Use set -e to exit immediately if any command fails.
set -e

echo "--- Starting local development workflow for $IMAGE_NAME ---"

# 1. Check for gcloud credentials
# Determine the correct path for gcloud credentials based on the OS.
if [[ "$OSTYPE" == "darwin"* || "$OSTYPE" == "linux-gnu"* ]]; then
    GCLOUD_CONFIG_PATH="$HOME/.config/gcloud"
elif [[ "$OSTYPE" == "msys"* || "$OSTYPE" == "cygwin"* ]]; then
    GCLOUD_CONFIG_PATH="$APPDATA/gcloud"
else
    echo "Unsupported OS: $OSTYPE. Please set GCLOUD_CONFIG_PATH manually."
    exit 1
fi

if [ ! -d "$GCLOUD_CONFIG_PATH" ]; then
    echo "ERROR: Google Cloud credentials not found at $GCLOUD_CONFIG_PATH"
    echo "Please run 'gcloud auth application-default login' first."
    exit 1
fi

echo "Found gcloud credentials at: $GCLOUD_CONFIG_PATH"


# 2. Stop and remove any existing container with the same name.
# The '|| true' part ensures that the script doesn't fail if the container doesn't exist.
echo "--- Cleaning up old containers (if any) ---"
docker stop $CONTAINER_NAME || true
docker rm $CONTAINER_NAME || true


# 3. Build the Docker image from the current directory.
echo "--- Building Docker image: $IMAGE_NAME ---"
docker build -t $IMAGE_NAME .


# 4. Run the new Docker container.
#    - Mounts the gcloud credentials from your local machine.
#    - Sets the GOOGLE_APPLICATION_CREDENTIALS environment variable.
#    - Maps port 8080 to access the application.
echo "--- Running new container: $CONTAINER_NAME ---"
echo "Access the application at http://localhost:8080"
docker run --rm -it \
  --name $CONTAINER_NAME \
  -p 8080:8080 \
  -e GOOGLE_APPLICATION_CREDENTIALS=/tmp/keys/application_default_credentials.json \
  -v "$GCLOUD_CONFIG_PATH":/tmp/keys \
  $IMAGE_NAME

echo "--- Container exited. Local development session ended. ---"
