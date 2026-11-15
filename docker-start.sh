#!/usr/bin/env bash
# ==============================================================================
# Retinal Disease Detection Using Deep Learning on OCT Images
# Shell Startup Script for Linux / macOS
# ==============================================================================

set -e

echo "====================================================================="
echo " Retinal Disease Detection Using Deep Learning on OCT Images"
echo " Docker Container Startup Script"
echo "====================================================================="
echo ""

if ! command -v docker &> /dev/null; then
    echo "[ERROR] Docker is not installed or not in PATH."
    exit 1
fi

echo "[INFO] Building and starting Retinal OCT Streamlit App container..."
docker compose up -d --build

echo ""
echo "====================================================================="
echo " SUCCESS: Application is running!"
echo " Open your browser to: http://localhost:8501"
echo ""
echo " Useful commands:"
echo "   View logs:   docker compose logs -f"
echo "   Stop app:    docker compose down"
echo "====================================================================="
