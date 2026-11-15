# ==============================================================================
# Dockerfile: Retinal Disease Detection Using Deep Learning on OCT Images
# Automated Classification & Explainable AI for Ophthalmic OCT B-scans
# ==============================================================================

FROM python:3.10-slim-bullseye

# Metadata
LABEL maintainer="Retinal OCT AI Research Team" \
      version="1.0" \
      description="Automated Retinal Disease Detection (CNV, DME, Drusen, Normal) with Grad-CAM Explainability"

# Set environment variables:
# - Prevent Python from writing .pyc files
# - Force unbuffered stdout/stderr streaming
# - Configure Streamlit server parameters for Docker containerization
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Install essential system dependencies (curl for healthcheck, libgl for image processing)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgl1 \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Step 1: Install CPU-optimized PyTorch and TorchVision first.
# This prevents downloading ~2.5GB of CUDA binaries, reducing image size from ~6GB to ~1.8GB
# and drastically cutting container build time. (For GPU support, see DOCKER.md)
RUN pip install --no-cache-dir torch>=2.0.0 torchvision>=0.15.0 --index-url https://download.pytorch.org/whl/cpu

# Step 2: Install remaining application dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Step 3: Create dedicated non-root user for security best practices
RUN groupadd -r appuser && useradd -r -g appuser -u 1000 -m -s /bin/bash appuser

# Step 4: Copy application code, pre-trained models, sample assets, and benchmark reports
COPY src/ /app/src/
COPY assets/ /app/assets/
COPY outputs/ /app/outputs/
COPY models/ /app/models/
COPY app.py /app/app.py
COPY README.md /app/README.md

# Assign ownership to appuser
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose Streamlit default port
EXPOSE 8501

# Healthcheck to verify the web service is responding
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Launch the Streamlit web dashboard
ENTRYPOINT ["streamlit", "run", "app.py"]
CMD ["--server.port=8501", "--server.address=0.0.0.0"]
