# ==============================================================================
# Dockerfile: Retinal Disease Detection Using Deep Learning on OCT Images
# Automated Classification & Explainable AI for Ophthalmic OCT B-scans
# ==============================================================================

FROM python:3.10-slim-bookworm

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

# Step 1: Upgrade packaging tools and install CPU-optimized PyTorch and TorchVision.
# Using --extra-index-url allows pip to resolve general build tools (flit_core, etc.) from PyPI
# while fetching lightweight CPU-only PyTorch wheels instead of 2.5GB CUDA packages.
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu

# Step 2: Install remaining application dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

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
