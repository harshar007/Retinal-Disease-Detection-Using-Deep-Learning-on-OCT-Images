# 🐳 Docker Deployment Guide: Retinal OCT Disease Detection

Automated Classification & Explainable AI for Ophthalmic Optical Coherence Tomography (OCT)

---

## 📋 Overview

This repository includes a production-grade containerization setup for running the **Retinal Disease Detection Streamlit Web Application** and the **Training & Evaluation Pipeline** in isolated, reproducible Docker environments.

### Key Container Features
- **Lightweight & Fast**: Uses `python:3.10-slim-bullseye` with CPU-optimized PyTorch wheels (`--index-url https://download.pytorch.org/whl/cpu`), cutting the image size from ~6.5 GB down to ~1.8 GB and building in minutes.
- **Security Best Practices**: Runs under an unprivileged user (`appuser`, UID 1000) instead of `root`.
- **Pre-packaged & Ready-to-Run**: Includes serialized models (`models/*.pkl`), clinical evaluation assets, and diagnostic benchmark reports.
- **Dynamic Volume Mounting**: Mounts local `./models`, `./assets`, and `./outputs` directories for instant model switching without rebuilding images.
- **Built-in Healthcheck**: Automatically polls Streamlit's `/_stcore/health` endpoint.
- **Multi-Service Architecture**: Supports both the live interactive Streamlit UI and on-demand model training pipelines via Docker Compose profiles.

---

## 🛠️ Prerequisites

Ensure you have one of the following installed:
1. **Docker Desktop** (Windows / macOS): [Download Docker Desktop](https://www.docker.com/products/docker-desktop/)
   - *Windows note*: Ensure Docker Desktop is running and WSL 2 backend is enabled.
2. **Docker Engine & Docker Compose v2** (Linux):
   ```bash
   sudo apt-get update && sudo apt-get install docker.io docker-compose-plugin
   ```

---

## 🚀 Quick Start (Recommended: Docker Compose)

### 1. Start the Application
To build the image and launch the web dashboard in the background:

```bash
docker compose up -d --build
```

*(On Windows, you can also double-click [docker-start.bat](file:///P:/Retinal%20Disease%20Detection%20Using%20Deep%20Learning%20on%20OCT%20Images/docker-start.bat) or run [docker-start.ps1](file:///P:/Retinal%20Disease%20Detection%20Using%20Deep%20Learning%20on%20OCT%20Images/docker-start.ps1))*

### 2. Access the Application
Open your web browser and navigate to:
```
http://localhost:8501
```

### 3. Check Container Status & Health
```bash
docker compose ps
```
The status will indicate `Up` with `(healthy)`.

### 4. View Application Logs
```bash
docker compose logs -f retinal-oct-app
```

### 5. Stop the Application
```bash
docker compose down
```

---

## 💻 Standalone Docker CLI Usage

If you prefer using pure `docker` commands without Docker Compose:

### 1. Build the Docker Image
```bash
docker build -t retinal-oct-detection:latest .
```

### 2. Run the Container
```bash
docker run -d \
  --name retinal-oct-app \
  -p 8501:8501 \
  -v "${PWD}/models:/app/models:ro" \
  -v "${PWD}/outputs:/app/outputs:ro" \
  -v "${PWD}/assets:/app/assets:ro" \
  retinal-oct-detection:latest
```

### 3. Manage the Container
- **Stop**: `docker stop retinal-oct-app`
- **Restart**: `docker start retinal-oct-app`
- **Remove**: `docker rm -f retinal-oct-app`
- **Follow logs**: `docker logs -f retinal-oct-app`

---

## 🔄 Running the Training Pipeline in Docker

To run the end-to-end model training, activation function benchmarking, and `.pkl` model export inside an isolated container:

```bash
docker compose --profile train run --rm retinal-oct-train
```

Trained models and evaluation charts will be written directly to your host's `./models` and `./outputs` directories via volume mounts.

---

## ⚡ NVIDIA GPU Acceleration (Optional)

By default, the container runs on CPU inference (fast and portable for all environments). If you have an NVIDIA GPU and want GPU acceleration:

### Prerequisites:
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed.

### Launch with GPU Compose Override:
```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
```

---

## 📁 File Structure Reference

```
Retinal Disease Detection Using Deep Learning on OCT Images/
├── Dockerfile                  # Multi-layer, CPU-optimized, non-root container spec
├── .dockerignore               # Filters out caches, VCS, and raw .pt checkpoints
├── docker-compose.yml          # Web application & training pipeline services
├── docker-compose.gpu.yml      # NVIDIA GPU device reservation overlay
├── docker-start.bat            # One-click Windows launcher
├── docker-start.ps1            # PowerShell launcher
├── docker-start.sh             # Linux / macOS launcher
├── DOCKER.md                   # This deployment documentation
├── app.py                      # Streamlit clinical diagnostic UI
├── requirements.txt            # Python dependencies
└── models/                     # Pickled model packages & checkpoints
```

---

## ❓ Troubleshooting

### 1. `failed to connect to the docker API / daemon not running`
- **Cause**: Docker Desktop is not actively running.
- **Fix**: Open Docker Desktop from the Windows Start menu or taskbar and wait until the whale icon is steady.

### 2. `port 8501 is already in use`
- **Cause**: Another service or a local Streamlit instance is bound to port 8501.
- **Fix**: Change the port mapping in `docker-compose.yml` (e.g. `"8502:8501"`), then navigate to `http://localhost:8502`.

### 3. Out-Of-Memory (OOM) during VGG16 Inference
- **Cause**: VGG16 is a large architecture (~138M parameters).
- **Fix**: In Docker Desktop Settings → Resources, allocate at least 4 GB of RAM to WSL 2 / Docker engine.
