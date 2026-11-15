# =====================================================================
# Retinal Disease Detection Using Deep Learning on OCT Images
# PowerShell Docker Launcher
# =====================================================================

Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host " Retinal Disease Detection Using Deep Learning on OCT Images" -ForegroundColor Cyan
Write-Host " Docker Container Startup Script" -ForegroundColor Cyan
Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host ""

# Check Docker CLI
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Host "[ERROR] Docker CLI not found. Please install Docker Desktop." -ForegroundColor Red
    exit 1
}

Write-Host "[INFO] Building and starting Retinal OCT Streamlit App container..." -ForegroundColor Yellow
docker compose up -d --build

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "[ERROR] Failed to start Docker container. Please ensure Docker Desktop is running." -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host ""
Write-Host "=====================================================================" -ForegroundColor Green
Write-Host " SUCCESS: Application is running!" -ForegroundColor Green
Write-Host " Open your browser to: http://localhost:8502" -ForegroundColor Green
Write-Host ""
Write-Host " Commands:" -ForegroundColor Gray
Write-Host "   View logs:   docker compose logs -f" -ForegroundColor Gray
Write-Host "   Stop app:    docker compose down" -ForegroundColor Gray
Write-Host "=====================================================================" -ForegroundColor Green
