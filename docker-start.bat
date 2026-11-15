@echo off
TITLE Retinal OCT AI - Docker Launcher
echo =====================================================================
echo  Retinal Disease Detection Using Deep Learning on OCT Images
echo  Docker Container Startup Script
echo =====================================================================
echo.

REM Check if Docker is installed
where docker >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Docker CLI not found. Please install Docker Desktop from https://www.docker.com/products/docker-desktop
    pause
    exit /b 1
)

echo [INFO] Building and starting Retinal OCT Streamlit App container...
echo.
docker compose up -d --build

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Failed to start Docker container. Please verify Docker Desktop is running.
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo =====================================================================
echo  SUCCESS: Application is running!
echo  Open your web browser and navigate to:
echo     http://localhost:8501
echo.
echo  To view container logs:
echo     docker compose logs -f
echo.
echo  To stop the application:
echo     docker compose down
echo =====================================================================
echo.
pause
