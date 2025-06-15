@echo off
color 0A
echo ========================================
echo  Table Extraction and UQ Setup
echo ========================================
echo.

REM Check if Docker is running
echo Checking Docker...
docker version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Docker is not running or not installed!
    echo.
    echo Please:
    echo 1. Install Docker Desktop from: https://www.docker.com/products/docker-desktop/
    echo 2. Start Docker Desktop
    echo 3. Run this script again
    echo.
    pause
    exit /b 1
)
echo ✅ Docker is running

REM Create data directories
echo.
echo Creating data directories...
if not exist "data" mkdir data
if not exist "data\input_images" mkdir data\input_images
if not exist "data\calibration_data" mkdir data\calibration_data
if not exist "models_cache" mkdir models_cache

REM Create domain directories
for %%d in (MatSci Biology CompSci ICDAR) do (
    if not exist "data\input_images\%%d" mkdir data\input_images\%%d
    if not exist "data\input_images\%%d\images" mkdir data\input_images\%%d\images
)

echo ✅ Directories created

REM Check for required files
echo.
echo Checking required files...
if not exist "Dockerfile" (
    echo ❌ Missing Dockerfile
    pause
    exit /b 1
)
if not exist "docker-compose.yml" (
    echo ❌ Missing docker-compose.yml
    pause
    exit /b 1
)
if not exist "requirements.txt" (
    echo ❌ Missing requirements.txt
    pause
    exit /b 1
)
if not exist "src\tsr_ocr.py" (
    echo ❌ Missing src\tsr_ocr.py
    echo Please add your table extraction script to src\ folder
    pause
    exit /b 1
)
echo ✅ Required files found

REM Build Docker image
echo.
echo Building Docker image (this may take 5-10 minutes)...
docker-compose build

if %errorlevel% equ 0 (
    echo ✅ Docker image built successfully!
) else (
    echo ❌ Docker build failed!
    pause
    exit /b 1
)

echo.
echo ========================================
echo  🎉 Setup Complete!
echo ========================================
echo.
echo Next steps:
echo 1. Add your images to data\input_images\[domain]\images\ folders
echo 2. Add domains_test_images.json to data\ folder
echo 3. Run: validate.bat (optional)
echo 4. Run: compute_calibration.bat
echo 5. Run: run_app.bat
echo.
pause