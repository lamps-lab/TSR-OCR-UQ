@echo off
color 0B
echo ========================================
echo  Computing Calibration Data
echo ========================================
echo.

REM Check if container exists, if not start it
docker ps -a | findstr table_extraction_uq_app >nul
if %errorlevel% neq 0 (
    echo Starting Docker container...
    docker-compose up -d
    timeout /t 10 >nul
) else (
    REM Check if container is running
    docker ps | findstr table_extraction_uq_app >nul
    if %errorlevel% neq 0 (
        echo Starting Docker container...
        docker-compose start
        timeout /t 5 >nul
    )
)

REM Run validation first
echo Running setup validation...
docker exec table_extraction_uq_app python3 /app/src/validate_setup.py

echo.
set /p continue="Continue with calibration computation? (y/n): "
if /i not "%continue%"=="y" (
    echo Calibration computation cancelled.
    pause
    exit /b 0
)

echo.
echo Computing APS conformal calibration data...
echo This may take 10-30 minutes depending on the number of images...
echo.

REM Run calibration computation
docker exec table_extraction_uq_app python3 /app/src/compute_calibration_data.py

if %errorlevel% equ 0 (
    echo.
    echo ✅ Calibration computation completed!
    echo.
    echo Generated files:
    echo   - data/calibration_data/calibration_scores_aps.npy
    echo   - data/calibration_data/calibration_metadata.json
    echo.
    echo You can now run the Streamlit app: run_app.bat
) else (
    echo ❌ Calibration computation failed!
    echo Check the error messages above.
)

echo.
pause