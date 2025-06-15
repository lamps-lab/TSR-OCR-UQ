@echo off
color 0C
echo ========================================
echo  Starting Streamlit App
echo ========================================
echo.

REM Check if calibration data exists
if not exist "data\calibration_data\calibration_scores_aps.npy" (
    echo ❌ Calibration data not found!
    echo Please run: compute_calibration.bat first
    pause
    exit /b 1
)

REM Start container if not running
docker ps | findstr table_extraction_uq_app >nul
if %errorlevel% neq 0 (
    echo Starting Docker container...
    docker-compose up -d
    timeout /t 10 >nul
)

echo Starting Streamlit app...
echo.
echo 🌐 Opening app at: http://localhost:8501
echo.
echo Press Ctrl+C to stop the app
echo.

REM Run Streamlit app
docker exec table_extraction_uq_app streamlit run /app/src/streamlit_app.py --server.port=8501 --server.address=0.0.0.0