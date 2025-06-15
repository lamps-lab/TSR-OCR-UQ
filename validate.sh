@echo off
color 0F
echo ========================================
echo  Setup Validation
echo ========================================
echo.

REM Check if container is running
docker ps | findstr table_extraction_uq_app >nul
if %errorlevel% neq 0 (
    echo Starting Docker container for validation...
    docker-compose up -d
    timeout /t 10 >nul
)

echo Running setup validation...
echo.

docker exec table_extraction_uq_app python3 /app/src/validate_setup.py

echo.
pause