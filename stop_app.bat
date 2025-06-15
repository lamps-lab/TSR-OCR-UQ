@echo off
color 0E
echo ========================================
echo  Stopping Application
echo ========================================
echo.

echo Stopping Docker containers...
docker-compose down

if %errorlevel% equ 0 (
    echo ✅ Application stopped successfully!
) else (
    echo ❌ Error stopping application!
)

echo.
pause