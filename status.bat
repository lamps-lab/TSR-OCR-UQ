@echo off
color 0F
echo ========================================
echo  Project Status Check
echo ========================================
echo.

echo Docker Status:
docker version >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ Docker is running
    
    docker ps | findstr table_extraction_uq_app >nul
    if %errorlevel% equ 0 (
        echo ✅ Container is running
    ) else (
        echo ❌ Container is not running
    )
) else (
    echo ❌ Docker is not running
)

echo.
echo Directory Structure:
if exist "data\input_images" (
    echo ✅ data\input_images\ exists
    for %%d in (MatSci Biology CompSci ICDAR) do (
        if exist "data\input_images\%%d\images" (
            for /f %%i in ('dir /b data\input_images\%%d\images\*.* 2^>nul ^| find /c /v ""') do (
                echo   %%d: %%i images
            )
        ) else (
            echo   ❌ %%d\images\ missing
        )
    )
) else (
    echo ❌ data\input_images\ missing
)

echo.
echo Required Files:
if exist "data\domains_with_thresholds.json" (
    echo ✅ Test images JSON found
) else (
    echo ❌ domains_with_thresholds.json missing
)

if exist "src\tsr_ocr.py" (echo ✅ tsr_ocr.py) else (echo ❌ tsr_ocr.py missing)
if exist "src\utils.py" (echo ✅ utils.py) else (echo ❌ utils.py missing)
if exist "src\score_functions.py" (echo ✅ score_functions.py) else (echo ❌ score_functions.py missing)

echo.
echo Calibration Data:
if exist "data\calibration_data\calibration_scores_aps.npy" (
    echo ✅ APS calibration data computed
) else (
    echo ❌ APS calibration data not computed
    echo    Run: compute_calibration.bat
)

echo.
pause