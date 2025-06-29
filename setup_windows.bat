@echo off
echo ========================================
echo    NeuroVandick Windows Setup
echo ========================================
echo.

echo Checking Python installation...
python --version
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

echo.
echo Installing/updating dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)

echo.
echo Running Windows setup script...
python setup_windows.py
if errorlevel 1 (
    echo ERROR: Setup script failed
    pause
    exit /b 1
)

echo.
echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo IMPORTANT: You need to increase your paging file size:
echo 1. Press Win + Pause/Break
echo 2. Click "Advanced system settings"
echo 3. Under Performance, click "Settings"
echo 4. Click "Advanced" tab
echo 5. Under Virtual memory, click "Change"
echo 6. Uncheck "Automatically manage paging file size"
echo 7. Select your system drive (usually C:)
echo 8. Choose "Custom size"
echo 9. Set Initial size to 16384 MB (16 GB)
echo 10. Set Maximum size to 32768 MB (32 GB)
echo 11. Click "Set" then "OK"
echo 12. Restart your computer
echo.
echo After restart, you can run:
echo   python src/main_memory_optimized.py
echo.
pause 