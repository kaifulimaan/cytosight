@echo off
REM Quick Start Script for CytoSight Backend (Windows)
REM Run this to set up and start the backend locally

setlocal enabledelayedexpansion

echo.
echo ==================================================
echo 🚀 CytoSight Backend - Local Development Setup
echo ==================================================
echo.

REM Check Python version
echo 1️⃣  Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found. Please install Python 3.9 or higher.
    echo    Download from: https://www.python.org/downloads/
    pause
    exit /b 1
)
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo    ✅ Found: Python %PYTHON_VERSION%
echo.

REM Check if .env exists
echo 2️⃣  Checking configuration...
if not exist .env (
    echo    ⚠️  .env file not found!
    echo.
    echo    📋 Please create .env with:
    echo       copy .env.example .env
    echo.
    echo    Then fill in your Supabase credentials:
    echo       - SUPABASE_URL
    echo       - SUPABASE_KEY
    echo       - SUPABASE_SERVICE_KEY
    echo       - DATABASE_URL
    echo.
    echo    Get these from: https://supabase.com/dashboard
    pause
    exit /b 1
)
echo    ✅ .env file found
echo.

REM Create virtual environment if it doesn't exist
echo 3️⃣  Setting up virtual environment...
if not exist venv (
    echo    Creating venv...
    python -m venv venv
)
echo    ✅ Virtual environment ready
echo.

REM Activate virtual environment
echo 4️⃣  Activating virtual environment...
call venv\Scripts\activate.bat
echo    ✅ Activated
echo.

REM Install dependencies
echo 5️⃣  Installing dependencies...
pip install -q -r requirements.txt
if errorlevel 1 (
    echo    ❌ Failed to install dependencies
    pause
    exit /b 1
)
echo    ✅ Dependencies installed
echo.

REM Initialize database
echo 6️⃣  Initializing database...
python init_db.py
if errorlevel 1 (
    echo    ⚠️  Database initialization had issues - continue anyway
)
echo.

REM Start backend
echo 7️⃣  Starting backend server...
echo.
echo ==================================================
echo 🎉 Backend starting on http://localhost:8000
echo ==================================================
echo.
echo 📖 Documentation:       http://localhost:8000/docs
echo 🔗 Frontend:            https://cytosight.lovable.app
echo 📊 Swagger UI:          http://localhost:8000/docs
echo 📝 ReDoc:               http://localhost:8000/redoc
echo.
echo Press Ctrl+C to stop
echo.

REM Start uvicorn
uvicorn app.main:app --reload --port 8000
