#!/bin/bash
# Quick Start Script for CytoSight Backend
# Run this to set up and start the backend locally

set -e  # Exit on error

echo ""
echo "=================================================="
echo "🚀 CytoSight Backend - Local Development Setup"
echo "=================================================="
echo ""

# Check Python version
echo "1️⃣  Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    if ! command -v python &> /dev/null; then
        echo "❌ Python not found. Please install Python 3.9 or higher."
        exit 1
    fi
    PYTHON_CMD=python
else
    PYTHON_CMD=python3
fi

PYTHON_VERSION=$($PYTHON_CMD --version)
echo "   ✅ Found: $PYTHON_VERSION"
echo ""

# Check if .env exists
echo "2️⃣  Checking configuration..."
if [ ! -f .env ]; then
    echo "   ⚠️  .env file not found!"
    echo ""
    echo "   📋 Please create .env with:"
    echo "      cp .env.example .env"
    echo ""
    echo "   Then fill in your Supabase credentials:"
    echo "      - SUPABASE_URL"
    echo "      - SUPABASE_KEY"
    echo "      - SUPABASE_SERVICE_KEY"
    echo "      - DATABASE_URL"
    echo ""
    echo "   Get these from: https://supabase.com/dashboard"
    exit 1
fi
echo "   ✅ .env file found"
echo ""

# Create virtual environment if it doesn't exist
echo "3️⃣  Setting up virtual environment..."
if [ ! -d venv ]; then
    echo "   Creating venv..."
    $PYTHON_CMD -m venv venv
fi
echo "   ✅ Virtual environment ready"
echo ""

# Activate virtual environment
echo "4️⃣  Activating virtual environment..."
source venv/bin/activate
echo "   ✅ Activated"
echo ""

# Install dependencies
echo "5️⃣  Installing dependencies..."
pip install -q -r requirements.txt
echo "   ✅ Dependencies installed"
echo ""

# Initialize database
echo "6️⃣  Initializing database..."
$PYTHON_CMD init_db.py
echo ""

# Start backend
echo "7️⃣  Starting backend server..."
echo ""
echo "=================================================="
echo "🎉 Backend starting on http://localhost:8000"
echo "=================================================="
echo ""
echo "📖 Documentation:       http://localhost:8000/docs"
echo "🔗 Frontend:            https://cytosight.lovable.app"
echo "📊 Swagger UI:          http://localhost:8000/docs"
echo "📝 ReDoc:               http://localhost:8000/redoc"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Start uvicorn
uvicorn app.main:app --reload --port 8000
