@echo off
REM Complete setup and start script for AI Recommendation System (Windows)

echo 🚀 Setting up AI Recommendation System...

REM Check if we're in the right directory
if not exist "ai_server.py" (
    echo ❌ Please run this script from the ai-recommend directory
    pause
    exit /b 1
)

REM Step 1: Install Python dependencies
echo 📦 Installing Python dependencies...
pip install -r requirements_ai_server.txt

REM Step 2: Check if models exist
echo 🔍 Checking for trained models...
if not exist "models\hybrid_model" (
    echo ⚠️  No trained models found. Training models now...
    echo This may take several minutes...
    python train_model_fixed.py
    
    if %errorlevel% neq 0 (
        echo ❌ Model training failed. Please check the logs.
        pause
        exit /b 1
    )
    echo ✅ Model training completed successfully
) else (
    echo ✅ Trained models found
)

REM Step 3: Set environment variables
echo 🔧 Setting up environment...
set AI_SERVER_PORT=5001
set AI_SERVER_HOST=0.0.0.0
set MODEL_SAVE_PATH=./models

REM Step 4: Start AI Server
echo 🚀 Starting AI Server on port %AI_SERVER_PORT%...
echo AI Server will run in this window
echo To stop: Close this window or Ctrl+C

echo.
echo 🎉 AI Recommendation System is ready!
echo.
echo 📡 AI Server: http://localhost:%AI_SERVER_PORT%
echo 📊 Health Check: curl http://localhost:%AI_SERVER_PORT%/health
echo 📈 Stats: curl http://localhost:%AI_SERVER_PORT%/stats
echo.
echo 🔗 Next: Start your main API server (watch-shop-be)
echo    cd ..\watch-shop-be ^&^& npm start
echo.

REM Start AI server
python ai_server.py
