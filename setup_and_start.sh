#!/bin/bash

# Complete setup and start script for AI Recommendation System

echo "🚀 Setting up AI Recommendation System..."

# Check if we're in the right directory
if [ ! -f "ai_server.py" ]; then
    echo "❌ Please run this script from the ai-recommend directory"
    exit 1
fi

# Step 1: Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install -r requirements_ai_server.txt

# Step 2: Check if models exist
echo "🔍 Checking for trained models..."
if [ ! -d "models/hybrid_model" ]; then
    echo "⚠️  No trained models found. Training models now..."
    echo "This may take several minutes..."
    python train_model_fixed.py
    
    if [ $? -eq 0 ]; then
        echo "✅ Model training completed successfully"
    else
        echo "❌ Model training failed. Please check the logs."
        exit 1
    fi
else
    echo "✅ Trained models found"
fi

# Step 3: Set environment variables
echo "🔧 Setting up environment..."
export AI_SERVER_PORT=${AI_SERVER_PORT:-5001}
export AI_SERVER_HOST=${AI_SERVER_HOST:-0.0.0.0}
export MODEL_SAVE_PATH=${MODEL_SAVE_PATH:-./models}

# Step 4: Start AI Server
echo "🚀 Starting AI Server on port $AI_SERVER_PORT..."
echo "AI Server will run in the background"
echo "To stop: kill the process or Ctrl+C"

# Start AI server in background
nohup python ai_server.py > ai_server.log 2>&1 &
AI_PID=$!

echo "✅ AI Server started with PID: $AI_PID"
echo "📝 Logs are being written to ai_server.log"

# Wait a moment for server to start
sleep 3

# Test if server is running
echo "🧪 Testing AI Server..."
if curl -s http://localhost:$AI_SERVER_PORT/health > /dev/null; then
    echo "✅ AI Server is running and healthy"
else
    echo "❌ AI Server failed to start. Check ai_server.log for details"
    exit 1
fi

echo ""
echo "🎉 AI Recommendation System is ready!"
echo ""
echo "📡 AI Server: http://localhost:$AI_SERVER_PORT"
echo "📊 Health Check: curl http://localhost:$AI_SERVER_PORT/health"
echo "📈 Stats: curl http://localhost:$AI_SERVER_PORT/stats"
echo ""
echo "🔗 Next: Start your main API server (watch-shop-be)"
echo "   cd ../watch-shop-be && npm start"
echo ""
echo "📝 To view logs: tail -f ai_server.log"
echo "🛑 To stop AI server: kill $AI_PID"
