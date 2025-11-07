#!/usr/bin/env python3
"""
Scheduler for automatic model training and recommendation generation
"""

import schedule
import time
import subprocess
import sys
import os
import requests
from datetime import datetime
from config import Config

def train_model():
    """Train the recommendation model"""
    print(f"[{datetime.now()}] Starting scheduled model training...")
    
    try:
        # Run training script
        result = subprocess.run([
            sys.executable, 
            os.path.join(os.path.dirname(__file__), 'train_model_fixed.py')
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"[{datetime.now()}] Model training completed successfully")
            
            # Reload models in AI server
            reload_ai_models()
            
            # Generate recommendations after training
            generate_recommendations()
        else:
            print(f"[{datetime.now()}] Model training failed: {result.stderr}")
            
    except Exception as e:
        print(f"[{datetime.now()}] Error during model training: {str(e)}")

def reload_ai_models():
    """Reload models in AI server"""
    print(f"[{datetime.now()}] Reloading models in AI server...")
    
    try:
        # Call AI server reload endpoint
        response = requests.post('http://localhost:5001/reload-models', timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                print(f"[{datetime.now()}] Models reloaded successfully in AI server")
            else:
                print(f"[{datetime.now()}] Failed to reload models: {result.get('message')}")
        else:
            print(f"[{datetime.now()}] AI server reload failed with status: {response.status_code}")
            
    except requests.exceptions.RequestException as e:
        print(f"[{datetime.now()}] Error connecting to AI server: {str(e)}")
    except Exception as e:
        print(f"[{datetime.now()}] Error during model reload: {str(e)}")

def generate_recommendations():
    """Generate recommendations for all users"""
    print(f"[{datetime.now()}] Starting scheduled recommendation generation...")
    
    try:
        # Run recommendation generation script
        result = subprocess.run([
            sys.executable, 
            os.path.join(os.path.dirname(__file__), 'train_model_fixed.py'),
            '--generate-recommendations'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"[{datetime.now()}] Recommendation generation completed successfully")
        else:
            print(f"[{datetime.now()}] Recommendation generation failed: {result.stderr}")
            
    except Exception as e:
        print(f"[{datetime.now()}] Error during recommendation generation: {str(e)}")

def main():
    """Main scheduler function"""
    print("Starting recommendation system scheduler...")
    
    # Schedule model training every 6 hours
    schedule.every(15).minutes.do(train_model)
    
    # Schedule recommendation generation every 2 hours
    schedule.every(5).minutes.do(generate_recommendations)
    
    # Run initial training
    print("Running initial model training...")
    train_model()
    
    print("Scheduler started. Press Ctrl+C to stop.")
    
    try:
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    except KeyboardInterrupt:
        print("\nScheduler stopped by user")
    except Exception as e:
        print(f"Scheduler error: {str(e)}")

if __name__ == "__main__":
    main()
