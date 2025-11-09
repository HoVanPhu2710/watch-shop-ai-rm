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

# Global state for monitoring
training_state = {
    'is_training': False,
    'last_training_start': None,
    'last_training_end': None,
    'last_training_status': None,
    'last_training_error': None,
    'training_count': 0
}

recommendation_state = {
    'is_generating': False,
    'last_generation_start': None,
    'last_generation_end': None,
    'last_generation_status': None,
    'generation_count': 0
}

def train_model():
    """Train the recommendation model"""
    training_state['is_training'] = True
    training_state['last_training_start'] = datetime.now()
    training_state['last_training_status'] = 'running'
    training_state['last_training_error'] = None
    
    print(f"[{datetime.now()}] ========================================")
    print(f"[{datetime.now()}] 🚀 Starting scheduled model training...")
    print(f"[{datetime.now()}] Training count: {training_state['training_count'] + 1}")
    print(f"[{datetime.now()}] ========================================")
    
    try:
        # Run training script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        print(f"[{datetime.now()}] Running training script: {os.path.join(script_dir, 'train_model_fixed.py')}")
        
        result = subprocess.run([
            sys.executable, 
            os.path.join(script_dir, 'train_model_fixed.py')
        ], capture_output=True, text=True, cwd=script_dir)
        
        training_state['last_training_end'] = datetime.now()
        
        if result.returncode == 0:
            training_state['last_training_status'] = 'success'
            training_state['training_count'] += 1
            print(f"[{datetime.now()}] ✅ Model training completed successfully")
            print(f"[{datetime.now()}] Training output (first 1000 chars):")
            print(result.stdout[:1000] if result.stdout else "No stdout output")
            
            # Check if models were actually saved
            script_dir = os.path.dirname(os.path.abspath(__file__))
            model_save_path = os.path.join(os.path.dirname(script_dir), Config.MODEL_SAVE_PATH.lstrip('./'))
            model_path = os.path.join(model_save_path, 'hybrid_model')
            encoder_path = os.path.join(model_save_path, 'encoders')
            
            import os
            if os.path.exists(model_path):
                model_files = os.listdir(model_path)
                print(f"[{datetime.now()}] 📁 Model files created: {len(model_files)} files")
                print(f"[{datetime.now()}] Model files: {model_files}")
            else:
                print(f"[{datetime.now()}] ⚠️ WARNING: Model path does not exist after training: {model_path}")
            
            if os.path.exists(encoder_path):
                encoder_files = os.listdir(encoder_path)
                print(f"[{datetime.now()}] 📁 Encoder files created: {len(encoder_files)} files")
                print(f"[{datetime.now()}] Encoder files: {encoder_files}")
            else:
                print(f"[{datetime.now()}] ⚠️ WARNING: Encoder path does not exist after training: {encoder_path}")
            
            # Reload models in AI server
            print(f"[{datetime.now()}] 🔄 Reloading models in AI server...")
            reload_ai_models()
            
            # Generate recommendations after training
            print(f"[{datetime.now()}] 📊 Generating recommendations after training...")
            generate_recommendations()
        else:
            training_state['last_training_status'] = 'failed'
            training_state['last_training_error'] = result.stderr
            print(f"[{datetime.now()}] ❌ Model training failed with return code: {result.returncode}")
            print(f"[{datetime.now()}] ========== STDERR ==========")
            print(result.stderr if result.stderr else "No stderr output")
            print(f"[{datetime.now()}] ========== STDOUT ==========")
            print(result.stdout if result.stdout else "No stdout output")
            print(f"[{datetime.now()}] ===========================")
            
    except Exception as e:
        training_state['last_training_status'] = 'error'
        training_state['last_training_error'] = str(e)
        print(f"[{datetime.now()}] ❌ Error during model training: {str(e)}")
        import traceback
        print(f"[{datetime.now()}] Traceback: {traceback.format_exc()}")
    finally:
        training_state['is_training'] = False
        duration = (training_state['last_training_end'] - training_state['last_training_start']).total_seconds() if training_state['last_training_end'] else None
        print(f"[{datetime.now()}] Training duration: {duration} seconds" if duration else "")
        print(f"[{datetime.now()}] ========================================")

def reload_ai_models():
    """Reload models in AI server"""
    print(f"[{datetime.now()}] 🔄 Reloading models in AI server...")
    
    try:
        # Get AI server URL from environment variable
        ai_server_url = os.getenv('AI_SERVER_URL', 'http://localhost:5001')
        if not ai_server_url.startswith('http'):
            ai_server_url = f'http://{ai_server_url}'
        
        # Call AI server reload endpoint
        reload_url = f'{ai_server_url}/reload-models'
        print(f"[{datetime.now()}] 📡 Calling reload endpoint: {reload_url}")
        
        response = requests.post(reload_url, timeout=30)
        print(f"[{datetime.now()}] 📡 Response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                print(f"[{datetime.now()}] ✅ Models reloaded successfully in AI server")
            else:
                print(f"[{datetime.now()}] ❌ Failed to reload models: {result.get('message')}")
        else:
            print(f"[{datetime.now()}] ❌ AI server reload failed with status: {response.status_code}")
            print(f"[{datetime.now()}] Response: {response.text[:200]}")
            
    except requests.exceptions.Timeout:
        print(f"[{datetime.now()}] ⏱️ Timeout connecting to AI server (30s)")
    except requests.exceptions.ConnectionError as e:
        print(f"[{datetime.now()}] 🔌 Connection error to AI server: {str(e)}")
        print(f"[{datetime.now()}] ⚠️ AI_SERVER_URL might be incorrect or server is down")
    except requests.exceptions.RequestException as e:
        print(f"[{datetime.now()}] ❌ Error connecting to AI server: {str(e)}")
    except Exception as e:
        print(f"[{datetime.now()}] ❌ Error during model reload: {str(e)}")

def generate_recommendations():
    """Generate recommendations for all users"""
    recommendation_state['is_generating'] = True
    recommendation_state['last_generation_start'] = datetime.now()
    recommendation_state['last_generation_status'] = 'running'
    
    print(f"[{datetime.now()}] 📊 Starting scheduled recommendation generation...")
    print(f"[{datetime.now()}] Generation count: {recommendation_state['generation_count'] + 1}")
    
    try:
        # Run recommendation generation script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        result = subprocess.run([
            sys.executable, 
            os.path.join(script_dir, 'train_model_fixed.py'),
            '--generate-recommendations'
        ], capture_output=True, text=True, cwd=script_dir)
        
        recommendation_state['last_generation_end'] = datetime.now()
        
        if result.returncode == 0:
            recommendation_state['last_generation_status'] = 'success'
            recommendation_state['generation_count'] += 1
            print(f"[{datetime.now()}] ✅ Recommendation generation completed successfully")
        else:
            recommendation_state['last_generation_status'] = 'failed'
            print(f"[{datetime.now()}] ❌ Recommendation generation failed: {result.stderr[:500]}")
            
    except Exception as e:
        recommendation_state['last_generation_status'] = 'error'
        print(f"[{datetime.now()}] ❌ Error during recommendation generation: {str(e)}")
    finally:
        recommendation_state['is_generating'] = False
        duration = (recommendation_state['last_generation_end'] - recommendation_state['last_generation_start']).total_seconds() if recommendation_state['last_generation_end'] else None
        print(f"[{datetime.now()}] Generation duration: {duration} seconds" if duration else "")

def main():
    """Main scheduler function"""
    print("Starting recommendation system scheduler...")
    
    # Get intervals from environment variables
    training_interval_minutes = int(os.getenv('TRAINING_INTERVAL_MINUTES', '15'))  # Default 15 minutes
    recommendation_interval_minutes = int(os.getenv('RECOMMENDATION_INTERVAL_MINUTES', '5'))  # Default 5 minutes
    
    # Schedule model training
    if training_interval_minutes >= 60:
        schedule.every(training_interval_minutes // 60).hours.do(train_model)
    else:
        schedule.every(training_interval_minutes).minutes.do(train_model)
    
    # Schedule recommendation generation
    if recommendation_interval_minutes >= 60:
        schedule.every(recommendation_interval_minutes // 60).hours.do(generate_recommendations)
    else:
        schedule.every(recommendation_interval_minutes).minutes.do(generate_recommendations)
    
    print(f"✅ Training scheduled every {training_interval_minutes} minutes")
    print(f"✅ Recommendation generation scheduled every {recommendation_interval_minutes} minutes")
    print(f"📅 Next training: {schedule.next_run() if hasattr(schedule, 'next_run') else 'N/A'}")
    
    # Run initial training
    print("=" * 60)
    print("🚀 Running initial model training...")
    print("=" * 60)
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
