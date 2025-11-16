#!/usr/bin/env python3
"""
Training script for the hybrid recommendation system - Fixed version
"""

import os
import sys
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from database import db
from data_processor_fixed import DataProcessor
from hybrid_model import HybridRecommendationModel
from config import Config

def main():
    """Main training function"""
    import sys
    
    print("=" * 80)
    print("Starting hybrid recommendation model training...")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Script location: {os.path.abspath(__file__)}")
    print(f"MODEL_SAVE_PATH from config: {Config.MODEL_SAVE_PATH}")
    print("=" * 80)
    sys.stdout.flush()
    
    start_time = time.time()
    
    try:
        # Initialize data processor
        print("Initializing data processor...")
        processor = DataProcessor()
        
        # Load and preprocess data
        print("Loading interaction data...")
        interactions_df = processor.load_interaction_data()
        
        if len(interactions_df) == 0:
            print("[ERROR] No interaction data found. Exiting...")
            print("Please ensure database has interaction data.")
            sys.stdout.flush()
            return
        
        print(f"[OK] Loaded {len(interactions_df)} interactions")
        sys.stdout.flush()
        
        print("Loading user features...")
        user_features_df = processor.load_user_features()
        
        print("Loading item features...")
        item_features_df = processor.load_item_features()
        
        # Preprocess data
        print("Preprocessing interaction data...")
        interactions_processed = processor.preprocess_interactions(interactions_df)
        
        print("Preprocessing user features...")
        user_features_processed = processor.preprocess_user_features(user_features_df)
        
        print("Preprocessing item features...")
        item_features_processed = processor.preprocess_item_features(item_features_df)
        
        # Get dimensions (use engineered numeric feature columns only)
        n_users = len(processor.user_encoder.classes_)
        n_items = len(processor.item_encoder.classes_)

        user_feature_cols = [
            'age_group_encoded',
            'gender_preference_encoded',
            'price_range_encoded',
            'brand_preferences_count',
            'category_preferences_count',
            'style_preferences_count',
        ]
        item_feature_cols = [
            'price_tier_encoded',
            'gender_target_encoded',
            'size_category_encoded',
            'style_tags_count',
            'material_tags_count',
            'color_tags_count',
            'movement_type_tags_count',
            'price_normalized',
            'rating_normalized',
        ]

        user_feature_dim = len(user_feature_cols)
        item_feature_dim = len(item_feature_cols)
        
        print(f"Data dimensions: {n_users} users, {n_items} items")
        print(f"User features: {user_feature_dim}, Item features: {item_feature_dim}")
        
        # Initialize hybrid model
        hybrid_model = HybridRecommendationModel(
            n_users=n_users,
            n_items=n_items,
            user_feature_dim=user_feature_dim,
            item_feature_dim=item_feature_dim
        )
        
        # Save the model - resolve absolute path FIRST (before training)
        if not os.path.isabs(Config.MODEL_SAVE_PATH):
            # If relative path, make it relative to the script directory
            script_dir = os.path.dirname(os.path.abspath(__file__))
            base_dir = os.path.dirname(script_dir)  # Go up one level from src/
            model_save_path = os.path.join(base_dir, Config.MODEL_SAVE_PATH.lstrip('./'))
        else:
            model_save_path = Config.MODEL_SAVE_PATH
        
        # Ensure directory exists
        os.makedirs(model_save_path, exist_ok=True)
        print(f"Model save path: {model_save_path}")
        
        # Train the model
        print("Training hybrid model...")
        
        training_histories = hybrid_model.train(
            interactions_processed,
            user_features_processed,
            item_features_processed
        )
        
        # Get training durations from hybrid_model.train return value
        cf_duration = training_histories.get('cf_duration_seconds', 0)
        cbf_duration = training_histories.get('cbf_duration_seconds', 0)
        hybrid_duration = int(time.time() - start_time)  # Total time
        
        # Count model parameters
        cf_params = hybrid_model.collaborative_model.model.count_params() if hybrid_model.collaborative_model.model else 0
        cbf_params = hybrid_model.content_based_model.model.count_params() if hybrid_model.content_based_model.model else 0
        hybrid_params = cf_params + cbf_params  # Hybrid combines both
        
        # Calculate convergence epochs (epoch where val_loss stops improving significantly)
        def find_convergence_epoch(val_losses, patience=5):
            """Find epoch where validation loss stops improving"""
            if len(val_losses) < patience + 1:
                return len(val_losses)
            best_loss = min(val_losses)
            best_epoch = val_losses.index(best_loss)
            # Check if loss doesn't improve for 'patience' epochs after best
            for i in range(best_epoch + patience, len(val_losses)):
                if val_losses[i] < best_loss * 1.01:  # Within 1% of best
                    return i
            return min(best_epoch + patience, len(val_losses))
        
        cf_val_losses = training_histories['cf_history'].history.get('val_loss', [])
        cbf_val_losses = training_histories['cbf_history'].history.get('val_loss', [])
        cf_convergence = find_convergence_epoch(cf_val_losses) if cf_val_losses else len(cf_val_losses)
        cbf_convergence = find_convergence_epoch(cbf_val_losses) if cbf_val_losses else len(cbf_val_losses)
        # Hybrid convergence is max of both (both need to converge)
        hybrid_convergence = max(cf_convergence, cbf_convergence)
        
        # Save training histories for plotting
        import json
        history_save_path = os.path.join(model_save_path, 'training_histories.json')
        histories_data = {
            'cf': {
                'loss': training_histories['cf_history'].history.get('loss', []),
                'val_loss': training_histories['cf_history'].history.get('val_loss', []),
                'mae': training_histories['cf_history'].history.get('mae', []),
                'training_duration_seconds': cf_duration,
                'convergence_epoch': cf_convergence,
                'num_parameters': cf_params
            },
            'cbf': {
                'loss': training_histories['cbf_history'].history.get('loss', []),
                'val_loss': training_histories['cbf_history'].history.get('val_loss', []),
                'mae': training_histories['cbf_history'].history.get('mae', []),
                'training_duration_seconds': cbf_duration,
                'convergence_epoch': cbf_convergence,
                'num_parameters': cbf_params
            },
            'hybrid': {
                'training_duration_seconds': hybrid_duration,
                'convergence_epoch': hybrid_convergence,
                'num_parameters': hybrid_params
            }
        }
        with open(history_save_path, 'w') as f:
            json.dump(histories_data, f, indent=2)
        print(f"[OK] Training histories saved to: {history_save_path}")
        
        # Save the model
        print(f"Saving models to: {model_save_path}")
        
        model_path = os.path.join(model_save_path, 'hybrid_model')
        print(f"Model path: {model_path}")
        os.makedirs(model_path, exist_ok=True)
        
        print("Saving hybrid model...")
        hybrid_model.save_models(model_path)
        print(f"[OK] Hybrid model saved to: {model_path}")
        
        # Save encoders for later use
        encoder_path = os.path.join(model_save_path, 'encoders')
        os.makedirs(encoder_path, exist_ok=True)
        print(f"Encoder path: {encoder_path}")
        
        import joblib
        print("Saving encoders...")
        joblib.dump(processor.user_encoder, os.path.join(encoder_path, 'user_encoder.pkl'))
        print("[OK] user_encoder.pkl saved")
        joblib.dump(processor.item_encoder, os.path.join(encoder_path, 'item_encoder.pkl'))
        print("[OK] item_encoder.pkl saved")
        joblib.dump(processor.scaler, os.path.join(encoder_path, 'scaler.pkl'))
        print("[OK] scaler.pkl saved")
        
        # Verify files were saved
        model_files = os.listdir(model_path) if os.path.exists(model_path) else []
        encoder_files = os.listdir(encoder_path) if os.path.exists(encoder_path) else []
        print(f"[OK] Model files saved: {len(model_files)} files")
        print(f"[OK] Encoder files saved: {len(encoder_files)} files")
        
        # Calculate training duration
        training_duration = int(time.time() - start_time)
        
        # Save training history to database
        save_training_history(
            model_type='hybrid',
            algorithm_version='1.0',
            training_data_size=len(interactions_processed),
            training_duration_seconds=training_duration,
            model_file_path=model_path
        )
        
        print(f"Training completed successfully in {training_duration} seconds")
        print(f"Model saved to: {model_path}")
        print(f"Encoders saved to: {encoder_path}")
        sys.stdout.flush()
        
        # Generate training plots after successful training
        print("\n" + "="*80)
        print("Generating training plots...")
        print("="*80)
        generate_training_plots()
        
    except Exception as e:
        print(f"Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        sys.exit(1)
    
    finally:
        db.close()

def save_training_history(model_type, algorithm_version, training_data_size, 
                         training_duration_seconds, model_file_path):
    """Save training history to database (optional - table may not exist)"""
    try:
        # Check if table exists first
        check_query = """
        SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name = 'model_training_history'
        )
        """
        table_exists = db.execute_query(check_query)
        
        if len(table_exists) == 0 or not table_exists.iloc[0][0]:
            # Table doesn't exist, skip saving history (not critical)
            return
        
        query = """
        INSERT INTO model_training_history 
        (model_type, algorithm_version, training_data_size, training_duration_seconds, model_file_path, created_at)
        VALUES (%s, %s, %s, %s, %s, %s)
        """
        
        params = (
            model_type,
            algorithm_version,
            training_data_size,
            training_duration_seconds,
            model_file_path,
            datetime.now()
        )
        
        db.execute_insert(query, params)
        print("Training history saved to database")
        
    except Exception as e:
        # Silently ignore - training history is optional
        pass

def generate_recommendations():
    """Generate recommendations for all users and save to database - Fixed version"""
    print("Generating recommendations...")
    
    try:
        # Resolve model path - same logic as training
        if not os.path.isabs(Config.MODEL_SAVE_PATH):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            base_dir = os.path.dirname(script_dir)
            model_save_path = os.path.join(base_dir, Config.MODEL_SAVE_PATH.lstrip('./'))
        else:
            model_save_path = Config.MODEL_SAVE_PATH
        
        # Load the trained model
        model_path = os.path.join(model_save_path, 'hybrid_model')
        print(f"Loading model from: {model_path}")
        hybrid_model = HybridRecommendationModel(1, 1, 1, 1)  # Dummy dimensions
        hybrid_model.load_models(model_path)
        
        # Load encoders
        encoder_path = os.path.join(model_save_path, 'encoders')
        print(f"Loading encoders from: {encoder_path}")
        import joblib
        user_encoder = joblib.load(os.path.join(encoder_path, 'user_encoder.pkl'))
        item_encoder = joblib.load(os.path.join(encoder_path, 'item_encoder.pkl'))
        
        # Load user and item features
        processor = DataProcessor()
        user_features_df = processor.load_user_features()
        item_features_df = processor.load_item_features()
        
        user_features_processed = processor.preprocess_user_features(user_features_df)
        item_features_processed = processor.preprocess_item_features(item_features_df)
        
        # Get all users from database (including new users)
        all_users_query = "SELECT id FROM users WHERE del_flag = '0'"
        all_users_df = db.execute_query(all_users_query)
        all_user_ids = all_users_df['id'].tolist()
        
        # Clear expired recommendations
        clear_query = "DELETE FROM recommendations WHERE expires_at < to_char(NOW(), 'YYYYMMDDHH24MISS')"
        db.execute_insert(clear_query)
        
        # Generate recommendations for each user
        recommendations_count = 0
        for user_id in all_user_ids:
            try:
                user_id = int(user_id)
                
                # Get user features or create default
                user_features = user_features_processed[user_features_processed['user_id'] == user_id]
                if len(user_features) == 0:
                    # Create default features for new users
                    print(f"Creating default features for new user {user_id}")
                    default_features = processor.create_default_user_features(user_id)
                    user_feature_vector = np.array([
                        default_features['age_group_encoded'],
                        default_features['gender_preference_encoded'],
                        default_features['price_range_encoded'],
                        default_features['brand_preferences_count'],
                        default_features['category_preferences_count'],
                        default_features['style_preferences_count']
                    ])
                else:
                    user_feature_vector = user_features.drop('user_id', axis=1).iloc[0].values
                
                # Get user's existing interactions to exclude
                interactions_query = """
                SELECT DISTINCT watch_id FROM user_interactions 
                WHERE user_id = %s AND del_flag = '0'
                """
                existing_items = db.execute_query(interactions_query, (int(user_id),))
                exclude_items = existing_items['watch_id'].tolist() if len(existing_items) > 0 else []
                
                # Get recommendations
                recommendations = hybrid_model.get_hybrid_recommendations(
                    user_id=user_id,
                    user_features=user_feature_vector,
                    all_item_features=item_features_processed.set_index('watch_id'),
                    n_recommendations=Config.MAX_RECOMMENDATIONS,
                    exclude_items=exclude_items
                )
                
                # Save recommendations to database
                expires_at_dt = datetime.now() + timedelta(hours=Config.RECOMMENDATION_EXPIRY_HOURS)
                expires_at = expires_at_dt.strftime('%Y%m%d%H%M%S')
                
                for item_id, score in recommendations:
                    insert_query = """
                    INSERT INTO recommendations 
                    (user_id, watch_id, recommendation_score, recommendation_type, algorithm_version, expires_at, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """
                    
                    params = (
                        user_id,
                        item_id,
                        float(score),
                        'hybrid',
                        '1.0',
                        expires_at,
                        datetime.now().strftime('%Y%m%d%H%M%S')
                    )
                    
                    db.execute_insert(insert_query, params)
                    recommendations_count += 1
                
                print(f"Generated {len(recommendations)} recommendations for user {user_id}")
                
            except Exception as e:
                print(f"Failed to generate recommendations for user {user_id}: {str(e)}")
                continue
        
        print(f"Generated {recommendations_count} total recommendations")
        
    except Exception as e:
        print(f"Failed to generate recommendations: {str(e)}")
        import traceback
        traceback.print_exc()

def generate_training_plots():
    """Generate training plots after model training"""
    try:
        print("Generating training plots...")
        # Add scripts directory to path
        scripts_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scripts')
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        
        # Change to scripts directory to run plot script
        original_dir = os.getcwd()
        try:
            os.chdir(scripts_dir)
            from plot_training_results import main as plot_main
            plot_main()
            print("Training plots generated successfully!")
        finally:
            os.chdir(original_dir)
    except Exception as e:
        print(f"Failed to generate training plots: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--generate-recommendations":
        generate_recommendations()
    elif len(sys.argv) > 1 and sys.argv[1] == "--generate-plots":
        generate_training_plots()
    else:
        main()
        # Plots are generated inside main() after successful training
