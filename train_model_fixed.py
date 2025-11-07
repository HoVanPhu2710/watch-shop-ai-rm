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
    print("Starting hybrid recommendation model training...")
    start_time = time.time()
    
    try:
        # Initialize data processor
        processor = DataProcessor()
        
        # Load and preprocess data
        print("Loading interaction data...")
        interactions_df = processor.load_interaction_data()
        
        if len(interactions_df) == 0:
            print("No interaction data found. Exiting...")
            return
        
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
        
        # Train the model
        print("Training hybrid model...")
        hybrid_model.train(
            interactions_processed,
            user_features_processed,
            item_features_processed
        )
        
        # Save the model
        model_path = os.path.join(Config.MODEL_SAVE_PATH, 'hybrid_model')
        hybrid_model.save_models(model_path)
        
        # Save encoders for later use
        encoder_path = os.path.join(Config.MODEL_SAVE_PATH, 'encoders')
        os.makedirs(encoder_path, exist_ok=True)
        
        import joblib
        joblib.dump(processor.user_encoder, os.path.join(encoder_path, 'user_encoder.pkl'))
        joblib.dump(processor.item_encoder, os.path.join(encoder_path, 'item_encoder.pkl'))
        joblib.dump(processor.scaler, os.path.join(encoder_path, 'scaler.pkl'))
        
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
        
    except Exception as e:
        print(f"Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    finally:
        db.close()

def save_training_history(model_type, algorithm_version, training_data_size, 
                         training_duration_seconds, model_file_path):
    """Save training history to database"""
    try:
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
        print(f"Failed to save training history: {str(e)}")

def generate_recommendations():
    """Generate recommendations for all users and save to database - Fixed version"""
    print("Generating recommendations...")
    
    try:
        # Load the trained model
        model_path = os.path.join(Config.MODEL_SAVE_PATH, 'hybrid_model')
        hybrid_model = HybridRecommendationModel(1, 1, 1, 1)  # Dummy dimensions
        hybrid_model.load_models(model_path)
        
        # Load encoders
        encoder_path = os.path.join(Config.MODEL_SAVE_PATH, 'encoders')
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
        from plot_training_results import main as plot_main
        plot_main()
        print("Training plots generated successfully!")
    except Exception as e:
        print(f"Failed to generate training plots: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--generate-recommendations":
        generate_recommendations()
    elif len(sys.argv) > 1 and sys.argv[1] == "--generate-plots":
        generate_training_plots()
    else:
        main()
        # Generate plots after training
        generate_training_plots()
