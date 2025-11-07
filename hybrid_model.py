import numpy as np
import pandas as pd
from collaborative_filtering import CollaborativeFilteringModel
from content_based_filtering import ContentBasedFilteringModel
from config import Config
import os
import joblib

class HybridRecommendationModel:
    def __init__(self, n_users, n_items, user_feature_dim, item_feature_dim):
        self.n_users = n_users
        self.n_items = n_items
        self.user_feature_dim = user_feature_dim
        self.item_feature_dim = item_feature_dim
        
        # Initialize individual models
        self.collaborative_model = CollaborativeFilteringModel(n_users, n_items)
        self.content_based_model = ContentBasedFilteringModel(user_feature_dim, item_feature_dim)
        
        # Hybrid weights
        self.collaborative_weight = Config.COLLABORATIVE_WEIGHT
        self.content_based_weight = Config.CONTENT_BASED_WEIGHT
        
    def train(self, interactions_df, user_features_df, item_features_df):
        """Train both models"""
        print("Training Collaborative Filtering Model...")
        self.collaborative_model.train(
            interactions_df[['user_encoded', 'item_encoded']],
            interactions_df['normalized_score'],
            interactions_df[['user_encoded', 'item_encoded']],  # Using same data for validation
            interactions_df['normalized_score']
        )
        
        print("Training Content-Based Filtering Model...")
        # Prepare content-based training data: keep ONLY engineered numeric columns
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

        # Build feature lookup tables indexed by ids
        user_features_table = (
            user_features_df[['user_id'] + user_feature_cols]
            .set_index('user_id')
            .fillna(0)
        )
        item_features_table = (
            item_features_df[['watch_id'] + item_feature_cols]
            .set_index('watch_id')
            .fillna(0)
        )

        # Create training pairs
        user_item_pairs = []
        user_features_list = []
        item_features_list = []
        scores = []
        
        for _, row in interactions_df.iterrows():
            user_id = row['user_id']
            item_id = row['watch_id']
            score = row['normalized_score']
            
            # Get user features vector (numeric only)
            if user_id in user_features_table.index:
                user_feature_vector = user_features_table.loc[user_id].values
            else:
                user_feature_vector = np.zeros(self.user_feature_dim)
            
            # Get item features vector (numeric only)
            if item_id in item_features_table.index:
                item_feature_vector = item_features_table.loc[item_id].values
            else:
                item_feature_vector = np.zeros(self.item_feature_dim)
            
            user_item_pairs.append((user_id, item_id))
            user_features_list.append(user_feature_vector)
            item_features_list.append(item_feature_vector)
            scores.append(score)
        
        # Convert to numpy arrays
        user_features_array = np.array(user_features_list)
        item_features_array = np.array(item_features_list)
        scores_array = np.array(scores)
        
        # Train content-based model
        self.content_based_model.train(
            {'user_features': user_features_array, 'item_features': item_features_array},
            scores_array,
            {'user_features': user_features_array, 'item_features': item_features_array},
            scores_array
        )
        
        print("Both models trained successfully!")
    
    def predict_hybrid(self, user_id, item_id, user_features, item_features):
        """Make hybrid prediction combining both models"""
        # Collaborative filtering prediction
        try:
            collab_pred = self.collaborative_model.predict([user_id], [item_id])[0]
        except:
            collab_pred = 0.0
        
        # Content-based prediction
        try:
            content_pred = self.content_based_model.predict(
                user_features.reshape(1, -1),
                item_features.reshape(1, -1)
            )[0]
        except:
            content_pred = 0.0
        
        # Weighted combination
        hybrid_pred = (
            self.collaborative_weight * collab_pred +
            self.content_based_weight * content_pred
        )
        
        return hybrid_pred
    
    def get_hybrid_recommendations(self, user_id, user_features, all_item_features, 
                                 n_recommendations=10, exclude_items=None):
        """Get hybrid recommendations for a user"""
        recommendations = []
        
        for item_id, item_features in all_item_features.iterrows():
            if exclude_items and item_id in exclude_items:
                continue
                
            try:
                score = self.predict_hybrid(user_id, item_id, user_features, item_features.values)
                recommendations.append((item_id, score))
            except:
                continue
        
        # Sort by score and return top recommendations
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:n_recommendations]
    
    def save_models(self, model_path):
        """Save both models"""
        os.makedirs(model_path, exist_ok=True)
        
        # Save collaborative filtering model
        collab_path = os.path.join(model_path, 'collaborative')
        self.collaborative_model.save_model(collab_path)
        
        # Save content-based model
        content_path = os.path.join(model_path, 'content_based')
        self.content_based_model.save_model(content_path)
        
        # Save hybrid model metadata
        metadata = {
            'n_users': self.n_users,
            'n_items': self.n_items,
            'user_feature_dim': self.user_feature_dim,
            'item_feature_dim': self.item_feature_dim,
            'collaborative_weight': self.collaborative_weight,
            'content_based_weight': self.content_based_weight
        }
        
        joblib.dump(metadata, os.path.join(model_path, 'hybrid_metadata.pkl'))
        
        print(f"Hybrid model saved to {model_path}")
    
    def load_models(self, model_path):
        """Load both models"""
        # Load metadata
        metadata = joblib.load(os.path.join(model_path, 'hybrid_metadata.pkl'))
        self.n_users = metadata['n_users']
        self.n_items = metadata['n_items']
        self.user_feature_dim = metadata['user_feature_dim']
        self.item_feature_dim = metadata['item_feature_dim']
        self.collaborative_weight = metadata['collaborative_weight']
        self.content_based_weight = metadata['content_based_weight']
        
        # Load collaborative filtering model
        collab_path = os.path.join(model_path, 'collaborative')
        self.collaborative_model = CollaborativeFilteringModel(self.n_users, self.n_items)
        self.collaborative_model.load_model(collab_path)
        
        # Load content-based model
        content_path = os.path.join(model_path, 'content_based')
        self.content_based_model = ContentBasedFilteringModel(
            self.user_feature_dim, 
            self.item_feature_dim
        )
        self.content_based_model.load_model(content_path)
        
        print(f"Hybrid model loaded from {model_path}")
    
    def evaluate_hybrid(self, test_data, user_features_df, item_features_df):
        """Evaluate hybrid model performance"""
        predictions = []
        actual_scores = []
        
        for _, row in test_data.iterrows():
            user_id = row['user_id']
            item_id = row['watch_id']
            actual_score = row['normalized_score']
            
            # Get user features
            user_features = user_features_df[user_features_df['user_id'] == user_id]
            if len(user_features) > 0:
                user_feature_vector = user_features.drop('user_id', axis=1).iloc[0].values
            else:
                user_feature_vector = np.zeros(self.user_feature_dim)
            
            # Get item features
            item_features = item_features_df[item_features_df['watch_id'] == item_id]
            if len(item_features) > 0:
                item_feature_vector = item_features.drop('watch_id', axis=1).iloc[0].values
            else:
                item_feature_vector = np.zeros(self.item_feature_dim)
            
            try:
                pred_score = self.predict_hybrid(user_id, item_id, user_feature_vector, item_feature_vector)
                predictions.append(pred_score)
                actual_scores.append(actual_score)
            except:
                continue
        
        if len(predictions) == 0:
            return {'mse': float('inf'), 'mae': float('inf'), 'rmse': float('inf')}
        
        # Calculate metrics
        mse = np.mean((np.array(predictions) - np.array(actual_scores)) ** 2)
        mae = np.mean(np.abs(np.array(predictions) - np.array(actual_scores)))
        rmse = np.sqrt(mse)
        
        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse
        }
