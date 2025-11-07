import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from database import db
from config import Config

class DataProcessor:
    def __init__(self):
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
        self.scaler = MinMaxScaler()
        self.tfidf_vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        
    def load_interaction_data(self):
        """Load user interaction data from database"""
        query = """
        SELECT 
            ui.user_id,
            ui.watch_id,
            ui.interaction_type,
            ui.score,
            ui.created_at,
            w.name as watch_name,
            w.category_id,
            w.brand_id,
            w.gender,
            w.base_price,
            w.rating
        FROM user_interactions ui
        JOIN watches w ON ui.watch_id = w.id
        WHERE ui.del_flag = '0'
        ORDER BY ui.created_at
        """
        
        df = db.execute_query(query)
        print(f"Loaded {len(df)} interactions")
        return df
    
    def load_user_features(self):
        """Load user features directly from users table (simplified schema)"""
        query = """
        SELECT 
            u.id AS user_id,
            u.age_group,
            u.gender_preference,
            u.price_range_preference,
            u.brand_preferences,
            u.category_preferences,
            u.style_preferences,
            u.gender,
            u.date_of_birth
        FROM users u
        WHERE u.del_flag = '0'
        """

        df = db.execute_query(query)
        print(f"Loaded {len(df)} user features")
        return df
    
    def load_item_features(self):
        """Load item features directly from watches table (simplified schema)"""
        query = """
        SELECT 
            w.id AS watch_id,
            w.price_tier,
            w.gender_target,
            w.style_tags,
            w.material_tags,
            w.color_tags,
            w.size_category,
            w.movement_type_tags,
            w.name,
            w.category_id,
            w.brand_id,
            w.gender,
            w.base_price,
            w.rating
        FROM watches w
        WHERE w.del_flag = '0'
        """

        df = db.execute_query(query)
        print(f"Loaded {len(df)} item features")
        return df
    
    def preprocess_interactions(self, df):
        """Preprocess interaction data with more lenient filtering"""
        # Filter users and items with minimum interactions (more lenient)
        user_counts = df['user_id'].value_counts()
        item_counts = df['watch_id'].value_counts()
        
        # Use lower thresholds for new users
        min_user_interactions = max(1, Config.MIN_INTERACTIONS_PER_USER // 2)
        min_item_interactions = max(1, Config.MIN_INTERACTIONS_PER_ITEM // 2)
        
        valid_users = user_counts[user_counts >= min_user_interactions].index
        valid_items = item_counts[item_counts >= min_item_interactions].index
        
        df_filtered = df[
            (df['user_id'].isin(valid_users)) & 
            (df['watch_id'].isin(valid_items))
        ].copy()
        
        print(f"Filtered to {len(df_filtered)} interactions from {len(valid_users)} users and {len(valid_items)} items")
        
        # Encode user and item IDs
        df_filtered['user_encoded'] = self.user_encoder.fit_transform(df_filtered['user_id'])
        df_filtered['item_encoded'] = self.item_encoder.fit_transform(df_filtered['watch_id'])
        
        # Normalize scores
        df_filtered['normalized_score'] = df_filtered['score'] / 6.0  # Max score is 6
        
        return df_filtered
    
    def preprocess_user_features(self, df):
        """Preprocess user features for content-based filtering with better null handling"""
        df_processed = df.copy()
        
        # Handle age group with default for new users
        age_mapping = {'18-25': 0, '26-35': 1, '36-45': 2, '46+': 3}
        df_processed['age_group_encoded'] = df_processed['age_group'].map(age_mapping).fillna(1)  # Default to 26-35
        
        # Handle gender preference with default
        gender_mapping = {'M': 0, 'F': 1, 'U': 2}
        df_processed['gender_preference_encoded'] = df_processed['gender_preference'].map(gender_mapping).fillna(2)  # Default to Unisex
        
        # Handle price range preference with default
        price_mapping = {'budget': 0, 'mid': 1, 'premium': 2, 'luxury': 3}
        df_processed['price_range_encoded'] = df_processed['price_range_preference'].map(price_mapping).fillna(1)  # Default to mid
        
        # Process JSON/JSONB fields with better null handling
        def _count_json_array(v):
            if v is None or (isinstance(v, float) and pd.isna(v)):
                return 0
            if isinstance(v, (list, tuple)):
                return len(v)
            try:
                parsed = json.loads(v)
                return len(parsed) if isinstance(parsed, (list, tuple)) else 0
            except Exception:
                return 0

        for col in ['brand_preferences', 'category_preferences', 'style_preferences']:
            df_processed[f'{col}_count'] = df_processed[col].apply(_count_json_array)
        
        return df_processed
    
    def preprocess_item_features(self, df):
        """Preprocess item features for content-based filtering"""
        df_processed = df.copy()
        
        # Handle price tier with default
        price_tier_mapping = {'budget': 0, 'mid': 1, 'premium': 2, 'luxury': 3}
        df_processed['price_tier_encoded'] = df_processed['price_tier'].map(price_tier_mapping).fillna(1)  # Default to mid
        
        # Handle gender target with default
        gender_mapping = {'M': 0, 'F': 1, 'U': 2}
        df_processed['gender_target_encoded'] = df_processed['gender_target'].map(gender_mapping).fillna(2)  # Default to Unisex
        
        # Handle size category with default
        size_mapping = {'small': 0, 'medium': 1, 'large': 2}
        df_processed['size_category_encoded'] = df_processed['size_category'].map(size_mapping).fillna(1)  # Default to medium
        
        # Process JSON/JSONB fields
        def _count_json_array_item(v):
            if v is None or (isinstance(v, float) and pd.isna(v)):
                return 0
            if isinstance(v, (list, tuple)):
                return len(v)
            try:
                parsed = json.loads(v)
                return len(parsed) if isinstance(parsed, (list, tuple)) else 0
            except Exception:
                return 0

        for col in ['style_tags', 'material_tags', 'color_tags', 'movement_type_tags']:
            df_processed[f'{col}_count'] = df_processed[col].apply(_count_json_array_item)
        
        # Normalize price and rating with better null handling
        df_processed['price_normalized'] = self.scaler.fit_transform(df_processed[['base_price']].fillna(0))
        df_processed['rating_normalized'] = self.scaler.fit_transform(df_processed[['rating']].fillna(0))
        
        return df_processed
    
    def create_user_item_matrix(self, interactions_df):
        """Create user-item interaction matrix"""
        matrix = interactions_df.pivot_table(
            index='user_encoded',
            columns='item_encoded',
            values='normalized_score',
            fill_value=0
        )
        return matrix
    
    def get_user_features_matrix(self, user_features_df):
        """Get user features matrix for content-based filtering"""
        feature_cols = [
            'age_group_encoded',
            'gender_preference_encoded',
            'price_range_encoded',
            'brand_preferences_count',
            'category_preferences_count',
            'style_preferences_count'
        ]
        
        return user_features_df[['user_id'] + feature_cols].set_index('user_id')
    
    def get_item_features_matrix(self, item_features_df):
        """Get item features matrix for content-based filtering"""
        feature_cols = [
            'price_tier_encoded',
            'gender_target_encoded',
            'size_category_encoded',
            'style_tags_count',
            'material_tags_count',
            'color_tags_count',
            'movement_type_tags_count',
            'price_normalized',
            'rating_normalized'
        ]
        
        return item_features_df[['watch_id'] + feature_cols].set_index('watch_id')
    
    def create_default_user_features(self, user_id):
        """Create default features for new users"""
        return {
            'user_id': user_id,
            'age_group_encoded': 1,  # 26-35
            'gender_preference_encoded': 2,  # Unisex
            'price_range_encoded': 1,  # mid
            'brand_preferences_count': 0,
            'category_preferences_count': 0,
            'style_preferences_count': 0
        }
