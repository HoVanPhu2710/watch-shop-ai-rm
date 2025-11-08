#!/usr/bin/env python3
"""
AI Recommendation Server - Real-time recommendation service
Loads trained models and serves recommendations via API
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
from datetime import datetime
import logging

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from database import db
from config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Initialize database connection lazily (will be initialized on first use)

class AIRecommendationServer:
    def __init__(self):
        self.hybrid_model = None
        self.user_encoder = None
        self.item_encoder = None
        self.scaler = None
        self.user_features_cache = {}
        self.item_features_cache = {}
        self.last_model_load = None
        self.model_loaded = False
        
    def load_models(self):
        """Load trained models and encoders"""
        try:
            logger.info("Loading AI models...")
            
            # Load hybrid model
            model_path = os.path.join(Config.MODEL_SAVE_PATH, 'hybrid_model')
            if not os.path.exists(model_path):
                logger.error(f"Model path not found: {model_path}")
                return False
                
            # Load encoders
            encoder_path = os.path.join(Config.MODEL_SAVE_PATH, 'encoders')
            self.user_encoder = joblib.load(os.path.join(encoder_path, 'user_encoder.pkl'))
            self.item_encoder = joblib.load(os.path.join(encoder_path, 'item_encoder.pkl'))
            self.scaler = joblib.load(os.path.join(encoder_path, 'scaler.pkl'))
            
            # Load hybrid model metadata
            metadata = joblib.load(os.path.join(model_path, 'hybrid_metadata.pkl'))
            
            # Initialize hybrid model
            from hybrid_model import HybridRecommendationModel
            self.hybrid_model = HybridRecommendationModel(
                n_users=metadata['n_users'],
                n_items=metadata['n_items'],
                user_feature_dim=metadata['user_feature_dim'],
                item_feature_dim=metadata['item_feature_dim']
            )
            
            # Load the actual model
            self.hybrid_model.load_models(model_path)
            
            # Load user and item features
            self.load_features_cache()
            
            self.model_loaded = True
            self.last_model_load = datetime.now()
            logger.info("Models loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load models: {str(e)}")
            return False
    
    def load_features_cache(self):
        """Load and cache user and item features"""
        try:
            logger.info("Loading features cache...")
            
            # Load user features directly from users table
            user_query = """
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
            user_features_df = db.execute_query(user_query)
            self.user_features_cache = self.preprocess_user_features(user_features_df)
            
            # Load item features directly from watches table
            item_query = """
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
                w.rating,
                w.sold
            FROM watches w
            WHERE w.del_flag = '0' AND w.status = true
            """
            item_features_df = db.execute_query(item_query)
            self.item_features_cache = self.preprocess_item_features(item_features_df)
            
            logger.info(f"Loaded {len(self.user_features_cache)} users and {len(self.item_features_cache)} items")
            
            # Debug: Log cache details
            if len(self.user_features_cache) > 0:
                logger.info(f"User features sample: {self.user_features_cache.head()}")
            else:
                logger.warning("User features cache is empty!")
                
            if len(self.item_features_cache) > 0:
                logger.info(f"Item features sample: {self.item_features_cache.head()}")
            else:
                logger.warning("Item features cache is empty!")
            
        except Exception as e:
            logger.error(f"Failed to load features cache: {str(e)}")
            logger.error(f"Exception type: {type(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
    
    def preprocess_user_features(self, df):
        """Preprocess user features"""
        df_processed = df.copy()
        
        # Handle age group with default
        age_mapping = {'18-25': 0, '26-35': 1, '36-45': 2, '46+': 3}
        df_processed['age_group_encoded'] = df_processed['age_group'].map(age_mapping).fillna(1)
        
        # Handle gender preference with default
        gender_mapping = {'M': 0, 'F': 1, 'U': 2}
        df_processed['gender_preference_encoded'] = df_processed['gender_preference'].map(gender_mapping).fillna(2)
        
        # Handle price range preference with default
        price_mapping = {'budget': 0, 'mid': 1, 'premium': 2, 'luxury': 3}
        df_processed['price_range_encoded'] = df_processed['price_range_preference'].map(price_mapping).fillna(1)
        
        # Process JSON fields
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
        
        return df_processed.set_index('user_id')
    
    def preprocess_item_features(self, df):
        """Preprocess item features"""
        df_processed = df.copy()
        
        # Handle price tier with default
        price_tier_mapping = {'budget': 0, 'mid': 1, 'premium': 2, 'luxury': 3}
        df_processed['price_tier_encoded'] = df_processed['price_tier'].map(price_tier_mapping).fillna(1)
        
        # Handle gender target with default
        gender_mapping = {'M': 0, 'F': 1, 'U': 2}
        df_processed['gender_target_encoded'] = df_processed['gender_target'].map(gender_mapping).fillna(2)
        
        # Handle size category with default
        size_mapping = {'small': 0, 'medium': 1, 'large': 2}
        df_processed['size_category_encoded'] = df_processed['size_category'].map(size_mapping).fillna(1)
        
        # Process JSON fields
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
        
        # Normalize price and rating - handle feature mismatch
        try:
            # Try to use the scaler with the expected features
            if hasattr(self.scaler, 'feature_names_in_'):
                # Check what features the scaler expects
                expected_features = self.scaler.feature_names_in_
                if 'rating' in expected_features and 'base_price' not in expected_features:
                    # Use rating instead of base_price for normalization
                    df_processed['price_normalized'] = self.scaler.transform(df_processed[['rating']].fillna(0))
                    df_processed['rating_normalized'] = self.scaler.transform(df_processed[['rating']].fillna(0))
                else:
                    # Use base_price if it's expected
                    df_processed['price_normalized'] = self.scaler.transform(df_processed[['base_price']].fillna(0))
                    df_processed['rating_normalized'] = self.scaler.transform(df_processed[['rating']].fillna(0))
            else:
                # Fallback: use base_price and rating
                df_processed['price_normalized'] = self.scaler.transform(df_processed[['base_price']].fillna(0))
                df_processed['rating_normalized'] = self.scaler.transform(df_processed[['rating']].fillna(0))
        except Exception as e:
            logger.warning(f"Scaler transform failed: {e}. Using default values.")
            # Fallback: use simple normalization
            df_processed['price_normalized'] = df_processed['base_price'].fillna(0) / 1000000.0  # Normalize to millions
            df_processed['rating_normalized'] = df_processed['rating'].fillna(0) / 5.0  # Normalize to 0-1
        
        return df_processed.set_index('watch_id')
    
    def create_default_user_features(self, user_id):
        """Create default features for new users"""
        return np.array([1, 2, 1, 0, 0, 0])  # age_group, gender_preference, price_range, brand_count, category_count, style_count
    
    def get_anonymous_recommendations(self, limit=10, profile='general'):
        """Get recommendations for anonymous users using default profiles"""
        # Try to load models if not loaded yet
        if not self.model_loaded:
            logger.warning("Models not loaded, attempting to load now...")
            if not self.load_models():
                return {"error": "Model not loaded. Please check logs and ensure models are available."}
        
        try:
            # Define default profiles
            default_profiles = {
                "general": [1, 2, 1, 0, 0, 0],      # Nam, 26-35, mid price
                "young_male": [0, 0, 0, 0, 0, 0],    # Nam, 18-25, budget
                "young_female": [0, 1, 0, 0, 0, 0],  # Nữ, 18-25, budget
                "mature_male": [2, 0, 2, 0, 0, 0],   # Nam, 36-45, premium
                "mature_female": [2, 1, 2, 0, 0, 0], # Nữ, 36-45, premium
                "luxury": [3, 2, 3, 0, 0, 0]         # Unisex, 46+, luxury
            }
            
            # Get user features for the profile
            user_features = np.array(default_profiles.get(profile, default_profiles["general"]))
            
            # Get all item features
            all_item_features = self.item_features_cache
            
            if len(all_item_features) == 0:
                # Fallback: return popular items if no items available
                logger.warning(f"No items available in cache for anonymous user. Returning popular items.")
                popular_items = self.get_popular_items(limit)
                return {
                    "recommendations": popular_items,
                    "message": "No items in cache, showing popular items",
                    "fallback": True,
                    "profile": profile
                }
            
            # Get recommendations using the model
            recommendations = self.hybrid_model.get_hybrid_recommendations(
                user_id=hash(profile) % 10000,  # Generate user_id from profile
                user_features=user_features,
                all_item_features=all_item_features,
                n_recommendations=limit,
                exclude_items=[]
            )
            
            # Format recommendations with item details
            formatted_recommendations = []
            for item_id, score in recommendations:
                if hasattr(self.item_features_cache, 'index') and item_id in self.item_features_cache.index:
                    item_data = self.item_features_cache.loc[item_id]
                    
                    # Get additional watch details
                    watch_details = self.get_watch_details(item_id)
                    
                    formatted_recommendations.append({
                        "watch_id": int(item_id),
                        "score": float(score),
                        "name": item_data.get('name', 'Unknown'),
                        "description": watch_details.get('description', ''),
                        "base_price": float(item_data.get('base_price', 0)),
                        "rating": float(item_data.get('rating', 0)) if item_data.get('rating') else 0.0,
                        "sold": int(item_data.get('sold', 0)) if item_data.get('sold') else 0,
                        "brand": watch_details.get('brand', {}),
                        "category": watch_details.get('category', {}),
                        "price_tier": item_data.get('price_tier', 'mid'),
                        "gender_target": item_data.get('gender_target', 'U'),
                        "size_category": item_data.get('size_category', 'medium'),
                        "style_tags": watch_details.get('style_tags', []),
                        "material_tags": watch_details.get('material_tags', []),
                        "color_tags": watch_details.get('color_tags', []),
                        "movement_type_tags": watch_details.get('movement_type_tags', []),
                        "images": watch_details.get('images', []),
                        "is_ai_recommended": True,
                        "profile_used": profile
                    })
            
            return {
                "recommendations": formatted_recommendations,
                "profile": profile,
                "count": len(formatted_recommendations),
                "user_type": "anonymous"
            }
            
        except Exception as e:
            logger.error(f"Failed to get anonymous recommendations: {str(e)}")
            return {"error": str(e)}
    
    def get_watch_details(self, watch_id):
        """Get detailed watch information"""
        try:
            query = """
            SELECT 
                w.description, w.thumbnail, w.slider,
                w.style_tags, w.material_tags, w.color_tags, w.movement_type_tags,
                b.name as brand_name, b.id as brand_id,
                c.name as category_name, c.id as category_id
            FROM watches w
            LEFT JOIN brands b ON w.brand_id = b.id
            LEFT JOIN categorys c ON w.category_id = c.id
            WHERE w.id = %s AND w.del_flag = '0'
            """
            result_df = db.execute_query(query, (int(watch_id),))
            
            if len(result_df) == 0:
                return {}
            
            row = result_df.iloc[0]
            
            # Process images
            images = []
            if row['thumbnail']:
                images.append(row['thumbnail'])
            if row['slider']:
                # slider is JSON array of image URLs
                try:
                    import json
                    slider_images = json.loads(row['slider']) if isinstance(row['slider'], str) else row['slider']
                    if isinstance(slider_images, list):
                        images.extend(slider_images)
                except:
                    pass
            
            # Process tags
            def parse_tags(tag_str):
                if not tag_str:
                    return []
                try:
                    import json
                    return json.loads(tag_str) if isinstance(tag_str, str) else tag_str
                except:
                    return []
            
            return {
                'description': row['description'] or '',
                'images': images,
                'style_tags': parse_tags(row['style_tags']),
                'material_tags': parse_tags(row['material_tags']),
                'color_tags': parse_tags(row['color_tags']),
                'movement_type_tags': parse_tags(row['movement_type_tags']),
                'brand': {
                    'id': int(row['brand_id']) if row['brand_id'] else None,
                    'name': row['brand_name'] or 'Unknown'
                },
                'category': {
                    'id': int(row['category_id']) if row['category_id'] else None,
                    'name': row['category_name'] or 'Unknown'
                }
            }
        except Exception as e:
            logger.error(f"Failed to get watch details for {watch_id}: {str(e)}")
            return {}
    
    def get_popular_items(self, limit=10):
        """Get popular items as fallback with full information"""
        try:
            query = """
            SELECT 
                w.id, w.name, w.base_price, w.rating, w.sold, w.description,
                w.price_tier, w.gender_target, w.size_category,
                w.style_tags, w.material_tags, w.color_tags, w.movement_type_tags,
                b.name as brand_name, c.name as category_name,
                w.thumbnail, w.slider
            FROM watches w
            LEFT JOIN brands b ON w.brand_id = b.id
            LEFT JOIN categorys c ON w.category_id = c.id
            WHERE w.del_flag = '0' AND w.status = true
            ORDER BY w.sold DESC, w.rating DESC
            LIMIT %s
            """
            popular_df = db.execute_query(query, (limit,))
            
            popular_items = []
            for _, row in popular_df.iterrows():
                # Process images
                images = []
                if row['thumbnail']:
                    images.append(row['thumbnail'])
                if row['slider']:
                    # slider is JSON array of image URLs
                    try:
                        import json
                        slider_images = json.loads(row['slider']) if isinstance(row['slider'], str) else row['slider']
                        if isinstance(slider_images, list):
                            images.extend(slider_images)
                    except:
                        pass
                
                # Process tags
                def parse_tags(tag_str):
                    if not tag_str:
                        return []
                    try:
                        import json
                        return json.loads(tag_str) if isinstance(tag_str, str) else tag_str
                    except:
                        return []
                
                popular_items.append({
                    "watch_id": int(row['id']),
                    "score": 0.5,  # Default score for popular items
                    "name": row['name'],
                    "description": row['description'] or "",
                    "base_price": float(row['base_price']),
                    "rating": float(row['rating']) if row['rating'] else 0.0,
                    "sold": int(row['sold']) if row['sold'] else 0,
                    "brand": {
                        "name": row['brand_name'] or "Unknown"
                    },
                    "category": {
                        "name": row['category_name'] or "Unknown"
                    },
                    "price_tier": row['price_tier'] or "mid",
                    "gender_target": row['gender_target'] or "U",
                    "size_category": row['size_category'] or "medium",
                    "style_tags": parse_tags(row['style_tags']),
                    "material_tags": parse_tags(row['material_tags']),
                    "color_tags": parse_tags(row['color_tags']),
                    "movement_type_tags": parse_tags(row['movement_type_tags']),
                    "images": images,
                    "is_popular": True
                })
            
            return popular_items
        except Exception as e:
            logger.error(f"Failed to get popular items: {str(e)}")
            return []
    
    def get_user_interactions_with_scores(self, user_id):
        """Get user's interaction history with scores for enhanced recommendations"""
        try:
            query = """
            SELECT watch_id, MAX(score) as max_score, COUNT(*) as interaction_count
            FROM user_interactions 
            WHERE user_id = %s AND del_flag = '0'
            GROUP BY watch_id
            ORDER BY max_score DESC, interaction_count DESC
            """
            interactions_df = db.execute_query(query, (int(user_id),))
            
            # Convert to dict for easy lookup
            interaction_scores = {}
            for _, row in interactions_df.iterrows():
                interaction_scores[int(row['watch_id'])] = {
                    'max_score': int(row['max_score']),
                    'interaction_count': int(row['interaction_count'])
                }
            
            return interaction_scores
        except Exception as e:
            logger.error(f"Failed to get user interactions with scores: {str(e)}")
            return {}
    
    def get_user_interactions(self, user_id):
        """Get user's existing interactions to exclude from recommendations"""
        try:
            query = """
            SELECT DISTINCT watch_id FROM user_interactions 
            WHERE user_id = %s AND del_flag = '0'
            """
            interactions_df = db.execute_query(query, (int(user_id),))
            return interactions_df['watch_id'].tolist() if len(interactions_df) > 0 else []
        except Exception as e:
            logger.error(f"Failed to get user interactions: {str(e)}")
            return []
    
    def get_recommendations(self, user_id, limit=10, exclude_interactions=False):
        """Get recommendations for a user"""
        # Try to load models if not loaded yet
        if not self.model_loaded:
            logger.warning("Models not loaded, attempting to load now...")
            if not self.load_models():
                return {"error": "Model not loaded. Please check logs and ensure models are available."}
        
        try:
            # Get user features
            if hasattr(self.user_features_cache, 'index') and user_id in self.user_features_cache.index:
                user_features = self.user_features_cache.loc[user_id].values
            else:
                # Create default features for new users
                user_features = self.create_default_user_features(user_id)
            
            # Get user interaction history for enhanced recommendations
            user_interactions = self.get_user_interactions_with_scores(user_id)
            
            # Get all item features
            all_item_features = self.item_features_cache
            
            if len(all_item_features) == 0:
                # Fallback: return popular items if no items available
                logger.warning(f"No items available in cache for user {user_id}. Returning popular items.")
                popular_items = self.get_popular_items(limit)
                return {
                    "recommendations": popular_items,
                    "message": "No items in cache, showing popular items",
                    "fallback": True
                }
            
            # Get recommendations
            recommendations = self.hybrid_model.get_hybrid_recommendations(
                user_id=user_id,
                user_features=user_features,
                all_item_features=all_item_features,
                n_recommendations=limit,
                exclude_items=[]  # Don't exclude any items
            )
            
            # Enhance recommendations with interaction history
            enhanced_recommendations = []
            for item_id, score in recommendations:
                enhanced_score = float(score)
                
                # Boost score for items with interaction history
                if item_id in user_interactions:
                    interaction_data = user_interactions[item_id]
                    # Boost score based on interaction strength and frequency
                    boost_factor = 1 + (interaction_data['max_score'] / 10.0) + (interaction_data['interaction_count'] * 0.1)
                    enhanced_score = min(enhanced_score * boost_factor, 1.0)  # Cap at 1.0
                
                enhanced_recommendations.append((item_id, enhanced_score))
            
            # Sort by enhanced score
            enhanced_recommendations.sort(key=lambda x: x[1], reverse=True)
            recommendations = enhanced_recommendations[:limit]
            
            # Format recommendations with item details
            formatted_recommendations = []
            for item_id, score in recommendations:
                if hasattr(self.item_features_cache, 'index') and item_id in self.item_features_cache.index:
                    item_data = self.item_features_cache.loc[item_id]
                    # Add interaction info if available
                    interaction_info = None
                    if item_id in user_interactions:
                        interaction_data = user_interactions[item_id]
                        interaction_info = {
                            "max_interaction_score": interaction_data['max_score'],
                            "interaction_count": interaction_data['interaction_count'],
                            "has_interaction_history": True
                        }
                    
                    # Get additional watch details
                    watch_details = self.get_watch_details(item_id)
                    
                    formatted_recommendations.append({
                        "watch_id": int(item_id),
                        "score": float(score),
                        "name": item_data.get('name', 'Unknown'),
                        "description": watch_details.get('description', ''),
                        "base_price": float(item_data.get('base_price', 0)),
                        "rating": float(item_data.get('rating', 0)) if item_data.get('rating') else 0.0,
                        "sold": int(item_data.get('sold', 0)) if item_data.get('sold') else 0,
                        "brand": watch_details.get('brand', {}),
                        "category": watch_details.get('category', {}),
                        "price_tier": item_data.get('price_tier', 'mid'),
                        "gender_target": item_data.get('gender_target', 'U'),
                        "size_category": item_data.get('size_category', 'medium'),
                        "style_tags": watch_details.get('style_tags', []),
                        "material_tags": watch_details.get('material_tags', []),
                        "color_tags": watch_details.get('color_tags', []),
                        "movement_type_tags": watch_details.get('movement_type_tags', []),
                        "images": watch_details.get('images', []),
                        "interaction_info": interaction_info,
                        "is_ai_recommended": True
                    })
            
            return {
                "recommendations": formatted_recommendations,
                "user_id": user_id,
                "count": len(formatted_recommendations)
            }
            
        except Exception as e:
            logger.error(f"Failed to get recommendations for user {user_id}: {str(e)}")
            return {"error": str(e)}
    
    def get_similar_items(self, watch_id, limit=10):
        """Get similar items for a watch"""
        # Try to load models if not loaded yet
        if not self.model_loaded:
            logger.warning("Models not loaded, attempting to load now...")
            if not self.load_models():
                return {"error": "Model not loaded. Please check logs and ensure models are available."}
        
        try:
            if watch_id not in self.item_features_cache.index:
                return {"error": "Watch not found"}
            
            # Get item features
            item_features = self.item_features_cache.loc[watch_id].values
            
            # Get all other items
            other_items = self.item_features_cache[self.item_features_cache.index != watch_id]
            
            if len(other_items) == 0:
                return {"similar_items": [], "message": "No similar items found"}
            
            # Calculate similarities (simple cosine similarity)
            similarities = []
            for other_id, other_features in other_items.iterrows():
                similarity = np.dot(item_features, other_features.values) / (
                    np.linalg.norm(item_features) * np.linalg.norm(other_features.values)
                )
                similarities.append((other_id, similarity))
            
            # Sort by similarity
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Format results
            similar_items = []
            for item_id, similarity in similarities[:limit]:
                item_data = self.item_features_cache.loc[item_id]
                similar_items.append({
                    "watch_id": int(item_id),
                    "similarity": float(similarity),
                    "name": item_data.get('name', 'Unknown'),
                    "base_price": float(item_data.get('base_price', 0)),
                    "rating": float(item_data.get('rating', 0))
                })
            
            return {
                "similar_items": similar_items,
                "watch_id": watch_id,
                "count": len(similar_items)
            }
            
        except Exception as e:
            logger.error(f"Failed to get similar items for watch {watch_id}: {str(e)}")
            return {"error": str(e)}

# Initialize AI server
ai_server = AIRecommendationServer()

# Load models on app initialization (for gunicorn)
# This will be called when the module is imported
def init_models():
    """Initialize models in background thread"""
    import threading
    def load():
        try:
            logger.info("Attempting to load models on startup...")
            if ai_server.load_models():
                logger.info("Models loaded successfully on startup")
            else:
                logger.warning("Failed to load models on startup. Models will be loaded on first request.")
        except Exception as e:
            logger.error(f"Error loading models on startup: {str(e)}")
    
    # Start loading in background thread to not block app startup
    thread = threading.Thread(target=load, daemon=True)
    thread.start()

# Try to load models on startup
try:
    init_models()
except Exception as e:
    logger.error(f"Failed to initialize models: {str(e)}")

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": ai_server.model_loaded,
        "last_model_load": ai_server.last_model_load.isoformat() if ai_server.last_model_load else None
    })

@app.route('/recommendations/<int:user_id>', methods=['GET'])
def get_recommendations(user_id):
    """Get recommendations for a user"""
    limit = request.args.get('limit', 10, type=int)
    exclude_interactions = request.args.get('exclude_interactions', 'true').lower() == 'true'
    
    result = ai_server.get_recommendations(user_id, limit, exclude_interactions)
    return jsonify(result)

@app.route('/recommendations/anonymous', methods=['GET'])
def get_anonymous_recommendations():
    """Get recommendations for anonymous users using default profiles"""
    limit = request.args.get('limit', 10, type=int)
    profile = request.args.get('profile', 'general', type=str)
    
    result = ai_server.get_anonymous_recommendations(limit, profile)
    return jsonify(result)

@app.route('/similar/<int:watch_id>', methods=['GET'])
def get_similar_items(watch_id):
    """Get similar items for a watch"""
    limit = request.args.get('limit', 10, type=int)
    
    result = ai_server.get_similar_items(watch_id, limit)
    return jsonify(result)

@app.route('/reload-models', methods=['POST'])
def reload_models():
    """Reload models (for updates)"""
    success = ai_server.load_models()
    return jsonify({
        "success": success,
        "message": "Models reloaded successfully" if success else "Failed to reload models"
    })

@app.route('/stats', methods=['GET'])
def get_stats():
    """Get server statistics"""
    return jsonify({
        "model_loaded": ai_server.model_loaded,
        "users_cached": len(ai_server.user_features_cache),
        "items_cached": len(ai_server.item_features_cache),
        "last_model_load": ai_server.last_model_load.isoformat() if ai_server.last_model_load else None
    })

def main():
    """Main function to start the AI server"""
    logger.info("Starting AI Recommendation Server...")
    
    # Load models on startup
    if not ai_server.load_models():
        logger.error("Failed to load models. Exiting...")
        sys.exit(1)
    
    # Start Flask server
    port = int(os.getenv('AI_SERVER_PORT', '5001'))
    host = os.getenv('AI_SERVER_HOST', '0.0.0.0')
    
    logger.info(f"AI Server starting on {host}:{port}")
    app.run(host=host, port=port, debug=False)

if __name__ == "__main__":
    main()
