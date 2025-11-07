import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import joblib
import os
from config import Config

class ContentBasedFilteringModel:
    def __init__(self, user_feature_dim, item_feature_dim):
        self.user_feature_dim = user_feature_dim
        self.item_feature_dim = item_feature_dim
        self.model = None
        self.user_features_scaler = None
        self.item_features_scaler = None
        
    def build_model(self):
        """Build the content-based filtering model"""
        # User features input
        user_input = Input(shape=(self.user_feature_dim,), name='user_features')
        user_dense1 = Dense(64, activation='relu')(user_input)
        user_dropout1 = Dropout(0.2)(user_dense1)
        user_dense2 = Dense(32, activation='relu')(user_dropout1)
        
        # Item features input
        item_input = Input(shape=(self.item_feature_dim,), name='item_features')
        item_dense1 = Dense(64, activation='relu')(item_input)
        item_dropout1 = Dropout(0.2)(item_dense1)
        item_dense2 = Dense(32, activation='relu')(item_dropout1)
        
        # Concatenate user and item features
        concat = Concatenate()([user_dense2, item_dense2])
        
        # Dense layers
        dense1 = Dense(64, activation='relu')(concat)
        dropout1 = Dropout(0.2)(dense1)
        dense2 = Dense(32, activation='relu')(dropout1)
        dropout2 = Dropout(0.2)(dense2)
        dense3 = Dense(16, activation='relu')(dropout2)
        
        # Output layer
        output = Dense(1, activation='sigmoid')(dense3)
        
        # Create model
        self.model = Model(inputs=[user_input, item_input], outputs=output)
        self.model.compile(
            optimizer=Adam(learning_rate=Config.LEARNING_RATE),
            loss='mse',
            metrics=['mae']
        )
        
        return self.model
    
    def train(self, X_train, y_train, X_val, y_val):
        """Train the content-based filtering model"""
        if self.model is None:
            self.build_model()
        
        # Callbacks
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6)
        ]
        
        # Train model
        history = self.model.fit(
            [X_train['user_features'], X_train['item_features']],
            y_train,
            validation_data=([X_val['user_features'], X_val['item_features']], y_val),
            epochs=Config.EPOCHS,
            batch_size=Config.BATCH_SIZE,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def predict(self, user_features, item_features):
        """Make predictions for user-item pairs"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        predictions = self.model.predict([user_features, item_features])
        return predictions.flatten()
    
    def get_recommendations(self, user_features, all_item_features, n_recommendations=10, exclude_items=None):
        """Get top recommendations for a user based on content similarity"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Get all items
        all_items = list(all_item_features.index)
        if exclude_items is not None:
            all_items = [item for item in all_items if item not in exclude_items]
        
        # Create user-item pairs
        user_features_repeated = np.tile(user_features, (len(all_items), 1))
        item_features_list = all_item_features.loc[all_items].values
        
        # Get predictions
        predictions = self.predict(user_features_repeated, item_features_list)
        
        # Get top recommendations
        top_indices = np.argsort(predictions)[::-1][:n_recommendations]
        top_items = [all_items[i] for i in top_indices]
        top_scores = predictions[top_indices]
        
        return list(zip(top_items, top_scores))
    
    def calculate_item_similarity(self, item_features):
        """Calculate item-item similarity matrix"""
        similarity_matrix = cosine_similarity(item_features)
        return similarity_matrix
    
    def get_similar_items(self, item_id, item_features, similarity_matrix, n_similar=10):
        """Get similar items based on content features"""
        if item_id not in item_features.index:
            return []
        
        item_idx = item_features.index.get_loc(item_id)
        similarities = similarity_matrix[item_idx]
        
        # Get top similar items (excluding the item itself)
        top_indices = np.argsort(similarities)[::-1][1:n_similar+1]
        similar_items = item_features.index[top_indices]
        similar_scores = similarities[top_indices]
        
        return list(zip(similar_items, similar_scores))
    
    def save_model(self, model_path):
        """Save the model and scalers"""
        os.makedirs(model_path, exist_ok=True)
        
        # Save model
        self.model.save(os.path.join(model_path, 'content_based_model.h5'))
        
        # Save scalers
        if self.user_features_scaler is not None:
            joblib.dump(self.user_features_scaler, os.path.join(model_path, 'user_features_scaler.pkl'))
        if self.item_features_scaler is not None:
            joblib.dump(self.item_features_scaler, os.path.join(model_path, 'item_features_scaler.pkl'))
        
        print(f"Model saved to {model_path}")
    
    def load_model(self, model_path):
        """Load the model and scalers"""
        # Load model
        self.model = tf.keras.models.load_model(os.path.join(model_path, 'content_based_model.h5'))
        
        # Load scalers
        user_scaler_path = os.path.join(model_path, 'user_features_scaler.pkl')
        item_scaler_path = os.path.join(model_path, 'item_features_scaler.pkl')
        
        if os.path.exists(user_scaler_path):
            self.user_features_scaler = joblib.load(user_scaler_path)
        if os.path.exists(item_scaler_path):
            self.item_features_scaler = joblib.load(item_scaler_path)
        
        print(f"Model loaded from {model_path}")
    
    def evaluate(self, X_test, y_test):
        """Evaluate the model"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        predictions = self.model.predict([X_test['user_features'], X_test['item_features']])
        predictions = predictions.flatten()
        
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mse)
        
        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse
        }
