import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import joblib
import os
from config import Config

class CollaborativeFilteringModel:
    def __init__(self, n_users, n_items, embedding_dim=50):
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.model = None
        self.user_encoder = None
        self.item_encoder = None
        
    def build_model(self):
        """Build the collaborative filtering model"""
        # User input
        user_input = Input(shape=(), name='user_input')
        user_embedding = Embedding(self.n_users, self.embedding_dim, name='user_embedding')(user_input)
        user_flat = Flatten()(user_embedding)
        
        # Item input
        item_input = Input(shape=(), name='item_input')
        item_embedding = Embedding(self.n_items, self.embedding_dim, name='item_embedding')(item_input)
        item_flat = Flatten()(item_embedding)
        
        # Concatenate embeddings
        concat = Concatenate()([user_flat, item_flat])
        
        # Dense layers
        dense1 = Dense(128, activation='relu')(concat)
        dropout1 = Dropout(0.2)(dense1)
        dense2 = Dense(64, activation='relu')(dropout1)
        dropout2 = Dropout(0.2)(dense2)
        dense3 = Dense(32, activation='relu')(dropout2)
        
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
        """Train the collaborative filtering model"""
        if self.model is None:
            self.build_model()
        
        # Callbacks
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6)
        ]
        
        # Train model
        history = self.model.fit(
            [X_train['user_encoded'], X_train['item_encoded']],
            y_train,
            validation_data=([X_val['user_encoded'], X_val['item_encoded']], y_val),
            epochs=Config.EPOCHS,
            batch_size=Config.BATCH_SIZE,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def predict(self, user_ids, item_ids):
        """Make predictions for user-item pairs"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Encode user and item IDs
        user_encoded = self.user_encoder.transform(user_ids)
        item_encoded = self.item_encoder.transform(item_ids)
        
        predictions = self.model.predict([user_encoded, item_encoded])
        return predictions.flatten()
    
    def get_recommendations(self, user_id, n_recommendations=10, exclude_items=None):
        """Get top recommendations for a user"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Get all items
        all_items = np.arange(self.n_items)
        if exclude_items is not None:
            exclude_encoded = self.item_encoder.transform(exclude_items)
            all_items = np.setdiff1d(all_items, exclude_encoded)
        
        # Create user-item pairs
        user_ids = np.full(len(all_items), user_id)
        item_ids = all_items
        
        # Get predictions
        predictions = self.predict(user_ids, self.item_encoder.inverse_transform(item_ids))
        
        # Get top recommendations
        top_indices = np.argsort(predictions)[::-1][:n_recommendations]
        top_items = self.item_encoder.inverse_transform(all_items[top_indices])
        top_scores = predictions[top_indices]
        
        return list(zip(top_items, top_scores))
    
    def save_model(self, model_path):
        """Save the model and encoders"""
        os.makedirs(model_path, exist_ok=True)
        
        # Save model
        self.model.save(os.path.join(model_path, 'collaborative_model.h5'))
        
        # Save encoders
        joblib.dump(self.user_encoder, os.path.join(model_path, 'user_encoder.pkl'))
        joblib.dump(self.item_encoder, os.path.join(model_path, 'item_encoder.pkl'))
        
        print(f"Model saved to {model_path}")
    
    def load_model(self, model_path):
        """Load the model and encoders"""
        # Load model
        self.model = tf.keras.models.load_model(os.path.join(model_path, 'collaborative_model.h5'))
        
        # Load encoders
        self.user_encoder = joblib.load(os.path.join(model_path, 'user_encoder.pkl'))
        self.item_encoder = joblib.load(os.path.join(model_path, 'item_encoder.pkl'))
        
        print(f"Model loaded from {model_path}")
    
    def evaluate(self, X_test, y_test):
        """Evaluate the model"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        predictions = self.model.predict([X_test['user_encoded'], X_test['item_encoded']])
        predictions = predictions.flatten()
        
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mse)
        
        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse
        }
