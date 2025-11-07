import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Database configuration
    DB_HOST = os.getenv('DB_HOST', 'localhost')
    DB_PORT = os.getenv('DB_PORT', '5432')
    DB_NAME = os.getenv('DB_NAME', 'watch_shop')
    DB_USER = os.getenv('DB_USER', 'postgres')
    DB_PASSWORD = os.getenv('DB_PASSWORD', 'password')
    
    # Model configuration
    MODEL_SAVE_PATH = os.getenv('MODEL_SAVE_PATH', './models')
    BATCH_SIZE = int(os.getenv('BATCH_SIZE', '32'))
    EPOCHS = int(os.getenv('EPOCHS', '50'))
    LEARNING_RATE = float(os.getenv('LEARNING_RATE', '0.001'))
    
    # Recommendation configuration
    MAX_RECOMMENDATIONS = int(os.getenv('MAX_RECOMMENDATIONS', '10'))
    RECOMMENDATION_EXPIRY_HOURS = int(os.getenv('RECOMMENDATION_EXPIRY_HOURS', '24'))
    
    # Training configuration
    MIN_INTERACTIONS_PER_USER = int(os.getenv('MIN_INTERACTIONS_PER_USER', '5'))
    MIN_INTERACTIONS_PER_ITEM = int(os.getenv('MIN_INTERACTIONS_PER_ITEM', '3'))
    TRAIN_TEST_SPLIT = float(os.getenv('TRAIN_TEST_SPLIT', '0.8'))
    
    # Hybrid model weights
    COLLABORATIVE_WEIGHT = float(os.getenv('COLLABORATIVE_WEIGHT', '0.6'))
    CONTENT_BASED_WEIGHT = float(os.getenv('CONTENT_BASED_WEIGHT', '0.4'))
