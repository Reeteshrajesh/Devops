import os
import sys
import logging

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.sentiment_model import SentimentClassifier

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_training_data():
    """
    Load training data for the sentiment model.
    In a real-world scenario, this would typically load data from a file or database.
    """
    X_train = [
        "I love this product, it's amazing!",
        "This is terrible, very disappointed.",
        "Great experience, would recommend.",
        "Worst purchase ever.",
        "Absolutely fantastic service!",
        "The quality is poor and disappointing.",
        "Incredible performance, exceeded my expectations.",
        "Waste of money, do not buy.",
        "Smooth and efficient, works perfectly.",
        "Frustrating and complicated to use.",
        "Excellent customer support",
        "Shipping was slow and product was damaged",
        "Best purchase I've made this year",
        "Complete waste of money",
        "Highly recommend this product"
    ]
    y_train = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    
    return X_train, y_train

def train_and_save_model(output_path='sentiment_model.joblib'):
    """
    Train the sentiment classification model and save it.
    
    Args:
        output_path (str): Path to save the trained model
    """
    try:
        # Load training data
        X_train, y_train = load_training_data()
        
        # Create and train classifier
        classifier = SentimentClassifier()
        classifier.train(X_train, y_train)
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save the model
        classifier.save_model(output_path)
        
        logger.info(f"Model trained and saved successfully to {output_path}")
        
        # Perform a quick test
        test_texts = [
            "This is an amazing product",
            "I'm really disappointed with this purchase"
        ]
        predictions = classifier.predict(test_texts)
        logger.info("Test Predictions:")
        for text, pred in zip(test_texts, predictions):
            logger.info(f"'{text}': {'Positive' if pred == 1 else 'Negative'}")
        
    except Exception as e:
        logger.error(f"Error training model: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Default model path
    model_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
        'sentiment_model.joblib'
    )
    
    train_and_save_model(model_path)