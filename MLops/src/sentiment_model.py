import numpy as np
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from typing import List, Union

class SentimentClassifier:
    """
    A machine learning classifier for sentiment analysis.
    
    Attributes:
        model (Pipeline): Scikit-learn pipeline with vectorizer and classifier
    """
    
    def __init__(self):
        """
        Initialize the sentiment classifier with a pipeline.
        The pipeline includes a CountVectorizer and Multinomial Naive Bayes classifier.
        """
        self.model = Pipeline([
            ('vectorizer', CountVectorizer()),
            ('classifier', MultinomialNB())
        ])
    
    def train(self, X_train: List[str], y_train: List[int]) -> None:
        """
        Train the sentiment classification model.
        
        Args:
            X_train (List[str]): List of text samples for training
            y_train (List[int]): Corresponding sentiment labels (0: Negative, 1: Positive)
        """
        self.model.fit(X_train, y_train)
    
    def predict(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Predict sentiment for given texts.
        
        Args:
            texts (Union[str, List[str]]): Text or list of texts to classify
            
        Returns:
            numpy.ndarray: Predicted sentiment labels (0: Negative, 1: Positive)
        """
        # Convert single string to list for consistent processing
        if isinstance(texts, str):
            texts = [texts]
        return self.model.predict(texts)
    
    def predict_proba(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Predict sentiment probabilities for given texts.
        
        Args:
            texts (Union[str, List[str]]): Text or list of texts to classify
            
        Returns:
            numpy.ndarray: Probability of each class (Negative, Positive)
        """
        # Convert single string to list for consistent processing
        if isinstance(texts, str):
            texts = [texts]
        return self.model.predict_proba(texts)
    
    def save_model(self, filepath: str) -> None:
        """
        Save the trained model to a file.
        
        Args:
            filepath (str): Path to save the model
        """
        joblib.dump(self.model, filepath)
    
    @classmethod
    def load_model(cls, filepath: str) -> 'SentimentClassifier':
        """
        Load a pre-trained model.
        
        Args:
            filepath (str): Path to the saved model
            
        Returns:
            SentimentClassifier: Loaded sentiment classification model
        """
        model = joblib.load(filepath)
        loaded_cls = cls()
        loaded_cls.model = model
        return loaded_cls

def train_example_model() -> SentimentClassifier:
    """
    Train a sample sentiment classification model.
    
    Returns:
        SentimentClassifier: Trained sentiment classifier
    """
    # Comprehensive training data
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
        "Frustrating and complicated to use."
    ]
    y_train = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]

    # Create and train classifier
    classifier = SentimentClassifier()
    classifier.train(X_train, y_train)
    
    return classifier

if __name__ == "__main__":
    # Train and save the model
    classifier = train_example_model()
    classifier.save_model('sentiment_model.joblib')
    
    # Test prediction
    test_texts = [
        "This is a wonderful product",
        "I hate this product"
    ]
    predictions = classifier.predict(test_texts)
    probabilities = classifier.predict_proba(test_texts)
    
    print("Predictions:", predictions)
    print("Probabilities:")
    for text, pred, prob in zip(test_texts, predictions, probabilities):
        print(f"Text: {text}")
        print(f"Prediction: {'Positive' if pred == 1 else 'Negative'}")
        print(f"Probability: {prob}")