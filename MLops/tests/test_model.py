import os
import pytest
import numpy as np
from src.sentiment_model import SentimentClassifier

class TestSentimentModel:
    @pytest.fixture
    def classifier(self):
        """
        Fixture to create a trained sentiment classifier
        """
        classifier = SentimentClassifier()
        X_train = [
            "I love this product",
            "This is terrible",
            "Great experience",
            "Worst purchase ever"
        ]
        y_train = [1, 0, 1, 0]
        classifier.train(X_train, y_train)
        return classifier

    def test_predict(self, classifier):
        """
        Test prediction functionality
        """
        test_texts = [
            "Amazing product",
            "Horrible experience"
        ]
        predictions = classifier.predict(test_texts)
        
        assert len(predictions) == len(test_texts)
        assert all(pred in [0, 1] for pred in predictions)

    def test_predict_proba(self, classifier):
        """
        Test prediction probabilities
        """
        test_texts = [
            "Amazing product",
            "Horrible experience"
        ]
        probabilities = classifier.predict_proba(test_texts)
        
        assert probabilities.shape[0] == len(test_texts)
        assert probabilities.shape[1] == 2  # Negative and Positive classes
        assert np.allclose(probabilities.sum(axis=1), 1.0)

    def test_save_and_load_model(self, classifier, tmp_path):
        """
        Test model saving and loading
        """
        # Create a temporary file path
        model_path = os.path.join(tmp_path, 'test_model.joblib')
        
        # Save the model
        classifier.save_model(model_path)
        
        # Load the model
        loaded_classifier = SentimentClassifier.load_model(model_path)
        
        # Test loaded model prediction
        test_texts = ["Amazing product", "Horrible experience"]
        original_predictions = classifier.predict(test_texts)
        loaded_predictions = loaded_classifier.predict(test_texts)
        
        np.testing.assert_array_equal(original_predictions, loaded_predictions)