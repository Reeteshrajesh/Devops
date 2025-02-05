import os
import logging
from flask import Flask, request, jsonify
from werkzeug.exceptions import HTTPException
from src.sentiment_model import SentimentClassifier

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Load the pre-trained model
try:
    MODEL_PATH = 'sentiment_model.joblib'
    classifier = SentimentClassifier.load_model(MODEL_PATH)
    logger.info(f"Model successfully loaded from {MODEL_PATH}")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    classifier = None

@app.route('/predict', methods=['POST'])
def predict_sentiment():
    """
    Predict sentiment for input texts.
    
    Expected JSON input:
    {
        "texts": ["list of texts to classify"],
        "include_proba": false  # Optional flag to include probabilities
    }
    
    Returns:
    {
        "predictions": [list of sentiment labels],
        "labels": [list of label names],
        "probabilities": [optional probabilities]
    }
    """
    try:
        # Validate input data
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400
        
        data = request.get_json()
        texts = data.get('texts', [])
        include_proba = data.get('include_proba', False)
        
        # Validate texts
        if not texts:
            return jsonify({"error": "No texts provided"}), 400
        
        if not isinstance(texts, list):
            texts = [texts]
        
        # Predict
        predictions = classifier.predict(texts)
        labels = ["Positive" if pred == 1 else "Negative" for pred in predictions]
        
        # Prepare response
        response = {
            "predictions": predictions.tolist(),
            "labels": labels
        }
        
        # Optionally include probabilities
        if include_proba:
            probabilities = classifier.predict_proba(texts)
            response["probabilities"] = probabilities.tolist()
        
        logger.info(f"Processed {len(texts)} text(s)")
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """
    Basic health check endpoint.
    
    Returns:
    {
        "status": "healthy",
        "model_loaded": bool
    }
    """
    return jsonify({
        "status": "healthy", 
        "model_loaded": classifier is not None
    }), 200

@app.errorhandler(HTTPException)
def handle_http_exception(e):
    """
    Handle HTTP exceptions with custom error response
    """
    response = {
        "error": str(e.description),
        "code": e.code
    }
    logger.warning(f"HTTP Exception: {response}")
    return jsonify(response), e.code

@app.errorhandler(Exception)
def handle_global_exception(e):
    """
    Global exception handler
    """
    logger.error(f"Unhandled exception: {e}", exc_info=True)
    return jsonify({
        "error": "Internal server error",
        "details": str(e)
    }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)