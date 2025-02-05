# Sentiment Analysis ML Model Deployment

## Project Overview

A production-ready machine learning project demonstrating sentiment analysis model deployment using Docker and Kubernetes.

## Features

- Sentiment classification using Naive Bayes
- Flask-based REST API
- Docker containerization
- Kubernetes deployment
- Comprehensive testing
- CI/CD ready

## Prerequisites

- Python 3.9+
- Docker
- Kubernetes (minikube/Docker Desktop)
- kubectl

## Project Structure

```
sentiment-analysis-deployment/
│
├── src/                    # Source code
│   ├── __init__.py
│   ├── sentiment_model.py  # ML Model
│   └── flask_api.py        # API Endpoints
│
├── tests/                  # Unit and integration tests
│   ├── test_model.py
│   └── test_api.py
│
├── scripts/                # Utility scripts
│   ├── train_model.py
│   └── deploy.sh
│
├── k8s/                    # Kubernetes configs
│   ├── sentiment-deployment.yaml
│   └── sentiment-service.yaml
│
├── requirements/           # Dependency files
│   ├── base.txt
│   ├── dev.txt
│   └── prod.txt
│
├── Dockerfile              # Docker image config
└── README.md               # Project documentation
```

## Local Development Setup

### 1. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements/dev.txt
```

### 3. Train Model

```bash
python scripts/train_model.py
```

### 4. Run Tests

```bash
pytest tests/
```

## Docker Deployment

### Build Docker Image

```bash
docker build -t sentiment-analysis-api:v1.0.0 .
```

### Run Docker Container

```bash
docker run -p 5000:5000 sentiment-analysis-api:v1.0.0
```

## Kubernetes Deployment

### Deploy to Kubernetes

```bash
kubectl apply -f k8s/sentiment-deployment.yaml
kubectl apply -f k8s/sentiment-service.yaml
```

### Verify Deployment

```bash
kubectl get deployments
kubectl get services
kubectl get pods
```

## API Usage

### Predict Sentiment

```bash
curl -X POST http://localhost:5000/predict \
     -H "Content-Type: application/json" \
     -d '{"texts":["I love this product!", "This is terrible."]}'
```

## Contributing

1. Fork the repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Create Pull Request

## License

MIT License

## Contact

[Your Contact Information]
