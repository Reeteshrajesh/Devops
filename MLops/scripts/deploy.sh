#!/bin/bash

# Deployment script for Sentiment Analysis Model

# Exit on any error
set -e

# Color codes for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Docker image and tag
IMAGE_NAME="sentiment-analysis-api"
IMAGE_TAG="v1.0.0"
FULL_IMAGE_NAME="${IMAGE_NAME}:${IMAGE_TAG}"

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Validate dependencies
validate_dependencies() {
    echo -e "${YELLOW}Checking required dependencies...${NC}"
    
    # Check for required tools
    local deps=("docker" "kubectl" "python3")
    for dep in "${deps[@]}"; do
        if ! command_exists "$dep"; then
            echo -e "${RED}Error: $dep is not installed.${NC}"
            exit 1
        fi
    done
    
    echo -e "${GREEN}All dependencies verified.${NC}"
}

# Train the model
train_model() {
    echo -e "${YELLOW}Training sentiment analysis model...${NC}"
    python3 "${PROJECT_ROOT}/scripts/train_model.py"
    echo -e "${GREEN}Model training complete.${NC}"
}

# Build Docker image
build_docker_image() {
    echo -e "${YELLOW}Building Docker image...${NC}"
    docker build -t "${FULL_IMAGE_NAME}" "${PROJECT_ROOT}"
    echo -e "${GREEN}Docker image built successfully.${NC}"
}

# Push to Docker registry (optional, commented out)
# Uncomment and configure for your specific registry
push_to_registry() {
    echo -e "${YELLOW}Pushing image to registry...${NC}"
    # docker login
    # docker push "${FULL_IMAGE_NAME}"
    echo -e "${GREEN}Image pushed to registry.${NC}"
}

# Deploy to Kubernetes
deploy_to_kubernetes() {
    echo -e "${YELLOW}Deploying to Kubernetes...${NC}"
    
    # Apply Kubernetes configurations
    kubectl apply -f "${PROJECT_ROOT}/k8s/sentiment-deployment.yaml"
    kubectl apply -f "${PROJECT_ROOT}/k8s/sentiment-service.yaml"
    
    # Verify deployment
    echo -e "${YELLOW}Verifying deployment...${NC}"
    kubectl get deployments
    kubectl get services
    kubectl get pods
    
    echo -e "${GREEN}Deployment complete.${NC}"
}

# Main deployment workflow
main() {
    echo -e "${GREEN}Starting Sentiment Analysis Model Deployment${NC}"
    
    validate_dependencies
    train_model
    build_docker_image
    # Uncomment if you want to push to a registry
    # push_to_registry
    deploy_to_kubernetes
    
    echo -e "${GREEN}Deployment Workflow Complete!${NC}"
}

# Run the main function
main