#!/bin/bash

# Face Yoga API - Google Cloud Run Deployment Script

set -e

echo "🚀 Starting Face Yoga API deployment to Google Cloud Run..."

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo "❌ Google Cloud SDK is not installed. Please install it first."
    exit 1
fi

# Set project variables
PROJECT_ID=${1:-"your-project-id"}
REGION=${2:-"us-central1"}
SERVICE_NAME="face-yoga-api"

echo "📋 Project ID: $PROJECT_ID"
echo "🌍 Region: $REGION"
echo "🔧 Service: $SERVICE_NAME"

# Set the project
gcloud config set project $PROJECT_ID

# Enable required APIs
echo "🔧 Enabling required Google Cloud APIs..."
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com

# Build and deploy using Cloud Build
echo "🏗️ Building and deploying with Cloud Build..."
gcloud builds submit --config cloudbuild.yaml

echo "✅ Deployment completed successfully!"
echo "🌐 Your Face Yoga API is now running on Google Cloud Run"

# Get the service URL
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --region=$REGION --format="value(status.url)")
echo "🔗 Service URL: $SERVICE_URL"

echo "🧪 Testing the deployment..."
curl -f "$SERVICE_URL/health" || echo "⚠️ Health check failed - service might still be starting"

echo "🎉 Face Yoga API with MediaPipe + OpenCV is live!"
