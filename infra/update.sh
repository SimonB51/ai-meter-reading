#!/usr/bin/env bash
# Ship a new version: build -> push -> point the service at the new image (rolling deploy).
cd "$(dirname "$0")/.."
source infra/config.sh
SERVICE_ARN="$(cat infra/.service_arn)"

say "Build & push ${IMAGE}"
aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "$ECR"
docker buildx build --platform linux/amd64 -t "$IMAGE" --push .

say "Update service"
aws ecs update-express-gateway-service --region "$REGION" --service-arn "$SERVICE_ARN" \
  --primary-container "{\"image\": \"${IMAGE}\", \"containerPort\": 8080,
    \"awsLogsConfiguration\": {\"logGroup\": \"${LOG_GROUP}\", \"logStreamPrefix\": \"api\"},
    \"environment\": [
      {\"name\": \"STORAGE\", \"value\": \"aws\"},
      {\"name\": \"S3_BUCKET\", \"value\": \"${BUCKET}\"},
      {\"name\": \"DDB_TABLE\", \"value\": \"${TABLE}\"},
      {\"name\": \"AWS_REGION\", \"value\": \"${REGION}\"},
      {\"name\": \"REVIEW_THRESHOLD\", \"value\": \"${REVIEW_THRESHOLD:-0.5}\"}]}" \
  --query 'service.serviceArn' --output text
