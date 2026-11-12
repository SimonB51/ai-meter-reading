#!/usr/bin/env bash
# First deployment of the AI Meter Reading API on AWS.
#
#   S3 (photos)  +  DynamoDB (readings)  +  ECR (image)  +  ECS Express Mode (Fargate + ALB + autoscaling)
#   CloudWatch Logs come with it.
#
# Prereqs: AWS CLI v2 (recent: `aws ecs create-express-gateway-service help` must work),
#          `aws configure` done with YOUR credentials, Docker running.
# Usage:   ./infra/deploy.sh            (re-runnable: existing resources are skipped)
cd "$(dirname "$0")/.."
source infra/config.sh

say "Account ${ACCOUNT} / region ${REGION}"

# ---------- storage ----------
say "S3 bucket ${BUCKET} (private, encrypted)"
if ! aws s3api head-bucket --bucket "$BUCKET" 2>/dev/null; then
  if [ "$REGION" = "us-east-1" ]; then
    aws s3api create-bucket --bucket "$BUCKET" --region "$REGION"
  else
    aws s3api create-bucket --bucket "$BUCKET" --region "$REGION" \
      --create-bucket-configuration LocationConstraint="$REGION"
  fi
  aws s3api put-public-access-block --bucket "$BUCKET" --public-access-block-configuration \
    BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true
fi

say "DynamoDB table ${TABLE} (on-demand)"
if ! aws dynamodb describe-table --table-name "$TABLE" --region "$REGION" >/dev/null 2>&1; then
  aws dynamodb create-table --table-name "$TABLE" --region "$REGION" \
    --attribute-definitions AttributeName=id,AttributeType=S \
    --key-schema AttributeName=id,KeyType=HASH \
    --billing-mode PAY_PER_REQUEST >/dev/null
  aws dynamodb wait table-exists --table-name "$TABLE" --region "$REGION"
fi

# ---------- IAM ----------
say "IAM roles"
ecs_tasks_trust='{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"ecs-tasks.amazonaws.com"},"Action":"sts:AssumeRole"}]}'
ecs_trust='{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"ecs.amazonaws.com"},"Action":"sts:AssumeRole"}]}'

aws iam get-role --role-name "$EXEC_ROLE" >/dev/null 2>&1 || {
  aws iam create-role --role-name "$EXEC_ROLE" --assume-role-policy-document "$ecs_tasks_trust" >/dev/null
  aws iam attach-role-policy --role-name "$EXEC_ROLE" \
    --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy; }

aws iam get-role --role-name "$INFRA_ROLE" >/dev/null 2>&1 || {
  aws iam create-role --role-name "$INFRA_ROLE" --assume-role-policy-document "$ecs_trust" >/dev/null
  aws iam attach-role-policy --role-name "$INFRA_ROLE" \
    --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSInfrastructureRoleforExpressGatewayServices; }

# least privilege: the app can only touch its own bucket prefix and its own table
aws iam get-role --role-name "$TASK_ROLE" >/dev/null 2>&1 || \
  aws iam create-role --role-name "$TASK_ROLE" --assume-role-policy-document "$ecs_tasks_trust" >/dev/null
aws iam put-role-policy --role-name "$TASK_ROLE" --policy-name app-data-access --policy-document "{
  \"Version\": \"2012-10-17\",
  \"Statement\": [
    {\"Effect\": \"Allow\", \"Action\": [\"s3:PutObject\", \"s3:GetObject\"],
     \"Resource\": \"arn:aws:s3:::${BUCKET}/photos/*\"},
    {\"Effect\": \"Allow\", \"Action\": [\"dynamodb:PutItem\", \"dynamodb:GetItem\", \"dynamodb:Scan\"],
     \"Resource\": \"arn:aws:dynamodb:${REGION}:${ACCOUNT}:table/${TABLE}\"}
  ]}"

aws logs create-log-group --log-group-name "$LOG_GROUP" --region "$REGION" 2>/dev/null || true
aws logs put-retention-policy --log-group-name "$LOG_GROUP" --retention-in-days 14 --region "$REGION"

# ---------- image ----------
say "Build & push image ${IMAGE}"
aws ecr describe-repositories --repository-names "$REPO" --region "$REGION" >/dev/null 2>&1 || \
  aws ecr create-repository --repository-name "$REPO" --region "$REGION" \
    --image-scanning-configuration scanOnPush=true >/dev/null
aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "$ECR"
# --platform matters on Apple Silicon: Fargate runs x86_64 here
docker buildx build --platform linux/amd64 -t "$IMAGE" --push .

# ---------- service ----------
say "ECS Express Mode service (1 vCPU / 2 GB, 1-2 tasks)"
sleep 10   # IAM is eventually consistent
CONTAINER=$(cat <<JSON
{"image": "${IMAGE}", "containerPort": 8080,
 "awsLogsConfiguration": {"logGroup": "${LOG_GROUP}", "logStreamPrefix": "api"},
 "environment": [
   {"name": "STORAGE", "value": "aws"},
   {"name": "S3_BUCKET", "value": "${BUCKET}"},
   {"name": "DDB_TABLE", "value": "${TABLE}"},
   {"name": "AWS_REGION", "value": "${REGION}"},
   {"name": "REVIEW_THRESHOLD", "value": "0.5"}]}
JSON
)
SERVICE_ARN=$(aws ecs create-express-gateway-service --region "$REGION" \
  --service-name "$APP" \
  --execution-role-arn "arn:aws:iam::${ACCOUNT}:role/${EXEC_ROLE}" \
  --infrastructure-role-arn "arn:aws:iam::${ACCOUNT}:role/${INFRA_ROLE}" \
  --task-role-arn "arn:aws:iam::${ACCOUNT}:role/${TASK_ROLE}" \
  --health-check-path /health \
  --cpu 1024 --memory 2048 \
  --primary-container "$CONTAINER" \
  --scaling-target '{"minTaskCount":1,"maxTaskCount":2,"autoScalingMetric":"AVERAGE_CPU","autoScalingTargetValue":60}' \
  --query 'service.serviceArn' --output text)
echo "$SERVICE_ARN" > infra/.service_arn

ENDPOINT=$(aws ecs describe-express-gateway-service --service-arn "$SERVICE_ARN" --region "$REGION" \
  --query 'service.activeConfigurations[0].ingressPaths[0].endpoint' --output text)
say "Done. The service takes ~5 min to become healthy."
echo "URL:  https://${ENDPOINT#https://}"
echo "Logs: aws logs tail ${LOG_GROUP} --follow --region ${REGION}"
