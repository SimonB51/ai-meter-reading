#!/usr/bin/env bash
# Tear everything down (stops all costs). Asks before deleting data.
cd "$(dirname "$0")/.."
source infra/config.sh

if [ "${CONFIRM:-}" != "yes" ]; then
  read -r -p "Delete the service, the ${BUCKET} bucket (photos) and the ${TABLE} table? [y/N] " ok
  [ "$ok" = "y" ] || exit 1
fi

ARN="$(get_service_arn)"
if [ -n "$ARN" ]; then
  say "Service"
  aws ecs delete-express-gateway-service --service-arn "$ARN" --region "$REGION" >/dev/null
  aws ssm delete-parameter --name "$PARAM" --region "$REGION" || true
fi
say "S3";        aws s3 rb "s3://${BUCKET}" --force || true
say "DynamoDB";  aws dynamodb delete-table --table-name "$TABLE" --region "$REGION" >/dev/null || true
say "ECR";       aws ecr delete-repository --repository-name "$REPO" --force --region "$REGION" >/dev/null || true
say "Logs";      aws logs delete-log-group --log-group-name "$LOG_GROUP" --region "$REGION" || true
say "Task role"
aws iam delete-role-policy --role-name "$TASK_ROLE" --policy-name app-data-access || true
aws iam delete-role --role-name "$TASK_ROLE" || true
echo "(The two shared ECS roles are kept - they cost nothing.)"
