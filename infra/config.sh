# Shared settings for the deploy scripts. Edit REGION / APP if you want.
set -euo pipefail

export AWS_PAGER=""
REGION="${REGION:-us-east-1}"
APP="${APP:-meter-reading}"
ACCOUNT="$(aws sts get-caller-identity --query Account --output text)"

BUCKET="${APP}-photos-${ACCOUNT}"          # bucket names are global -> suffix with account id
TABLE="${APP}-readings"
REPO="${APP}"
ECR="${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com"
IMAGE_TAG="${IMAGE_TAG:-$(git rev-parse --short HEAD 2>/dev/null || date +%Y%m%d%H%M)}"
IMAGE="${ECR}/${REPO}:${IMAGE_TAG}"

EXEC_ROLE="ecsTaskExecutionRole"
INFRA_ROLE="ecsInfrastructureRoleForExpressServices"
TASK_ROLE="${APP}-task-role"
LOG_GROUP="/ecs/${APP}"

say() { printf "\n\033[1;32m==> %s\033[0m\n" "$*"; }

# The service ARN is kept in SSM Parameter Store so that any machine (your Mac, GitHub Actions)
# can find the running service.
PARAM="/${APP}/service-arn"
get_service_arn() { aws ssm get-parameter --name "$PARAM" --region "$REGION" --query Parameter.Value --output text 2>/dev/null || true; }
save_service_arn() { aws ssm put-parameter --name "$PARAM" --value "$1" --type String --overwrite --region "$REGION" >/dev/null; }
