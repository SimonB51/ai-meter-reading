#!/usr/bin/env bash
# Show service status + public URL.
cd "$(dirname "$0")/.."
source infra/config.sh
aws ecs describe-express-gateway-service --service-arn "$(get_service_arn)" --region "$REGION" \
  --query '{status: service.status.statusCode, url: service.activeConfigurations[0].ingressPaths[0].endpoint}'
