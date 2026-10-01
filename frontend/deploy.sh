#!/usr/bin/env bash
# Publish the card page to S3 + CloudFront.
# Run from anywhere: ./frontend/deploy.sh  (needs `terraform apply` done in ../infra)
set -euo pipefail
cd "$(dirname "$0")"

tf() { terraform -chdir=../infra output -raw "$1"; }
BUCKET=$(tf site_bucket)
DIST=$(tf cloudfront_id)

# A single page with no build step: short cache, so a new version shows up quickly
aws s3 cp index.html "s3://$BUCKET/index.html" \
  --content-type "text/html; charset=utf-8" --cache-control "public,max-age=60"

# Clear the CloudFront cache so the new page is served right away
aws cloudfront create-invalidation --distribution-id "$DIST" --paths "/*" >/dev/null

echo "Deployed: $(tf site_url)"
