# BERT student behind an HTTP API: POST /extract-bert (adapted from bias-api)

variable "bert_image_tag" {
  # No default on purpose: a default would roll the Lambda back to an old image
  type = string
}

resource "aws_ecr_repository" "bert" {
  name         = "pt-bio-extract-bert"
  force_delete = true # allows destroy even with images inside

  image_scanning_configuration {
    scan_on_push = true
  }
}

# Keep only the 3 most recent images (storage = cost)
resource "aws_ecr_lifecycle_policy" "bert" {
  repository = aws_ecr_repository.bert.name
  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "keep 3 images"
      selection = {
        tagStatus   = "any"
        countType   = "imageCountMoreThan"
        countNumber = 3
      }
      action = { type = "expire" }
    }]
  })
}

data "aws_iam_policy_document" "lambda_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["lambda.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "bert" {
  name               = "pt-bio-extract-bert-lambda"
  assume_role_policy = data.aws_iam_policy_document.lambda_assume.json
}

# The Lambdas download their model from the data bucket, and may read nothing else
resource "aws_iam_role_policy" "bert_models" {
  name = "read-bert-models"
  role = aws_iam_role.bert.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = ["s3:GetObject"]
      Resource = ["${aws_s3_bucket.data.arn}/v1/models/bert-token/*"]
    }]
  })
}

resource "aws_iam_role_policy_attachment" "bert_logs" {
  role       = aws_iam_role.bert.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

# One Lambda per model, all from the same image: MODEL_FILE picks which ONNX file it loads
locals {
  bert_models = {
    "bert" = {
      route = "POST /extract-bert"
      file  = "model.onnx"       # fp32: exact, but a slow cold start
    }
    "bert-quant" = {
      route = "POST /extract-bert-quant"
      file  = "model.quant.onnx" # quantized: smaller, faster cold start
    }
  }
}

# Created by Terraform so that destroy removes it
resource "aws_cloudwatch_log_group" "bert" {
  for_each          = local.bert_models
  name              = "/aws/lambda/pt-bio-extract-${each.key}"
  retention_in_days = 7
}

resource "aws_lambda_function" "bert" {
  for_each      = local.bert_models
  function_name = "pt-bio-extract-${each.key}"
  role          = aws_iam_role.bert.arn
  package_type  = "Image"
  image_uri     = "${aws_ecr_repository.bert.repository_url}:${var.bert_image_tag}"
  architectures = ["x86_64"] # built on an x86 PC: no emulation needed
  memory_size   = 3008 # CPU scales with memory: a faster cold start
  timeout       = 90   # if init misses its 10 s, the retry inside the first call can finish

  environment {
    variables = {
      MODEL_FILE = each.value.file
      MODEL_S3   = "s3://${aws_s3_bucket.data.id}/v1/models/bert-token/${each.value.file}"
    }
  }

  depends_on = [
    aws_iam_role_policy_attachment.bert_logs,
    aws_iam_role_policy.bert_models,
    aws_cloudwatch_log_group.bert,
  ]
}

resource "aws_apigatewayv2_api" "http" {
  name          = "pt-bio-extract-http"
  protocol_type = "HTTP"

  # The card page (frontend/index.html) calls the API from the browser
  cors_configuration {
    allow_origins = var.cors_origins
    allow_methods = ["POST", "OPTIONS"]
    allow_headers = ["content-type"]
  }
}

variable "cors_origins" {
  description = "Sites allowed to call the API from a browser"
  type        = list(string)
  default     = ["*"] # any site, including the page opened from disk; narrow it once it is hosted
}

resource "aws_apigatewayv2_integration" "bert" {
  for_each               = local.bert_models
  api_id                 = aws_apigatewayv2_api.http.id
  integration_type       = "AWS_PROXY"
  integration_uri        = aws_lambda_function.bert[each.key].invoke_arn
  payload_format_version = "2.0"
}

resource "aws_apigatewayv2_route" "bert" {
  for_each  = local.bert_models
  api_id    = aws_apigatewayv2_api.http.id
  route_key = each.value.route
  target    = "integrations/${aws_apigatewayv2_integration.bert[each.key].id}"
}

# These used to be single resources: rename them in the state instead of re-creating them
moved {
  from = aws_cloudwatch_log_group.bert
  to   = aws_cloudwatch_log_group.bert["bert"]
}
moved {
  from = aws_lambda_function.bert
  to   = aws_lambda_function.bert["bert"]
}
moved {
  from = aws_apigatewayv2_integration.bert
  to   = aws_apigatewayv2_integration.bert["bert"]
}
moved {
  from = aws_apigatewayv2_route.bert
  to   = aws_apigatewayv2_route.bert["bert"]
}
moved {
  from = aws_lambda_permission.bert_apigw
  to   = aws_lambda_permission.bert_apigw["bert"]
}

resource "aws_apigatewayv2_stage" "default" {
  api_id      = aws_apigatewayv2_api.http.id
  name        = "$default"
  auto_deploy = true

  # The API has no key: a low request limit protects your credits
  default_route_settings {
    throttling_burst_limit = 5
    throttling_rate_limit  = 2
  }
}

resource "aws_lambda_permission" "bert_apigw" {
  for_each      = local.bert_models
  statement_id  = "AllowAPIGatewayInvoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.bert[each.key].function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_apigatewayv2_api.http.execution_arn}/*/*"
}

output "bert_ecr_url" {
  value = aws_ecr_repository.bert.repository_url
}

output "api_base_url" {
  description = "Paste it into the card page (frontend/index.html)"
  value       = aws_apigatewayv2_api.http.api_endpoint
}

output "bert_api_url" {
  value = "${aws_apigatewayv2_api.http.api_endpoint}/extract-bert"
}
