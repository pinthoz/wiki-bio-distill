# The BERT API on an always-on EC2 server, next to the Lambdas: POST /extract-bert-ec2.
# Same code (app_bert/app.py, wrapped by app_bert/server.py), same API key, same API Gateway.
# Off by default: the server costs money every hour it exists (about 20 USD a month).
#   terraform apply -var bert_image_tag=$TAG -var ec2_api=true    # create
#   terraform apply -var bert_image_tag=$TAG -var ec2_api=false   # destroy
#
# Request path: CloudFront -> API Gateway (API key) -> http://<instance>:8080 with a secret
# header that only API Gateway knows, so calling the instance directly gets a 403.

variable "ec2_api" {
  description = "Create the EC2 server for the BERT API"
  type        = bool
  default     = false
}

variable "ec2_api_instance_type" {
  type    = string
  default = "t3.small" # 2 vCPUs, 2 GB RAM: enough for the fp32 model (~1 GB in use)
}

variable "ec2_api_model_file" {
  type    = string
  default = "model.onnx" # fp32: the fastest per request; "model.quant.onnx" needs less RAM
}

data "aws_region" "current" {}

# Latest Amazon Linux 2023, looked up through AWS's public SSM parameter
data "aws_ssm_parameter" "al2023" {
  name = "/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64"
}

# Shared secret between API Gateway and the server, kept in SSM; the instance reads it at boot
resource "random_password" "origin_secret" {
  length  = 40
  special = false
}

resource "aws_ssm_parameter" "origin_secret" {
  name  = "/pt-bio-extract/origin-secret"
  type  = "SecureString"
  value = random_password.origin_secret.result
}

resource "aws_iam_role" "api_ec2" {
  name               = "pt-bio-extract-api-ec2"
  assume_role_policy = data.aws_iam_policy_document.ec2_assume.json
}

# The server reads its model files and the secret, and nothing else
resource "aws_iam_role_policy" "api_ec2" {
  name = "model-and-secret"
  role = aws_iam_role.api_ec2.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["s3:GetObject"]
        Resource = ["${aws_s3_bucket.data.arn}/v1/models/bert-token/*"]
      },
      {
        Effect   = "Allow"
        Action   = ["ssm:GetParameter"]
        Resource = [aws_ssm_parameter.origin_secret.arn]
      },
    ]
  })
}

# A shell through SSM Session Manager, without SSH
resource "aws_iam_role_policy_attachment" "api_ec2_ssm" {
  role       = aws_iam_role.api_ec2.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

resource "aws_iam_instance_profile" "api_ec2" {
  name = "pt-bio-extract-api-ec2"
  role = aws_iam_role.api_ec2.name
}

# Port 8080 is open because API Gateway has no fixed IP addresses; the secret header
# is what keeps everyone else out
resource "aws_security_group" "api_ec2" {
  name_prefix = "pt-bio-extract-api-ec2-"
  description = "BERT API server: 8080 from API Gateway (checked by a secret header)"

  ingress {
    from_port   = 8080
    to_port     = 8080
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_instance" "api" {
  count                  = var.ec2_api ? 1 : 0
  ami                    = data.aws_ssm_parameter.al2023.value
  instance_type          = var.ec2_api_instance_type
  iam_instance_profile   = aws_iam_instance_profile.api_ec2.name
  vpc_security_group_ids = [aws_security_group.api_ec2.id]

  metadata_options {
    http_tokens = "required" # IMDSv2 only
  }

  root_block_device {
    volume_size = 16
    volume_type = "gp3"
  }

  # Runs once, at first boot (cloud-init): installs the API and starts it as a systemd service.
  # Its log is in /var/log/cloud-init-output.log on the instance
  user_data_replace_on_change = true
  user_data                   = <<-EOT
    #!/bin/bash
    set -euxo pipefail
    REGION=${data.aws_region.current.region}

    # Python 3.12, as in the Lambda image: the pinned numpy needs it
    dnf install -y python3.12 python3.12-pip git
    git clone https://github.com/pinthoz/wiki-bio-distill.git /opt/bertapi
    python3.12 -m venv /opt/bertapi/venv
    /opt/bertapi/venv/bin/pip install -q -r /opt/bertapi/app_bert/requirements.txt

    mkdir -p /opt/model
    for f in ${var.ec2_api_model_file} tokenizer.json lemma_map.json; do
      aws s3 cp "s3://${aws_s3_bucket.data.id}/v1/models/bert-token/$f" /opt/model/ --region "$REGION"
    done

    SECRET=$(aws ssm get-parameter --name "${aws_ssm_parameter.origin_secret.name}" \
      --with-decryption --query Parameter.Value --output text --region "$REGION")
    printf 'ORIGIN_SECRET=%s\n' "$SECRET" > /etc/bertapi.env
    chmod 600 /etc/bertapi.env

    useradd --system --no-create-home bertapi || true
    cat > /etc/systemd/system/bertapi.service <<'UNIT'
    [Unit]
    Description=BERT biography extraction API
    After=network-online.target

    [Service]
    User=bertapi
    WorkingDirectory=/opt/bertapi
    Environment=MODEL_DIR=/opt/model
    Environment=MODEL_FILE=${var.ec2_api_model_file}
    Environment=PYTHONPATH=/opt/bertapi:/opt/bertapi/app_bert
    EnvironmentFile=/etc/bertapi.env
    ExecStart=/opt/bertapi/venv/bin/python /opt/bertapi/app_bert/server.py
    Restart=always

    [Install]
    WantedBy=multi-user.target
    UNIT

    systemctl daemon-reload
    systemctl enable --now bertapi
  EOT

  tags = {
    Name = "pt-bio-extract-api"
  }
}

# API Gateway forwards the request to the server, adding the secret header
resource "aws_apigatewayv2_integration" "ec2" {
  count                  = var.ec2_api ? 1 : 0
  api_id                 = aws_apigatewayv2_api.http.id
  integration_type       = "HTTP_PROXY"
  integration_method     = "POST"
  integration_uri        = "http://${aws_instance.api[0].public_dns}:8080/extract"
  payload_format_version = "1.0" # the only format HTTP_PROXY integrations take
  timeout_milliseconds   = 29000

  request_parameters = {
    "overwrite:header.x-origin-verify" = random_password.origin_secret.result
  }
}

resource "aws_apigatewayv2_route" "ec2" {
  count              = var.ec2_api ? 1 : 0
  api_id             = aws_apigatewayv2_api.http.id
  route_key          = "POST /extract-bert-ec2"
  target             = "integrations/${aws_apigatewayv2_integration.ec2[0].id}"
  authorization_type = "CUSTOM"
  authorizer_id      = aws_apigatewayv2_authorizer.api_key.id
}

output "ec2_api_instance_id" {
  value = one(aws_instance.api[*].id)
}

output "ec2_api_direct_url" {
  description = "For testing only: without the x-origin-verify header it answers 403"
  value       = var.ec2_api ? "http://${aws_instance.api[0].public_dns}:8080" : null
}
