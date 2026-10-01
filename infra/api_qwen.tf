# The Qwen student (GGUF Q4_K_M) on an EC2 server with llama.cpp: POST /extract-qwen.
# Same pattern as api_ec2.tf (secret header from API Gateway, API key, SSM instead of SSH).
# A generative model needs CPU: each request takes seconds. Off by default, and expensive
# while it is on (a c7i.xlarge is about 0.20 USD an hour, ~140 USD a month):
#   terraform apply -var bert_image_tag=$TAG -var qwen_api=true    # create
#   terraform apply -var bert_image_tag=$TAG -var qwen_api=false   # destroy

variable "qwen_api" {
  description = "Create the EC2 server for the Qwen student"
  type        = bool
  default     = false
}

variable "qwen_instance_type" {
  type    = string
  default = "c7i.xlarge" # 4 vCPUs, 8 GB: llama.cpp uses every core for each request
}

resource "aws_iam_role" "api_qwen" {
  name               = "pt-bio-extract-api-qwen"
  assume_role_policy = data.aws_iam_policy_document.ec2_assume.json
}

# The server reads its GGUF model and the origin secret, and nothing else
resource "aws_iam_role_policy" "api_qwen" {
  name = "model-and-secret"
  role = aws_iam_role.api_qwen.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["s3:GetObject"]
        Resource = ["${aws_s3_bucket.data.arn}/v1/models/qwen1.5b-lora-r16/*"]
      },
      {
        Effect   = "Allow"
        Action   = ["ssm:GetParameter"]
        Resource = [aws_ssm_parameter.origin_secret.arn]
      },
    ]
  })
}

resource "aws_iam_role_policy_attachment" "api_qwen_ssm" {
  role       = aws_iam_role.api_qwen.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

resource "aws_iam_instance_profile" "api_qwen" {
  name = "pt-bio-extract-api-qwen"
  role = aws_iam_role.api_qwen.name
}

resource "aws_instance" "qwen" {
  count                  = var.qwen_api ? 1 : 0
  ami                    = data.aws_ssm_parameter.al2023.value
  instance_type          = var.qwen_instance_type
  iam_instance_profile   = aws_iam_instance_profile.api_qwen.name
  vpc_security_group_ids = [aws_security_group.api_ec2.id] # port 8080, checked by the secret

  metadata_options {
    http_tokens = "required" # IMDSv2 only
  }

  root_block_device {
    volume_size = 20
    volume_type = "gp3"
  }

  # First boot: builds llama-cpp-python (compiles llama.cpp: about 5-10 minutes), downloads
  # the model and starts the API as a systemd service. Log: /var/log/cloud-init-output.log
  user_data_replace_on_change = true
  user_data                   = <<-EOT
    #!/bin/bash
    set -euxo pipefail
    REGION=${data.aws_region.current.region}

    dnf install -y python3.12 python3.12-pip python3.12-devel gcc-c++ cmake git
    git clone https://github.com/pinthoz/wiki-bio-distill.git /opt/qwenapi
    python3.12 -m venv /opt/qwenapi/venv
    /opt/qwenapi/venv/bin/pip install -q -r /opt/qwenapi/app_qwen/requirements.txt

    mkdir -p /opt/model
    aws s3 cp "s3://${aws_s3_bucket.data.id}/v1/models/qwen1.5b-lora-r16/student-q4_k_m.gguf" /opt/model/ --region "$REGION"

    SECRET=$(aws ssm get-parameter --name "${aws_ssm_parameter.origin_secret.name}" \
      --with-decryption --query Parameter.Value --output text --region "$REGION")
    printf 'ORIGIN_SECRET=%s\n' "$SECRET" > /etc/qwenapi.env
    chmod 600 /etc/qwenapi.env

    useradd --system --no-create-home qwenapi || true
    cat > /etc/systemd/system/qwenapi.service <<'UNIT'
    [Unit]
    Description=Qwen student biography extraction API
    After=network-online.target

    [Service]
    User=qwenapi
    WorkingDirectory=/opt/qwenapi
    Environment=MODEL_PATH=/opt/model/student-q4_k_m.gguf
    Environment=PYTHONPATH=/opt/qwenapi:/opt/qwenapi/app_qwen
    EnvironmentFile=/etc/qwenapi.env
    ExecStart=/opt/qwenapi/venv/bin/python /opt/qwenapi/app_qwen/server.py
    Restart=always

    [Install]
    WantedBy=multi-user.target
    UNIT

    systemctl daemon-reload
    systemctl enable --now qwenapi
  EOT

  tags = {
    Name = "pt-bio-extract-qwen"
  }
}

resource "aws_apigatewayv2_integration" "qwen" {
  count                  = var.qwen_api ? 1 : 0
  api_id                 = aws_apigatewayv2_api.http.id
  integration_type       = "HTTP_PROXY"
  integration_method     = "POST"
  integration_uri        = "http://${aws_instance.qwen[0].public_dns}:8080/extract"
  payload_format_version = "1.0"
  timeout_milliseconds   = 29000 # the API Gateway maximum: a generation must finish in 29 s

  request_parameters = {
    "overwrite:header.x-origin-verify" = random_password.origin_secret.result
  }
}

resource "aws_apigatewayv2_route" "qwen" {
  count              = var.qwen_api ? 1 : 0
  api_id             = aws_apigatewayv2_api.http.id
  route_key          = "POST /extract-qwen"
  target             = "integrations/${aws_apigatewayv2_integration.qwen[0].id}"
  authorization_type = "CUSTOM"
  authorizer_id      = aws_apigatewayv2_authorizer.api_key.id
}

output "qwen_instance_id" {
  value = one(aws_instance.qwen[*].id)
}
