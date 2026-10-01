# GPU instance for training the student (train_qlora.py). Off by default: it costs money while
# it exists, so create it only to train and destroy it straight after:
#   terraform apply -var bert_image_tag=$TAG -var train_instance=true    # create
#   terraform apply -var bert_image_tag=$TAG -var train_instance=false   # destroy
# No SSH: you connect with SSM Session Manager, so no key pair and no open port.

variable "train_instance" {
  description = "Create the GPU training instance"
  type        = bool
  default     = false
}

variable "train_spot" {
  description = "Spot (much cheaper, can be interrupted) instead of On-Demand"
  type        = bool
  default     = true
}

variable "train_instance_type" {
  type    = string
  default = "g4dn.xlarge" # 1 NVIDIA T4 (16 GB), 4 vCPUs, 16 GB RAM: the same GPU as Colab
}

# AWS Deep Learning AMI: Ubuntu with the NVIDIA driver, CUDA and PyTorch already installed
data "aws_ami" "dlami" {
  count       = var.train_instance ? 1 : 0
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["Deep Learning OSS Nvidia Driver AMI GPU PyTorch * (Ubuntu 22.04) *"]
  }
  filter {
    name   = "architecture"
    values = ["x86_64"]
  }
}

data "aws_iam_policy_document" "ec2_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ec2.amazonaws.com"]
    }
  }
}

# The instance's identity: it reads the data and writes the results, with no access keys on it
resource "aws_iam_role" "train" {
  name               = "pt-bio-extract-train-ec2"
  assume_role_policy = data.aws_iam_policy_document.ec2_assume.json
}

resource "aws_iam_role_policy" "train_s3" {
  name = "data-bucket"
  role = aws_iam_role.train.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["s3:ListBucket"]
        Resource = [aws_s3_bucket.data.arn]
      },
      {
        Effect   = "Allow"
        Action   = ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"] # delete: `aws s3 sync --delete`
        Resource = ["${aws_s3_bucket.data.arn}/*"]
      },
    ]
  })
}

# Lets SSM Session Manager open a shell on the instance (replaces SSH)
resource "aws_iam_role_policy_attachment" "train_ssm" {
  role       = aws_iam_role.train.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

resource "aws_iam_instance_profile" "train" {
  name = "pt-bio-extract-train-ec2"
  role = aws_iam_role.train.name
}

# No inbound rule at all: SSM connects from the inside out
resource "aws_security_group" "train" {
  name_prefix = "pt-bio-extract-train-"
  description = "Training instance: outbound only (Hugging Face, S3, SSM)"

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_instance" "train" {
  count                  = var.train_instance ? 1 : 0
  ami                    = data.aws_ami.dlami[0].id
  instance_type          = var.train_instance_type
  iam_instance_profile   = aws_iam_instance_profile.train.name
  vpc_security_group_ids = [aws_security_group.train.id]

  # `sudo shutdown -h now` on the instance deletes it, instead of leaving a stopped one behind
  instance_initiated_shutdown_behavior = "terminate"

  dynamic "instance_market_options" {
    for_each = var.train_spot ? [1] : []
    content {
      market_type = "spot"
      spot_options {
        spot_instance_type             = "one-time"
        instance_interruption_behavior = "terminate"
      }
    }
  }

  metadata_options {
    http_tokens = "required" # IMDSv2 only
  }

  root_block_device {
    volume_size           = 150 # the AMI alone takes most of 100 GB
    volume_type           = "gp3"
    delete_on_termination = true
  }

  tags = {
    Name = "pt-bio-extract-train"
  }
}

output "train_instance_id" {
  value = one(aws_instance.train[*].id)
}
