resource "aws_s3_bucket" "data" {
  bucket_prefix = "pt-bio-extract-data-"
  force_destroy = true
}

resource "aws_s3_bucket_versioning" "data" {
  bucket = aws_s3_bucket.data.id
  versioning_configuration {
    status = "Enabled"   # each version of the dataset will be recoverable
  }
}

resource "aws_s3_bucket_public_access_block" "data" {
  bucket                  = aws_s3_bucket.data.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Identity used by the Colab notebook: only sees this bucket
resource "aws_iam_user" "colab" {
  name = "pt-bio-extract-colab"
}

resource "aws_iam_user_policy" "colab" {
  user = aws_iam_user.colab.name
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = ["s3:GetObject", "s3:PutObject", "s3:ListBucket"]
      Resource = [aws_s3_bucket.data.arn, "${aws_s3_bucket.data.arn}/*"]
    }]
  })
}

output "data_bucket" {
  value = aws_s3_bucket.data.id
}