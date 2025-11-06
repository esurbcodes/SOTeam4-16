# src/aws/s3_utils.py
import os
import boto3

# ---- Environment setup ----
def _bucket() -> str:
    b = os.getenv("S3_BUCKET")
    if not b:
        raise RuntimeError("S3_BUCKET not set")
    return b

def _client():
    """
    Create an S3 client.
    - In AWS Lambda: boto3 auto-detects AWS_REGION from the runtime.
    - Locally: your AWS CLI config or .env provides credentials and region.
    """
    return boto3.client("s3")

# ---- Core operations ----
def upload_to_s3(key: str, data: bytes) -> str:
    """Upload bytes to S3 and return s3:// URI."""
    client = _client()
    bucket = _bucket()
    client.put_object(Bucket=bucket, Key=key, Body=data)
    return f"s3://{bucket}/{key}"
    return f"s3://{bucket}/{key}"

def download_from_s3(key: str) -> bytes:
    """Download bytes from S3."""
    client = _client()
    bucket = _bucket()
    obj = client.get_object(Bucket=bucket, Key=key)
    return obj["Body"].read()

# ---- Aliases (for compatibility) ----
def put_bytes(key: str, data: bytes) -> str:
    return upload_to_s3(key, data)

def get_bytes(key: str) -> bytes:
    return download_from_s3(key)
