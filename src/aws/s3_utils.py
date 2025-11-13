# src/aws/s3_utils.py
import os
import boto3

def _bucket() -> str:
    b = os.getenv("S3_BUCKET")
    if not b:
        raise RuntimeError("S3_BUCKET not set")
    return b

def _client():
    """
    Create an S3 client.
    - In Lambda: boto3 auto-detects region from the runtime env (AWS_REGION).
    - Locally: use AWS config or the AWS_REGION env var if present.
    """
    region = os.getenv("AWS_REGION")  # present in Lambda; optional locally
    return boto3.client("s3", region_name=region) if region else boto3.client("s3")

def upload_to_s3(key: str, data: bytes) -> str:
    """Upload bytes to S3 and return s3:// URI."""
    client = _client()
    bucket = _bucket()
    client.put_object(Bucket=bucket, Key=key, Body=data)
    return f"s3://{bucket}/{key}"

def download_from_s3(key: str) -> bytes:
    """Download bytes from S3."""
    client = _client()
    bucket = _bucket()
    obj = client.get_object(Bucket=bucket, Key=key)
    return obj["Body"].read()

# Compatibility aliases
put_bytes = upload_to_s3
get_bytes = download_from_s3
