import boto3
from pathlib import Path
from io import BytesIO
from llm_agents_introduction.types import URL


def upload_to_s3(
    data: bytes, item_name: str, bucket: str = "genai-agents-intro"
) -> URL:
    s3 = boto3.client("s3")
    s3.upload_fileobj(BytesIO(data), bucket, item_name)

    return f"https://{bucket}.s3.amazonaws.com/{item_name}"


def upload_file_to_s3(file_path: str, bucket: str = "genai-agents-intro") -> URL:
    file_name = Path(file_path).name
    s3 = boto3.client("s3")

    s3.upload_file(file_path, bucket, file_name)

    return f"https://{bucket}.s3.amazonaws.com/{file_name}"
