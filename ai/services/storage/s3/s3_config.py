import os

AWS_REGION_NAME = os.getenv("AWS_REGION_NAME", "us-west-2")
AWS_BUCKET_NAME = os.getenv("AWS_BUCKET_NAME", "geni-bucket")

print(f"AWS REGION NAME: {AWS_REGION_NAME}")
print(f"AWS BUCKET NAME: {AWS_BUCKET_NAME}")