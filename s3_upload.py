import boto3
import argparse
import os

def upload_folder_to_s3(local_folder, bucket, s3_folder, aws_access_key, aws_secret_key, region):
    s3_client = boto3.client(
        's3',
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key,
        region_name=region
    )

    for root, dirs, files in os.walk(local_folder):
        for file in files:
            local_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_path, local_folder)
            s3_key = os.path.join(s3_folder, relative_path).replace("\\", "/")  # for Windows compatibility

            print(f"Uploading {local_path} to s3://{bucket}/{s3_key}")
            s3_client.upload_file(local_path, bucket, s3_key)

def main():
    parser = argparse.ArgumentParser(description="Upload local folder to S3 using Boto3 with CLI credentials")
    parser.add_argument('--access-key', required=True, help='AWS Access Key ID')
    parser.add_argument('--secret-key', required=True, help='AWS Secret Access Key')
    parser.add_argument('--region', required=True, help='AWS Region')
    parser.add_argument('--bucket', required=True, help='S3 bucket name')
    parser.add_argument('--local-folder', required=True, help='Local folder to upload')
    parser.add_argument('--s3-folder', required=True, help='Target S3 folder path')

    args = parser.parse_args()

    upload_folder_to_s3(
        args.local_folder,
        args.bucket,
        args.s3_folder,
        args.access_key,
        args.secret_key,
        args.region
    )

if __name__ == '__main__':
    main()
