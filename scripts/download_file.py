import boto3
import argparse
import os
from omegaconf import OmegaConf
from botocore.exceptions import ClientError


def download_file_from_s3(bucket_name: str, s3_key: str, local_file_path: str):
    """
    Downloads a file from an S3 bucket.

    Args:
        bucket_name (str): The name of the S3 bucket.
        s3_key (str): The source key (path) in the S3 bucket.
        local_file_path (str): The local path to save the downloaded file.
    """
    s3_client = boto3.client('s3')

    # Ensure the local directory exists
    local_dir = os.path.dirname(local_file_path)
    if local_dir:
        os.makedirs(local_dir, exist_ok=True)
    
    print(f"Downloading s3://{bucket_name}/{s3_key} to '{local_file_path}'")
    try:
        s3_client.download_file(bucket_name, s3_key, local_file_path)
        print(f"  -> Download successful.")

    except ClientError as e:
        if e.response['Error']['Code'] == '404':
            print(f"  !! Error: The file was not found at s3://{bucket_name}/{s3_key}")
        else:
            print(f"  !! An error occurred: {e}")
    except Exception as e:
        print(f"  !! An error occurred: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Download a single file from S3.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--config-path",
        default="conf/config.yaml",
        help="Path to the project config file (default: conf/config.yaml)."
    )
    parser.add_argument(
        "--s3-key", 
        required=True, 
        help="The key (path) of the file in the S3 bucket to download."
    )
    parser.add_argument(
        "--file",
        required=True,
        help="The local path to save the downloaded file."
    )
    
    args = parser.parse_args()

    # Load configuration from YAML file
    try:
        config = OmegaConf.load(args.config_path)
        bucket_name = config.s3.bucket
    except FileNotFoundError:
        print(f"Error: Config file not found at '{args.config_path}'")
        exit(1)
    except Exception as e:
        print(f"Error reading or parsing config file: {e}")
        exit(1)

    download_file_from_s3(bucket_name, args.s3_key, args.file) 

"""
Example of usage:
PROJECT_ROOT=/home/vime725h/LatentDiffusion \
python -m scripts.download_file \
    --s3-key "checkpoints/autoencoder-num_latents=128-wikipedia-final-128/100000.pth" \
    --file "checkpoints/autoencoder-num_latents=128-wikipedia-final-128/100000.pth"
"""