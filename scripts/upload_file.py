import boto3
import argparse
from pathlib import Path
import os
from omegaconf import OmegaConf


def upload_file_to_s3(bucket_name: str, local_file_path: str, s3_key: str):
    """
    Uploads a single file to an S3 bucket and makes it public.

    Args:
        bucket_name (str): The name of the S3 bucket.
        local_file_path (str): The path to the local file to upload.
        s3_key (str): The destination key (path) in the S3 bucket.
    """
    s3_client = boto3.client('s3')
    
    print(f"Uploading '{local_file_path}' to s3://{bucket_name}/{s3_key}")
    try:
        s3_client.upload_file(local_file_path, bucket_name, s3_key)
        
        region = s3_client.get_bucket_location(Bucket=bucket_name)['LocationConstraint']
        
        # Format the public URL based on the bucket's region
        if region is None: # Handles us-east-1 which has a different URL format
            url = f"https://{bucket_name}.s3.amazonaws.com/{s3_key}"
        else:
            url = f"https://{bucket_name}.s3.{region}.amazonaws.com/{s3_key}"
            
        print(f"  -> Upload successful.")
        print(f"  -> Public URL: {url}")

    except FileNotFoundError:
        print(f"  !! Error: The file was not found at '{local_file_path}'")
    except Exception as e:
        print(f"  !! An error occurred: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Upload a single file to S3, using its path as the S3 key.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--config-path",
        default="conf/config.yaml",
        help="Path to the project config file (default: conf/config.yaml)."
    )
    parser.add_argument(
        "--file", 
        required=True, 
        help="Path to the local file to upload."
    )
    parser.add_argument(
        "--s3-key",
        required=False,
        help="The key (path) for the file in the S3 bucket.\n"
             "If not provided, the local file path will be used as the key.\n"
             "Example: checkpoints/diffusion_checkpoints/model.pth"
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

    # Default the S3 key to the local file path if not provided
    s3_key = args.s3_key if args.s3_key else args.file

    upload_file_to_s3(bucket_name, args.file, s3_key) 

# python scripts/upload_file.py --file "checkpoints/autoencoder-num_latents=128-wikipedia-final-128/100000.pth"
