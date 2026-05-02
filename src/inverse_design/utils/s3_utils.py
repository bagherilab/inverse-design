import os
import json
import pandas as pd
from pathlib import Path
from typing import Union, Optional
import logging
from urllib.parse import urlparse
import boto3
from botocore.exceptions import ClientError
import tempfile
import xml.etree.ElementTree as ET
import zipfile

logger = logging.getLogger(__name__)


def is_s3_path(path: str) -> bool:
    """Check if a path is an S3 path."""
    return path.startswith("s3://")


def ensure_dir_exists(path: str) -> None:
    """
    Create directory if it doesn't exist. Works for both local and S3 paths.

    Parameters:
    -----------
    path : str
        Directory path (local or S3)
    """
    if is_s3_path(path):
        _ensure_s3_dir_exists(path)
    else:
        os.makedirs(path, exist_ok=True)


def parse_s3_url(s3_url: str) -> tuple:
    """Parse S3 URL into bucket and key components"""
    parsed = urlparse(s3_url)
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    return bucket, key


def _ensure_s3_dir_exists(s3_path: str) -> None:
    """
    Create S3 directory by uploading an empty object with trailing slash.

    Parameters:
    -----------
    s3_path : str
        S3 path (e.g., 's3://bucket/path/')
    """
    try:
        # Parse S3 path
        if not s3_path.endswith("/"):
            s3_path += "/"

        bucket_name, key = parse_s3_url(s3_path)

        # Create S3 client
        s3_client = boto3.client("s3")

        # Upload empty object to create "directory"
        try:
            s3_client.put_object(Bucket=bucket_name, Key=key)
            logger.info(f"Created S3 directory: {s3_path}")
        except ClientError as e:
            if e.response["Error"]["Code"] != "NoSuchBucket":
                raise
            logger.error(f"S3 bucket {bucket_name} does not exist")

    except ImportError:
        logger.error("boto3 is required for S3 operations. Install with: pip install boto3")
        raise
    except Exception as e:
        logger.error(f"Error creating S3 directory {s3_path}: {e}")
        raise


def path_exists(path: str) -> bool:
    """
    Check if a path exists. Works for both local and S3 paths.

    Parameters:
    -----------
    path : str
        Path to check (local or S3)

    Returns:
    --------
    bool
        True if path exists, False otherwise
    """
    if is_s3_path(path):
        return _s3_path_exists(path)
    else:
        return os.path.exists(path)


def _s3_path_exists(s3_path: str) -> bool:
    """
    Check if S3 path exists.

    Parameters:
    -----------
    s3_path : str
        S3 path to check

    Returns:
    --------
    bool
        True if path exists, False otherwise
    """
    try:
        import boto3
        from botocore.exceptions import ClientError

        # Parse S3 path
        bucket_name, key = s3_path.replace("s3://", "").split("/", 1)

        # Create S3 client
        s3_client = boto3.client("s3")

        try:
            # Check if object exists
            s3_client.head_object(Bucket=bucket_name, Key=key)
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                return False
            else:
                raise

    except ImportError:
        logger.error("boto3 is required for S3 operations. Install with: pip install boto3")
        raise
    except Exception as e:
        logger.error(f"Error checking S3 path {s3_path}: {e}")
        return False


def save_json(data: dict, file_path: str, **kwargs) -> None:
    """
    Save data to JSON file. Works for both local and S3 paths.

    Parameters:
    -----------
    data : dict
        Data to save
    file_path : str
        File path (local or S3)
    **kwargs
        Additional arguments for json.dump
    """
    if is_s3_path(file_path):
        _save_json_to_s3(data, file_path, **kwargs)
    else:
        with open(file_path, "w") as f:
            json.dump(data, f, **kwargs)


def _save_json_to_s3(data: dict, s3_path: str, **kwargs) -> None:
    """
    Save JSON data to S3.

    Parameters:
    -----------
    data : dict
        Data to save
    s3_path : str
        S3 file path
    **kwargs
        Additional arguments for json.dump
    """
    try:
        import boto3

        # Parse S3 path
        bucket_name, key = s3_path.replace("s3://", "").split("/", 1)

        # Create S3 client
        s3_client = boto3.client("s3")

        # Convert data to JSON string
        json_str = json.dumps(data, **kwargs)

        # Upload to S3
        s3_client.put_object(
            Bucket=bucket_name,
            Key=key,
            Body=json_str.encode("utf-8"),
            ContentType="application/json",
        )
        logger.info(f"Saved JSON to S3: {s3_path}")

    except ImportError:
        logger.error("boto3 is required for S3 operations. Install with: pip install boto3")
        raise
    except Exception as e:
        logger.error(f"Error saving JSON to S3 {s3_path}: {e}")
        raise


def read_csv(file_path: str, **kwargs) -> pd.DataFrame:
    """
    Read CSV file. Works for both local and S3 paths.

    Parameters:
    -----------
    file_path : str
        File path (local or S3)
    **kwargs
        Additional arguments for pd.read_csv

    Returns:
    --------
    pd.DataFrame
        Loaded data
    """
    if is_s3_path(file_path):
        return _read_csv_from_s3(file_path, **kwargs)
    else:
        return pd.read_csv(file_path, **kwargs)


def _read_csv_from_s3(s3_path: str, **kwargs) -> pd.DataFrame:
    """
    Read CSV file from S3.

    Parameters:
    -----------
    s3_path : str
        S3 file path
    **kwargs
        Additional arguments for pd.read_csv

    Returns:
    --------
    pd.DataFrame
        Loaded data
    """
    try:
        import boto3
        import io

        # Parse S3 path
        bucket_name, key = s3_path.replace("s3://", "").split("/", 1)

        # Create S3 client
        s3_client = boto3.client("s3")

        # Download file content
        response = s3_client.get_object(Bucket=bucket_name, Key=key)
        content = response["Body"].read()

        # Read CSV from content
        df = pd.read_csv(io.BytesIO(content), **kwargs)
        logger.info(f"Read CSV from S3: {s3_path}")
        return df

    except ImportError:
        logger.error("boto3 is required for S3 operations. Install with: pip install boto3")
        raise
    except Exception as e:
        logger.error(f"Error reading CSV from S3 {s3_path}: {e}")
        raise


def list_files(directory_path: str, pattern: str = "*") -> list:
    """
    List files in directory. Works for both local and S3 paths.

    Parameters:
    -----------
    directory_path : str
        Directory path (local or S3)
    pattern : str
        File pattern to match (default: "*")

    Returns:
    --------
    list
        List of file names
    """
    if is_s3_path(directory_path):
        return _list_s3_files(directory_path, pattern)
    else:
        return [f.name for f in Path(directory_path).glob(pattern)]


def _list_s3_files(s3_path: str, pattern: str = "*") -> list:
    """
    List files in S3 directory.

    Parameters:
    -----------
    s3_path : str
        S3 directory path
    pattern : str
        File pattern to match

    Returns:
    --------
    list
        List of file names
    """
    try:
        import boto3
        from fnmatch import fnmatch

        # Parse S3 path
        bucket_name, prefix = s3_path.replace("s3://", "").split("/", 1)
        if not prefix.endswith("/"):
            prefix += "/"

        # Create S3 client
        s3_client = boto3.client("s3")

        # List objects
        response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)

        if "Contents" not in response:
            return []

        # Filter by pattern
        files = []
        for obj in response["Contents"]:
            key = obj["Key"]
            filename = key.replace(prefix, "")
            if filename and fnmatch(filename, pattern):
                files.append(filename)

        return files

    except ImportError:
        logger.error("boto3 is required for S3 operations. Install with: pip install boto3")
        raise
    except Exception as e:
        logger.error(f"Error listing S3 files in {s3_path}: {e}")
        return []


def save_dataframe(df: pd.DataFrame, file_path: str, **kwargs) -> None:
    """
    Save DataFrame to CSV file. Works for both local and S3 paths.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to save
    file_path : str
        File path (local or S3)
    **kwargs
        Additional arguments for df.to_csv
    """
    if is_s3_path(file_path):
        _save_dataframe_to_s3(df, file_path, **kwargs)
    else:
        df.to_csv(file_path, **kwargs)


def _save_dataframe_to_s3(df: pd.DataFrame, s3_path: str, **kwargs) -> None:
    """
    Save DataFrame to S3 as CSV.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to save
    s3_path : str
        S3 file path
    **kwargs
        Additional arguments for df.to_csv
    """
    try:
        import boto3
        import io

        # Parse S3 path
        bucket_name, key = s3_path.replace("s3://", "").split("/", 1)

        # Create S3 client
        s3_client = boto3.client("s3")

        # Convert DataFrame to CSV string
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, **kwargs)
        csv_content = csv_buffer.getvalue()

        # Upload to S3
        s3_client.put_object(
            Bucket=bucket_name, Key=key, Body=csv_content.encode("utf-8"), ContentType="text/csv"
        )
        logger.info(f"Saved DataFrame to S3: {s3_path}")

    except ImportError:
        logger.error("boto3 is required for S3 operations. Install with: pip install boto3")
        raise
    except Exception as e:
        logger.error(f"Error saving DataFrame to S3 {s3_path}: {e}")
        raise


def download_s3_files(s3_input_dir: str, local_temp_dir: str, pattern: str = "*.xml") -> list:
    """Download files from S3 directory to local temp directory"""
    bucket, key_prefix = parse_s3_url(s3_input_dir)
    s3_client = boto3.client("s3")

    # List objects in S3 with the prefix
    try:
        response = s3_client.list_objects_v2(Bucket=bucket, Prefix=key_prefix)
        if "Contents" not in response:
            return []
    except ClientError as e:
        raise FileNotFoundError(f"Cannot access S3 path {s3_input_dir}: {e}") from e

    downloaded_files = []
    for obj in response["Contents"]:
        key = obj["Key"]
        # Filter for XML files (or other patterns)
        if key.endswith(".xml") or "input_" in key:
            # Download to local temp directory
            local_filename = os.path.basename(key)
            local_filepath = os.path.join(local_temp_dir, local_filename)

            try:
                s3_client.download_file(bucket, key, local_filepath)
                downloaded_files.append(Path(local_filepath))
            except ClientError as e:
                logging.warning(f"Failed to download {key}: {e}")

    return downloaded_files


def upload_results_to_s3(local_results_dir: str, s3_output_dir: str):
    """Upload all result files from local directory to S3"""
    bucket, key_prefix = parse_s3_url(s3_output_dir)
    s3_client = boto3.client("s3")

    # Walk through local results directory and upload all files
    for root, dirs, files in os.walk(local_results_dir):
        for file in files:
            local_file_path = os.path.join(root, file)
            # Create S3 key maintaining directory structure
            relative_path = os.path.relpath(local_file_path, local_results_dir)
            s3_key = f"{key_prefix}/{relative_path}".replace("\\", "/")  # Handle Windows paths

            try:
                s3_client.upload_file(local_file_path, bucket, s3_key)
                logging.debug(f"Uploaded {local_file_path} to s3://{bucket}/{s3_key}")
            except ClientError as e:
                logging.error(f"Failed to upload {local_file_path}: {e}")


def upload_xml_to_s3(tree: ET.ElementTree, file_path: str):
    # For S3, write to temporary file first, then upload
    with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as temp_file:
        tree.write(temp_file.name, encoding="utf-8", xml_declaration=True)

        # Upload to S3
        parsed = urlparse(file_path)
        bucket = parsed.netloc
        key = parsed.path.lstrip("/")

        s3_client = boto3.client("s3")
        s3_client.upload_file(temp_file.name, bucket, key)

        # Clean up temp file
        os.unlink(temp_file.name)


def download_entire_s3_directory(s3_path: str, local_dir: str, verbose: bool = True):
    """
    Download entire S3 directory structure with all files to local directory

    Args:
        s3_path: S3 URL (e.g., 's3://bucket/path/to/directory')
        local_dir: Local directory to download to
        verbose: Print progress information
    """
    bucket, prefix = parse_s3_url(s3_path)
    s3_client = boto3.client("s3")

    # Ensure prefix ends with / for proper directory listing
    if prefix and not prefix.endswith("/"):
        prefix += "/"

    # Create local directory
    os.makedirs(local_dir, exist_ok=True)

    downloaded_files = []

    try:
        if verbose:
            print(f"Downloading from s3://{bucket}/{prefix} to {local_dir}")

        # Use paginator to handle large directories
        paginator = s3_client.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=bucket, Prefix=prefix)

        for page in pages:
            if "Contents" in page:
                for obj in page["Contents"]:
                    key = obj["Key"]

                    # Skip directory markers
                    if key.endswith("/"):
                        continue

                    # Create local file path maintaining directory structure
                    relative_path = key[len(prefix) :] if prefix else key
                    local_file_path = os.path.join(local_dir, relative_path)

                    # Create directory structure
                    local_file_dir = os.path.dirname(local_file_path)
                    if local_file_dir:
                        os.makedirs(local_file_dir, exist_ok=True)

                    # Download file
                    try:
                        s3_client.download_file(bucket, key, local_file_path)
                        downloaded_files.append(local_file_path)
                    except ClientError as e:
                        print(f"Failed to download {key}: {e}")

        if verbose:
            print(f"Download complete! {len(downloaded_files)} files downloaded to {local_dir}")

        return downloaded_files

    except ClientError as e:
        raise FileNotFoundError(f"Cannot access S3 path {s3_path}: {e}") from e


def upload_results_back_to_s3(local_dir: str, s3_path: str, file_patterns: list = None):
    """
    Upload specific result files back to S3

    Args:
        local_dir: Local directory containing results
        s3_path: S3 path to upload to
        file_patterns: List of file patterns to upload (e.g., ['*.csv', '*.json'])
    """
    if file_patterns is None:
        file_patterns = ["final_metrics.csv", "all_param_df.csv"]

    bucket, prefix = parse_s3_url(s3_path)
    s3_client = boto3.client("s3")

    uploaded_files = []

    for pattern in file_patterns:
        if "*" in pattern:
            # Handle glob patterns
            files = list(Path(local_dir).glob(f"**/{pattern}"))
        else:
            # Handle specific file names
            files = list(Path(local_dir).glob(f"**/{pattern}"))
            if not files:
                # Try direct path
                direct_path = Path(local_dir) / pattern
                if direct_path.exists():
                    files = [direct_path]

        for local_file in files:
            # Create S3 key maintaining relative structure
            relative_path = local_file.relative_to(local_dir)
            s3_key = (
                f"{prefix}/{relative_path}".replace("\\", "/")
                if prefix
                else str(relative_path).replace("\\", "/")
            )

            try:
                s3_client.upload_file(str(local_file), bucket, s3_key)
                uploaded_files.append(f"s3://{bucket}/{s3_key}")
                print(f"Uploaded {relative_path} to S3")
            except ClientError as e:
                print(f"Failed to upload {local_file}: {e}")

    return uploaded_files


def upload_file_to_s3(file_path, s3_destination, aws_profile=None):
    """
    Upload a file to S3.

    Args:
        file_path (str): Path to the local file to upload
        s3_destination (str): S3 destination in format 's3://bucket-name/path/to/file'
                             or 'bucket-name/path/to/file'
        aws_profile (str): AWS profile name (optional)

    Returns:
        bool: True if upload successful, False otherwise

    Example:
        upload_file_to_s3('/local/path/file.zip', 's3://my-bucket/results/file.zip')
        upload_file_to_s3('/local/path/file.zip', 'my-bucket/results/file.zip')
    """

    # Check if local file exists
    if not os.path.exists(file_path):
        print(f"Error: Local file {file_path} does not exist")
        return False

    # Parse S3 destination
    if s3_destination.startswith("s3://"):
        parsed = urlparse(s3_destination)
        bucket_name = parsed.netloc
        s3_key = parsed.path.lstrip("/")
    else:
        # Assume format is bucket-name/path/to/file
        parts = s3_destination.split("/", 1)
        if len(parts) < 2:
            print(f"Error: Invalid S3 destination format: {s3_destination}")
            print("Use format: 's3://bucket-name/path/to/file' or 'bucket-name/path/to/file'")
            return False
        bucket_name = parts[0]
        s3_key = parts[1]

    try:
        # Create S3 client
        if aws_profile:
            session = boto3.Session(profile_name=aws_profile)
            s3_client = session.client("s3")
        else:
            s3_client = boto3.client("s3")

        # Get file size for progress (optional)
        file_size = os.path.getsize(file_path)
        print(f"Uploading {file_path} ({file_size:,} bytes) to s3://{bucket_name}/{s3_key}")

        # Upload file
        s3_client.upload_file(file_path, bucket_name, s3_key)

        print(f"✓ Successfully uploaded to s3://{bucket_name}/{s3_key}")
        return True

    except ClientError as e:
        print(f"✗ Failed to upload {file_path}: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error uploading {file_path}: {e}")
        return False
