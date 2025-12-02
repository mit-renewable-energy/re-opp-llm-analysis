"""
S3 Data Management Utility for Renewable Energy Dispute Characterization

This module provides functions to:
1. Upload local data files to S3
2. Download data from S3 on demand with local caching
3. Sync entire data directories to/from S3

Usage:
    from utils.s3_data import ensure_data_available, sync_to_s3

    # Ensure a file is available locally (downloads from S3 if missing)
    path = ensure_data_available("data/final/analysis_with_relevance.pkl")

    # Upload all local data to S3
    sync_to_s3(source_dir="/path/to/pcloud/data")
"""

import os
import json
import boto3
from pathlib import Path
from typing import Optional, List, Union
from botocore.exceptions import ClientError
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class S3Config:
    """S3 configuration settings"""
    BUCKET = "mitrenewableenergy"
    REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")

    # Data directories to sync (S3 structure now mirrors local repo)
    DATA_DIRS = ["data", "archive", "viz"]

    @classmethod
    def get_s3_path(cls, local_path: str) -> str:
        """
        Convert local path to S3 key.

        S3 structure now mirrors local repo structure exactly:
        - data/raw/*, data/processed/*, data/final/*
        - data/processed/results/* (per-plant JSONs)
        - archive/*
        - viz/*
        """
        # Remove leading ./ or / if present
        clean_path = local_path.lstrip("./").lstrip("/")
        return clean_path

    @classmethod
    def get_local_path(cls, s3_key: str) -> str:
        """Convert S3 key to local path (relative to project root)"""
        # S3 structure mirrors local, so path is the same
        return s3_key


def get_s3_client():
    """Create and return a boto3 S3 client using credentials from .env"""
    return boto3.client(
        's3',
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1")
    )


def get_s3_resource():
    """Create and return a boto3 S3 resource using credentials from .env"""
    return boto3.resource(
        's3',
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1")
    )


def upload_file(local_path: Union[str, Path], s3_key: Optional[str] = None,
                show_progress: bool = True) -> bool:
    """
    Upload a single file to S3.

    Args:
        local_path: Path to the local file
        s3_key: S3 key (path in bucket). If None, derives from local_path
        show_progress: Whether to print progress

    Returns:
        True if successful, False otherwise
    """
    local_path = Path(local_path)
    if not local_path.exists():
        print(f"Error: File not found: {local_path}")
        return False

    if s3_key is None:
        s3_key = S3Config.get_s3_path(str(local_path))

    s3 = get_s3_client()
    try:
        if show_progress:
            print(f"Uploading: {local_path} -> s3://{S3Config.BUCKET}/{s3_key}")
        s3.upload_file(str(local_path), S3Config.BUCKET, s3_key)
        return True
    except ClientError as e:
        print(f"Error uploading {local_path}: {e}")
        return False


def upload_directory(local_dir: Union[str, Path], s3_prefix: Optional[str] = None,
                     extensions: Optional[List[str]] = None,
                     show_progress: bool = True) -> int:
    """
    Upload an entire directory to S3.

    Args:
        local_dir: Path to local directory
        s3_prefix: S3 prefix (path in bucket). If None, derives from local_dir
        extensions: List of file extensions to include (e.g., ['.csv', '.pkl']).
                   If None, uploads all files.
        show_progress: Whether to print progress

    Returns:
        Number of files uploaded
    """
    local_dir = Path(local_dir)
    if not local_dir.exists():
        print(f"Error: Directory not found: {local_dir}")
        return 0

    if s3_prefix is None:
        s3_prefix = S3Config.get_s3_path(str(local_dir))

    uploaded = 0
    for root, dirs, files in os.walk(local_dir):
        # Skip __pycache__ and hidden directories
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']

        for file in files:
            # Skip hidden files
            if file.startswith('.'):
                continue

            # Filter by extension if specified
            if extensions and not any(file.endswith(ext) for ext in extensions):
                continue

            local_path = Path(root) / file
            # Calculate relative path from local_dir
            rel_path = local_path.relative_to(local_dir)
            s3_key = f"{s3_prefix}/{rel_path}"

            if upload_file(local_path, s3_key, show_progress):
                uploaded += 1

    print(f"Uploaded {uploaded} files from {local_dir}")
    return uploaded


def download_file(s3_key: str, local_path: Optional[Union[str, Path]] = None,
                  show_progress: bool = True) -> Optional[Path]:
    """
    Download a file from S3.

    Args:
        s3_key: S3 key (path in bucket)
        local_path: Local path to save to. If None, derives from s3_key
        show_progress: Whether to print progress

    Returns:
        Path to downloaded file, or None if failed
    """
    if local_path is None:
        local_path = Path(S3Config.get_local_path(s3_key))
    else:
        local_path = Path(local_path)

    # Create parent directories if needed
    local_path.parent.mkdir(parents=True, exist_ok=True)

    s3 = get_s3_client()
    try:
        if show_progress:
            print(f"Downloading: s3://{S3Config.BUCKET}/{s3_key} -> {local_path}")
        s3.download_file(S3Config.BUCKET, s3_key, str(local_path))
        return local_path
    except ClientError as e:
        if e.response['Error']['Code'] == '404':
            print(f"File not found in S3: {s3_key}")
        else:
            print(f"Error downloading {s3_key}: {e}")
        return None


def file_exists_in_s3(s3_key: str) -> bool:
    """Check if a file exists in S3"""
    s3 = get_s3_client()
    try:
        s3.head_object(Bucket=S3Config.BUCKET, Key=s3_key)
        return True
    except ClientError:
        return False


def ensure_data_available(filename: str, project_root: Optional[Path] = None) -> Optional[Path]:
    """
    Ensure a data file is available locally. Downloads from S3 if missing.

    This is the main entry point for scripts that need data files.
    It checks if the file exists locally, and if not, downloads it from S3.

    Args:
        filename: Relative path to the file (e.g., "data/final/analysis.pkl")
        project_root: Project root directory. If None, uses current directory.

    Returns:
        Path to the local file, or None if not available
    """
    if project_root is None:
        project_root = Path.cwd()
    else:
        project_root = Path(project_root)

    local_path = project_root / filename

    # If file exists locally, return it
    if local_path.exists():
        return local_path

    # Try to download from S3
    s3_key = S3Config.get_s3_path(filename)

    # Check if USE_S3 is enabled (default to True if file doesn't exist locally)
    use_s3 = os.getenv("USE_S3", "true").lower() == "true"

    if not use_s3:
        print(f"File not found locally and USE_S3 is disabled: {filename}")
        return None

    print(f"File not found locally, downloading from S3: {filename}")
    result = download_file(s3_key, local_path)

    return result


def list_s3_contents(prefix: Optional[str] = None, show_sizes: bool = True) -> List[dict]:
    """
    List contents of the S3 bucket.

    Args:
        prefix: Prefix to filter by (e.g., "data/", "archive/")
        show_sizes: Whether to print file sizes

    Returns:
        List of dicts with 'Key', 'Size', 'LastModified'
    """
    s3 = get_s3_client()

    contents = []
    paginator = s3.get_paginator('list_objects_v2')

    paginate_args = {'Bucket': S3Config.BUCKET}
    if prefix:
        paginate_args['Prefix'] = prefix

    try:
        for page in paginator.paginate(**paginate_args):
            if 'Contents' in page:
                for obj in page['Contents']:
                    contents.append({
                        'Key': obj['Key'],
                        'Size': obj['Size'],
                        'LastModified': obj['LastModified']
                    })
                    if show_sizes:
                        size_mb = obj['Size'] / (1024 * 1024)
                        print(f"  {obj['Key']} ({size_mb:.2f} MB)")
    except ClientError as e:
        print(f"Error listing S3 contents: {e}")

    print(f"\nTotal: {len(contents)} files")
    return contents


def sync_to_s3(source_dir: Optional[Union[str, Path]] = None,
               directories: Optional[List[str]] = None,
               dry_run: bool = False) -> int:
    """
    Upload all data files from source directory to S3.

    Args:
        source_dir: Source directory containing data. If None, uses current directory.
        directories: List of subdirectories to sync. If None, uses S3Config.DATA_DIRS
        dry_run: If True, only print what would be uploaded without actually uploading

    Returns:
        Number of files uploaded (or would be uploaded if dry_run)
    """
    if source_dir is None:
        source_dir = Path.cwd()
    else:
        source_dir = Path(source_dir)

    if directories is None:
        directories = S3Config.DATA_DIRS

    print(f"Syncing to S3: s3://{S3Config.BUCKET}/")
    print(f"Source: {source_dir}")
    print(f"Directories: {directories}")
    if dry_run:
        print("DRY RUN - no files will be uploaded")
    print("-" * 50)

    total_uploaded = 0

    for dir_name in directories:
        dir_path = source_dir / dir_name
        if dir_path.exists():
            print(f"\nProcessing: {dir_name}/")
            if dry_run:
                # Count files that would be uploaded
                count = sum(1 for _ in dir_path.rglob('*') if _.is_file() and not _.name.startswith('.'))
                print(f"  Would upload {count} files")
                total_uploaded += count
            else:
                count = upload_directory(dir_path)
                total_uploaded += count
        else:
            print(f"Skipping (not found): {dir_name}/")

    print("-" * 50)
    print(f"Total: {total_uploaded} files {'would be ' if dry_run else ''}uploaded")
    return total_uploaded


def sync_from_s3(target_dir: Optional[Union[str, Path]] = None,
                 directories: Optional[List[str]] = None,
                 dry_run: bool = False) -> int:
    """
    Download all data files from S3 to target directory.

    Args:
        target_dir: Target directory to download to. If None, uses current directory.
        directories: List of subdirectories to sync. If None, uses S3Config.DATA_DIRS
        dry_run: If True, only print what would be downloaded without actually downloading

    Returns:
        Number of files downloaded (or would be downloaded if dry_run)
    """
    if target_dir is None:
        target_dir = Path.cwd()
    else:
        target_dir = Path(target_dir)

    if directories is None:
        directories = S3Config.DATA_DIRS

    print(f"Syncing from S3: s3://{S3Config.BUCKET}/")
    print(f"Target: {target_dir}")
    print(f"Directories: {directories}")
    if dry_run:
        print("DRY RUN - no files will be downloaded")
    print("-" * 50)

    s3 = get_s3_client()
    total_downloaded = 0

    for dir_name in directories:
        prefix = dir_name
        print(f"\nProcessing: {dir_name}/")

        paginator = s3.get_paginator('list_objects_v2')
        try:
            for page in paginator.paginate(Bucket=S3Config.BUCKET, Prefix=prefix):
                if 'Contents' in page:
                    for obj in page['Contents']:
                        s3_key = obj['Key']
                        local_rel_path = S3Config.get_local_path(s3_key)
                        local_path = target_dir / local_rel_path

                        if dry_run:
                            print(f"  Would download: {s3_key}")
                            total_downloaded += 1
                        else:
                            if download_file(s3_key, local_path, show_progress=True):
                                total_downloaded += 1
        except ClientError as e:
            print(f"Error listing {prefix}: {e}")

    print("-" * 50)
    print(f"Total: {total_downloaded} files {'would be ' if dry_run else ''}downloaded")
    return total_downloaded


# CLI interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="S3 Data Management for Renewable Energy Project")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # List command
    list_parser = subparsers.add_parser("list", help="List S3 contents")
    list_parser.add_argument("--prefix", help="Filter by prefix")

    # Upload command
    upload_parser = subparsers.add_parser("upload", help="Upload data to S3")
    upload_parser.add_argument("--source", help="Source directory")
    upload_parser.add_argument("--dirs", nargs="+", help="Directories to upload")
    upload_parser.add_argument("--dry-run", action="store_true", help="Don't actually upload")

    # Download command
    download_parser = subparsers.add_parser("download", help="Download data from S3")
    download_parser.add_argument("--target", help="Target directory")
    download_parser.add_argument("--dirs", nargs="+", help="Directories to download")
    download_parser.add_argument("--dry-run", action="store_true", help="Don't actually download")

    # Ensure command
    ensure_parser = subparsers.add_parser("ensure", help="Ensure a file is available")
    ensure_parser.add_argument("filename", help="File to ensure is available")

    args = parser.parse_args()

    if args.command == "list":
        list_s3_contents(args.prefix)
    elif args.command == "upload":
        sync_to_s3(args.source, args.dirs, args.dry_run)
    elif args.command == "download":
        sync_from_s3(args.target, args.dirs, args.dry_run)
    elif args.command == "ensure":
        result = ensure_data_available(args.filename)
        if result:
            print(f"File available at: {result}")
        else:
            print("File not available")
    else:
        parser.print_help()
