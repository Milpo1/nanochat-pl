#!/usr/bin/env python3
import argparse
import sys
import os
from google.cloud import storage

def download_gs(key_path, source_uri, dest_dir):
    """
    Downloads files from a gs:// URI to a local directory.
    Uses match_glob, so source_uri can be a single file or include wildcards (*).
    """
    
    # 1. Parse the GS URI (e.g., gs://my-bucket/folder/*.csv)
    if not source_uri.startswith("gs://"):
        print("Error: Source must start with gs://")
        sys.exit(1)
    
    # Strip 'gs://' and split bucket from path
    path_without_scheme = source_uri[5:]
    if "/" not in path_without_scheme:
        print("Error: Invalid format. Use gs://bucket-name/path/to/file")
        sys.exit(1)

    bucket_name, blob_pattern = path_without_scheme.split("/", 1)

    # 2. Check Key File
    if not os.path.exists(key_path):
        print(f"Error: Key file not found at {key_path}")
        sys.exit(1)

    try:
        # 3. Initialize Client
        client = storage.Client.from_service_account_json(key_path)

        print(f"Searching bucket '{bucket_name}' for pattern: {blob_pattern}")

        # 4. List files (match_glob works for specific files AND wildcards)
        blobs = list(client.list_blobs(bucket_name, match_glob=blob_pattern))

        if not blobs:
            print("No files matched the pattern.")
            return

        # 5. Prepare Destination Directory
        os.makedirs(dest_dir, exist_ok=True)

        # 6. Download Loop
        for blob in blobs:
            # We flatten the download (ignore remote folders, just take filename)
            filename = os.path.basename(blob.name)
            local_path = os.path.join(dest_dir, filename)
            
            print(f"Downloading: {filename}")
            blob.download_to_filename(local_path)

        print(f"Done. Downloaded {len(blobs)} files to '{dest_dir}'.")

    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple GCP Fetcher")
    
    parser.add_argument("--key", required=True, help="Path to JSON key file")
    parser.add_argument("--src", required=True, help="GS URI (e.g. gs://bucket/data/*.csv)")
    parser.add_argument("--dest", required=True, help="Local destination directory")

    args = parser.parse_args()

    download_gs(args.key, args.src, args.dest)