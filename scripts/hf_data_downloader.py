import argparse
import os
import time
import requests
from pathlib import Path
from huggingface_hub import snapshot_download


def download_dataset(repo_id, revision, local_dir, allow_patterns, num_workers=16):
    """method download files as an snapshots

    Args:
        repo_id (str): huggingface dataset repo id
        local_dir (str): destination folder to store files
        allow_patterns (str): file patter
        num_workers (str): number of workers to download files
        
    Eg:
    If hf repo contains files in parquer format use pattern like
        `.parquet`
    If in sub-folder of parquet format
        `**.parquet`
    
    Increase the num workers to enjoy the parallelism
    """
    print(f"Downloading dataset from {repo_id}...")
    max_retries = 5
    retry_delay = 10  # seconds
    for attempt in range(max_retries):
        try:
            snapshot_download(
                repo_id,
                repo_type="dataset",
                revision=revision,
                local_dir=local_dir,
                allow_patterns=allow_patterns,
                resume_download=True,
                max_workers=num_workers
            )
            break
        except requests.exceptions.ReadTimeout:
            if attempt < max_retries - 1:
                print(f"Timeout occurred. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                raise
    print(f"Dataset downloaded to {local_dir}")
    
    
    
def main(args, seed=42):
    # prepare workspace
    working_dir = args.working_dir
    if working_dir is None:
        working_dir = ""
    
    working_dir = Path(working_dir)
    src_dir = working_dir / args.dataset
    
    if not src_dir.exists():
        print(f"Creating source directory {src_dir}")
        src_dir.mkdir(parents=True)
    
    # download dataset
    download_dataset(
        repo_id=args.dataset,
        local_dir=str(src_dir),
        allow_patterns=args.allowed_pattern,
        num_workers=args.num_workers,
        revision=args.revision
    )

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--working_dir", type=str, default="data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allowed_pattern", type=str, default=None)
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=16)

    args = parser.parse_args()

    main(args, args.seed)