import requests
import time
import requests
from huggingface_hub import snapshot_download

def download_dataset_from_huggingface(repo_id, revision, local_dir, allow_patterns, num_workers=16):
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
    
    if allow_patterns is not None:
        allow_patterns = allow_patterns.split(",")
        
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
    