import argparse
from data_processing.downloaders import download_dataset_from_huggingface
from pathlib import Path

    
    
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
    download_dataset_from_huggingface(
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
    parser.add_argument("--allowed_pattern", type=str, default=None, help="multiple patterns can be provided as comma seperated format")
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=16)

    args = parser.parse_args()

    main(args, args.seed)