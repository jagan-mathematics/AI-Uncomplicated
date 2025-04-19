import argparse
import os
from pathlib import Path

from data_processing.parquet.convertors import parquet_to_jsonl
from data_processing.terashuf.setup import setup_terashuf, execute_terashuf, run_command
from scripts.hf_data_downloader import download_dataset_from_huggingface


def get_sample_file(path):
    for path, subdirs, files in os.walk(path):
        for name in files:
            return os.path.join(path, name)


def get_extension(file):
    filename, file_extension = os.path.splitext(file)
    return file_extension


def main(args):
    working_dir = args.data_dir
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

    out_dir = f"{src_dir}_shuffled"
    os.makedirs(out_dir, exist_ok=True)
    work_dir = src_dir  # Directory of this Python file
    prefix = f"{args.dataset}.chunk."

    sample_file = get_sample_file(work_dir)
    orig_extension = get_extension(sample_file)

    allow_patterns = ""

    is_parquet = False
    if "parquet" in orig_extension:
        is_parquet = True
        orig_extension = ".jsonl"
    elif orig_extension == ".jsonl":
        orig_extension = ".jsonl"
    else:
        raise NotImplementedError(f"Not implemented extension type {orig_extension}")

    cat_command = "cat {}"
    suffix = ".jsonl"
    k_validation = 10000

    if is_parquet:
        parquet_to_jsonl(args.dataset, work_dir, src_dir, src_dir)

    terashuf_dir = setup_terashuf(work_dir)
    os.environ["MEMORY"] = f"{args.memory}"
    os.environ["SEED"] = f"{args.seed}"

    execute_terashuf(terashuf_dir, src_dir, orig_extension, cat_command, args.nchunks, out_dir, suffix, prefix)

    validation_file = f"{out_dir}/{args.dataset}.val{suffix}"
    for i in range(args.nchunks):
        chunk_file = f"{out_dir}/{prefix}{i:02d}{suffix}"
        run_command(f"head -n {k_validation} {chunk_file} >> {validation_file}")
        run_command(f"sed -i '1,{k_validation}d' {chunk_file}")

    print("All tasks completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--allowed_pattern", type=str, default=None,
                        help="multiple patterns can be provided as comma seperated format")
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--memory", type=float, default=8)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--nchunks", type=int, default=32)

    args = parser.parse_args()

    main(args)
