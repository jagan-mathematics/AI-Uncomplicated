import os
import subprocess


def run_command(command):
    print(f"Running: {command}")
    subprocess.run(command, shell=True, check=True)


def setup_terashuf(work_dir):
    terashuf_dir = os.path.join(work_dir, "terashuf")
    terashuf_executable = os.path.join(terashuf_dir, "terashuf")

    if os.path.exists(terashuf_executable):
        print("terashuf executable already exists. Skipping setup.")
        return terashuf_dir

    print("Setting up terashuf...")
    run_command(f"git clone https://github.com/alexandres/terashuf {terashuf_dir}")
    run_command(f"make -C {terashuf_dir}")
    return terashuf_dir


def execute_terashuf(terashuf_dir, src_dir, orig_extension, cat_command, nchunks, out_dir, suffix, prefix):
    terashuf_executable = os.path.join(terashuf_dir, "terashuf")
    run_command(
        f"ulimit -n 100000 && "
        f"find {src_dir} -type f -name '*{orig_extension}' -print0 | xargs -0 -I {{}} sh -c '{cat_command}' | {terashuf_executable} | "
        f"split -n r/{nchunks} -d --suffix-length 2 --additional-suffix {suffix} - {out_dir}/{prefix}"
        # "; trap 'echo \"Caught signal 13, exiting with code 1\"; exit 1' SIGPIPE;"
    )