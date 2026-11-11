"""Maintainer script: upload checkpoints to the public HF dataset.

Usage (needs a write token in HF_TOKEN):
    uv run python scripts/upload_artifacts.py <local_dir>

``<local_dir>`` mirrors the dataset layout,
``<experiment>/<run_name>/<file>.pt`` with an optional ``<file>.json``
metadata file next to it. Files already in the dataset are overwritten.
"""

import argparse

from huggingface_hub import HfApi

from shared.artifacts import HF_REPO_ID


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("local_dir")
    args = p.parse_args()
    api = HfApi()
    api.create_repo(HF_REPO_ID, repo_type="dataset", exist_ok=True)
    api.upload_folder(
        repo_id=HF_REPO_ID, repo_type="dataset", folder_path=args.local_dir
    )
    print(f"Uploaded {args.local_dir} to https://huggingface.co/datasets/{HF_REPO_ID}")


if __name__ == "__main__":
    main()
