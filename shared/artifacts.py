"""Released checkpoints, hosted in a public Hugging Face dataset."""

from huggingface_hub import hf_hub_download

HF_REPO_ID = "brendanlong/learned-position-increments-experiment"
HF_PREFIX = "hf:"


def artifact_path(relpath: str) -> str:
    """Download ``<experiment>/<run_name>/<file>``; return the local path."""
    return hf_hub_download(repo_id=HF_REPO_ID, repo_type="dataset", filename=relpath)


def resolve_checkpoint(source: str) -> str:
    """Resolve a local path or an ``hf:<relpath>`` reference to a local file."""
    if source.startswith(HF_PREFIX):
        return artifact_path(source.removeprefix(HF_PREFIX))
    return source
