"""Shared checkpoint upload utilities for wandb artifact storage."""

from pathlib import Path

import wandb


def upload_checkpoint_to_wandb(
    checkpoint_path: Path,
    artifact_name: str,
    artifact_type: str = "model",
    metadata: dict[str, object] | None = None,
) -> None:
    """Upload a checkpoint file to wandb as an artifact.

    Must be called while a wandb run is active (before wandb.finish()).

    Args:
        checkpoint_path: Path to the .pt checkpoint file.
        artifact_name: Name for the wandb artifact (e.g. "chain-2hop-ws").
        artifact_type: Artifact type (default: "model").
        metadata: Optional metadata dict attached to the artifact.
    """
    run = wandb.run
    if run is None:
        print("WARNING: No active wandb run — skipping checkpoint upload.")
        return

    artifact = wandb.Artifact(
        name=artifact_name,
        type=artifact_type,
        metadata=metadata or {},
    )
    artifact.add_file(str(checkpoint_path))
    run.log_artifact(artifact)
    print(f"Uploaded checkpoint to wandb artifact: {artifact_name}")
