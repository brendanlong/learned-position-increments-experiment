#!/bin/bash
# Launch training via SkyPilot.
#
# Usage:
#   ./train.sh <remote|vast|local> [--gpu GPU|--vram N] [--disk-size N] [--module MODULE] <selective_pe|delta_graft> [-- extra args for train.py]
#
# By default, picks the cheapest Ampere+ GPU with >= 8GB VRAM.
# Use --vram to raise the minimum, or --gpu to force a specific type.
#
# Examples:
#   # Cheapest GPU with >= 8GB VRAM on RunPod:
#   ./train.sh remote selective_pe -- [additional args]
#
#   # Cheapest GPU with >= 24GB VRAM on Vast.ai:
#   ./train.sh vast --vram 24 selective_pe -- [additional args]
#
#   # Force a specific GPU:
#   ./train.sh vast --gpu H100 delta_graft -- [additional args]
#
#   # More disk space for large checkpoints:
#   ./train.sh remote --disk-size 100 delta_graft -- [additional args]
#
#   # Local training on RTX 3060 Ti:
#   ./train.sh local selective_pe -- [additional args]
#
#   # Manage the local queue:
#   sky queue local-gpu          # show queued/running jobs
#   sky cancel local-gpu JOB_ID  # cancel a specific job
#   sky cancel local-gpu --all   # cancel all queued jobs
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ---------------------------------------------------------------------------
# GPU catalog: Ampere+ only (compute capability >= 8.0)
# Format: NAME:MIN_VRAM_GB
# VRAM is the *minimum* across SkyPilot variants (e.g. RTX3060 includes
# the 8 GB 3060 Ti, so we list 8 not 12).
# ---------------------------------------------------------------------------
GPU_CATALOG=(
    "RTX3060:8"
    "RTX3070:8"
    "RTX3080:10"
    "RTX3090:24"
    "RTX4060:8"
    "RTX4070:12"
    "RTX4080:16"
    "RTX4090:24"
    "RTX5060:8"
    "RTX5070:12"
    "RTX5090:32"
    "RTX4500-Ada:24"
    "RTXA5000:24"
    "RTXA6000:48"
    "RTX5880-Ada:48"
    "RTX6000-Ada:48"
    "RTXPRO6000WS:96"
    "A40:48"
    "L40S:45"
    "A100:40"
    "H100:80"
    "H200:141"
)

IMAGE_ID="docker:runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04"

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
INFRA=""
GPU=""
VRAM=8
DISK_SIZE=50
MODULE=""
EXPERIMENT=""
TRAIN_ARGS=""

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
if [[ $# -lt 2 ]]; then
    echo "Usage: ./train.sh <remote|vast|local> [options] <selective_pe|delta_graft> [-- args...]"
    echo "Options: --gpu GPU | --vram N (default 8) | --disk-size N (default 50) | --module MODULE"
    exit 1
fi

INFRA="$1"; shift

while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu) GPU="$2"; shift 2 ;;
        --vram) VRAM="$2"; shift 2 ;;
        --disk-size) DISK_SIZE="$2"; shift 2 ;;
        --module) MODULE="$2"; shift 2 ;;
        --) shift; TRAIN_ARGS="$*"; break ;;
        selective_pe|delta_graft) EXPERIMENT="$1"; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ -z "$EXPERIMENT" ]]; then
    echo "Error: must specify experiment (selective_pe or delta_graft)"
    exit 1
fi

# Select task YAML fragment (workdir/setup/run/envs — no resources block)
case "$EXPERIMENT" in
    selective_pe) TASK_YAML="$SCRIPT_DIR/skypilot/train-selective-pe.yaml" ;;
    delta_graft) TASK_YAML="$SCRIPT_DIR/skypilot/train-delta-graft.yaml" ;;
    *) echo "Unknown experiment: $EXPERIMENT"; exit 1 ;;
esac

# ---------------------------------------------------------------------------
# Generate the resources YAML block
# ---------------------------------------------------------------------------
generate_resources() {
    if [[ -n "$GPU" ]]; then
        # Specific GPU requested
        cat <<EOF
resources:
  accelerators: {${GPU}: 1}
  use_spot: false
  disk_size: ${DISK_SIZE}
  image_id: ${IMAGE_ID}
EOF
    else
        # Pick cheapest GPU with >= VRAM GB
        local found=0
        echo "resources:"
        echo "  any_of:"
        for entry in "${GPU_CATALOG[@]}"; do
            local name="${entry%%:*}"
            local vram="${entry##*:}"
            if (( vram >= VRAM )); then
                found=1
                cat <<EOF
    - accelerators: {${name}: 1}
      use_spot: false
      disk_size: ${DISK_SIZE}
      image_id: ${IMAGE_ID}
EOF
            fi
        done
        if (( found == 0 )); then
            echo "Error: no Ampere+ GPUs with >= ${VRAM}GB VRAM in catalog" >&2
            exit 1
        fi
    fi
}

# ---------------------------------------------------------------------------
# Build the full task YAML = generated resources + base task fragment
# ---------------------------------------------------------------------------
TEMP_YAML=$(mktemp /tmp/sky-task-XXXXXX.yaml)
cleanup() { rm -f "$TEMP_YAML"; }
trap cleanup EXIT

{
    echo "name: ${EXPERIMENT}-train"
    echo ""
    generate_resources
    echo ""
    cat "$TASK_YAML"
} > "$TEMP_YAML"

# ---------------------------------------------------------------------------
# Build env / secret arguments
# ---------------------------------------------------------------------------
ENV_ARGS=()
SECRET_ARGS=()

if [[ -n "$MODULE" ]]; then
    ENV_ARGS+=(--env "TRAIN_MODULE=$MODULE")
fi
if [[ -n "$TRAIN_ARGS" ]]; then
    ENV_ARGS+=(--env "TRAIN_ARGS=$TRAIN_ARGS")
fi
# Optional human-friendly run name; namespaces the S3 checkpoint path so
# queued/concurrent runs don't overwrite each other. Falls back to the
# SkyPilot task ID in the task YAML when unset.
if [[ -n "${RUN_NAME:-}" ]]; then
    ENV_ARGS+=(--env "RUN_NAME=$RUN_NAME")
fi

# Forward secrets (redacted in SkyPilot dashboard)
for var in WANDB_API_KEY AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY; do
    if [[ -n "${!var:-}" ]]; then
        SECRET_ARGS+=(--secret "$var")
    fi
done

# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------
echo ">>> Launching $EXPERIMENT training ($INFRA)"
echo "  Task: $TASK_YAML"
echo "  Module: ${MODULE:-<default>}"
if [[ -n "$GPU" ]]; then
    echo "  GPU: $GPU"
else
    echo "  Min VRAM: ${VRAM}GB (cheapest Ampere+ GPU)"
fi
echo "  Disk: ${DISK_SIZE}GB"
echo "  Train args: ${TRAIN_ARGS:-<defaults>}"
echo ""

if [[ "$INFRA" == "local" ]]; then
    # Local: submit to persistent local-gpu cluster (creates it if needed)
    # Jobs queue up and run one after another (FIFO)
    # Override GPU to match local hardware
    LOCAL_ARGS=(--infra kubernetes --gpus RTX3060-TI:1)
    if sky status local-gpu 2>/dev/null | grep -q "UP"; then
        echo "  Queuing on existing local-gpu cluster"
        sky exec local-gpu "$TEMP_YAML" -d \
            "${LOCAL_ARGS[@]}" ${ENV_ARGS[@]+"${ENV_ARGS[@]}"} ${SECRET_ARGS[@]+"${SECRET_ARGS[@]}"}
    else
        echo "  Creating local-gpu cluster and running first job"
        sky launch "$TEMP_YAML" \
            "${LOCAL_ARGS[@]}" \
            --cluster local-gpu -d --yes \
            ${ENV_ARGS[@]+"${ENV_ARGS[@]}"} ${SECRET_ARGS[@]+"${SECRET_ARGS[@]}"}
    fi

elif [[ "$INFRA" == "remote" ]]; then
    # RunPod: managed jobs for auto-provisioning and auto-shutdown
    sky jobs launch "$TEMP_YAML" --yes \
        --infra runpod \
        ${ENV_ARGS[@]+"${ENV_ARGS[@]}"} ${SECRET_ARGS[@]+"${SECRET_ARGS[@]}"}

elif [[ "$INFRA" == "vast" ]]; then
    # Vast.ai: sky launch with --down for auto-teardown
    # (managed jobs don't work yet due to controller SDK version mismatch,
    # see https://github.com/skypilot-org/skypilot/issues/9362)
    sky launch "$TEMP_YAML" --yes \
        --infra vast --down \
        ${ENV_ARGS[@]+"${ENV_ARGS[@]}"} ${SECRET_ARGS[@]+"${SECRET_ARGS[@]}"}

else
    echo "Error: infra must be 'remote', 'vast', or 'local'"
    exit 1
fi
