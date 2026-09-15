#!/bin/bash
#
# GRPO training job.  Usage:
#   sbatch --time=HH:MM:SS scripts/train.sbatch venv_train/bin/python -m contract.train_grpo --out runs/x ...
#
#SBATCH --job-name=contract_train
#SBATCH --account=lingo
#SBATCH --partition=lingo-h100
#SBATCH --qos=lingo-main
#SBATCH --time=08:00:00
#SBATCH --output=/data/scratch-oc40/jda/contract/logs/%x_%j.log
#SBATCH --error=/data/scratch-oc40/jda/contract/logs/%x_%j.err
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --open-mode=append

cd /data/scratch-oc40/jda/contract
export HF_HOME=/data/scratch-oc40/jda/contract/hf_cache
export UV_CACHE_DIR=/data/scratch-oc40/jda/contract/.uv_cache
# Training uses its own venv: TRL pins an older vLLM (<=0.27.1) than the serving venv has (0.28),
# so trl[vllm] is resolved separately.  Built inside the job (downloads must not run on the login node).
if [ ! -x venv_train/bin/python ]; then
    bin/uv venv --python /usr/bin/python3.10 venv_train
    bin/uv pip install --python venv_train/bin/python "trl[vllm,peft]" datasets
    # flashinfer 0.6.16 is not Python-3.10 compatible and breaks vLLM's compile path; vLLM works without it.
    bin/uv pip uninstall --python venv_train/bin/python flashinfer-python
fi
export HF_HUB_OFFLINE=1
# vLLM otherwise imports flashinfer (removed above) when deciding on its sampler kernel.
export VLLM_USE_FLASHINFER_SAMPLER=0
# Job-specific ports so that several training jobs can share the node.
export MASTER_PORT=$((30000 + SLURM_JOB_ID % 10000))
export VLLM_GROUP_PORT=$((40000 + SLURM_JOB_ID % 10000))
"$@"
