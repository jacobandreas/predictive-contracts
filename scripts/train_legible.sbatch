#!/bin/bash
#
# Legibility training job: two GPUs on one node.  GPU 1 serves the frozen monitor (vLLM, base model); GPU 0 trains.
#   sbatch --time=24:00:00 scripts/train_legible.sbatch venv_train/bin/python -m contract.legible.train --reward legibility ...
#
#SBATCH --job-name=contract_legible
#SBATCH --account=lingo
#SBATCH --partition=lingo-h100
#SBATCH --qos=lingo-main
#SBATCH --time=24:00:00
#SBATCH --output=/data/scratch-oc40/jda/contract/logs/%x_%j.log
#SBATCH --error=/data/scratch-oc40/jda/contract/logs/%x_%j.err
#SBATCH --gpus=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --open-mode=append

cd /data/scratch-oc40/jda/contract
export HF_HOME=/data/scratch-oc40/jda/contract/hf_cache
export HF_HUB_OFFLINE=1
export UV_CACHE_DIR=/data/scratch-oc40/jda/contract/.uv_cache
export VLLM_USE_FLASHINFER_SAMPLER=0
export MASTER_PORT=$((30000 + SLURM_JOB_ID % 10000))
export VLLM_GROUP_PORT=$((40000 + SLURM_JOB_ID % 10000))
echo "GPUs: $(nvidia-smi -L)"

# the monitor: the serving venv's vLLM on the second GPU, thinking off at request time, same model as the policy's base
MONITOR_MODEL=${MONITOR_MODEL:-Qwen/Qwen3-4B}
PORT=$((20000 + SLURM_JOB_ID % 10000))
CUDA_VISIBLE_DEVICES=1 venv/bin/vllm serve $MONITOR_MODEL --port $PORT --dtype bfloat16 --max-model-len 16384 \
    --gpu-memory-utilization 0.85 --enable-prefix-caching --max-num-seqs 256 > logs/monitor_$SLURM_JOB_ID.log 2>&1 &
MONITOR_PID=$!
until curl -sf localhost:$PORT/health > /dev/null; do
    sleep 10
    kill -0 $MONITOR_PID 2>/dev/null || { echo "monitor vllm died, see logs/monitor_$SLURM_JOB_ID.log"; exit 1; }
done
export MONITOR_URL=http://localhost:$PORT/v1
echo "monitor ready at $MONITOR_URL"

CUDA_VISIBLE_DEVICES=0 "$@"
kill $MONITOR_PID
