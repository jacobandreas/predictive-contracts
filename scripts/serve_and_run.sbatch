#!/bin/bash
#
# Start a vLLM server for Qwen3-4B on one GPU, wait until it is healthy, then run
# the command given as arguments against it.  Usage:
#   sbatch --time=HH:MM:SS scripts/serve_and_run.sbatch venv/bin/python -m contract.run_tasks ...
#
#SBATCH --job-name=contract
#SBATCH --account=lingo
#SBATCH --partition=lingo-h100
#SBATCH --qos=lingo-main
#SBATCH --time=01:00:00
#SBATCH --output=/data/scratch-oc40/jda/contract/logs/%x_%j.log
#SBATCH --error=/data/scratch-oc40/jda/contract/logs/%x_%j.err
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

cd /data/scratch-oc40/jda/contract
export HF_HOME=/data/scratch-oc40/jda/contract/hf_cache
export HF_HUB_OFFLINE=1
MODEL=${MODEL:-Qwen/Qwen3-4B}
echo "GPU: $(nvidia-smi -L)"   # so a bad device can be identified if vLLM fails to start
PORT=$((20000 + SLURM_JOB_ID % 10000))   # the node is shared, so pick a job-specific port
export LLM_BASE_URL=http://localhost:$PORT/v1
# Optional LoRA adapter: LORA=/path/to/adapter makes the model available under the name "rl".
LORA_ARGS=""; [ -n "$LORA" ] && LORA_ARGS="--enable-lora --max-lora-rank 64 --lora-modules rl=$LORA"

venv/bin/vllm serve $MODEL --port $PORT --dtype bfloat16 --max-model-len ${MAX_MODEL_LEN:-16384} \
    --gpu-memory-utilization 0.85 --reasoning-parser qwen3 --enable-prefix-caching \
    --max-num-seqs 256 $LORA_ARGS > logs/vllm_$SLURM_JOB_ID.log 2>&1 &
VLLM_PID=$!
until curl -sf localhost:$PORT/health > /dev/null; do
    sleep 10
    kill -0 $VLLM_PID 2>/dev/null || { echo "vllm died, see logs/vllm_$SLURM_JOB_ID.log"; exit 1; }
done
echo "vllm ready on port $PORT"

"$@"
kill $VLLM_PID
