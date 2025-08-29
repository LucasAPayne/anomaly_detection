#!/bin/bash -l

#SBATCH --job-name=kgg
#SBATCH -D .
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=2
#SBATCH --partition=gpu
#SBATCH --time=0-06:00:00
#SBATCH --output=tests/logs/output_%j.txt
#SBATCH --error=tests/logs/error_%j.txt

GPUS_PER_NODE=2

source /etc/profile.d/modules.sh

module load cuda
module load python/3.10
module load gcc

source venv/bin/activate

export NCCL_TIMEOUT=1800 # 30 minutes
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_TRACE_BUFFER_SIZE=131072

MASTER_ADDR=$(scontrol show hostnames $SLURM_NODELIST | head -n 1)
MASTER_PORT=29500
RANK=$SLURM_PROCID
LOCAL_RANK=$SLURM_LOCALID

export LAUNCHER="accelerate launch \
  --num_processes $SLURM_NNODES \
  --num_machines $SLURM_NNODES \
  --rdzv_backend c10d \
  --main_process_ip $MASTER_ADDR \
  --main_process_port $MASTER_PORT \
  "

export SCRIPT="tests/llm_stability.py"
export SCRIPT_ARGS="-i 1"

export CMD="$LAUNCHER $SCRIPT $SCRIPT_ARGS"
srun $CMD
