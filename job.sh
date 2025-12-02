#!/bin/bash -l

#SBATCH --job-name=kgg
#SBATCH -D .
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=2
#SBATCH --partition=gpu
#SBATCH --time=0-24:00:00
#SBATCH --output=tests/logs/output_%j.txt
#SBATCH --error=tests/logs/error_%j.txt

GPUS_PER_NODE=2

source /etc/profile.d/modules.sh

module load cuda
module load python/3.10
module load gcc
module load mpich

source venv/bin/activate

# python tests/llm_stability.py -i 2
# python demo_ait.py
python tests/llm_stability.py -i 1
