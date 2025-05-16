#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-node=1                      # GPUs used
#SBATCH --constraint="GPU_GEN:ADA"
#SBATCH --mem=16G
#SBATCH --mail-type=BEGIN,END,FAIL    # 2. Send email upon events (Options: NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=u3011649@connect.hku.hk     #    Email address to receive notification
#SBATCH --time=20:00:00             # 7. Job execution duration limit day-hour:min:sec
#SBATCH --output=out/%x_%j.out            # 8. Standard output log as $job_name_$job_id.out
#SBATCH --error=out/%x_%j.err 
##SBATCH --exclusive                  # 独占整个节点


# Load the environment module for Nvidia CUDA
module load cuda
module load anaconda
conda activate fedllm

export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes=1 main_my.py 
   

