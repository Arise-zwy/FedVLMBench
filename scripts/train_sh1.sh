#!/bin/bash
#######################################################################################
###                                                                                   #
### slurm-pytorch.submit:                                                             #
### Sample SLURM job script for running pytorch with GPU.                             #
###                                                                                   #
### Usage:                                                                            #
###    cd to directory containing python script, then:                                #
###    sbatch slurm-pytorch.submit                                                    #
###                                                                                   #
### - Written by David Pong, HKU ITS (2024-06-11)                                     #
###                                                                                   #
#######################################################################################

#SBATCH --job-name=pytorch                     # Job name
##SBATCH --mail-type=END,FAIL                  # Mail events
##SBATCH --mail-user=abc@email                 # Set your email address
#SBATCH --partition=gpu_shared               # Specific Partition (gpu/gpu_shared)
#SBATCH --qos=normal                           # Specific QoS (debug/normal)
#SBATCH --time=20:00:00                      # Wall time limit (days-hrs:min:sec)
#SBATCH --nodes=1                              # Single compute node used
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4                      # CPUs used
#SBATCH --gpus-per-node=1                      # GPUs used
#SBATCH --output=out/%x_%j.out            # 8. Standard output log as $job_name_$job_id.out
#SBATCH --error=out/%x_%j.err 

module load anaconda
conda activate fedllm

export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes=1 main_my.py 
   
