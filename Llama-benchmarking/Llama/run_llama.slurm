#!/bin/bash

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=JobExample4       #Set the job name to "JobExample4"
#SBATCH --time=12:00:00              #Set the wall clock limit to 1hr and 30min
#SBATCH --ntasks=1                   #Request 1 task
#SBATCH --mem=32G                  #Request 2560MB (2.5GB) per node
#SBATCH --output=Example4Out.%j      #Send stdout/err to "Example4Out.[jobID]"
#SBATCH --gres=gpu:a100:1                 #Request 1 GPU per node can be 1 or 2
#SBATCH --partition=gpu              #Request the GPU partition/queue

# Login to Hugging Face (set your token via: huggingface-cli login)
# huggingface-cli login --token <YOUR_HF_TOKEN>

#First Executable Line
conda activate video_tree
cd "$(dirname "$0")"
CUDA_VISIBLE_DEVICES=0 python llama_batch_processing.py
