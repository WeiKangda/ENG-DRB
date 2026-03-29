#!/bin/bash
#SBATCH --job-name=disrpt_task12
#SBATCH --output=disrpt_task12_%j.out
#SBATCH --error=disrpt_task12_%j.err
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1 
#SBATCH --mem=64G
#SBATCH --partition=gpu_debug

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export HF_HOME=${HF_HOME:-~/.cache/huggingface}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-~/.cache/huggingface}

# Activate your conda environment (adjust name as needed)
conda activate disrpt
# cd to the project root (adjust if your path differs)
cd "$(dirname "$0")/.."

# Create results directory
mkdir -p data/result/eng.tutorial.annotation
mkdir -p logs/task12

echo "Starting DISRPT Task 1&2 (Segmentation + Connective Detection) training..."
echo "Timestamp: $(date)"
echo "Dataset: eng.tutorial.annotation"
echo "Model: BiLSTM + CRF with RoBERTa encoder"

# Run Task 1&2 training
python3 task12.py \
    --do_train \
    --do_dev \
    --do_test \
    --dataset=eng.tutorial.annotation \
    --model_type="bilstm+crf" \
    --encoder_type="roberta" \
    --max_seq_length=512 \
    --train_batch_size=16 \
    --eval_batch_size=32 \
    --learning_rate=3e-5 \
    --dropout=0.1 \
    --num_train_epochs=10 \
    --warmup_ratio=0.06 \
    --weight_decay=0.1 \
    --max_grad_norm=2.0 \
    --seed=106524

echo "Task 1&2 training completed."
echo "Timestamp: $(date)"

# Backup results
echo "Backing up results..."
cp -r data/result/eng.tutorial.annotation ./logs/task12/results_$(date +%Y%m%d_%H%M%S)
cp *.out *.err ./logs/task12/ 2>/dev/null || true

echo "Results saved to: data/result/eng.tutorial.annotation/"
echo "Logs saved to: ./logs/task12/"
echo "Job completed successfully."