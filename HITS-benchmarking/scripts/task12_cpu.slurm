#!/bin/bash
#SBATCH --job-name=disrpt_task12_cpu
#SBATCH --output=disrpt_task12_cpu_%j.out
#SBATCH --error=disrpt_task12_cpu_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --partition=cpu

# Activate your conda environment (adjust name as needed)
conda activate disrpt
# cd to the project root (adjust if your path differs)
cd "$(dirname "$0")/.."

# Create results directory
mkdir -p data/result/eng.tutorial.annotation
mkdir -p logs/task12_cpu

# Install requirements if needed
echo "Installing requirements..."
pip install pytorch-crf transformers==4.21.3 scikit-learn pandas tqdm numpy sentencepiece --quiet

echo "Starting DISRPT Task 1&2 (CPU Training) - Segmentation + Connective Detection..."
echo "Timestamp: $(date)"
echo "Dataset: eng.tutorial.annotation"
echo "Model: BiLSTM + CRF with BERT encoder (CPU optimized)"

# Run Task 1&2 training with CPU-friendly settings
python3 task12.py \
    --do_train \
    --do_dev \
    --do_test \
    --dataset=eng.tutorial.annotation \
    --model_type="bilstm+crf" \
    --encoder_type="bert" \
    --max_seq_length=256 \
    --train_batch_size=4 \
    --eval_batch_size=8 \
    --learning_rate=2e-5 \
    --dropout=0.1 \
    --num_train_epochs=5 \
    --warmup_ratio=0.1 \
    --weight_decay=0.01 \
    --max_grad_norm=1.0 \
    --seed=106524

echo "Task 1&2 CPU training completed."
echo "Timestamp: $(date)"

# Backup results
echo "Backing up results..."
cp -r data/result/eng.tutorial.annotation ./logs/task12_cpu/results_$(date +%Y%m%d_%H%M%S)
cp *.out *.err ./logs/task12_cpu/ 2>/dev/null || true

echo "Results saved to: data/result/eng.tutorial.annotation/"
echo "Logs saved to: ./logs/task12_cpu/"
echo "CPU job completed successfully."