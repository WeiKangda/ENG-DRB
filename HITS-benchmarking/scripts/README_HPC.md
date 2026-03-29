# DISRPT HPC Training Guide

This directory contains SLURM scripts for running DISRPT discourse parsing models on HPC systems.

## Files Overview

- `task12_tutorial.slurm` - GPU training for Task 1&2 (Segmentation + Connective Detection)
- `task3_tutorial.slurm` - GPU training for Task 3 (Relation Classification)  
- `task12_cpu.slurm` - CPU fallback training for Task 1&2
- `submit_jobs.sh` - Convenience script to submit all jobs
- `README_HPC.md` - This guide

## Prerequisites

1. **Data Setup**: Ensure your data is preprocessed and available at:
   ```
   data/dataset/eng.tutorial.annotation/
   ├── eng.tutorial.annotation_train.json
   ├── eng.tutorial.annotation_dev.json
   └── eng.tutorial.annotation_test.json
   ```

2. **Environment**: Conda environment with Python 3.8+ and dependencies from `requirements.txt`

3. **Upload Files**: Copy the project to your HPC scratch space:
   ```bash
   scp -r HITS-benchmarking/ username@hpc-login:/scratch/user/<your-username>/
   ```

## Quick Start

1. **Submit both tasks**:
   ```bash
   cd /path/to/HITS-benchmarking
   chmod +x scripts/submit_jobs.sh
   ./scripts/submit_jobs.sh
   ```

2. **Or submit individually**:
   ```bash
   sbatch scripts/task12_tutorial.slurm  # Task 1&2
   sbatch scripts/task3_tutorial.slurm   # Task 3
   ```

3. **Monitor jobs**:
   ```bash
   squeue -u $USER
   tail -f disrpt_task12_*.out
   ```

## Job Configuration

### Task 1&2 (Segmentation + Connective Detection)
- **Model**: BiLSTM + CRF with RoBERTa encoder
- **Time**: 12 hours
- **GPU**: H100 x1
- **Memory**: 64GB
- **Batch Size**: 16 (train), 32 (eval)
- **Epochs**: 10

### Task 3 (Relation Classification)
- **Model**: RoBERTa-based classifier
- **Time**: 8 hours  
- **GPU**: H100 x1
- **Memory**: 64GB
- **Batch Size**: 16 (train), 32 (eval)
- **Epochs**: 5

### CPU Fallback (Task 1&2 only)
- **Model**: BiLSTM + CRF with BERT encoder
- **Time**: 24 hours
- **CPUs**: 16 cores
- **Memory**: 128GB
- **Batch Size**: 4 (train), 8 (eval)
- **Epochs**: 5

## Expected Results

After successful training, results will be saved in:
```
data/result/eng.tutorial.annotation/
├── pytorch_model.bin           # Best model checkpoint
├── pytorch_model_final.bin     # Final model
├── tokenizer_config.json       # Tokenizer config
└── training_logs.txt           # Training logs
```

## Troubleshooting

### Common Issues:

1. **Out of Memory**:
   - Reduce `train_batch_size` to 8 or 4
   - Reduce `max_seq_length` to 256 or 128
   - Use CPU version for Task 1&2

2. **Time Limit Exceeded**:
   - Reduce `num_train_epochs`
   - Request more time with `--time=24:00:00`

3. **Missing Dependencies**:
   - Jobs will auto-install requirements
   - For manual install: `pip install pytorch-crf transformers==4.21.3`

4. **CUDA Errors**:
   - Use CPU version: `sbatch scripts/task12_cpu.slurm`
   - Check GPU availability: `nvidia-smi`

### Debugging Commands:

```bash
# Check job status
squeue -j <job_id>

# View live output
tail -f disrpt_task12_<job_id>.out

# Cancel job
scancel <job_id>

# Check available resources
sinfo -p gpu
scontrol show partition gpu
```

## Performance Expectations

### Task 1&2 (Segmentation):
- **Training Time**: 8-12 hours on H100
- **Expected F1**: 70-85% for connective detection
- **Memory Usage**: ~40GB GPU memory

### Task 3 (Relations):
- **Training Time**: 4-8 hours on H100  
- **Expected Accuracy**: 60-80% for relation classification
- **Memory Usage**: ~30GB GPU memory

## Customization

To modify hyperparameters, edit the SLURM files:

```bash
# In task12_tutorial.slurm, modify:
--train_batch_size=8 \        # Reduce if OOM
--num_train_epochs=15 \       # Increase for better performance
--learning_rate=5e-5 \        # Adjust learning rate
--max_seq_length=256 \        # Reduce for speed
```

## Results Analysis

After training, check these key metrics in the output logs:
- **Segmentation F1**: Discourse boundary detection accuracy
- **Connective F1**: Discourse connective identification accuracy  
- **Relation Accuracy**: Discourse relation classification accuracy
- **Training Loss**: Should decrease consistently

Your tutorial annotation dataset contains:
- **Train**: 8 documents, ~4,285 relations
- **Dev**: 1 document, ~58 relations
- **Test**: 3 documents, ~340 relations

This provides a solid foundation for learning discourse parsing on tutorial/instructional content!