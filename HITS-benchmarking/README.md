# HITS Benchmarking on ENG-DRB

Benchmarking the [HITS at DISRPT 2023](https://aclanthology.org/2023.disrpt-1.4.pdf) discourse parsing system on the [ENG-DRB](https://doi.org/10.57967/hf/6895) dataset.

This codebase implements three discourse parsing tasks using transformer-based models:
- **Task 1:** Discourse Segmentation (EDU boundary detection)
- **Task 2:** Connective Detection (discourse marker identification)
- **Task 3:** Relation Classification (discourse relation labeling)

## Project Structure

```
HITS-benchmarking/
├── task12.py              # Training/evaluation for Tasks 1 & 2
├── task3.py               # Training/evaluation for Task 3
├── models.py              # Model architectures (BiLSTM+CRF, Transformer classifiers)
├── utils.py               # Data processing utilities
├── task_dataset.py        # PyTorch Dataset classes
├── preprocessing.py       # Raw data to JSON conversion
├── seg_eval.py            # Segmentation F-score evaluation
├── rel_eval.py            # Relation classification accuracy evaluation
├── run_task12.sh          # Shell script for Tasks 1 & 2
├── run_task3.sh           # Shell script for Task 3
├── run_task3_missing.sh   # Task 3 for datasets without training data
├── eval_task3.sh          # Task 3 evaluation only
├── data/
│   └── config/
│       └── rel_config.json  # Per-corpus hyperparameters
└── scripts/               # SLURM scripts for HPC training
```

## Requirements

- Python >= 3.8
- PyTorch >= 1.12
- Transformers >= 4.21

Install all dependencies:

```bash
pip install -r requirements.txt
```

## Data

The datasets are included under `data/dataset/`:

| Dataset | Description |
|---------|-------------|
| `eng.tutorial.annotation` | ENG-DRB tutorial annotation (train/dev/test splits) |

### Setup

1. Create required directories:
   ```bash
   mkdir -p data/result data/logs data/embeddings
   ```

2. Generate preprocessed `.json` files (if missing):
   ```bash
   python3 preprocessing.py
   ```

## Training & Evaluation

### Task 1 & 2: Discourse Segmentation + Connective Detection

```bash
python3 task12.py \
    --do_train \
    --dataset=eng.tutorial.annotation \
    --max_seq_length=512 \
    --model_type="bilstm+crf" \
    --encoder_type="roberta" \
    --pretrained_path="roberta-base" \
    --learning_rate=3e-5 \
    --train_batch_size=16 \
    --num_train_epochs=10
```

Or run all datasets at once:
```bash
sh run_task12.sh
```

### Task 3: Relation Classification

```bash
python3 task3.py \
    --do_train \
    --dataset=eng.tutorial.annotation \
    --encoder_type="roberta" \
    --pretrained_path="roberta-base" \
    --learning_rate=2e-5 \
    --num_train_epochs=5
```

Or run all datasets at once:
```bash
sh run_task3.sh
```

### HPC (SLURM) Training

See [scripts/README_HPC.md](scripts/README_HPC.md) for instructions on running jobs on a SLURM cluster.

## Model Architectures

| Task | Model | Description |
|------|-------|-------------|
| 1 & 2 | BiLSTM + CRF | Transformer encoder + BiLSTM (2 layers, bidirectional) + CRF |
| 3 | BaseRelClassifier | Transformer encoder + linear classifier |

Supported encoders: RoBERTa, BERT, XLM-RoBERTa, CamemBERT, ELECTRA.

## Results on ENG-DRB (Tutorial Annotation)

| Task | Dev | Test |
|------|-----|------|
| Task 1 & 2 (F1) | 100.0% | 98.62% |
| Task 3 (Accuracy) | 80.70% | 63.73% |

## Citation

If you use this code, please cite both the HITS system and the ENG-DRB dataset:

```bibtex
@inproceedings{liu-etal-2023-hits,
    title = "{HITS} at {DISRPT} 2023: Discourse Segmentation, Connective Detection, and Relation Classification",
    author = "Liu, Wei and Fan, Yi and Strube, Michael",
    booktitle = "Proceedings of the 3rd Shared Task on Discourse Relation Parsing and Treebanking (DISRPT 2023)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "The Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.disrpt-1.4",
    doi = "10.18653/v1/2023.disrpt-1.4",
    pages = "43--49",
}
```
