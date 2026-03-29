# Llama Model Benchmarking Pipeline

This directory contains a comprehensive pipeline for benchmarking Llama models on discourse relation identification tasks. The pipeline combines LLM processing, postprocessing, and evaluation into a single Python script.

## Features

- **Llama Model Integration**: Uses Hugging Face Transformers to load and run Llama models locally
- **Sliding Window Processing**: Processes documents using configurable sliding windows
- **Postprocessing**: Removes duplicates and merges partially agreed senses
- **Evaluation**: Computes precision, recall, and F1 scores with both partial agreement and exact match metrics
- **Configurable**: Supports various Llama models and processing parameters

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Model Access

To use Llama models, you need to:

1. **Request access** to Llama models on Hugging Face:
   - Go to https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
   - Click "Request access" and follow the instructions
   - Wait for approval from Meta

2. **Login to Hugging Face**:
   ```bash
   huggingface-cli login
   ```

### 3. Create System Prompt

```bash
python llama_benchmarking_pipeline.py --create-prompt
```

This creates a default `system_prompt.txt` file. You can modify this file to customize the prompt for your specific task.

## Usage

### Basic Usage

```bash
python llama_benchmarking_pipeline.py
```

### Advanced Usage

```bash
python llama_benchmarking_pipeline.py \
    --model "meta-llama/Llama-2-13b-chat-hf" \
    --input "../revised_data.jsonl" \
    --gold "../revised_data.jsonl" \
    --output-dir "results" \
    --window-size 20 \
    --step 10 \
    --max-tokens 3000 \
    --temperature 0.7
```

### Command Line Arguments

- `--model`: HuggingFace model name (default: "meta-llama/Llama-2-7b-chat-hf")
- `--input`: Input data file path (default: "../revised_data.jsonl")
- `--gold`: Gold standard data file path (default: "../revised_data.jsonl")
- `--output-dir`: Output directory (default: "results")
- `--window-size`: Sliding window size (default: 20)
- `--step`: Sliding window step (default: 10)
- `--max-tokens`: Maximum tokens for generation (default: 3000)
- `--temperature`: Generation temperature (default: 0.7)
- `--create-prompt`: Create default system prompt file

## Pipeline Steps

The pipeline consists of three main steps:

### 1. Document Processing
- Loads the Llama model and tokenizer
- Processes each document using sliding windows
- Generates discourse relation predictions for each window
- Saves raw outputs to `results/llama_raw_output.jsonl`

### 2. Postprocessing
- Removes exact duplicate predictions
- Merges partially agreed senses
- Resolves conflicts by keeping highest confidence predictions
- Saves processed outputs to `results/llama_processed_output.jsonl`

### 3. Evaluation
- Compares predictions against gold standard
- Computes precision, recall, and F1 scores
- Uses both partial agreement and exact match metrics
- Saves evaluation results to `results/evaluation_results.json`

## Output Files

- `results/llama_raw_output.jsonl`: Raw model predictions
- `results/llama_processed_output.jsonl`: Postprocessed predictions
- `results/evaluation_results.json`: Evaluation metrics and scores

## Supported Models

The pipeline supports any Llama model available on Hugging Face, including:

- `meta-llama/Llama-2-7b-chat-hf`
- `meta-llama/Llama-2-13b-chat-hf`
- `meta-llama/Llama-2-70b-chat-hf`
- `meta-llama/Llama-3-8b-instruct`
- `meta-llama/Llama-3-70b-instruct`

## Hardware Requirements

- **GPU**: Recommended for faster processing (CUDA-compatible)
- **RAM**: At least 16GB for 7B models, 32GB+ for larger models
- **Storage**: 10-50GB depending on model size

## Troubleshooting

### Common Issues

1. **Model Access Denied**: Ensure you have been granted access to Llama models on Hugging Face
2. **Out of Memory**: Reduce batch size or use a smaller model
3. **CUDA Errors**: Ensure PyTorch is installed with CUDA support

### Performance Tips

- Use GPU acceleration when available
- Adjust window size and step based on your document characteristics
- Consider using quantization for larger models if memory is limited

## Customization

### Modifying the System Prompt

Edit `system_prompt.txt` to customize the instruction given to the model:

```bash
nano system_prompt.txt
```

### Adding New Models

To use a different model, simply change the `--model` parameter:

```bash
python llama_benchmarking_pipeline.py --model "your-model-name"
```

### Custom Evaluation Metrics

Modify the `Evaluator` class in the script to add custom evaluation metrics.

## Example Results

The pipeline outputs evaluation results in the following format:

```json
{
  "partial_agreement": {
    "overall_scores": {
      "precision": 0.3491,
      "recall": 0.5022,
      "f1": 0.4119,
      "total_tp": 445.4179,
      "total_fp": 830.5821,
      "total_fn": 441.5821
    },
    "per_item_scores": {
      "document_name": {
        "precision": 0.3793,
        "recall": 0.5231,
        "f1": 0.4398
      }
    }
  },
  "exact_match": {
    "overall_scores": {
      "precision": 0.1897,
      "recall": 0.2734,
      "f1": 0.224
    }
  }
}
```

## License

This pipeline is provided for research purposes. Please ensure you comply with the license terms of the Llama models you use. 