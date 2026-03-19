# ENG-DRB benchmark release

This repository releases reproducible benchmarking code for **ENG-DRB**, with a shared data / scoring pipeline and two provider-specific backends:

- **OpenAI backend**: prepares request JSONL and runs through the OpenAI **Batch API**
- **Claude backend**: sends **direct requests window by window** and logs raw responses locally

The benchmark data are loaded directly from Hugging Face:

- dataset: `ChengZhangPNW/ENG-DRB`
- hub page: <https://huggingface.co/datasets/ChengZhangPNW/ENG-DRB>

## What is shared across providers

These parts are common to both OpenAI and Claude:

- load ENG-DRB from Hugging Face
- filter relations as `implicit` or `non_implicit`
- apply the same sliding-window setup over `Spans`
- merge window-level predictions back to document level
- deduplicate overlapping predictions
- compute the same evaluation scores

## What differs across providers

Only the provider-specific inference layer changes.

### OpenAI

- model path in this release: `o4-mini-2025-04-16`
- execution style: **Batch API**
- workflow: create request JSONL → submit batch → wait for completion → download raw results → merge / deduplicate / evaluate

### Claude

- model path in this release: `claude-3-7-sonnet-20250219`
- execution style: **direct API calls per sliding window**
- workflow: load gold JSONL → send one request per window → save raw responses JSONL → merge / deduplicate / evaluate

This difference is intentional and reflects the original benchmarking code you used. The Claude path in this repository is **not** a separate benchmark package; it is the same benchmark with a different backend adapter.

## Relation labels

This repository uses the clearer public label:

- `implicit` = implicit only
- `non_implicit` = everything not marked `implicit`, which in ENG-DRB means **explicit + AltLex**

The old label `explicit` is still accepted internally as a deprecated alias of `non_implicit` for backward compatibility.

## Repository layout

```text
.
├── notebooks/
│   ├── ENG_DRB_quickstart.ipynb
│   ├── ENG_DRB_openai_reproduce.ipynb
│   └── ENG_DRB_claude_reproduce.ipynb
├── prompts/
│   ├── non_implicit.txt
│   ├── explicit.txt        # deprecated alias
│   └── implicit.txt
├── scripts/
│   ├── run_benchmark.py
│   ├── run_openai_benchmark.py
│   └── openai_batch_roundtrip.py
├── src/
│   └── eng_drb_benchmark/
│       ├── __init__.py
│       ├── batch.py
│       ├── data.py
│       ├── evaluate.py
│       ├── postprocess.py
│       └── providers/
│           ├── __init__.py
│           ├── claude.py
│           └── openai.py
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Install

```bash
python -m pip install -e .
```

Or install dependencies directly:

```bash
python -m pip install -r requirements.txt
```

## Quickstart notebooks

- `notebooks/ENG_DRB_quickstart.ipynb` loads ENG-DRB and exports benchmark-ready JSONL files.
- `notebooks/ENG_DRB_openai_reproduce.ipynb` covers the OpenAI Batch path.
- `notebooks/ENG_DRB_claude_reproduce.ipynb` covers the Claude direct-request path.

## Reproducing the OpenAI backend

Prepare gold JSONL and Batch request JSONL for `non_implicit`:

```bash
python scripts/run_benchmark.py   --provider openai   --relation-type non_implicit   --prompt-file prompts/non_implicit.txt   --output-dir results/openai_non_implicit
```

Submit the Batch job:

```bash
export OPENAI_API_KEY=your_key_here
python scripts/openai_batch_roundtrip.py submit   results/openai_non_implicit/requests_non_implicit.jsonl
```

Download the results after the batch finishes:

```bash
python scripts/openai_batch_roundtrip.py download   BATCH_ID   results/openai_non_implicit/batch_results.jsonl
```

Merge, deduplicate, and score:

```bash
python scripts/run_benchmark.py   --provider openai   --relation-type non_implicit   --prompt-file prompts/non_implicit.txt   --output-dir results/openai_non_implicit   --batch-results results/openai_non_implicit/batch_results.jsonl
```

Repeat the same pattern for `implicit` by switching the relation type and prompt file.

## Reproducing the Claude backend

Run the full Claude path directly for `non_implicit`:

```bash
export ANTHROPIC_API_KEY=your_key_here
python scripts/run_benchmark.py   --provider claude   --relation-type non_implicit   --prompt-file prompts/non_implicit.txt   --output-dir results/claude_non_implicit
```

That command will:

1. load ENG-DRB from Hugging Face,
2. export `gold_non_implicit.jsonl`,
3. run Claude inference window by window,
4. write raw responses to `claude_raw_non_implicit.jsonl`,
5. merge and deduplicate predictions,
6. compute evaluation scores.

If you already have a Claude raw-results file and only want to post-process / evaluate it:

```bash
python scripts/run_benchmark.py   --provider claude   --relation-type non_implicit   --prompt-file prompts/non_implicit.txt   --output-dir results/claude_non_implicit   --batch-results results/claude_non_implicit/claude_raw_non_implicit.jsonl
```

## Important reproducibility note

This repo is designed to make reproduction much easier, but the backends are not identical operationally:

- **OpenAI** requires an API key plus asynchronous Batch submission and download.
- **Claude** requires an API key plus direct online inference for each window.

So the repository is unified at the benchmark level, not at the provider-runtime level.

## Benchmark fidelity note

The prompt files under `prompts/` are included verbatim for fidelity to the original benchmark setup.
