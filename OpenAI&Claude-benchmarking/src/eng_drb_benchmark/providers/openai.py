from __future__ import annotations

import os
from pathlib import Path

from openai import OpenAI

DEFAULT_OPENAI_MODEL = "o4-mini-2025-04-16"


def _make_client(api_key: str | None = None) -> OpenAI:
    api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("openai_api_key")
    if not api_key:
        raise RuntimeError("OpenAI API key not found. Set OPENAI_API_KEY.")
    return OpenAI(api_key=api_key)


def submit_openai_batch(requests_jsonl: str | Path, *, completion_window: str = "24h", api_key: str | None = None) -> str:
    client = _make_client(api_key)
    requests_jsonl = Path(requests_jsonl)
    upload = client.files.create(file=requests_jsonl.open("rb"), purpose="batch")
    batch_job = client.batches.create(input_file_id=upload.id, endpoint="/v1/chat/completions", completion_window=completion_window)
    return batch_job.id


def download_openai_batch_results(batch_id: str, output_path: str | Path, *, api_key: str | None = None) -> Path:
    client = _make_client(api_key)
    batch_job = client.batches.retrieve(batch_id)
    if batch_job.status != "completed":
        raise RuntimeError(f"Batch {batch_id} is not completed. Current status: {batch_job.status}")
    if not batch_job.output_file_id:
        raise RuntimeError(f"Batch {batch_id} completed without an output file id.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    content = client.files.content(batch_job.output_file_id).content
    output_path.write_bytes(content)
    return output_path
