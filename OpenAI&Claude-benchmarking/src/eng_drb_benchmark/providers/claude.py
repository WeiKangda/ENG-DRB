from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Iterable

from anthropic import Anthropic, APIStatusError, RateLimitError

from eng_drb_benchmark.batch import iter_sliding_windows

DEFAULT_CLAUDE_MODEL = "claude-3-7-sonnet-20250219"
Record = dict[str, Any]


def _make_client(api_key: str | None = None) -> Anthropic:
    api_key = api_key or os.getenv("ANTHROPIC_API_KEY") or os.getenv("claude_api_key")
    if not api_key:
        raise RuntimeError("Anthropic API key not found. Set ANTHROPIC_API_KEY.")
    return Anthropic(api_key=api_key)


def run_claude_requests(
    records: Iterable[Record],
    output_path: str | Path,
    prompt_text: str,
    *,
    model: str = DEFAULT_CLAUDE_MODEL,
    max_tokens: int = 3000,
    window_size: int = 20,
    step: int = 10,
    api_key: str | None = None,
    retry_wait_seconds: int = 60,
) -> Path:
    """Run the ENG-DRB sliding-window benchmark against Claude directly.

    This follows the original notebook behavior more closely than the OpenAI path:
    it sends one request per window and streams results into a JSONL log that can
    later be post-processed and evaluated.
    """
    if not prompt_text.strip():
        raise ValueError("prompt_text is empty")

    client = _make_client(api_key)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_windows = 0
    success_count = 0
    failed_count = 0

    with output_path.open("w", encoding="utf-8") as outfile:
        for _record, window, start_no, end_no, base_request_id in iter_sliding_windows(
            records, window_size=window_size, step=step
        ):
            total_windows += 1
            request_id = base_request_id.replace(
                f"_{start_no}-{end_no}", f"_spansection_{start_no}-{end_no}"
            )
            payload = {
                "model": model,
                "system": prompt_text,
                "messages": [{"role": "user", "content": json.dumps(window, ensure_ascii=False)}],
                "max_tokens": max_tokens,
            }
            response_obj: dict[str, Any] = {"id": request_id}

            try:
                try:
                    message_response = client.messages.create(**payload)
                except RateLimitError:
                    time.sleep(retry_wait_seconds)
                    message_response = client.messages.create(**payload)
                response_obj["response"] = message_response.model_dump()
                success_count += 1
            except APIStatusError as exc:
                response_obj["error"] = f"API Error {exc.status_code}: {exc.response.text}"
                failed_count += 1
            except Exception as exc:
                response_obj["error"] = str(exc)
                failed_count += 1

            outfile.write(json.dumps(response_obj, ensure_ascii=False) + "\n")
            outfile.flush()

    print(
        f"Claude processing complete. Total windows attempted: {total_windows}; "
        f"succeeded: {success_count}; failed: {failed_count}"
    )
    return output_path
