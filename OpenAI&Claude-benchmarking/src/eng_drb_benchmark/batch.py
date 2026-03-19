from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Iterator

Record = dict[str, Any]


def _safe_int_span(span_no: Any) -> int:
    try:
        return int(float(span_no))
    except (TypeError, ValueError):
        return 0


def iter_sliding_windows(
    records: Iterable[Record], *, window_size: int = 20, step: int = 10
) -> Iterator[tuple[Record, list[dict[str, Any]], int, int, str]]:
    """Yield document windows for provider-specific runners.

    Yields
    ------
    tuple
        (record, window, start_no, end_no, request_id)
    """
    if window_size <= 0 or step <= 0:
        raise ValueError("window_size and step must both be positive")

    for record in records:
        doc = record.get("Doc")
        spans = record.get("Spans")
        if not doc or not isinstance(spans, list) or not spans:
            continue

        for start_idx in range(0, len(spans), step):
            window = spans[start_idx : start_idx + window_size]
            if not window:
                continue
            start_no = _safe_int_span(window[0].get("span_no"))
            end_no = _safe_int_span(window[-1].get("span_no"))
            request_id = f"{doc}_{start_no}-{end_no}"
            yield record, window, start_no, end_no, request_id


def create_openai_batch_requests(
    records: Iterable[Record],
    output_path: str | Path,
    prompt_text: str,
    *,
    model: str = "o4-mini-2025-04-16",
    window_size: int = 20,
    step: int = 10,
    temperature: float = 0.0,
    top_p: float = 0.0,
    max_completion_tokens: int = 100_000,
) -> Path:
    """Create a JSONL file suitable for the OpenAI Batch API."""
    if not prompt_text.strip():
        raise ValueError("prompt_text is empty")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    request_count = 0
    with output_path.open("w", encoding="utf-8") as f_out:
        for _record, window, _start_no, _end_no, request_id in iter_sliding_windows(
            records, window_size=window_size, step=step
        ):
            request_obj = {
                "custom_id": request_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model,
                    "temperature": temperature,
                    "top_p": top_p,
                    "max_completion_tokens": max_completion_tokens,
                    "messages": [
                        {"role": "system", "content": prompt_text},
                        {"role": "user", "content": json.dumps(window, ensure_ascii=False)},
                    ],
                },
            }
            f_out.write(json.dumps(request_obj, ensure_ascii=False) + "\n")
            request_count += 1

    print(f"Wrote {request_count} batch requests to {output_path}")
    return output_path
