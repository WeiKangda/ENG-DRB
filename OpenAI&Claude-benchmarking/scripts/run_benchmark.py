from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from eng_drb_benchmark.batch import create_openai_batch_requests
from eng_drb_benchmark.data import export_gold_jsonl, load_eng_drb
from eng_drb_benchmark.evaluate import evaluate_from_files
from eng_drb_benchmark.postprocess import (
    deduplicate_prediction_file,
    merge_claude_results,
    merge_openai_batch_results,
)
from eng_drb_benchmark.providers.claude import DEFAULT_CLAUDE_MODEL, run_claude_requests
from eng_drb_benchmark.providers.openai import DEFAULT_OPENAI_MODEL


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare and evaluate an ENG-DRB benchmark run for OpenAI or Claude."
    )
    parser.add_argument("--provider", choices=["openai", "claude"], default="openai")
    parser.add_argument(
        "--split",
        default=None,
        help="Dataset split on Hugging Face. If omitted, the first available split is used.",
    )
    parser.add_argument(
        "--relation-type",
        default="non_implicit",
        help="Relation subset: all, implicit, or non_implicit. The deprecated alias explicit is still accepted.",
    )
    parser.add_argument("--prompt-file", required=True, help="Path to the system prompt text file.")
    parser.add_argument("--output-dir", default="results/run", help="Directory for generated files.")
    parser.add_argument(
        "--batch-results",
        help="Path to raw provider outputs JSONL. If omitted, the script prepares requests only for OpenAI, or runs inference directly for Claude.",
    )
    parser.add_argument("--window-size", type=int, default=20)
    parser.add_argument("--step", type=int, default=10)

    parser.add_argument("--openai-model", default=DEFAULT_OPENAI_MODEL)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.0)
    parser.add_argument("--max-completion-tokens", type=int, default=100_000)

    parser.add_argument("--claude-model", default=DEFAULT_CLAUDE_MODEL)
    parser.add_argument("--claude-max-tokens", type=int, default=3000)
    parser.add_argument("--retry-wait-seconds", type=int, default=60)
    return parser.parse_args()


def _normalize_relation_type(value: str) -> str:
    return "non_implicit" if value.lower() == "explicit" else value.lower()


def main() -> None:
    args = parse_args()
    args.relation_type = _normalize_relation_type(args.relation_type)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_dict = load_eng_drb()
    split_name = args.split or next(iter(dataset_dict.keys()))
    dataset = dataset_dict[split_name]

    gold_path = export_gold_jsonl(
        dataset,
        output_dir / f"gold_{args.relation_type}.jsonl",
        relation_type=args.relation_type,
    )
    prompt_text = Path(args.prompt_file).read_text(encoding="utf-8").strip()
    print(f"Gold file: {gold_path}")

    raw_results_path: Path | None = Path(args.batch_results) if args.batch_results else None

    if args.provider == "openai":
        request_path = create_openai_batch_requests(
            records=(json.loads(line) for line in gold_path.read_text(encoding="utf-8").splitlines() if line.strip()),
            output_path=output_dir / f"requests_{args.relation_type}.jsonl",
            prompt_text=prompt_text,
            model=args.openai_model,
            window_size=args.window_size,
            step=args.step,
            temperature=args.temperature,
            top_p=args.top_p,
            max_completion_tokens=args.max_completion_tokens,
        )
        print(f"Batch requests: {request_path}")
        if raw_results_path is None:
            return
        merged_path = merge_openai_batch_results(raw_results_path, output_dir / "pred_merged.jsonl")
    else:
        if raw_results_path is None:
            raw_results_path = run_claude_requests(
                records=(json.loads(line) for line in gold_path.read_text(encoding="utf-8").splitlines() if line.strip()),
                output_path=output_dir / f"claude_raw_{args.relation_type}.jsonl",
                prompt_text=prompt_text,
                model=args.claude_model,
                max_tokens=args.claude_max_tokens,
                window_size=args.window_size,
                step=args.step,
                retry_wait_seconds=args.retry_wait_seconds,
            )
        merged_path = merge_claude_results(raw_results_path, output_dir / "pred_merged.jsonl")

    dedup_path = deduplicate_prediction_file(merged_path, output_dir / "pred_dedup.jsonl")
    scores = evaluate_from_files(gold_path, dedup_path)
    with open(output_dir / "scores.json", "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2)
    print(json.dumps(scores["partial_agreement"]["overall_scores"], indent=2))
    print(json.dumps(scores["exact_match"]["overall_scores"], indent=2))
    print(f"Merged predictions: {merged_path}")
    print(f"Deduplicated predictions: {dedup_path}")
    print(f"Scores: {output_dir / 'scores.json'}")


if __name__ == "__main__":
    main()
