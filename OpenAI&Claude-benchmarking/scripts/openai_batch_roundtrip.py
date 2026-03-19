from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from eng_drb_benchmark.providers.openai import download_openai_batch_results, submit_openai_batch


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit or download an OpenAI Batch run for ENG-DRB.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    submit = subparsers.add_parser("submit")
    submit.add_argument("requests_jsonl")
    submit.add_argument("--completion-window", default="24h")

    download = subparsers.add_parser("download")
    download.add_argument("batch_id")
    download.add_argument("output_jsonl")

    args = parser.parse_args()
    if args.command == "submit":
        batch_id = submit_openai_batch(args.requests_jsonl, completion_window=args.completion_window)
        print(batch_id)
    else:
        output = download_openai_batch_results(args.batch_id, args.output_jsonl)
        print(output)


if __name__ == "__main__":
    main()
