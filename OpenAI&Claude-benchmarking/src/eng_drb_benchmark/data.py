from __future__ import annotations

import copy
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from datasets import Dataset, DatasetDict, get_dataset_split_names, load_dataset

DATASET_NAME = "ChengZhangPNW/ENG-DRB"


Record = dict[str, Any]


def get_available_splits(dataset_name: str = DATASET_NAME) -> list[str]:
    """Return split names published on the Hugging Face Hub."""
    return list(get_dataset_split_names(dataset_name))


def load_eng_drb(dataset_name: str = DATASET_NAME, split: str | None = None) -> Dataset | DatasetDict:
    """Load ENG-DRB from Hugging Face.

    If ``split`` is ``None``, the full DatasetDict is returned.
    Otherwise, the requested split is returned.
    """
    return load_dataset(dataset_name, split=split) if split else load_dataset(dataset_name)


def filter_record_senses(record: Record, relation_type: str = "all") -> Record:
    """Filter a record's senses by relation type.

    Parameters
    ----------
    record:
        A single ENG-DRB example containing a ``Senses`` field.
    relation_type:
        One of ``all``, ``implicit``, or ``non_implicit``.
        The deprecated alias ``explicit`` is also accepted for backward
        compatibility. In the ENG-DRB data, implicit relations are encoded as
        ``sense["explicit"] == "implicit"``; non-implicit relations are
        everything else, including explicit and AltLex senses.
    """
    relation_type = relation_type.lower()
    if relation_type == "explicit":
        relation_type = "non_implicit"
    if relation_type not in {"all", "implicit", "non_implicit"}:
        raise ValueError("relation_type must be one of: all, implicit, non_implicit")

    if relation_type == "all":
        return copy.deepcopy(record)

    new_record = copy.deepcopy(record)
    senses = record.get("Senses", []) or []

    if relation_type == "implicit":
        new_record["Senses"] = [s for s in senses if s.get("explicit") == "implicit"]
    else:
        new_record["Senses"] = [s for s in senses if s.get("explicit") != "implicit"]

    return new_record


def summarize_relation_counts(records: Iterable[Record]) -> dict[str, int]:
    """Count how many records and senses appear in a dataset slice."""
    doc_count = 0
    total_senses = 0
    implicit_senses = 0
    non_implicit_senses = 0
    sense_labels: Counter[str] = Counter()

    for record in records:
        doc_count += 1
        senses = record.get("Senses", []) or []
        total_senses += len(senses)
        for sense in senses:
            if sense.get("explicit") == "implicit":
                implicit_senses += 1
            else:
                non_implicit_senses += 1
            if "sense" in sense:
                sense_labels[str(sense["sense"])] += 1

    return {
        "documents": doc_count,
        "total_senses": total_senses,
        "implicit_senses": implicit_senses,
        "non_implicit_senses": non_implicit_senses,
        "unique_sense_labels": len(sense_labels),
    }


def export_gold_jsonl(
    dataset: Dataset,
    output_path: str | Path,
    relation_type: str = "all",
    keep_empty_records: bool = True,
) -> Path:
    """Export a Hugging Face split to JSONL in the format used by the benchmark code."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with output_path.open("w", encoding="utf-8") as f:
        for record in dataset:
            filtered = filter_record_senses(record, relation_type=relation_type)
            if keep_empty_records or filtered.get("Senses"):
                f.write(json.dumps(filtered, ensure_ascii=False) + "\n")
                written += 1

    if written == 0:
        raise ValueError(f"No records were written to {output_path}. Check the requested split/relation_type.")
    return output_path
