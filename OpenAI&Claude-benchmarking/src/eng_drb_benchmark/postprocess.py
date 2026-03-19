from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

Sense = dict[str, Any]

REQUIRED_SENSE_KEYS = {
    "Arg1_start": (int, float),
    "Arg1_end": (int, float),
    "Arg2_start": (int, float),
    "Arg2_end": (int, float),
    "sense": str,
    "explicit": str,
    "confidence": (int, float),
}

_OPENAI_ID_RE = re.compile(r"^(?P<doc>.+)_(?P<start>\d+)-(?P<end>\d+)$")
_CLAUDE_ID_RE = re.compile(r"^(?P<doc>.+)_spansection_(?P<start>\d+)-(?P<end>\d+)$")


def _extract_json_from_text(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    for start_marker in ("```json", "```"):
        start = text.find(start_marker)
        if start == -1:
            continue
        json_start = start + len(start_marker)
        if json_start < len(text) and text[json_start] == "\n":
            json_start += 1
        end = text.find("```", json_start)
        if end == -1:
            continue
        inner = text[json_start:end].strip()
        try:
            return json.loads(inner)
        except json.JSONDecodeError:
            continue
    raise ValueError("Could not parse JSON response directly or from a fenced code block.")


def _validate_sense_obj(sense_obj: dict[str, Any]) -> Sense:
    for key, expected_type in REQUIRED_SENSE_KEYS.items():
        if key not in sense_obj:
            raise ValueError(f"Missing key {key!r} in sense object")
        if not isinstance(sense_obj[key], expected_type):
            expected = (
                " or ".join(t.__name__ for t in expected_type)
                if isinstance(expected_type, tuple)
                else expected_type.__name__
            )
            raise ValueError(
                f"Key {key!r} has incorrect type; expected {expected}, found {type(sense_obj[key]).__name__}"
            )

    if sense_obj["Arg1_start"] > sense_obj["Arg1_end"]:
        raise ValueError("Arg1_start must be <= Arg1_end")
    if sense_obj["Arg2_start"] > sense_obj["Arg2_end"]:
        raise ValueError("Arg2_start must be <= Arg2_end")
    return sense_obj


def _doc_prefix_from_request_id(request_id: str) -> str:
    claude_match = _CLAUDE_ID_RE.match(request_id)
    if claude_match:
        return claude_match.group("doc")
    openai_match = _OPENAI_ID_RE.match(request_id)
    if openai_match:
        return openai_match.group("doc")
    return request_id


def _merge_result_file(input_file_path: str | Path, output_file_path: str | Path, *, provider: str) -> Path:
    input_file_path = Path(input_file_path)
    output_file_path = Path(output_file_path)
    output_file_path.parent.mkdir(parents=True, exist_ok=True)

    merged: dict[str, list[Sense]] = {}
    skipped = 0

    with input_file_path.open("r", encoding="utf-8") as infile:
        for line_number, line in enumerate(infile, start=1):
            if not line.strip():
                continue
            try:
                outer = json.loads(line)
                if provider == "openai":
                    request_id = outer["custom_id"]
                    content = (
                        outer.get("response", {})
                        .get("body", {})
                        .get("choices", [{}])[0]
                        .get("message", {})
                        .get("content")
                    )
                    if not content:
                        raise ValueError("Missing response.body.choices[0].message.content")
                elif provider == "claude":
                    request_id = outer["id"]
                    content = outer.get("response", {}).get("content", [{}])[0].get("text")
                    if not content:
                        raise ValueError("Missing response.content[0].text")
                else:
                    raise ValueError(f"Unsupported provider: {provider}")

                doc_id = _doc_prefix_from_request_id(request_id)
                inner = _extract_json_from_text(content)
                senses = inner.get("Senses")
                if not isinstance(senses, list):
                    raise ValueError("Inner JSON must contain a list-valued 'Senses' field")
                validated = [_validate_sense_obj(s) for s in senses]
                merged.setdefault(doc_id, []).extend(validated)
            except Exception as exc:
                skipped += 1
                print(f"Skipping line {line_number}: {exc}")

    with output_file_path.open("w", encoding="utf-8") as outfile:
        for doc_id, senses in merged.items():
            outfile.write(json.dumps({"id": doc_id, "Senses": senses}, ensure_ascii=False) + "\n")

    print(f"Merged {len(merged)} documents from {provider}; skipped {skipped} malformed lines.")
    return output_file_path


def merge_openai_batch_results(input_file_path: str | Path, output_file_path: str | Path) -> Path:
    return _merge_result_file(input_file_path, output_file_path, provider="openai")


def merge_claude_results(input_file_path: str | Path, output_file_path: str | Path) -> Path:
    return _merge_result_file(input_file_path, output_file_path, provider="claude")


def _check_overlap(start1: Any, end1: Any, start2: Any, end2: Any) -> bool:
    try:
        start1, end1, start2, end2 = float(start1), float(end1), float(start2), float(end2)
    except (TypeError, ValueError):
        return False
    if start1 > end1 or start2 > end2:
        return False
    return max(start1, start2) <= min(end1, end2)


def _are_partially_agreed(sense1: Sense, sense2: Sense) -> bool:
    if sense1.get("sense") != sense2.get("sense"):
        return False
    arg1_overlap = _check_overlap(
        sense1.get("Arg1_start"),
        sense1.get("Arg1_end"),
        sense2.get("Arg1_start"),
        sense2.get("Arg1_end"),
    )
    arg2_overlap = _check_overlap(
        sense1.get("Arg2_start"),
        sense1.get("Arg2_end"),
        sense2.get("Arg2_start"),
        sense2.get("Arg2_end"),
    )
    return arg1_overlap and arg2_overlap


def _merge_sense_objects(sense1: Sense, sense2: Sense) -> Sense:
    merged_sense = {
        "sense": sense1["sense"],
        "Arg1_start": min(float(sense1["Arg1_start"]), float(sense2["Arg1_start"])),
        "Arg1_end": max(float(sense1["Arg1_end"]), float(sense2["Arg1_end"])),
        "Arg2_start": min(float(sense1["Arg2_start"]), float(sense2["Arg2_start"])),
        "Arg2_end": max(float(sense1["Arg2_end"]), float(sense2["Arg2_end"])),
    }

    explicit1 = sense1.get("explicit", "")
    explicit2 = sense2.get("explicit", "")
    if explicit1 == explicit2:
        merged_sense["explicit"] = explicit1
    elif explicit1 == "implicit":
        merged_sense["explicit"] = explicit2
    elif explicit2 == "implicit":
        merged_sense["explicit"] = explicit1
    else:
        merged_sense["explicit"] = " | ".join(
            sorted(set(explicit1.split(" | ")) | set(explicit2.split(" | ")))
        ).strip()

    try:
        conf1 = float(sense1.get("confidence", 0.0))
    except (TypeError, ValueError):
        conf1 = 0.0
    try:
        conf2 = float(sense2.get("confidence", 0.0))
    except (TypeError, ValueError):
        conf2 = 0.0
    merged_sense["confidence"] = max(conf1, conf2)
    return merged_sense


def _deduplicate_record(json_obj: dict[str, Any]) -> dict[str, Any]:
    senses = json_obj.get("Senses", [])

    seen_sense_str = set()
    unique_senses: list[Sense] = []
    for sense in senses:
        sense_str = json.dumps(sense, sort_keys=True)
        if sense_str not in seen_sense_str:
            seen_sense_str.add(sense_str)
            unique_senses.append(sense)

    current_senses = unique_senses
    merged_occurred = True
    while merged_occurred:
        merged_occurred = False
        next_senses: list[Sense] = []
        used_indices = set()

        for i in range(len(current_senses)):
            if i in used_indices:
                continue
            merged_sense = current_senses[i]
            used_indices.add(i)
            for j in range(i + 1, len(current_senses)):
                if j in used_indices:
                    continue
                if _are_partially_agreed(merged_sense, current_senses[j]):
                    merged_sense = _merge_sense_objects(merged_sense, current_senses[j])
                    used_indices.add(j)
                    merged_occurred = True
            next_senses.append(merged_sense)
        current_senses = next_senses

    grouped_by_range: dict[str, list[Sense]] = {}
    for sense in current_senses:
        try:
            range_key = (
                f"{float(sense.get('Arg1_start', math.nan))}-{float(sense.get('Arg1_end', math.nan))}_"
                f"{float(sense.get('Arg2_start', math.nan))}-{float(sense.get('Arg2_end', math.nan))}"
            )
        except (TypeError, ValueError):
            range_key = "invalid_range"
        grouped_by_range.setdefault(range_key, []).append(sense)

    final_senses: list[Sense] = []
    for range_key, senses_list in grouped_by_range.items():
        if range_key == "invalid_range":
            final_senses.extend(senses_list)
            continue
        sorted_senses = sorted(
            senses_list,
            key=lambda s: float(s.get("confidence", -1.0)),
            reverse=True,
        )
        final_senses.append(sorted_senses[0])

    return {"id": json_obj.get("id"), "Senses": final_senses}


def deduplicate_prediction_file(input_file_path: str | Path, output_file_path: str | Path) -> Path:
    input_file_path = Path(input_file_path)
    output_file_path = Path(output_file_path)
    output_file_path.parent.mkdir(parents=True, exist_ok=True)

    processed = 0
    with input_file_path.open("r", encoding="utf-8") as infile, output_file_path.open(
        "w", encoding="utf-8"
    ) as outfile:
        for line in infile:
            if not line.strip():
                continue
            record = json.loads(line)
            processed_record = _deduplicate_record(record)
            outfile.write(json.dumps(processed_record, ensure_ascii=False) + "\n")
            processed += 1

    print(f"Deduplicated {processed} prediction records.")
    return output_file_path
