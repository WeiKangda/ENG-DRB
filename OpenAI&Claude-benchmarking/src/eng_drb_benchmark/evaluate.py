from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment


Sense = dict[str, Any]


def normalize_sense(sense_str: str) -> str:
    parts = sense_str.split(".")
    return ".".join(parts[:2]) if len(parts) > 2 else sense_str


def get_covered_span_nos(start: float | None, end: float | None, all_span_nos: list[float]) -> set[float]:
    if start is None or end is None or not all_span_nos:
        return set()
    covered = set()
    for span_no in sorted(all_span_nos):
        if start <= span_no <= end:
            covered.add(span_no)
        elif span_no > end:
            break
    return covered


def calculate_partial_agreement(gold_sense: Sense, pred_sense: Sense, all_span_nos: list[float]) -> float:
    gold_arg1 = get_covered_span_nos(gold_sense.get("Arg1_start"), gold_sense.get("Arg1_end"), all_span_nos)
    gold_arg2 = get_covered_span_nos(gold_sense.get("Arg2_start"), gold_sense.get("Arg2_end"), all_span_nos)
    pred_arg1 = get_covered_span_nos(pred_sense.get("Arg1_start"), pred_sense.get("Arg1_end"), all_span_nos)
    pred_arg2 = get_covered_span_nos(pred_sense.get("Arg2_start"), pred_sense.get("Arg2_end"), all_span_nos)

    numerator = len(gold_arg1 & pred_arg1) + len(gold_arg2 & pred_arg2)
    denominator = len(gold_arg1 | gold_arg2 | pred_arg1 | pred_arg2)
    return 0.0 if denominator == 0 else numerator / denominator


def load_data_and_spans(path: str | Path, id_field: str = "id", senses_field: str = "Senses", spans_field: str = "Spans"):
    senses_map: dict[str, list[Sense]] = {}
    span_nos_map: dict[str, list[float]] = {}

    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            key = obj.get(id_field)
            if key is None:
                continue

            processed_senses: list[Sense] = []
            for sense in obj.get(senses_field, []) or []:
                if not isinstance(sense, dict):
                    continue
                sense = dict(sense)
                sense.pop("confidence", None)
                sense["sense"] = normalize_sense(sense.get("sense", ""))
                processed_senses.append(sense)
            senses_map[key] = processed_senses

            spans = obj.get(spans_field, []) or []
            span_nos = sorted({span.get("span_no") for span in spans if isinstance(span, dict) and span.get("span_no") is not None})
            span_nos_map[key] = span_nos

    return senses_map, span_nos_map


def compute_scores(
    gold_senses_map: dict[str, list[Sense]],
    pred_senses_map: dict[str, list[Sense]],
    span_nos_map: dict[str, list[float]],
    *,
    use_partial_agreement: bool = True,
) -> dict[str, Any]:
    per_item_scores: dict[str, Any] = {}
    total_tp = 0.0
    total_fp = 0.0
    total_fn = 0.0
    total_gold_count = 0
    total_pred_count = 0

    all_keys = sorted(set(gold_senses_map) | set(pred_senses_map))
    for key in all_keys:
        gold_senses = gold_senses_map.get(key, [])
        pred_senses = pred_senses_map.get(key, [])
        span_nos = span_nos_map.get(key, [])

        num_gold = len(gold_senses)
        num_pred = len(pred_senses)
        total_gold_count += num_gold
        total_pred_count += num_pred

        item_tp = item_fp = item_fn = 0.0

        if num_gold == 0 and num_pred == 0:
            per_item_scores[key] = {
                "precision": 1.0,
                "recall": 1.0,
                "f1": 1.0,
                "tp": 0.0,
                "fp": 0.0,
                "fn": 0.0,
                "num_gold": 0,
                "num_pred": 0,
            }
            continue

        if use_partial_agreement:
            if num_gold > 0 and num_pred > 0:
                cost_matrix = np.full((num_gold, num_pred), 1.0)
                for i, gold_s in enumerate(gold_senses):
                    for j, pred_s in enumerate(pred_senses):
                        if gold_s.get("sense") == pred_s.get("sense"):
                            cost_matrix[i, j] = -calculate_partial_agreement(gold_s, pred_s, span_nos)
                row_ind, col_ind = linear_sum_assignment(cost_matrix)
                for r, c in zip(row_ind, col_ind):
                    if gold_senses[r].get("sense") == pred_senses[c].get("sense"):
                        item_tp += -cost_matrix[r, c]
                item_fp = num_pred - item_tp
                item_fn = num_gold - item_tp
            elif num_gold > 0:
                item_fn = float(num_gold)
            elif num_pred > 0:
                item_fp = float(num_pred)
        else:
            keys_for_exact_match = ("Arg1_start", "Arg1_end", "Arg2_start", "Arg2_end", "sense")
            gold_set = {tuple(s.get(k) for k in keys_for_exact_match) for s in gold_senses}
            pred_set = {tuple(s.get(k) for k in keys_for_exact_match) for s in pred_senses}
            item_tp = float(len(gold_set & pred_set))
            item_fp = float(len(pred_set - gold_set))
            item_fn = float(len(gold_set - pred_set))

        precision = item_tp / (item_tp + item_fp) if (item_tp + item_fp) > 0 else 0.0
        recall = item_tp / (item_tp + item_fn) if (item_tp + item_fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        per_item_scores[key] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "tp": round(item_tp, 4),
            "fp": round(item_fp, 4),
            "fn": round(item_fn, 4),
            "num_gold": num_gold,
            "num_pred": num_pred,
        }
        total_tp += item_tp
        total_fp += item_fp
        total_fn += item_fn

    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0.0

    return {
        "per_item_scores": per_item_scores,
        "overall_scores": {
            "precision": round(overall_precision, 4),
            "recall": round(overall_recall, 4),
            "f1": round(overall_f1, 4),
            "total_tp": round(total_tp, 4),
            "total_fp": round(total_fp, 4),
            "total_fn": round(total_fn, 4),
            "total_gold": total_gold_count,
            "total_pred": total_pred_count,
        },
    }


def evaluate_from_files(gold_path: str | Path, pred_path: str | Path) -> dict[str, Any]:
    gold_senses_map, gold_span_nos_map = load_data_and_spans(gold_path, id_field="Doc", senses_field="Senses", spans_field="Spans")
    pred_senses_map, _ = load_data_and_spans(pred_path, id_field="id", senses_field="Senses", spans_field="Spans")

    eval_doc_ids = sorted(gold_senses_map)
    pred_filtered = {k: pred_senses_map.get(k, []) for k in eval_doc_ids}
    span_filtered = {k: gold_span_nos_map.get(k, []) for k in eval_doc_ids}
    gold_filtered = {k: gold_senses_map[k] for k in eval_doc_ids}

    partial = compute_scores(gold_filtered, pred_filtered, span_filtered, use_partial_agreement=True)
    exact = compute_scores(gold_filtered, pred_filtered, span_filtered, use_partial_agreement=False)
    return {"partial_agreement": partial, "exact_match": exact}
