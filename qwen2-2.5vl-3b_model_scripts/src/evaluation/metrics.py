"""metrics.py — Evaluation metrics and logit preprocessing.

Contains metrics for both tasks:
  - Detector (bbox): iou, coverage, get_compute_metrics            [unchanged from original]
  - Extraction (fields): field_level_scores, document_exact_match,
    get_compute_metrics_extraction                                  [new, added for Pipeline 1]
"""

import json
import re
import numpy as np
from typing import Tuple, List, Dict, Any, Callable, Optional
from transformers import AutoTokenizer


def get_preprocess_logits_for_metrics() -> Callable:
    """Reduces per-token vocab logits to argmax token ids before Trainer accumulation to prevent OOM."""
    def _fn(logits, labels):
        """Extract the argmax token prediction from the output logits tensor."""
        if isinstance(logits, tuple):
            logits = logits[0]
        return logits.argmax(dim=-1)
    return _fn


def _safe_json_load(text: str) -> Optional[Any]:
    """Parse JSON from raw generated text after removing markdown fences."""
    clean = re.sub(r"```(?:json)?|```", "", text).strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        return None


def _extract_label_pred_pairs(preds: np.ndarray, labels: np.ndarray, tokenizer: AutoTokenizer) -> List[Tuple[str, str]]:
    """Extract text pairs from predicted logits and label tensors."""
    preds = preds[:, :-1]
    labels = labels[:, 1:]
    pairs = []

    for row_pred, row_label in zip(preds, labels):
        mask = row_label != -100
        if not mask.any():
            continue
        label_text = tokenizer.decode(row_label[mask], skip_special_tokens=True).strip()
        pred_text = tokenizer.decode(row_pred[mask], skip_special_tokens=True).strip()
        pairs.append((pred_text, label_text))

    return pairs


# ---------------------------------------------------------------------------
# Detector (bbox) metrics — unchanged
# ---------------------------------------------------------------------------

def iou(box_a: List[int], box_b: List[int]) -> float:
    """Compute intersection-over-union between two bboxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def coverage(pred_box: List[int], gt_box: List[int]) -> float:
    """Compute coverage of ground truth by a predicted bbox."""
    ax1, ay1, ax2, ay2 = pred_box
    bx1, by1, bx2, by2 = gt_box
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    gt_area = max(0, bx2 - bx1) * max(0, by2 - by1)
    return inter / gt_area if gt_area > 0 else 0.0


def get_compute_metrics(config: Dict[str, Any]) -> Callable:
    """Build the compute_metrics function used by the Trainer during validation for
    the detector task (bbox IoU/coverage)."""
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["model_id"])

    def compute(eval_pred: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
        """Decode predicted tokens and evaluate bounding box JSON outputs for IoU and Coverage."""
        preds, labels = eval_pred
        pairs = _extract_label_pred_pairs(np.asarray(preds), np.asarray(labels), tokenizer)
        ious, coverages, parsed = [], [], 0

        for pred_text, label_text in pairs:
            pred_json = _safe_json_load(pred_text)
            label_json = _safe_json_load(label_text)

            if pred_json is None or label_json is None:
                ious.append(0.0)
                coverages.append(0.0)
                continue

            parsed += 1
            try:
                ious.append(iou(pred_json["bbox"], label_json["bbox"]))
                coverages.append(coverage(pred_json["bbox"], label_json["bbox"]))
            except (KeyError, TypeError, ValueError):
                ious.append(0.0)
                coverages.append(0.0)

        n = max(1, len(pairs))
        return {
            "iou": float(np.mean(ious)) if ious else 0.0,
            "coverage": float(np.mean(coverages)) if coverages else 0.0,
            "parse_rate": parsed / n,
        }
    return compute


# ---------------------------------------------------------------------------
# Extraction (fields) metrics — new, added for Pipeline 1
# ---------------------------------------------------------------------------

# Field-level scoring for the extraction task now lives in field_matching.py
# (schema-aware, format-tolerant: punctuation/casing/null-alias/date-format/
# decimal/list-order insensitive). Re-exported here so existing imports of
# `from src.evaluation.metrics import field_level_scores, document_exact_match`
# keep working unchanged.
# field_level_scores_partial additionally expands list-of-object fields
# (diagnoses, procedures) into one partial-credit metric per sub-field
# instead of one collapsed all-or-nothing score per list item -- see
# field_matching.py's module docstring for the pairing/scoring algorithm.
from src.evaluation.field_matching import (  # noqa: E402,F401
    field_level_scores,
    field_level_scores_partial,
    document_exact_match,
)


def get_compute_metrics_extraction(config: Dict[str, Any]) -> Callable:
    """Build the compute_metrics function used by the Trainer during validation for
    the extraction task: per-field accuracy, document-level exact match, parse rate."""
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["model_id"])
    schema = config["fields"]

    def compute(eval_pred: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
        """Decode predicted tokens and evaluate field-JSON outputs against ground truth."""
        preds, labels = eval_pred
        pairs = _extract_label_pred_pairs(np.asarray(preds), np.asarray(labels), tokenizer)

        doc_matches, all_field_scores, parsed, scored = [], [], 0, 0

        for pred_text, label_text in pairs:
            label_json = _safe_json_load(label_text)
            if not isinstance(label_json, dict):
                # Malformed ground truth shouldn't happen post prepare_dataset_extraction.py,
                # but skip rather than silently mis-score if it does. Not counted toward
                # parse_rate since there's no valid ground truth to parse against.
                continue

            scored += 1
            pred_json = _safe_json_load(pred_text)
            if not isinstance(pred_json, dict):
                doc_matches.append(0.0)
                # Zero-fill via field_level_scores_partial(None, label_json, schema)
                # rather than a fixed schema-wide key list: this record's own
                # label_json may be sparse (not every schema field annotated), and
                # calling the same scoring function with an empty prediction
                # guarantees the zero-filled key set exactly matches whatever keys
                # a successfully-parsed row with this same label_json would have
                # produced -- so failures are weighted identically to successes,
                # regardless of how complete any individual record's ground truth is.
                zero_scores = field_level_scores_partial(None, label_json, schema)
                all_field_scores.extend(zero_scores.values())
                continue

            parsed += 1
            # Reporting/training-signal metric ("field_accuracy") uses partial credit
            # for list sub-fields. doc_exact_match stays strict/unchanged: it's derived
            # from the original all-or-nothing field_level_scores, not the expanded one,
            # so a doc only counts as an exact match if every list item fully matches
            # on every sub-field -- partial credit never masks that at the doc level.
            field_scores = field_level_scores_partial(pred_json, label_json, schema)
            strict_field_scores = field_level_scores(pred_json, label_json, schema)
            all_field_scores.extend(field_scores.values())
            doc_matches.append(1.0 if strict_field_scores and all(v == 1.0 for v in strict_field_scores.values()) else 0.0)

        n = max(1, scored)
        return {
            "field_accuracy": float(np.mean(all_field_scores)) if all_field_scores else 0.0,
            "doc_exact_match": float(np.mean(doc_matches)) if doc_matches else 0.0,
            "parse_rate": parsed / n,
        }
    return compute