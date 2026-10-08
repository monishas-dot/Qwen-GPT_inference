"""predict_extraction.py — resumable, folder-driven inference for the
full-page field-extraction task (Pipeline 1). Derived from evaluate_extraction.py.

Two changes from the eval script:
  1. Adds a `predict` split that reads images directly from a folder
     (--image_dir), so no ground-truth JSONL manifest is required. The
     existing train/val/test JSONL-based flow is unchanged and still works.
  2. Writes each row to per_file_report.jsonl as it's produced (flushed
     immediately) instead of only at the very end, and skips ids already
     present in that file on a re-run. A crash or interruption partway
     through no longer loses completed predictions -- just re-run the same
     command and it picks up where it left off.
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any, Set

import torch

# Disable cuDNN entirely. The Azure conda environment (vlm_extract) has a
# broken cuDNN library / known Turing-architecture FP16 Conv3D bug that
# throws CUDNN_STATUS_NOT_INITIALIZED. This forces PyTorch onto native CUDA
# kernels instead, which don't hit the bug. Must be set before any
# cuDNN-touching model code is imported below.
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False

from PIL import Image

from src.evaluation.metrics import field_level_scores, document_exact_match
from src.evaluation.eval_utils import load_config, load_split, dedupe_by_id
from src.inference.run_infer_detection import load_model, load_processor, run_model
from src.inference.run_infer_extraction import parse_fields
from src.preprocessing.preprocessing import Preprocessor
from src.utils.coordinate_scaling import resized_wh
from src.utils.field_schema import build_field_extraction_prompt

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")


def load_records_from_image_dir(image_dir: Path) -> List[Dict[str, Any]]:
    """Build records straight from a folder of images -- no JSONL manifest,
    no ground-truth fields. Recursive, so nested per-source subfolders (like
    the ones produced by pdf_to_page_images.py) are picked up too."""
    paths = sorted(
        p for p in image_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    return [{"id": p.stem, "image_path": str(p), "fields": None} for p in paths]


def load_already_done(report_path: Path) -> Set[str]:
    """Ids already present in an existing per_file_report.jsonl, so a re-run
    can skip them instead of redoing work."""
    if not report_path.exists():
        return set()
    done = set()
    with open(report_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # tolerate a truncated last line from a prior crash
            if "id" in row:
                done.add(row["id"])
    return done


def main() -> None:
    """Run inference: load config/model, infer on a split or folder, and
    save a resumable, incrementally-written report."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config_extractor_full_page.yaml")
    ap.add_argument("--adapter_model", required=True)
    ap.add_argument("--split", default="predict", choices=["train", "val", "test", "predict"])
    ap.add_argument("--image_dir", default=None,
                     help="Folder of images to run inference on directly (used when --split predict "
                          "and no *_jsonl manifest is configured/desired). Searched recursively.")
    ap.add_argument("--output_dir", default=None)
    args = ap.parse_args()

    config = load_config(args.config)
    schema = config["fields"]
    dtype = getattr(torch, config["model"]["torch_dtype"])
    base_id = config["model"]["model_id"]
    device_map = config["model"].get("device_map", "auto")
    attn_impl = config["model"].get("attn_implementation", "sdpa")
    
    # Fixed device configuration retrieval
    inference_device = config.get("inference", {}).get("device", "cpu")

    if args.split == "predict" and args.image_dir:
        image_dir = Path(args.image_dir)
        if not image_dir.is_dir():
            raise FileNotFoundError(f"--image_dir not found: {image_dir}")
        records = load_records_from_image_dir(image_dir)
        if not records:
            raise ValueError(f"No {IMAGE_EXTENSIONS} images found under {image_dir}")
    else:
        split_key = f"{args.split}_jsonl"
        if split_key not in config["paths"]:
            raise KeyError(
                f"config.paths has no '{split_key}'. Either re-run prepare_dataset_extraction.py, "
                f"or pass --image_dir to run directly on a folder of images."
            )
        records = dedupe_by_id(load_split(config["paths"][split_key]))
        if not records:
            raise ValueError(f"{config['paths'][split_key]} is empty.")

    output_dir = Path(args.output_dir or (Path(config["paths"]["infer_output_dir"]) / f"eval_{args.split}"))
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / "per_file_report.jsonl"
    already_done = load_already_done(report_path)
    if already_done:
        print(f"Resuming: {len(already_done)} record(s) already in {report_path}, skipping them.")
        records = [r for r in records if r["id"] not in already_done]

    if not records:
        print("Nothing left to do.")
    else:
        print("\nLoading base model + adapter...")
        model = load_model(base_id, dtype, device_map, attn_impl, args.adapter_model)
        proc = load_processor(base_id, config)
        pre = Preprocessor(config["augmentation"])
        prompt = build_field_extraction_prompt(schema, config["task_description"])
        system_prompt = config.get("system_prompt", "You are a precise medical document analyzer.").strip()
        
        # Safely extract max_new_tokens from config with a default fallback
        max_new_tokens = config.get("inference", {}).get("max_new_tokens", 512)
        
        min_pix = config["processor"]["min_pixels"]
        max_pix = config["processor"]["max_pixels"]

        print(f"\nRunning inference on {len(records)} '{args.split}' image(s)...\n")

        # Append mode: prior completed rows (if any) stay in the file untouched.
        with open(report_path, "a") as report_f:
            for i, r in enumerate(records):
                rid = r["id"]
                img_path = r["image_path"]
                gt_fields = r.get("fields")
                row = {"id": rid, "image_path": img_path}

                try:
                    raw_img = Image.open(img_path).convert("RGB")
                    img_w, img_h = raw_img.size

                    resized_w, resized_h = resized_wh(img_w, img_h, max_pix, max_pix)
                    prepped_img, _ = pre.run(raw_img, target_size=(resized_w, resized_h), augment=False)

                    raw_out = run_model(
                        model=model, proc=proc, img=prepped_img, prompt=prompt,
                        system_prompt=system_prompt, max_new_tokens=max_new_tokens,
                        device=inference_device
                    )

                    pred_fields = parse_fields(raw_out, schema)

                    field_scores = field_level_scores(pred_fields, gt_fields, schema) if gt_fields else None
                    doc_match = document_exact_match(pred_fields, gt_fields, schema) if gt_fields else None

                    row.update({
                        "predicted_fields": pred_fields,
                        "ground_truth_fields": gt_fields,
                        "field_scores": field_scores,
                        "doc_exact_match": doc_match,
                        "status": "OK",
                    })

                except Exception as e:
                    row.update({"status": "ERROR", "error": str(e), "field_scores": None, "doc_exact_match": None})

                report_f.write(json.dumps(row, default=str) + "\n")
                report_f.flush()

                status = row["status"]
                dm_str = f"{row['doc_exact_match']:.0f}" if row.get("doc_exact_match") is not None else "n/a"
                print(f"  [{i + 1}/{len(records)}] {status:8s} {rid}  doc_exact_match={dm_str}")

    # Re-read the full report (prior + this run's rows) to build the summary,
    # so the summary is always accurate even across multiple resumed runs.
    per_file = []
    if report_path.exists():
        with open(report_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        per_file.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

    per_field_totals: Dict[str, List[float]] = {f["name"]: [] for f in schema}
    for row in per_file:
        if row.get("field_scores"):
            for key, score in row["field_scores"].items():
                per_field_totals.setdefault(key, []).append(score)

    per_field_accuracy = {
        k: (sum(v) / len(v) if v else None) for k, v in per_field_totals.items()
    }

    doc_matches = [r["doc_exact_match"] for r in per_file if r.get("doc_exact_match")is not None]
    n_errors = sum(1 for r in per_file if r["status"] == "ERROR")

    summary = {
        "split": args.split,
        "n_total": len(per_file),
        "n_errors": n_errors,
        "doc_exact_match_rate": (sum(doc_matches) / len(doc_matches)) if doc_matches else None,
        "per_field_accuracy": per_field_accuracy,
    }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print(f"SUMMARY — split={args.split}")
    print("=" * 70)
    print(f"  Total: {summary['n_total']}   Errors: {n_errors}")
    if doc_matches:
        print(f"  Doc exact-match rate: {summary['doc_exact_match_rate']:.4f}")
    else:
        print("  No ground-truth fields provided -- doc_exact_match / per-field accuracy not computed.")
    for k, v in per_field_accuracy.items():
        v_str = f"{v:.4f}" if v is not None else "n/a"
        print(f"  Field accuracy [{k}]: {v_str}")


if __name__ == "__main__":
    main()
