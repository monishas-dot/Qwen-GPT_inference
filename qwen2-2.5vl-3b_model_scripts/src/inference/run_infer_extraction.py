"""run_infer_extract.py — Inference script for the full-page field-extraction task
(Pipeline 1). Reuses the task-agnostic pieces of run_infer.py (model/processor
loading, generation call, image-input handling, result writing) and replaces the
bbox-specific parse_bbox/crop_from_bbox with field-JSON parsing against the
config's field schema.
"""

import os
import json
import yaml
import torch

# Disable cuDNN entirely. The Azure conda environment appears to have a broken 
    # cuDNN library or is hitting a known Turing-architecture FP16 Conv3D bug.
    # This forces PyTorch to use native CUDA kernels which will not crash.
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False

import argparse
from pathlib import Path
from typing import Dict, Any, List

# comment out these 2 env vars to allow HF Hub downloads if needed
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from src.inference.run_infer_detection import (
    load_processor, load_model, run_model,
    load_gt, blank_result, iter_b64_inputs, iter_path_inputs, write_results,
)
from src.preprocessing.preprocessing import Preprocessor
from src.utils.coordinate_scaling import resized_wh
from src.utils.field_schema import build_field_extraction_prompt, reconcile_predicted_fields, parse_model_json


def load_config(path: str = "config_extractor_full_page.yaml") -> Dict[str, Any]:
    """Load YAML configuration for inference."""
    with open(path) as f:
        return yaml.safe_load(f)


def parse_fields(raw: str, schema: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Parse model output into a field dict, reconciled against the config's field
    schema (missing keys filled with None so downstream consumers/metrics get a
    fixed key set)."""
    return parse_model_json(raw, schema)


def run_inference(args: argparse.Namespace, config: Dict[str, Any]) -> None:
    """Orchestrate inference over inputs and collect results."""
    dtype = getattr(torch, config["model"]["torch_dtype"])
    base_id = config["model"]["model_id"]
    device_map = config["model"].get("device_map", "auto")
    attn_impl = config["model"].get("attn_implementation", "sdpa")

    valid_exts = set(config["data"].get("supported_image_extensions", {".png", ".jpg", ".jpeg", ".tiff", ".tif"}))
    schema = config["fields"]

    inference_device = config["inference"].get("device", "cuda")
    pre = Preprocessor(config["augmentation"])
    gt_by_id = load_gt(args.gt_jsonl)

    system_prompt = config.get(
        "system_prompt",
        "You are a precise medical document analyzer."
    ).strip()
    prompt = build_field_extraction_prompt(schema, config["task_description"])

    print("\nLoading model...")
    model = load_model(base_id, dtype, device_map, attn_impl, args.adapter_model)
    proc = load_processor(base_id, config)

    min_pix = config["processor"]["min_pixels"]
    max_pix = config["processor"]["max_pixels"]

    if args.images_jsonl and args.image_dir:
        # NEW: Read test records from JSONL, but load images from the local folder
        from PIL import Image
        inputs = []
        with open(args.images_jsonl, "r") as f:
            for line in f:
                if not line.strip(): continue
                rec = json.loads(line)
                rid = rec.get("id") or rec.get("external_ref")
                
                # Determine filename and append extension if missing
                filename = rec.get("image_name") or rec.get("external_ref") or rid
                if not any(filename.lower().endswith(ext) for ext in valid_exts):
                    filename += ".png"
                
                img_path = os.path.join(args.image_dir, filename)
                try:
                    raw_img = Image.open(img_path).convert("RGB")
                    load_error = None
                except Exception as e:
                    raw_img = None
                    load_error = str(e)
                
                inputs.append((rid, raw_img, filename, load_error))
        print(f"\n[TEST SET MODE] Running inference on {len(inputs)} test images loaded from folder...\n")

    elif args.image_dir:
        inputs = list(iter_path_inputs(args.image_dir, gt_by_id, valid_exts))
        print(f"\n[LOCAL TESTING MODE] Running inference on {len(inputs)} images from --image_dir...\n")
    else:
        inputs = list(iter_b64_inputs(args.images_jsonl, gt_by_id))
        print(f"\nRunning inference on {len(inputs)} images from --images_jsonl...\n")

    results = []
    for i, (rid, raw_img, image_ref, load_error) in enumerate(inputs):
        result = blank_result(rid, image_ref, gt_by_id)
        if load_error:
            result["errors"].append(f"image_load: {load_error}")
            print(f"  [{i+1}/{len(inputs)}] FAIL {rid} — could not load image: {load_error}")
            results.append(result)
            continue

        try:
            img_w, img_h = raw_img.size

            # Match the eval-time resize used in ExtractionDataset (fixed to
            # max_pixels, no resolution augmentation) — do_resize=False on the
            # processor means this pre-resize is the only resize that happens.
            resized_w, resized_h = resized_wh(img_w, img_h, max_pix, max_pix)
            prepped_img, _ = pre.run(raw_img, target_size=(resized_w, resized_h), augment=False)

            raw_out = run_model(
                model=model,
                proc=proc,
                img=prepped_img,
                prompt=prompt,
                system_prompt=system_prompt,
                max_new_tokens=config["inference"]["max_new_tokens"],
                device=inference_device
            )

            result["prediction"]["raw_output"] = raw_out
            fields = parse_fields(raw_out, schema)
            result["prediction"]["fields"] = fields

        except Exception as e:
            result["errors"].append(f"parse: {e}")

        status = "OK" if not result["errors"] else "FAIL"
        print(f"  [{i+1}/{len(inputs)}] {status} {rid}")
        results.append(result)

    if getattr(args, 'srf_flat_output', False):
        Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_path, "a") as f:
            for r in results:
                flat = {}
                flat["external_ref"] = r["id"]
                if r.get("ground_truth"):
                    flat["document_title"] = r["ground_truth"].get("document_title", "Unknown")
                    flat["total_pages"] = r["ground_truth"].get("total_pages", 1)
                    flat["srf"] = r["ground_truth"].get("srf", [1])
                else:
                    flat["document_title"] = "Unknown"
                    flat["total_pages"] = 1
                    flat["srf"] = [1]
                
                if "fields" in r.get("prediction", {}):
                    flat.update(r["prediction"]["fields"])
                
                if r.get("errors"):
                    flat["errors"] = r["errors"]
                
                f.write(json.dumps(flat, default=str) + "\n")
        n_ok = sum(1 for r in results if not r["errors"])
        print(f"\nDone. {n_ok}/{len(results)} clean.  Results -> {args.output_path}")
    else:
        write_results(results, args.output_path)


def main() -> None:
    """Parse CLI args and launch inference."""
    p = argparse.ArgumentParser()
    p.add_argument("--adapter_model", required=True)
    p.add_argument("--images_jsonl", default=None)
    p.add_argument("--image_dir", default=None)
    p.add_argument("--output_path", required=True)
    p.add_argument("--gt_jsonl", default=None)
    p.add_argument("--config", default="configs/config_extraction.yaml")
    p.add_argument("--srf_flat_output", action="store_true", help="Format the output JSONL to match the flat SRF GT format.")
    args = p.parse_args()

    if not args.images_jsonl and not args.image_dir:
        p.error("Provide exactly one of --images_jsonl or --image_dir.")

    config = load_config(args.config)
    run_inference(args, config)


if __name__ == "__main__":
    main()