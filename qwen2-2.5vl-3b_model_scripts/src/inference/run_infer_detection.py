"""run_infer.py — Inference script for the Detector."""

import os
import re
import io
import json
import base64
import binascii
import yaml
import torch
import argparse
from pathlib import Path
from PIL import Image
from typing import Dict, Any, List, Generator, Optional, Set

# Use setdefault so that a caller who has already set these env vars (e.g. to "0"
# for HF Hub access) is not silently overridden.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from src.preprocessing.preprocessing import Preprocessor
from src.utils.coordinate_scaling import resized_wh, scale_predicted_coords_to_raw_image
from src.utils.qwen_processor import build_processor, debug_check_processor


def load_config(path: str = "configs/config.yaml") -> Dict[str, Any]:
    """Load YAML configuration for inference."""
    with open(path) as f:
        return yaml.safe_load(f)


def load_processor(base_id: str, config: Dict[str, Any]) -> AutoProcessor:
    """Build and validate a processor with explicit resize bounds."""
    min_pixels = config["processor"]["min_pixels"]
    max_pixels = config["processor"]["max_pixels"]
    proc = build_processor(base_id, min_pixels, max_pixels)
    debug_check_processor(proc, base_id, min_pixels, max_pixels)
    return proc


def load_model(
    base_id: str,
    dtype: torch.dtype,
    device_map: str,
    attn_implementation: str,
    adapter_path: str
) -> torch.nn.Module:
    """Load the model for inference. If adapter_path holds a PEFT adapter,
    load the base model and attach the adapter. Otherwise adapter_path is
    treated as a full fine-tuned checkpoint (training used lora.enabled:
    false) and loaded directly."""
    is_adapter = (Path(adapter_path) / "adapter_config.json").exists()

    # Force safe device handling for Mac/CPU without accelerate device_map issues
    if device_map in ["auto", "cuda"] or not device_map:
        device_map = "mps" if torch.backends.mps.is_available() else "cpu"

    effective_device_map = None if device_map in ["cpu", "mps"] else device_map

    if is_adapter:
        print(f"  Loading base model {base_id} (once)...")
        base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            base_id,
            device_map=effective_device_map,
            torch_dtype=dtype,
            attn_implementation=attn_implementation
        )
        print(f"  Attaching adapter: {adapter_path}")
        model = PeftModel.from_pretrained(base, adapter_path)
    else:
        print(f"  Loading full fine-tuned model from {adapter_path} (no adapter_config.json found)...")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            adapter_path,
            device_map=effective_device_map,
            torch_dtype=dtype,
            attn_implementation=attn_implementation
        )

    if effective_device_map is None:
        target_device = device_map
        print(f"  Moving model to device: {target_device}")
        model.to(target_device)

    model.eval()
    return model


def run_model(
    model: PeftModel,
    proc: AutoProcessor,
    img: Image.Image,
    prompt: str,
    system_prompt: str,
    max_new_tokens: int,
    device: str
) -> str:
    """Run inference on an image with prompt and decode generated text."""
    # Ensure device is valid for macOS execution
    if device == "cuda" or not device:
        device = "mps" if torch.backends.mps.is_available() else "cpu"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": prompt}]},
    ]
    text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = proc(text=[text], images=[img], return_tensors="pt").to(device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=proc.tokenizer.eos_token_id,
            do_sample=False,
        )
    return proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]


def parse_bbox(raw: str, img_w: int, img_h: int, min_pixels: int, max_pixels: int) -> Dict[str, List[int]]:
    """Parse output into a bbox and map it back to raw image pixels."""
    clean = re.sub(r"```(?:json)?|```", "", raw).strip()
    try:
        data = json.loads(clean)
    except json.JSONDecodeError:
        nums = re.findall(r'\d+', clean)
        if len(nums) >= 4:
            data = {"bbox": [int(n) for n in nums[:4]]}
        else:
            raise ValueError(f"Cannot parse output: {raw[:200]}")

    if isinstance(data, list) and len(data) == 4:
        bbox_model = [int(v) for v in data]
    elif isinstance(data, dict) and "bbox" in data:
        bbox_model = [int(v) for v in data["bbox"]]
    else:
        raise ValueError(f"Unexpected schema: {data}")

    resized_w, resized_h = resized_wh(img_w, img_h, min_pixels, max_pixels)
    x1m, y1m, x2m, y2m = [max(0, v) for v in bbox_model]
    x1m, x2m = min(x1m, resized_w), min(x2m, resized_w)
    y1m, y2m = min(y1m, resized_h), min(y2m, resized_h)

    if x2m <= x1m or y2m <= y1m:
        raise ValueError(f"Degenerate bbox: {bbox_model}")

    raw_pixels = scale_predicted_coords_to_raw_image(
        [x1m, y1m, x2m, y2m], img_w, img_h, min_pixels, max_pixels
    )
    return {
        "model_space": [x1m, y1m, x2m, y2m],
        "pixels": raw_pixels,
    }


def crop_from_bbox(img: Image.Image, pixels: List[int], padding: int) -> Image.Image:
    """Crop the raw image using a predicted bbox and padding."""
    x1, y1, x2, y2 = pixels
    w, h = img.size
    return img.crop((max(0, x1 - padding), max(0, y1 - padding), min(w, x2 + padding), min(h, y2 + padding)))


def load_gt(gt_jsonl: Optional[str]) -> Dict[str, Any]:
    """Load optional ground truth rows keyed by record id from JSONL."""
    gt_by_id = {}
    if gt_jsonl:
        with open(gt_jsonl) as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    gt_by_id[r["id"]] = r
    return gt_by_id


def blank_result(rid: str, image_ref: str, gt_by_id: Dict[str, Any]) -> Dict[str, Any]:
    """Create an empty inference result skeleton for a record."""
    return {"id": rid, "image_ref": image_ref, "prediction": {}, "ground_truth": gt_by_id.get(rid), "errors": []}


def decode_b64_image(image_b64: str) -> Image.Image:
    """Decode a base64 image payload into a PIL Image."""
    if image_b64.startswith("data:"):
        _, _, image_b64 = image_b64.partition(",")
    try:
        raw = base64.b64decode(image_b64, validate=True)
    except (binascii.Error, ValueError) as e:
        raise ValueError(f"Invalid base64 image data: {e}")
    return Image.open(io.BytesIO(raw)).convert("RGB")


def iter_b64_inputs(images_jsonl: str, gt_by_id: Dict[str, Any]) -> Generator:
    """Yield records from a JSONL of base64 images, skipping invalid entries."""
    with open(images_jsonl) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rid = row["id"]
            if gt_by_id and rid not in gt_by_id:
                continue
            try:
                img = decode_b64_image(row["image_b64"])
            except Exception as e:
                yield rid, None, rid, str(e)
                continue
            yield rid, img, rid, None


def iter_path_inputs(image_dir: str, gt_by_id: Dict[str, Any], valid_extensions: Set[str]) -> Generator:
    """Yield local image files and their ids from a directory."""
    image_dir_path = Path(image_dir)
    image_files = sorted(f for f in image_dir_path.iterdir() if f.suffix.lower() in valid_extensions)
    for f in image_files:
        rid = f.stem
        if gt_by_id and rid not in gt_by_id:
            continue
        try:
            img = Image.open(f).convert("RGB")
        except Exception as e:
            yield rid, None, str(f), str(e)
            continue
        yield rid, img, str(f), None


def run_inference(args: argparse.Namespace, config: Dict[str, Any]) -> None:
    """Orchestrate inference over inputs and collect results."""
    dtype = getattr(torch, config["model"]["torch_dtype"])
    base_id = config["model"]["model_id"]
    device_map = config["model"].get("device_map", "auto")
    attn_impl = config["model"].get("attn_implementation", "sdpa")

    padding = config["data"]["crop_padding"]
    valid_exts = set(config["data"].get("supported_image_extensions", {".png", ".jpg", ".jpeg", ".tiff", ".tif"}))

    crops_dir = Path(config["paths"]["infer_output_dir"]) / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    inference_device = config.get("inference", {}).get("device", "mps")
    if inference_device == "cuda":
        inference_device = "mps" if torch.backends.mps.is_available() else "cpu"

    pre = Preprocessor(config["augmentation"])
    gt_by_id = load_gt(args.gt_jsonl)

    system_prompt = config.get(
        "system_prompt",
        "You are a precise medical document analyzer."
    ).strip()

    print("\nLoading model...")
    model = load_model(base_id, dtype, device_map, attn_impl, args.adapter_model)
    proc = load_processor(base_id, config)

    min_pix = config["processor"]["min_pixels"]
    max_pix = config["processor"]["max_pixels"]

    if args.image_dir:
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

            # Match the preprocessing resize step done in the dataset
            resized_w, resized_h = resized_wh(img_w, img_h, max_pix, max_pix)
            prepped_img, _ = pre.run(raw_img, target_size=(resized_w, resized_h), augment=False)

            raw_out = run_model(
                model=model,
                proc=proc,
                img=prepped_img,
                prompt=config["prompt"].strip(),
                system_prompt=system_prompt,
                max_new_tokens=config["inference"]["max_new_tokens"],
                device=inference_device
            )

            result["prediction"]["raw_output"] = raw_out
            bbox = parse_bbox(raw_out, img_w, img_h, min_pix, max_pix)

            result["prediction"].update({
                "predicted_bbox_model_space": bbox["model_space"],
                "predicted_bbox_pixels": bbox["pixels"]
            })

            crop = crop_from_bbox(raw_img, bbox["pixels"], padding)
            crop_path = crops_dir / f"{rid}_crop.png"
            crop.save(crop_path)

            buf = io.BytesIO()
            crop.save(buf, format="PNG")
            crop_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

            result["prediction"]["crop_b64"] = crop_b64
        except Exception as e:
            result["errors"].append(f"parse: {e}")

        status = "OK" if not result["errors"] else "FAIL"
        print(f"  [{i+1}/{len(inputs)}] {status} {rid}")
        results.append(result)

    write_results(results, args.output_path)
    print(f"\nDone. Crops -> {crops_dir}/")


def write_results(results: List[Dict[str, Any]], output_path: str) -> None:
    """Append result JSONL records to the output file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "a") as f:
        for r in results:
            f.write(json.dumps(r, default=str) + "\n")
    n_ok = sum(1 for r in results if not r["errors"])
    print(f"\nDone. {n_ok}/{len(results)} clean.  Results -> {output_path}")


def main() -> None:
    """Parse CLI args and launch inference."""
    p = argparse.ArgumentParser()
    p.add_argument("--adapter_model", required=True)
    p.add_argument("--images_jsonl", default=None)
    p.add_argument("--image_dir", default=None)
    p.add_argument("--output_path", required=True)
    p.add_argument("--gt_jsonl", default=None)
    p.add_argument("--config", default="configs/config.yaml")
    args = p.parse_args()

    if bool(args.images_jsonl) == bool(args.image_dir):
        p.error("Provide exactly one of --images_jsonl or --image_dir.")

    config = load_config(args.config)
    run_inference(args, config)


if __name__ == "__main__":
    main()