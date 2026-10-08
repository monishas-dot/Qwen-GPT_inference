"""run_openai_full_page.py — Async batch inference using GPT-5.2 through the AI gateway
(OpenAI-compatible API)
for the SAME full-page field-extraction task as the Qwen pipeline
(config_extractor_full_page.yaml), so outputs are directly comparable.

Saves predictions incrementally to disk, is resumable across crashes/restarts,
and displays a live progress bar. Includes token usage tracking per request
and a batch summary.
"""

import os
import sys
import json
import base64
import asyncio
import argparse
import httpx
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel, Field
from openai import AsyncOpenAI
import openai
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential
from dotenv import load_dotenv
from PIL import Image
import io

try:
    from tqdm.asyncio import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# Load credentials from key.env (current folder, next to this script, or project root).
# override=True so the values in key.env win over any stale variables left in the shell.
_here = Path(__file__).resolve().parent
for _p in (Path.cwd() / "key.env", _here / "key.env", _here.parent / "key.env"):
    if _p.exists():
        load_dotenv(dotenv_path=_p, override=True)
load_dotenv()

def _clean(v):
    return v.strip().strip('"').strip("'").strip() if v else None

# The API key can be stored under any of these names in key.env (first one found is used).
_KEY_NAMES = ("AI_GATEWAY_API_KEY", "AZURE_OPENAI_AD_TOKEN", "OPENAI_API_KEY")
_key_name = next((n for n in _KEY_NAMES if _clean(os.environ.get(n))), None)
AI_GATEWAY_API_KEY = _clean(os.environ.get(_key_name)) if _key_name else None

# Gateway URL (OpenAI-compatible, ends with /v1) and model (the gateway's deployment name).
AI_GATEWAY_BASE_URL = _clean(os.environ.get("AI_GATEWAY_BASE_URL")) or "https://ai-gw.infra.autonomize.dev/v1"
MODEL_NAME = _clean(os.environ.get("AI_GATEWAY_MODEL")) or _clean(os.environ.get("AZURE_OPENAI_DEPLOYMENT")) or "gpt-5.2"

if not AI_GATEWAY_API_KEY:
    print("[ERROR] No API key found. Put one of these lines in key.env (key after the =, nothing else on the line):\n"
          "        AI_GATEWAY_API_KEY=your_key", file=sys.stderr)
    sys.exit(1)
print(f"[INIT] API key loaded from {_key_name} ({len(AI_GATEWAY_API_KEY)} characters); base_url={AI_GATEWAY_BASE_URL}; model={MODEL_NAME}")

# ---------------------------------------------------------
# EXECUTION LIMITS
# ---------------------------------------------------------
IMAGE_OPTS = {"max_side": 1536, "detail": "auto", "quality": 90, "prompt": None}   # overridden by --max_side / --detail
CONCURRENCY_LIMIT = 5   # Set to 1 to prevent endpoint saturation on heavy images
RETRYABLE_ERRORS = (
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    httpx.ReadTimeout,
    asyncio.TimeoutError  # Catches the forced 120s execution kill
)

# ---------------------------------------------------------
# Pydantic schema — mirrors config_extractor_full_page.yaml's `fields:` list
# exactly (same names, same nesting) so GPT-5.2 and Qwen outputs line up
# field-for-field for comparison.
# ---------------------------------------------------------
class DiagnosisItem(BaseModel):
    diagnosisCode: Optional[str] = Field(default=None, description="ICD diagnosis code.")

class ProcedureItem(BaseModel):
    procedureCode: Optional[str] = Field(default=None, description="CPT/HCPCS procedure code.")
    dateOfServiceStart: Optional[str] = Field(default=None, description="Date of service start, MM/DD/YYYY; null if unreadable.")
    dateOfServiceEnd: Optional[str] = Field(default=None, description="Date of service end, MM/DD/YYYY; null if unreadable.")
    numberOfVisits: Optional[int] = Field(default=None, description="Number of visits/units requested for this procedure.")

class FullPageExtractionSchema(BaseModel):
    diagnoses: List[DiagnosisItem] = Field(default_factory=list)
    procedures: List[ProcedureItem] = Field(default_factory=list)
    servicing_provider_npi: Optional[str] = Field(default=None)
    type_of_service: Optional[str] = Field(default=None)
    member_id: Optional[str] = Field(default=None)
    requesting_provider_phone: Optional[str] = Field(default=None)
    acuity: Optional[str] = Field(default=None)
    date_of_birth: Optional[str] = Field(default=None)
    servicing_provider_name: Optional[str] = Field(default=None)
    requesting_provider_name: Optional[str] = Field(default=None)
    member_name: Optional[str] = Field(default=None)
    place_of_service: Optional[str] = Field(default=None)
    requesting_provider_npi: Optional[str] = Field(default=None)
    requesting_provider_fax: Optional[str] = Field(default=None)
    lob: Optional[str] = Field(default=None)
    servicing_provider_fax: Optional[str] = Field(default=None)
    servicing_provider_phone: Optional[str] = Field(default=None)
    servicing_provider_address: Optional[str] = Field(default=None)
    requesting_provider_address: Optional[str] = Field(default=None)

# System prompt
SYSTEM_PROMPT = (
    "You are a precise medical document analyzer. "
    "You extract structured field values from fax images."
)

# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------
def encode_image_to_base64(image_path: str, max_side: int = 1536) -> str:
    """Opens the PNG, downscales safely, and compresses to JPEG in memory."""
    with Image.open(image_path) as img:
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=IMAGE_OPTS["quality"])
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

def clean_json_string(raw_str: str) -> str:
    raw_str = raw_str.strip()
    if raw_str.startswith("```"):
        raw_str = raw_str.split("\n", 1)[1]
    if raw_str.endswith("```"):
        raw_str = raw_str.rsplit("\n", 1)[0]
    return raw_str.strip()

def extract_usage(completion) -> dict:
    """Extracts prompt, completion, and total token counts safely."""
    if hasattr(completion, "usage") and completion.usage:
        return {
            "prompt_tokens": getattr(completion.usage, "prompt_tokens", 0),
            "completion_tokens": getattr(completion.usage, "completion_tokens", 0),
            "total_tokens": getattr(completion.usage, "total_tokens", 0)
        }
    return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

def load_already_done(output_path: str) -> set:
    """Ids that already have a clean (error-free) row in the output file."""
    done = set()
    if not os.path.exists(output_path):
        return done
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("id") and not row.get("errors"):
                done.add(row["id"])
    return done

file_lock = asyncio.Lock()

# ---------------------------------------------------------
# Core Processing Pipeline
# ---------------------------------------------------------
async def process_single_image(client: AsyncOpenAI, image_path: str, sem: asyncio.Semaphore, out_f) -> dict:
    filename = os.path.basename(image_path)
    stem_id = Path(image_path).stem

    row_data = {
        "id": stem_id,
        "image_path": filename,
        "prediction": {},
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "errors": []
    }

    async with sem:
        try:
            image_b64 = encode_image_to_base64(image_path, IMAGE_OPTS["max_side"])
        except Exception as e:
            row_data["errors"].append(f"Encoding failed: {e}")
            async with file_lock:
                out_f.write(json.dumps(row_data) + "\n")
                out_f.flush()
            return row_data

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}", "detail": IMAGE_OPTS["detail"]}},
                    {"type": "text", "text": IMAGE_OPTS["prompt"]}
                ]
            }
        ]

        async def call_api(attempt_num):
            print(f" [DEBUG] Sending request for {filename} (Attempt {attempt_num})...")
            try:
                completion = await asyncio.wait_for(
                    client.beta.chat.completions.parse(
                        model=MODEL_NAME,
                        messages=messages,
                        response_format=FullPageExtractionSchema
                    ),
                    timeout=120.0
                )
                parsed_output = completion.choices[0].message.parsed
                if parsed_output is None:
                    raise ValueError("Structured output parse returned None.")

                usage_info = extract_usage(completion)
                print(f" [DEBUG] Success for {filename}! Tokens used: {usage_info['total_tokens']} "
                      f"(Prompt: {usage_info['prompt_tokens']}, Completion: {usage_info['completion_tokens']})")
                return parsed_output.model_dump(), completion.choices[0].message.content, usage_info

            except Exception as sdk_err:
                if isinstance(sdk_err, asyncio.TimeoutError):
                    print(f" [DEBUG] asyncio HARD TIMEOUT on {filename}.")
                    raise sdk_err

                if isinstance(sdk_err, (openai.PermissionDeniedError, openai.AuthenticationError)) or "403" in str(sdk_err) or "401" in str(sdk_err):
                    raise sdk_err
                if isinstance(sdk_err, RETRYABLE_ERRORS) or "429" in str(sdk_err):
                    print(f" [DEBUG] Network/Rate limit error on {filename}: {type(sdk_err).__name__}")
                    raise sdk_err

                print(f" [DEBUG] Fallback triggered for {filename} due to parse error.")
                json_messages = messages.copy()
                json_messages.append({"role": "user", "content": "Provide your output strictly in JSON format conforming to the expected schema."})

                completion = await asyncio.wait_for(
                    client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=json_messages,
                        response_format={"type": "json_object"}
                    ),
                    timeout=120.0
                )
                raw_txt = completion.choices[0].message.content
                usage_info = extract_usage(completion)
                print(f" [DEBUG] Fallback Success for {filename}! Tokens used: {usage_info['total_tokens']} "
                      f"(Prompt: {usage_info['prompt_tokens']}, Completion: {usage_info['completion_tokens']})")
                return json.loads(clean_json_string(raw_txt)), raw_txt, usage_info

        try:
            async for attempt in AsyncRetrying(
                retry=retry_if_exception_type(RETRYABLE_ERRORS),
                stop=stop_after_attempt(5),
                wait=wait_exponential(multiplier=2, min=5, max=30),
                reraise=True,
            ):
                with attempt:
                    result_dict, raw_text_response, usage_info = await call_api(attempt.retry_state.attempt_number)
        except Exception as e:
            print(f" [DEBUG] Final Failure for {filename}: {str(e)}")
            row_data["errors"].append(str(e))
            async with file_lock:
                out_f.write(json.dumps(row_data) + "\n")
                out_f.flush()
            return row_data

        row_data["prediction"] = {
            "raw_output": raw_text_response,
            "fields": result_dict,
        }
        row_data["usage"] = usage_info

        async with file_lock:
            out_f.write(json.dumps(row_data) + "\n")
            out_f.flush()
        return row_data

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", required=False, help="Directory containing images to process")
    parser.add_argument("--image_path", required=False, help="Path to a single image file to process")
    parser.add_argument("--output_path", required=True, help="Where to save the JSONL output")
    parser.add_argument("--max_side", type=int, default=1536, help="Longest image side in pixels sent to GPT")
    parser.add_argument("--detail", choices=["low", "high", "auto"], default="auto", help="Vision detail level")
    parser.add_argument("--prompt_file", default="full_page.txt", help="Filename inside the gpt_prompt folder or full path to the prompt file")
    parser.add_argument("--jpeg_quality", type=int, default=90, help="JPEG quality of the image sent to GPT")
    args = parser.parse_args()
    
    IMAGE_OPTS["max_side"], IMAGE_OPTS["detail"] = args.max_side, args.detail
    IMAGE_OPTS["quality"] = args.jpeg_quality

    # Resolve prompt path from the 'gpt_prompt' directory
    prompt_arg = Path(args.prompt_file)
    if prompt_arg.is_absolute() and prompt_arg.exists():
        prompt_path = prompt_arg
    else:
        prompt_path = _here / "gpt_prompt" / args.prompt_file
        if not prompt_path.exists():
            # Fallback if user passed a relative path outside gpt_prompt
            prompt_path = Path(args.prompt_file)

    if prompt_path.exists():
        IMAGE_OPTS["prompt"] = prompt_path.read_text(encoding="utf-8").strip()
        print(f"[INIT] Prompt loaded from {prompt_path} ({len(IMAGE_OPTS['prompt'])} chars)")
    else:
        print(f"[ERROR] Prompt file not found at {prompt_path}. Please place your prompts in the 'gpt_prompt' folder.", file=sys.stderr)
        sys.exit(1)

    print(f"[INIT] Image settings: max_side={args.max_side}, detail={args.detail}")

    if not args.image_dir and not args.image_path:
        print("[ERROR] You must provide either --image_dir or --image_path")
        sys.exit(1)

    timeout_config = httpx.Timeout(120.0)
    http_client = httpx.AsyncClient(timeout=timeout_config)

    client = AsyncOpenAI(
        base_url=AI_GATEWAY_BASE_URL,
        api_key=AI_GATEWAY_API_KEY,
        http_client=http_client,
        max_retries=0
    )

    image_paths = []
    if args.image_path:
        image_paths = [Path(args.image_path)]
        print(f"[INIT] Processing single image: {args.image_path}")
    elif args.image_dir:
        image_paths = sorted(
            p for p in Path(args.image_dir).rglob("*")
            if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg")
        )
        print(f"[INIT] Found {len(image_paths)} images to process using deployment model instance: {MODEL_NAME}")

    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)

    already_done = load_already_done(args.output_path)
    if already_done:
        before = len(image_paths)
        image_paths = [p for p in image_paths if p.stem not in already_done]
        print(f"[RESUME] {len(already_done)} image(s) already completed in {args.output_path}; "
              f"{before - len(image_paths)} skipped, {len(image_paths)} remaining.")

    if not image_paths:
        print("[DONE] Nothing left to process.")
        return

    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)

    print("[START] Beginning batch execution loop...")

    results = []
    with open(args.output_path, "a", encoding="utf-8") as out_f:
        tasks = [process_single_image(client, str(path), semaphore, out_f) for path in image_paths]

        if HAS_TQDM:
            results = await tqdm.gather(*tasks, desc="Processing images via AI gateway")
        else:
            for i, task in enumerate(asyncio.as_completed(tasks)):
                res = await task
                results.append(res)
                if i % 10 == 0:
                    print(f"[PROGRESS] Completed {i}/{len(image_paths)} image requests...")

    total_prompt_tokens = sum(r.get("usage", {}).get("prompt_tokens", 0) for r in results if r)
    total_completion_tokens = sum(r.get("usage", {}).get("completion_tokens", 0) for r in results if r)
    grand_total_tokens = sum(r.get("usage", {}).get("total_tokens", 0) for r in results if r)
    n_errors = sum(1 for r in results if r and r.get("errors"))

    print("\n" + "=" * 60)
    print(" BATCH EXECUTION SUMMARY & TOKEN USAGE")
    print("=" * 60)
    print(f" Files Processed (this run): {len(image_paths)}")
    print(f" Errors (this run):          {n_errors}")
    print(f" Total Prompt Tokens:        {total_prompt_tokens:,}")
    print(f" Total Completion Tokens:    {total_completion_tokens:,}")
    print(f" Grand Total Tokens:         {grand_total_tokens:,}")
    print("=" * 60)
    print(f"[SUCCESS] Output saved to: {args.output_path}")

if __name__ == "__main__":
    asyncio.run(main())
