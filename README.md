# Prompt experiments: GPT-5.2 vs. base Qwen2.5-VL-3B

This folder has everything needed to run SRF field extraction with two models, score both against our ground truth, and compare them:

- GPT-5.2 through Azure OpenAI
- The base (not fine-tuned) Qwen2.5-VL-3B-Instruct

It contains **code only**. Images, ground truth and credentials are not included (see [Inputs you need](#inputs-you-need)).

## Contents

```
prompt_ops_reqd/
├── run_openai_full_page.py        # 1. GPT-5.2 inference (Azure OpenAI)
├── qwen2-2.5vl-3b_model_scripts/  # 2. Qwen inference (run from inside this folder)
│   ├── configs/config_extraction.yaml   # Qwen prompt, field schema, match rules
│   └── src/evaluation/predict_extraction.py
├── evaluate_extraction_metrics.py # 3. Score Qwen + GPT against ground truth
├── compare_qwen_gpt.py            # 4. GPT vs. Qwen field-by-field comparison
├── generate_comparison_report.py  #    ...rendered as HTML
├── pdf_to_page_images.py          # 0. PDFs to page PNGs, if you only have PDFs
└── requirements.txt
```

## Inputs you need

1. **Page images** (PNG or JPG) for the documents you're scoring. Use `pdf_to_page_images.py` if you only have PDFs.
2. **The ground-truth JSONL**, i.e. the human-annotated gold set with per-field blur severity. Each row must have an `id` that matches the image file name without its extension.
3. **Azure OpenAI access** for the GPT-5.2 deployment.

Get the images and ground truth through our approved data channels, not by email or chat. Check the AI Usage Guide before putting any of this data into an AI tool.

## Setup

```bash
pip install -r requirements.txt
```

Qwen inference needs a CUDA GPU. The model loads in fp16 and was tuned for a T4. The config's `load_in_4bit`, `lora`, `augmentation` and `training` sections are left over from training and are ignored at inference. Run GPT inference and scoring anywhere.

Set the Azure credentials as environment variables:

```bash
export AZURE_OPENAI_ENDPOINT=...
export AZURE_OPENAI_API_VERSION=...
export AZURE_OPENAI_DEPLOYMENT=...      # the GPT-5.2 deployment name
export AZURE_OPENAI_AD_TOKEN=...
```

The script also reads a `key.env` file next to it if one exists. Don't create one, and never commit credentials.

## Running an experiment

Put each prompt variant in its own output folder (for example `runs/<prompt_name>/`) so results never overwrite each other.

### 0. (Optional) PDFs to images
```bash
python pdf_to_page_images.py <pdf_dir> <image_dir> --dpi 200
```

### 1. GPT-5.2 inference
```bash
python run_openai_full_page.py --image_dir <image_dir> --output_path runs/<name>/gpt.jsonl
```
- This only runs inference, with 5 requests in parallel (`CONCURRENCY_LIMIT`). It doesn't score anything.
- Re-running the same command resumes and skips images already done. Failed images are retried, but their old error line stays in the file, so remove duplicate `id`s before scoring.

### 2. Qwen base-model inference
```bash
cd qwen2-2.5vl-3b_model_scripts
python -m src.evaluation.predict_extraction \
  --config configs/config_extraction.yaml \
  --adapter_model Qwen/Qwen2.5-VL-3B-Instruct \
  --split predict \
  --image_dir <image_dir> \
  --output_dir ../runs/<name>/qwen
```
- Passing the Hugging Face model ID as `--adapter_model` loads the **base** model with no fine-tuned adapter.
- The run can be resumed. It writes `runs/<name>/qwen/per_file_report.jsonl`.

### 3. Score both against ground truth
Edit the four path constants near the top of `evaluate_extraction_metrics.py`:

```python
GT_JSONL   = "<ground_truth>.jsonl"
QWEN_JSONL = "runs/<name>/qwen/per_file_report.jsonl"
GPT_JSONL  = "runs/<name>/gpt.jsonl"
OUTPUT_DIR = "runs/<name>/metrics"
```
Then run `python evaluate_extraction_metrics.py`. It writes:
- `summary.txt`: accuracy per model, broken down by blur severity (none/low/medium)
- `per_field_severity_qwen.csv` and `per_field_severity_gpt.csv`: one row per field and severity
- `exclusions.json`: every file skipped, and why

Scoring rules:
- Files are joined on `id`.
- Fields whose ground-truth blur severity is `high` are left out for both models.
- `diagnoses` and `procedures` are scored per sub-field.
- Qwen rows with a status other than `OK`, and GPT rows with `errors`, are left out and listed in `exclusions.json`.

### 4. Compare GPT and Qwen side by side
```bash
python compare_qwen_gpt.py runs/<name>/qwen/per_file_report.jsonl runs/<name>/gpt.jsonl runs/<name>/compare
python generate_comparison_report.py runs/<name>/compare
```
Start with `mismatches_only.json`, or open the HTML report.

## Where to edit the prompts

The two models get their prompts from **different places**. If you want to test the same prompt on both, edit both.

| Model | Where to edit |
|---|---|
| GPT-5.2 | `run_openai_full_page.py`: `SYSTEM_PROMPT` and `FIELD_LIST_DESCRIPTION` (near the top). Images are resized so the longest side is 1536 px (`encode_image_to_base64`). |
| Qwen | `qwen2-2.5vl-3b_model_scripts/configs/config_extraction.yaml`: `system_prompt`, `task_description`, and the `fields:` list. The prompt is built from these by `src/utils/field_schema.py`. `inference.max_new_tokens` and `processor.max_pixels` also affect results. |

Change field **names** carefully. The field list in the Qwen config also controls how each field is scored (`match_type`). The scoring script and `compare_qwen_gpt.py` expect the same field names, so a renamed field will quietly score as missing.

A copy of the config for each prompt variant (for example `configs/config_extraction_<name>.yaml`, passed with `--config`) makes it easier to reproduce runs.