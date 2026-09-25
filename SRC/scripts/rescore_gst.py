"""Category C1 re-scoring: GST Invoices, via LLMGateway (vision).

Reproduces notebooks/hindsight_gst_invoice_cba.ipynb: downloads the
Roboflow Tax Invoice COCO dataset, builds ground truth via a guided
Sonnet-4.6 extraction pass (same as the original — GT construction is a
different task from unguided test extraction), then evaluates Haiku and
GPT-4o-mini unguided, and scores with the corrected cba_scorer.py.

Requires the LLMGateway key to have access to claude-sonnet-4-6 (for GT
construction) in addition to claude-haiku-4-5 and gpt-4o-mini.

Usage:
    LLMGATEWAY_API_KEY=... python3 scripts/rescore_gst.py --n-gt-candidates 10 --out /tmp/gst_pilot.json
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import random
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cba_scorer import score_corpus  # noqa: E402

GATEWAY_BASE_URL = "https://api.llmgateway.io/v1"
DATASET_DIR_NAME = "Tax-invoice-1"
ROBOFLOW_API_KEY = "ZJuS6aqkhbJLXlPDjlQK"

MAX_IMAGE_BYTES = 4_500_000
MAX_RETRIES = 5

COCO_TO_CONCEPT = {
    "CGST": "cgst", "SGST": "sgst", "IGST": "igst",
    "Initial Total Amount": "initial_total_amount",
    "Total Amount": "total_amount",
    "Round off": "round_off",
    "Invoice Number": "invoice_number",
    "Invoice Date": "invoice_date",
    "Ack No.": "ack_no",
    "Ack Date": "ack_date",
    "GST Number": "gst_number",
    "CIN No.": "cin_no",
    "Amount in Words": "amount_in_words",
}
EVAL_CONCEPTS = sorted(set(COCO_TO_CONCEPT.values()))

MODELS = {
    "haiku": "claude-haiku-4-5",
    "gpt4o-mini": "gpt-4o-mini",
}
GT_MODEL = "claude-sonnet-4-6"

GT_SYSTEM = """You are a document extraction system specialized in Indian GST tax invoices.
You will be told which specific fields are present in the invoice. Extract the exact text value for each.
Return a JSON object mapping each field name to its extracted value.
Only include fields from the provided list. Do not fabricate values — only extract text visible on the invoice."""

TEST_SYSTEM = """You are a document extraction system. Look at this Indian GST tax invoice and extract fields, mapping each to the most appropriate canonical concept.

Canonical concepts:
cgst, sgst, igst, initial_total_amount, total_amount, round_off, invoice_number, invoice_date, ack_no, ack_date, gst_number, cin_no, amount_in_words

Return a JSON array of objects. Each object must have:
- "label": the field label as shown on the invoice
- "value": the corresponding value as shown on the invoice
- "concept": the canonical concept name from the list above

Rules:
- Only include fields that clearly match a canonical concept
- Each concept may appear at most once
- Do not fabricate values — only use text visible on the invoice
- CGST = Central GST, SGST = State GST, IGST = Integrated GST
- initial_total_amount = taxable amount before tax, total_amount = final total after tax
- gst_number = GSTIN (15-character alphanumeric)"""


def ensure_dataset() -> str:
    if not os.path.isdir(DATASET_DIR_NAME):
        from roboflow import Roboflow

        rf = Roboflow(api_key=ROBOFLOW_API_KEY)
        project = rf.workspace("writer-information-6sirk").project("tax-invoice-ud2jo")
        dataset = project.version(1).download("coco")
        return dataset.location
    return DATASET_DIR_NAME


def load_coco_split(dataset_dir: str, split_dir: str) -> list:
    ann_path = os.path.join(dataset_dir, split_dir, "_annotations.coco.json")
    with open(ann_path) as f:
        data = json.load(f)
    cat_map = {c["id"]: c["name"] for c in data["categories"]}
    img_map = {img["id"]: img for img in data["images"]}
    img_anns = defaultdict(list)
    for ann in data["annotations"]:
        cat_name = cat_map[ann["category_id"]]
        img_anns[ann["image_id"]].append({"category": cat_name})

    docs = []
    for img_id, img_info in img_map.items():
        docs.append({
            "image_path": os.path.join(dataset_dir, split_dir, img_info["file_name"]),
            "file_name": img_info["file_name"],
            "annotations": img_anns.get(img_id, []),
        })
    return docs


def image_to_base64(image_path: str):
    from PIL import Image as PILImage

    file_size = os.path.getsize(image_path)
    if file_size <= MAX_IMAGE_BYTES:
        with open(image_path, "rb") as f:
            data = f.read()
        media_type = "image/jpeg" if image_path.lower().endswith((".jpg", ".jpeg")) else "image/png"
        return base64.b64encode(data).decode("utf-8"), media_type
    img = PILImage.open(image_path)
    for scale in [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]:
        resized = img.resize((int(img.width * scale), int(img.height * scale)), PILImage.LANCZOS)
        if resized.mode == "RGBA":
            resized = resized.convert("RGB")
        buf = io.BytesIO()
        resized.save(buf, format="JPEG", quality=85)
        if buf.tell() <= MAX_IMAGE_BYTES:
            return base64.b64encode(buf.getvalue()).decode("utf-8"), "image/jpeg"
    raise ValueError(f"Cannot shrink {image_path}")


def parse_json(raw: str):
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
    return json.loads(raw)


def call_with_retry(fn, max_retries=MAX_RETRIES, base_delay=2):
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            is_rate_limit = "429" in str(e) or "rate_limit" in str(e).lower()
            if attempt < max_retries - 1:
                wait = min(10 * (2 ** attempt), 120) if is_rate_limit else base_delay * (2 ** attempt)
                print(f"    retry {attempt + 1}/{max_retries - 1} in {wait:.0f}s: {type(e).__name__}: {e}")
                time.sleep(wait)
            else:
                raise


def vision_extract(client, model_id: str, image_path: str, system_prompt: str, instruction: str, expect_array: bool):
    b64, media_type = image_to_base64(image_path)

    def _call():
        response = client.chat.completions.create(
            model=model_id,
            max_tokens=1024,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{b64}"}},
                    {"type": "text", "text": instruction},
                ]},
            ],
        )
        content = response.choices[0].message.content
        usage = response.usage
        cost = getattr(usage, "cost", None) if usage else None
        return parse_json(content), cost

    return call_with_retry(_call)


def array_to_predictions(arr) -> dict:
    predictions = {}
    for item in arr:
        if not isinstance(item, dict):
            continue
        concept = item.get("concept", "")
        value = item.get("value", "")
        if concept and concept not in predictions:
            predictions[concept] = value
    return predictions


def normalize_gst_value(v) -> str:
    import re

    v = str(v).lower().strip()
    for sym in ["₹", "rs.", "rs", "inr", "/-"]:
        v = v.replace(sym, "")
    v = v.replace(",", "").strip()
    v = re.sub(r"\s+", " ", v)
    if re.fullmatch(r"-?\d+\.\d+", v):
        v = v.rstrip("0").rstrip(".")
    if v in ["-", "nil", "n/a", "na", "—", "–"]:
        v = "0"
    return v


def get_balance(api_key) -> float | None:
    import httpx
    try:
        r = httpx.get(f"{GATEWAY_BASE_URL}/credits", headers={"Authorization": f"Bearer {api_key}"}, timeout=10)
        return float(r.json()["balance"])
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-gt-candidates", type=int, default=10, help="How many qualified images to attempt GT construction on")
    parser.add_argument("--n-final-docs", type=int, default=None, help="Stop GT construction once this many valid-GT docs are built (default: same as --n-gt-candidates)")
    parser.add_argument("--out", default="/tmp/gst_pilot.json")
    parser.add_argument("--models", nargs="+", default=list(MODELS.keys()))
    args = parser.parse_args()
    n_final = args.n_final_docs or args.n_gt_candidates

    api_key = os.environ.get("LLMGATEWAY_API_KEY")
    if not api_key:
        raise SystemExit("LLMGATEWAY_API_KEY not set in environment")

    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url=GATEWAY_BASE_URL)

    balance_before = get_balance(api_key)
    print(f"Gateway balance before run: ${balance_before}")

    dataset_dir = ensure_dataset()
    all_docs = []
    for split in ["train", "valid", "test"]:
        all_docs.extend(load_coco_split(dataset_dir, split))
    print(f"Loaded {len(all_docs)} images")

    for doc in all_docs:
        present = {COCO_TO_CONCEPT.get(a["category"]) for a in doc["annotations"]}
        present.discard(None)
        doc["eval_concepts"] = sorted(present)

    qualified = [d for d in all_docs if len(d["eval_concepts"]) >= 3]
    print(f"Images with >= 3 eval concepts: {len(qualified)} / {len(all_docs)}")

    random.seed(42)
    candidates = qualified[:] if len(qualified) <= 150 else random.sample(qualified, 150)
    random.shuffle(candidates)
    candidates = candidates[:args.n_gt_candidates]
    print(f"Attempting GT construction on {len(candidates)} candidates (target: {n_final} valid docs)")

    gt_docs = []
    gt_errors = 0
    for i, doc in enumerate(candidates):
        try:
            concept_list = ", ".join(doc["eval_concepts"])
            user_msg = f"Extract the value for each of these fields present in this invoice: {concept_list}"
            gt_values, cost = vision_extract(client, GT_MODEL, doc["image_path"], GT_SYSTEM, user_msg, expect_array=False)
            gt = {}
            for concept in doc["eval_concepts"]:
                val = gt_values.get(concept, "")
                if isinstance(val, (int, float)):
                    val = str(val)
                if isinstance(val, str) and val.strip():
                    gt[concept] = val.strip()
            if len(gt) >= 3:
                doc["gt"] = gt
                gt_docs.append(doc)
                print(f"  [{i + 1}/{len(candidates)}] GT built ({len(gt_docs)} total), cost=${cost}")
            if len(gt_docs) >= n_final:
                break
            time.sleep(1.5)
        except Exception as e:
            gt_errors += 1
            print(f"  GT error on {doc['file_name']}: {e}")

    samples = gt_docs
    print(f"\nGT construction complete: {len(samples)} documents with valid GT ({gt_errors} errors)")

    results = {}
    for model_name in args.models:
        model_id = MODELS[model_name]
        print(f"\n=== {model_name} ({model_id}) ===")
        doc_results = []
        for i, s in enumerate(samples):
            arr, cost = vision_extract(client, model_id, s["image_path"], TEST_SYSTEM,
                                        "Extract all matching fields from this GST invoice.", expect_array=True)
            pred = array_to_predictions(arr)
            doc_results.append({"doc_id": s["file_name"], "ground_truth": s["gt"], "predicted": pred, "cost": cost})
            print(f"  [{i + 1}/{len(samples)}] {s['file_name']}: cost=${cost}")
            time.sleep(0.5)

        documents = []
        for d in doc_results:
            gt_items = [(c, normalize_gst_value(v)) for c, v in d["ground_truth"].items()]
            pred_items = [(c, normalize_gst_value(v)) for c, v in d["predicted"].items() if v not in (None, "")]
            documents.append((gt_items, pred_items))

        corpus = score_corpus(documents)
        results[model_name] = {"model_id": model_id, "scorer_result": corpus.to_dict(), "documents": doc_results}
        print(f"  DONE: Rec_val={corpus.rec_val:.4f} CBA_cond={corpus.cba_cond:.4f} "
              f"ACC_joint={corpus.acc_joint:.4f} Delta={corpus.delta:.4f} misbindings={corpus.n_misbindings}")

    balance_after = get_balance(api_key)
    print(f"\nGateway balance after run: ${balance_after}")
    if balance_before is not None and balance_after is not None:
        print(f"Actual spend this run: ${balance_before - balance_after:.6f}")

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results written to {args.out}")


if __name__ == "__main__":
    main()
