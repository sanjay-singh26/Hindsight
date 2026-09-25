"""Category C1 re-scoring: CORD, Track B (vision) only, via LLMGateway.

Reproduces notebooks/hindsight_cord_cba.ipynb's Track B (LLM Vision)
sampling (same random.seed(42), same 100-receipt sample, same 13-concept
schema) and scores with the corrected cba_scorer.py.

NOTE: Track A (Tesseract OCR + LLM) is NOT reproduced here — this
environment does not have a working Tesseract install (blocked on a
`sudo chown` for Homebrew that was not authorized), so only the
vision-only track is re-scored. The original paper's CORD numbers
combine or otherwise draw on both tracks; this re-score is a partial
one and should be labeled as vision-track-only when reported.

Usage:
    LLMGATEWAY_API_KEY=... python3 scripts/rescore_cord.py --n-samples 3 --out /tmp/cord_pilot.json
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
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cba_scorer import score_corpus  # noqa: E402

GATEWAY_BASE_URL = "https://api.llmgateway.io/v1"
IMG_DIR = "/tmp/cord_images"

EVAL_CONCEPTS = [
    "subtotal_price", "tax_price", "total_price",
    "cashprice", "changeprice", "creditcardprice", "emoneyprice", "otherprice",
    "menuqty_cnt", "itemsubtotal_cnt",
    "total_etc", "subtotal_etc", "tax_etc",
]

MODELS = {
    "haiku": "claude-haiku-4-5",
    "gpt4o-mini": "gpt-4o-mini",
}

CONCEPT_LIST = ", ".join(EVAL_CONCEPTS)
STAGE2_SYSTEM_VISION = f"""You are a receipt extraction and mapping system. Look at this receipt image and extract fields, mapping each to the most appropriate canonical concept.

Canonical concepts:
{CONCEPT_LIST}

Return a JSON array of objects. Each object must have:
- "label": the field label as shown on the receipt
- "value": the corresponding value as shown on the receipt
- "concept": the canonical concept name from the list above

Rules:
- Only include fields that clearly match a canonical concept
- Each concept may appear at most once
- Do not fabricate values — only use text visible on the receipt
- Note: receipts may be in Indonesian (Bahasa Indonesia)"""


def normalize_cord_value(v) -> str:
    """CORD-specific normalization: Indonesian '.' thousands separator."""
    v = str(v).lower().strip()
    v = v.replace("rp", "").replace("rp.", "").strip()
    v = v.replace(",", "")
    # "45.000" -> "45000" (Indonesian thousands separator), but keep genuine decimals
    import re
    if re.fullmatch(r"\d{1,3}(\.\d{3})+", v):
        v = v.replace(".", "")
    v = re.sub(r"\s+", " ", v).strip()
    return v


def flatten_cord_gt(gt_parse: dict) -> dict:
    flat = {}
    for section_key, section_val in gt_parse.items():
        if isinstance(section_val, dict):
            for field_key, field_val in section_val.items():
                if field_key in EVAL_CONCEPTS and field_key not in flat:
                    if isinstance(field_val, str) and field_val.strip():
                        flat[field_key] = field_val.strip()
                    elif isinstance(field_val, list):
                        for v in field_val:
                            if isinstance(v, str) and v.strip():
                                flat[field_key] = v.strip()
                                break
        elif isinstance(section_val, list):
            for item in section_val:
                if isinstance(item, dict):
                    for field_key, field_val in item.items():
                        if field_key in EVAL_CONCEPTS and field_key not in flat:
                            if isinstance(field_val, str) and field_val.strip():
                                flat[field_key] = field_val.strip()
    return flat


def build_samples(n_samples: int) -> list:
    from datasets import load_dataset
    from PIL import Image as PILImage

    dataset = load_dataset("naver-clova-ix/cord-v2", split="test")
    print(f"CORD test set: {len(dataset)} receipts")

    cord_data = []
    for i, sample in enumerate(dataset):
        gt_raw = json.loads(sample["ground_truth"])
        gt_parse = gt_raw.get("gt_parse", {})
        flat_gt = flatten_cord_gt(gt_parse)
        cord_data.append({"index": i, "image": sample["image"], "flat_gt": flat_gt, "n": len(flat_gt)})

    qualified = [d for d in cord_data if d["n"] >= 3]
    print(f"Receipts with >= 3 eval concepts: {len(qualified)} / {len(cord_data)}")

    random.seed(42)
    if len(qualified) > 100:
        sampled = random.sample(qualified, 100)
    else:
        sampled = qualified[:]
    sampled.sort(key=lambda d: d["index"])
    sampled = sampled[:n_samples]

    os.makedirs(IMG_DIR, exist_ok=True)
    samples = []
    for d in sampled:
        doc_id = f"cord_{d['index']:04d}"
        img_path = os.path.join(IMG_DIR, f"{doc_id}.png")
        img = d["image"]
        if isinstance(img, PILImage.Image):
            img.save(img_path)
        samples.append({"doc_id": doc_id, "image_path": img_path, "ground_truth": d["flat_gt"]})
    return samples


def encode_image(image_path: str, max_width: int = 1024) -> str:
    from PIL import Image as PILImage

    img = PILImage.open(image_path)
    if img.width > max_width:
        ratio = max_width / img.width
        img = img.resize((max_width, int(img.height * ratio)), PILImage.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.standard_b64encode(buf.getvalue()).decode("utf-8")


def parse_json_array(raw: str):
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
    return json.loads(raw)


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


def extract(client, model_id: str, image_path: str, max_retries=5):
    b64 = encode_image(image_path)
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_id,
                max_tokens=1024,
                messages=[
                    {"role": "system", "content": STAGE2_SYSTEM_VISION},
                    {"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                        {"type": "text", "text": "Extract and map the fields from this receipt."},
                    ]},
                ],
            )
            content = response.choices[0].message.content
            usage = response.usage
            cost = getattr(usage, "cost", None) if usage else None
            arr = parse_json_array(content)
            return array_to_predictions(arr), cost
        except Exception as e:
            is_rate_limit = "429" in str(e) or "rate_limit" in str(e).lower()
            if attempt < max_retries - 1:
                wait = 2 * (2 ** attempt) if is_rate_limit else 2
                print(f"    retry {attempt + 1}/{max_retries - 1} in {wait}s: {type(e).__name__}: {e}")
                time.sleep(wait)
            else:
                raise


def get_balance(api_key) -> float | None:
    import httpx
    try:
        r = httpx.get(f"{GATEWAY_BASE_URL}/credits", headers={"Authorization": f"Bearer {api_key}"}, timeout=10)
        return float(r.json()["balance"])
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=3)
    parser.add_argument("--out", default="/tmp/cord_pilot.json")
    parser.add_argument("--models", nargs="+", default=list(MODELS.keys()))
    args = parser.parse_args()

    api_key = os.environ.get("LLMGATEWAY_API_KEY")
    if not api_key:
        raise SystemExit("LLMGATEWAY_API_KEY not set in environment")

    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url=GATEWAY_BASE_URL)

    balance_before = get_balance(api_key)
    print(f"Gateway balance before run: ${balance_before}")

    samples = build_samples(args.n_samples)
    print(f"Sampled {len(samples)} CORD receipts (Track B / vision only)")

    results = {}
    for model_name in args.models:
        model_id = MODELS[model_name]
        print(f"\n=== {model_name} ({model_id}) ===")
        doc_results = []
        for i, s in enumerate(samples):
            pred, cost = extract(client, model_id, s["image_path"])
            doc_results.append({"doc_id": s["doc_id"], "ground_truth": s["ground_truth"], "predicted": pred, "cost": cost})
            if (i + 1) % 10 == 0 or i == 0:
                print(f"  [{i + 1}/{len(samples)}] {s['doc_id']}: cost=${cost}")
            time.sleep(0.5)

        documents = []
        for d in doc_results:
            gt_items = [(c, normalize_cord_value(v)) for c, v in d["ground_truth"].items()]
            pred_items = [(c, normalize_cord_value(v)) for c, v in d["predicted"].items() if v not in (None, "")]
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
