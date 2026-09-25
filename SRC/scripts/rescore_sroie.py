"""Category C1 re-scoring: SROIE, via LLMGateway (vision).

Reproduces notebooks/hindsight_sroie_cba.ipynb's sampling (same
random.seed(42), same 50-receipt sample, same 14-concept extraction
schema; GT backs only 4 concepts), extracts via vision through
LLMGateway's OpenAI-compatible endpoint for both models, and scores
with the corrected cba_scorer.py (standard value-first Hungarian
matching — this intentionally does not replicate the original
notebook's bespoke store_address containment-matching special case,
since standardizing scoring across domains is the point of C1).

Usage:
    LLMGATEWAY_API_KEY=... python3 scripts/rescore_sroie.py --n-samples 3 --out /tmp/sroie_pilot.json
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
IMG_DIR = "/tmp/sroie_images"

RECEIPT_ONTOLOGY_FAMILIES = {
    "merchant": ["store_name", "store_address", "store_city_state_zip", "store_phone"],
    "transaction": ["transaction_date", "transaction_time", "receipt_number", "cashier", "payment_method"],
    "amounts": ["subtotal", "tax_amount", "total", "amount_tendered", "change_due"],
}
ALL_CONCEPTS = [c for fam in RECEIPT_ONTOLOGY_FAMILIES.values() for c in fam]

SROIE_TO_CANONICAL = {
    "company": "store_name",
    "address": "store_address",
    "date": "transaction_date",
    "total": "total",
}
GT_CONCEPTS = list(SROIE_TO_CANONICAL.values())

MODELS = {
    "haiku": "claude-haiku-4-5",
    "gpt4o-mini": "gpt-4o-mini",
}

SYSTEM_PROMPT = """You are a receipt extraction system. Given a receipt image, extract values and map each to the correct canonical concept.

Canonical concepts:
- store_name: Name of the store or business
- store_address: Store street address
- store_city_state_zip: Store city, state/province, and postal code
- store_phone: Store phone number
- transaction_date: Date of the transaction
- transaction_time: Time of the transaction
- receipt_number: Receipt or transaction ID number
- cashier: Cashier name or ID
- payment_method: Payment method (cash, card, etc.)
- subtotal: Sum of items BEFORE tax
- tax_amount: Tax amount charged
- total: Final total amount INCLUDING tax
- amount_tendered: Amount the customer paid
- change_due: Change returned to customer

Return a JSON object with exactly these 14 keys.
Monetary values: numeric string without currency symbols or commas (e.g., "4.95").
Dates: preserve format as shown on receipt.
If a field is not present on the receipt, use "N/A".
Return ONLY valid JSON, no other text."""


def parse_sroie_entities(rec) -> dict:
    fields = rec.get("fields", {})
    if isinstance(fields, str):
        fields = json.loads(fields)
    return {
        "company": fields.get("COMPANY", fields.get("company", "")),
        "date": fields.get("DATE", fields.get("date", "")),
        "address": fields.get("ADDRESS", fields.get("address", "")),
        "total": fields.get("TOTAL", fields.get("total", "")),
    }


def build_samples(n_samples: int) -> list:
    from datasets import load_dataset
    from PIL import Image as PILImage

    ds = load_dataset("sizhkhy/SROIE")
    train_data = ds["train"]
    print(f"SROIE loaded: {len(train_data)} receipts")

    random.seed(42)
    os.makedirs(IMG_DIR, exist_ok=True)

    indices = list(range(len(train_data)))
    random.shuffle(indices)
    selected = sorted(indices[:50])[:n_samples]  # same first-50 selection as the paper, capped for pilots

    samples = []
    for i, idx in enumerate(selected):
        rec = train_data[idx]
        entities = parse_sroie_entities(rec)

        gt = {c: "N/A" for c in ALL_CONCEPTS}
        for sroie_key, canonical_key in SROIE_TO_CANONICAL.items():
            val = entities.get(sroie_key, "")
            if val and str(val).strip():
                gt[canonical_key] = str(val).strip()

        doc_id = f"sroie_{i:04d}"
        img_path = os.path.join(IMG_DIR, f"{doc_id}.png")
        img = rec["images"]
        if isinstance(img, PILImage.Image):
            img.save(img_path)
        elif isinstance(img, bytes):
            with open(img_path, "wb") as f:
                f.write(img)
        elif isinstance(img, dict) and "bytes" in img:
            with open(img_path, "wb") as f:
                f.write(img["bytes"])

        samples.append({"doc_id": doc_id, "image_path": img_path, "ground_truth": gt})
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


def parse_json_object(raw: str):
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
    return json.loads(raw)


def extract(client, model_id: str, image_path: str, max_retries=5):
    b64 = encode_image(image_path)
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_id,
                max_tokens=1024,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                        {"type": "text", "text": "Extract all fields from this receipt image."},
                    ]},
                ],
            )
            content = response.choices[0].message.content
            usage = response.usage
            cost = getattr(usage, "cost", None) if usage else None
            return parse_json_object(content), cost
        except Exception as e:
            is_rate_limit = "429" in str(e) or "rate_limit" in str(e).lower()
            if attempt < max_retries - 1:
                wait = min(10 * (2 ** attempt), 120) if is_rate_limit else 2 ** (attempt + 1)
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
    parser.add_argument("--out", default="/tmp/sroie_pilot.json")
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
    print(f"Sampled {len(samples)} receipts")

    results = {}
    for model_name in args.models:
        model_id = MODELS[model_name]
        print(f"\n=== {model_name} ({model_id}) ===")
        doc_results = []
        for i, s in enumerate(samples):
            pred, cost = extract(client, model_id, s["image_path"])
            doc_results.append({"doc_id": s["doc_id"], "ground_truth": s["ground_truth"], "predicted": pred, "cost": cost})
            print(f"  [{i + 1}/{len(samples)}] {s['doc_id']}: cost=${cost}")
            time.sleep(0.5)

        documents = []
        for d in doc_results:
            gt_items = [(c, v) for c, v in d["ground_truth"].items() if v != "N/A"]
            pred_items = [(c, v) for c, v in d["predicted"].items() if v not in (None, "", "N/A")]
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
