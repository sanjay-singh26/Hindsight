"""Category C1 re-scoring: Paystubs (Tesseract OCR + LLM text mapping), via LLMGateway.

Reproduces notebooks/hindsight_paystub_cba_v2.ipynb's stratified sampling
(same random.seed(42), same layout-stratified 50-paystub sample) and
two-stage extraction (local Tesseract OCR -> LLM text-only concept
mapping via LLMGateway), scored with the corrected cba_scorer.py.

Usage:
    LLMGATEWAY_API_KEY=... python3 scripts/rescore_paystubs.py --n-samples 3 --out /tmp/paystub_pilot.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from collections import defaultdict, Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cba_scorer import score_corpus  # noqa: E402

GATEWAY_BASE_URL = "https://api.llmgateway.io/v1"
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "paystubs")
MANIFEST_PATH = os.path.join(DATA_DIR, "manifest.json")
IMAGE_DIR = os.path.join(DATA_DIR, "images")

EVAL_CONCEPTS = [
    "employee_name", "employer_name",
    "pay_date", "pay_period_start", "pay_period_end", "pay_frequency",
    "gross_pay_current", "net_pay_current", "gross_pay_ytd", "net_pay_ytd",
    "federal_tax_withheld", "state_tax_withheld",
    "social_security_tax", "medicare_tax",
]

MODELS = {
    "haiku": "claude-haiku-4-5",
    "gpt4o-mini": "gpt-4o-mini",
}

STAGE2_SYSTEM = """You are a data mapping system. Given raw OCR text from a paystub document, identify fields and map each to the most appropriate canonical concept.

Canonical concepts:
employee_name, employer_name, pay_date, pay_period_start, pay_period_end, pay_frequency, gross_pay_current, net_pay_current, gross_pay_ytd, net_pay_ytd, federal_tax_withheld, state_tax_withheld, social_security_tax, medicare_tax

Return a JSON array of objects. Each object must have:
- "label": the field label exactly as it appears in the OCR text
- "value": the corresponding value exactly as it appears in the OCR text
- "concept": the canonical concept name from the list above

Rules:
- Only include fields that clearly match a canonical concept.
- Each concept may appear at most once.
- Do not fabricate values — only use text present in the OCR output.
- For monetary values, return the string as shown in the OCR text (with $ and commas if present)."""


def normalize_value(v) -> str:
    v = str(v).lower().strip().replace("$", "").replace(",", "")
    v = re.sub(r"\s+", " ", v)
    if re.fullmatch(r"-?\d+\.\d+", v):
        v = v.rstrip("0").rstrip(".")
    return v


def build_samples(n_samples: int) -> list:
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)

    # Filter out entries whose image file is missing/empty (a pre-existing data
    # issue: 6/200 paystub PNGs in this repo are 0 bytes) so the stratified
    # sample draws only from images that can actually be OCR'd.
    n_before = len(manifest)
    manifest = [
        ps for ps in manifest
        if os.path.getsize(os.path.join(IMAGE_DIR, ps["image_file"])) > 0
    ]
    n_skipped = n_before - len(manifest)
    if n_skipped:
        print(f"Skipping {n_skipped} paystub(s) with empty/missing image files")

    for ps in manifest:
        gt = {}
        for field in ps["fields"]:
            gt[field["canonical_concept_id"]] = field["extracted_value"]
        ps["_gt"] = gt
        ps["_gross_eq_regular"] = (
            normalize_value(gt.get("gross_pay_current", "")) == normalize_value(gt.get("regular_pay_current", ""))
            if "gross_pay_current" in gt and "regular_pay_current" in gt
            else False
        )

    by_layout = defaultdict(list)
    for ps in manifest:
        by_layout[ps["layout_template"]].append(ps)

    random.seed(42)
    sampled = []
    for layout, paystubs in sorted(by_layout.items()):
        n_per_layout = max(3, round(50 * len(paystubs) / len(manifest)))
        not_equal = [ps for ps in paystubs if not ps["_gross_eq_regular"]]
        equal = [ps for ps in paystubs if ps["_gross_eq_regular"]]
        n_not_equal = min(len(not_equal), max(n_per_layout // 2 + 1, n_per_layout - len(equal)))
        n_equal = min(len(equal), n_per_layout - n_not_equal)
        random.shuffle(not_equal)
        random.shuffle(equal)
        sampled.extend(not_equal[:n_not_equal])
        sampled.extend(equal[:n_equal])

    random.shuffle(sampled)
    if len(sampled) > 50:
        sampled = sampled[:50]
    elif len(sampled) < 50:
        remaining = [ps for ps in manifest if ps not in sampled]
        random.shuffle(remaining)
        sampled.extend(remaining[:50 - len(sampled)])

    sampled.sort(key=lambda ps: ps["paystub_id"])
    sampled = sampled[:n_samples]

    samples = []
    for ps in sampled:
        gt_eval = {k: v for k, v in ps["_gt"].items() if k in EVAL_CONCEPTS}
        samples.append({
            "doc_id": ps["paystub_id"],
            "image_path": os.path.join(IMAGE_DIR, ps["image_file"]),
            "ground_truth": gt_eval,
        })
    return samples


def stage1_tesseract(image_path: str) -> str:
    import pytesseract
    from PIL import Image

    img = Image.open(image_path)
    return pytesseract.image_to_string(img, config="--psm 6")


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


def stage2_call(client, model_id: str, ocr_text: str, max_retries=5):
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_id,
                max_tokens=1024,
                messages=[
                    {"role": "system", "content": STAGE2_SYSTEM},
                    {"role": "user", "content": f"Here is the raw OCR text from a paystub:\n\n{ocr_text}\n\nMap each field to the appropriate canonical concept."},
                ],
            )
            content = response.choices[0].message.content
            usage = response.usage
            cost = getattr(usage, "cost", None) if usage else None
            return array_to_predictions(parse_json_array(content)), cost
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
    parser.add_argument("--out", default="/tmp/paystub_pilot.json")
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
    print(f"Sampled {len(samples)} paystubs")

    ocr_cache = {}
    for s in samples:
        ocr_cache[s["doc_id"]] = stage1_tesseract(s["image_path"])
    print(f"Tesseract OCR complete for {len(ocr_cache)} images")
    sample0 = samples[0]["doc_id"]
    lines = [l for l in ocr_cache[sample0].strip().split("\n") if l.strip()]
    print(f"  Sanity check ({sample0}): {len(lines)} non-empty lines, first 3: {lines[:3]}")

    results = {}
    for model_name in args.models:
        model_id = MODELS[model_name]
        print(f"\n=== {model_name} ({model_id}) ===")
        doc_results = []
        for i, s in enumerate(samples):
            pred, cost = stage2_call(client, model_id, ocr_cache[s["doc_id"]])
            doc_results.append({"doc_id": s["doc_id"], "ground_truth": s["ground_truth"], "predicted": pred, "cost": cost})
            if (i + 1) % 10 == 0 or i == 0:
                print(f"  [{i + 1}/{len(samples)}] {s['doc_id']}: cost=${cost}")
            time.sleep(0.5)

        documents = []
        for d in doc_results:
            gt_items = [(c, normalize_value(v)) for c, v in d["ground_truth"].items()]
            pred_items = [(c, normalize_value(v)) for c, v in d["predicted"].items() if v not in (None, "")]
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
