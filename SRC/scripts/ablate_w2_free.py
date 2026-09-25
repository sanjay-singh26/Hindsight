"""Category C3 prompt-format ablation: W-2 under free-JSON output.

The C1 re-score already used W-2's original schema-constrained prompt
(Condition 2: a fixed JSON object with all 44 canonical keys pre-allocated).
This script runs the SAME 50 W-2 forms (identical random.seed(42) sampling,
via rescore_w2's build_samples) under Condition 1: a free-JSON-array prompt
requesting [{label, value, concept}] triples, forcing the model to both
discover which fields are present AND generate/select the concept name for
each, rather than filling 44 pre-given slots.

Usage:
    LLMGATEWAY_API_KEY=... python3 scripts/ablate_w2_free.py --n-samples 50 --out /tmp/w2_free.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import rescore_w2 as base  # noqa: E402
from cba_scorer import score_corpus  # noqa: E402

GATEWAY_BASE_URL = "https://api.llmgateway.io/v1"

CONCEPT_LIST = ", ".join(base.ALL_CONCEPTS)
FREE_SYSTEM_PROMPT = f"""You are a document AI extraction system. You extract structured data from W-2 tax form images.

Given an image of a W-2 form, extract all fields you can identify, mapping each to the most appropriate canonical concept.

Canonical concepts:
{CONCEPT_LIST}

Return a JSON array of objects. Each object must have:
- "label": the field label as shown on the form (e.g., "Box 1", "Wages, tips, other comp.")
- "value": the corresponding value as shown on the form
- "concept": the canonical concept name from the list above that best matches this field

Rules:
- Only include fields that clearly match a canonical concept from the list
- Each concept should appear at most once
- Do not fabricate values — only use text visible on the form
- Monetary values: numeric string WITHOUT $ or commas (e.g., "4523.67")
- Return ONLY valid JSON, no other text."""


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


def extract_free(client, model_id: str, image_path: str, max_retries=5):
    b64 = base.encode_image(image_path)
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_id,
                max_tokens=2048,
                messages=[
                    {"role": "system", "content": FREE_SYSTEM_PROMPT},
                    {"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                        {"type": "text", "text": "Extract all fields from this W-2 tax form image."},
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
                wait = min(10 * (2 ** attempt), 120) if is_rate_limit else 2 ** (attempt + 1)
                print(f"    retry {attempt + 1}/{max_retries - 1} in {wait}s: {type(e).__name__}: {e}")
                time.sleep(wait)
            else:
                raise


def get_balance(api_key):
    import httpx
    try:
        r = httpx.get(f"{GATEWAY_BASE_URL}/credits", headers={"Authorization": f"Bearer {api_key}"}, timeout=10)
        return float(r.json()["balance"])
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=50)
    parser.add_argument("--out", default="/tmp/w2_free.json")
    parser.add_argument("--models", nargs="+", default=["haiku", "gpt4o-mini"])
    args = parser.parse_args()

    api_key = os.environ.get("LLMGATEWAY_API_KEY")
    if not api_key:
        raise SystemExit("LLMGATEWAY_API_KEY not set in environment")

    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url=GATEWAY_BASE_URL)

    balance_before = get_balance(api_key)
    print(f"Gateway balance before run: ${balance_before}")

    samples = base.build_samples(args.n_samples)
    print(f"Sampled {len(samples)} W-2 forms (identical to the schema-constrained condition's samples)")

    results = {}
    for model_name in args.models:
        model_id = base.MODELS[model_name]
        print(f"\n=== {model_name} ({model_id}) — free JSON ===")
        doc_results = []
        for i, s in enumerate(samples):
            pred, cost = extract_free(client, model_id, s["image_path"])
            doc_results.append({"doc_id": s["doc_id"], "ground_truth": s["ground_truth"], "predicted": pred, "cost": cost})
            if (i + 1) % 10 == 0 or i == 0:
                print(f"  [{i + 1}/{len(samples)}] {s['doc_id']}: {len(pred)} fields, cost=${cost}")
            time.sleep(0.5)

        documents = []
        for d in doc_results:
            gt_items = [(c, v) for c, v in d["ground_truth"].items() if v != "N/A"]
            pred_items = [(c, v) for c, v in d["predicted"].items() if v not in (None, "", "N/A")]
            documents.append((gt_items, pred_items))

        corpus = score_corpus(documents)
        results[model_name] = {"model_id": model_id, "condition": "free_json", "scorer_result": corpus.to_dict(), "documents": doc_results}
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
