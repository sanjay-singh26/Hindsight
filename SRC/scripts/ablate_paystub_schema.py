"""Category C3 prompt-format ablation: Paystubs under schema-constrained output.

The C1 re-score already used Paystubs' original free-JSON-array prompt
(Condition 1: [{label, value, concept}] triples, Stage 2 mapping raw
Tesseract OCR text). This script runs the SAME 50 paystubs (identical
random.seed(42) sampling, via rescore_paystubs's build_samples, and the
same cached OCR text) under Condition 2: a schema-constrained prompt with
a fixed 14-key JSON object.

Usage:
    LLMGATEWAY_API_KEY=... python3 scripts/ablate_paystub_schema.py --n-samples 50 --out /tmp/paystub_schema.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import rescore_paystubs as base  # noqa: E402
from cba_scorer import score_corpus  # noqa: E402

GATEWAY_BASE_URL = "https://api.llmgateway.io/v1"

CONCEPT_LIST = ", ".join(base.EVAL_CONCEPTS)
SCHEMA_SYSTEM = f"""You are a data mapping system. Given raw OCR text from a paystub document, extract the value for each of the following fixed fields.

Fields (in order):
{CONCEPT_LIST}

Return ONLY a JSON object with exactly these {len(base.EVAL_CONCEPTS)} keys (use the exact field names above as keys). If a field is not present in the OCR text, use "N/A" as its value. Do not fabricate values — only use text present in the OCR output. For monetary values, return the string as shown in the OCR text (with $ and commas if present)."""


def parse_json_object(raw: str):
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
    return json.loads(raw)


def stage2_call_schema(client, model_id: str, ocr_text: str, max_retries=5):
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_id,
                max_tokens=1024,
                messages=[
                    {"role": "system", "content": SCHEMA_SYSTEM},
                    {"role": "user", "content": f"Here is the raw OCR text from a paystub:\n\n{ocr_text}\n\nFill in the JSON schema."},
                ],
            )
            content = response.choices[0].message.content
            usage = response.usage
            cost = getattr(usage, "cost", None) if usage else None
            return parse_json_object(content), cost
        except Exception as e:
            is_rate_limit = "429" in str(e) or "rate_limit" in str(e).lower()
            if attempt < max_retries - 1:
                wait = 2 * (2 ** attempt) if is_rate_limit else 2
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
    parser.add_argument("--out", default="/tmp/paystub_schema.json")
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
    print(f"Sampled {len(samples)} paystubs (identical to the free-JSON condition's samples)")

    ocr_cache = {}
    for s in samples:
        ocr_cache[s["doc_id"]] = base.stage1_tesseract(s["image_path"])
    print(f"Tesseract OCR complete for {len(ocr_cache)} images")

    results = {}
    for model_name in args.models:
        model_id = base.MODELS[model_name]
        print(f"\n=== {model_name} ({model_id}) — schema-constrained ===")
        doc_results = []
        for i, s in enumerate(samples):
            pred, cost = stage2_call_schema(client, model_id, ocr_cache[s["doc_id"]])
            doc_results.append({"doc_id": s["doc_id"], "ground_truth": s["ground_truth"], "predicted": pred, "cost": cost})
            if (i + 1) % 10 == 0 or i == 0:
                print(f"  [{i + 1}/{len(samples)}] {s['doc_id']}: cost=${cost}")
            time.sleep(0.5)

        documents = []
        for d in doc_results:
            gt_items = [(c, base.normalize_value(v)) for c, v in d["ground_truth"].items()]
            pred_items = [(c, base.normalize_value(v)) for c, v in d["predicted"].items() if v not in (None, "", "N/A")]
            documents.append((gt_items, pred_items))

        corpus = score_corpus(documents)
        results[model_name] = {"model_id": model_id, "condition": "schema_constrained", "scorer_result": corpus.to_dict(), "documents": doc_results}
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
