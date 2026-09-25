"""Category C3 prompt-format ablation: CUAD under schema-constrained output.

The C1 re-score already used CUAD's original "free JSON array of
{clause, concept}" prompt (Condition 1: Free JSON — the model discovers
which concept applies to each clause number and emits it as a label). This
script runs the SAME 50 bundles (identical random.seed(42) construction,
via rescore_cuad's build_bundles) under Condition 2: a schema-constrained
prompt that pre-allocates one fixed JSON key per canonical concept, and
asks the model to fill in which clause number belongs to each — the
model no longer emits a "concept" field at all; concept assignment is
implicit in which schema slot a clause number lands in.

Usage:
    LLMGATEWAY_API_KEY=... python3 scripts/ablate_cuad_schema.py --n-bundles 50 --out /tmp/cuad_schema.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import rescore_cuad as base  # noqa: E402
from cba_scorer import score_corpus  # noqa: E402

GATEWAY_BASE_URL = "https://api.llmgateway.io/v1"

SCHEMA_SYSTEM_PROMPT = """You are a legal contract clause classification system. Given a set of numbered clauses extracted from commercial contracts, determine which clause number corresponds to each of the following fixed categories.

Categories (in order):
- Agreement Date: The date the agreement was signed or made
- Effective Date: The date the agreement takes effect (may differ from signing date)
- Expiration Date: The date or conditions under which the agreement expires
- License Grant: Provisions granting a license to use IP, technology, or rights
- Non-Transferable License: Provisions restricting transfer or sublicensing of granted rights
- Irrevocable Or Perpetual License: Provisions granting irrevocable or perpetual license rights
- Exclusivity: Provisions granting exclusive rights or restricting competition
- Cap On Liability: Provisions limiting or capping a party's liability
- Uncapped Liability: Provisions carving out exceptions from liability caps (uncapped exposure)
- Liquidated Damages: Provisions specifying pre-determined damages for breach
- Termination For Convenience: Provisions allowing termination without cause
- Renewal Term: Provisions defining automatic renewal or extension terms
- Notice Period To Terminate Renewal: Provisions specifying notice required to prevent renewal
- Change Of Control: Provisions triggered by ownership changes or acquisitions
- Anti-Assignment: Provisions restricting assignment of the agreement

Return ONLY a JSON object with exactly these 15 keys (use the exact category names above as keys). The value for each key must be the integer clause number that best matches that category. If you are not confident any clause matches a category, still provide your best guess (every category must appear in every document)."""


def parse_json_object(raw: str):
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
    return json.loads(raw)


def classify_bundle_schema(client, model_id: str, bundle_text: str, max_retries=5, base_delay=2):
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_id,
                max_tokens=1024,
                messages=[
                    {"role": "system", "content": SCHEMA_SYSTEM_PROMPT},
                    {"role": "user", "content": f"Classify each clause below:\n\n{bundle_text}"},
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


def schema_obj_to_pred_mapping(obj: dict) -> dict:
    """Invert {concept: clause_num} -> {clause_num: concept} (first concept wins a given clause_num)."""
    pred = {}
    for concept, clause_num in obj.items():
        try:
            cn = int(clause_num)
        except (TypeError, ValueError):
            continue
        if cn not in pred:
            pred[cn] = concept
    return pred


def get_balance(api_key):
    import httpx
    try:
        r = httpx.get(f"{GATEWAY_BASE_URL}/credits", headers={"Authorization": f"Bearer {api_key}"}, timeout=10)
        return float(r.json()["balance"])
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-bundles", type=int, default=50)
    parser.add_argument("--out", default="/tmp/cuad_schema.json")
    parser.add_argument("--models", nargs="+", default=["haiku", "gpt4o-mini"])
    args = parser.parse_args()

    api_key = os.environ.get("LLMGATEWAY_API_KEY")
    if not api_key:
        raise SystemExit("LLMGATEWAY_API_KEY not set in environment")

    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url=GATEWAY_BASE_URL)

    balance_before = get_balance(api_key)
    print(f"Gateway balance before run: ${balance_before}")

    cuad_data = base.ensure_cuad_data()
    span_pool = base.build_span_pool(cuad_data)
    bundles = base.build_bundles(span_pool, args.n_bundles)
    print(f"Built {len(bundles)} bundles (identical to the free-JSON condition's bundles)")

    results = {}
    for model_name in args.models:
        model_id = base.MODELS[model_name]
        print(f"\n=== {model_name} ({model_id}) — schema-constrained ===")
        bundle_results = []
        for b in bundles:
            obj, cost = classify_bundle_schema(client, model_id, b["text"])
            pred_mapping = schema_obj_to_pred_mapping(obj)
            bundle_results.append({"bundle_id": b["bundle_id"], "gt_mapping": b["gt_mapping"], "pred_mapping": pred_mapping})
            if (b["bundle_id"] + 1) % 10 == 0 or b["bundle_id"] == 0:
                print(f"  bundle {b['bundle_id']}: {len(pred_mapping)}/15 slots filled, cost=${cost}")
            time.sleep(1.0)

        documents = []
        for br in bundle_results:
            gt_items = [(concept, str(cn)) for cn, concept in br["gt_mapping"].items()]
            pred_items = [(concept, str(cn)) for cn, concept in br["pred_mapping"].items()]
            documents.append((gt_items, pred_items))

        corpus = score_corpus(documents)
        results[model_name] = {"model_id": model_id, "condition": "schema_constrained", "scorer_result": corpus.to_dict(), "bundles": bundle_results}
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
