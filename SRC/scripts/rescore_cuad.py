"""Category C1 re-scoring pilot: CUAD, via LLMGateway.

Reproduces the exact bundle construction from
notebooks/hindsight_cuad_cba.ipynb (same EVAL_CONCEPTS, same span-pool
logic, same random.seed(42)) so results are comparable to the original
paper run, then classifies each bundle with both models through
LLMGateway (OpenAI-compatible endpoint), and scores with the corrected
cba_scorer.py.

Usage:
    LLMGATEWAY_API_KEY=... python3 scripts/rescore_cuad.py --n-bundles 3 --out /tmp/cuad_pilot.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cba_scorer import score_corpus  # noqa: E402

GATEWAY_BASE_URL = "https://api.llmgateway.io/v1"
CUAD_URL = "https://raw.githubusercontent.com/TheAtticusProject/cuad/main/data.zip"
CUAD_CACHE_DIR = "/tmp/cuad_data"
CUAD_PATH = os.path.join(CUAD_CACHE_DIR, "CUADv1.json")

EVAL_CONCEPTS = [
    "Agreement Date", "Effective Date", "Expiration Date",
    "License Grant", "Non-Transferable License", "Irrevocable Or Perpetual License", "Exclusivity",
    "Cap On Liability", "Uncapped Liability", "Liquidated Damages",
    "Termination For Convenience", "Renewal Term", "Notice Period To Terminate Renewal",
    "Change Of Control", "Anti-Assignment",
]

MAX_SPAN_LENGTH = 1500
CONTEXT_WINDOW = 200
SHORT_SPAN_THRESHOLD = 100

MODELS = {
    "haiku": "claude-haiku-4-5",
    "gpt4o-mini": "gpt-4o-mini",
}

SYSTEM_PROMPT = """You are a legal contract clause classification system. Given a set of numbered clauses extracted from commercial contracts, classify each clause into exactly one of the following categories.

Categories:
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

For short spans (e.g., dates), surrounding context is provided in the format:
  ...context before... >>>SPAN<<< ...context after...
Classify the highlighted >>>SPAN<<< text, using the context to determine its role.

Return a JSON array where each item has:
- "clause": the clause number (integer)
- "concept": the category name from the list above (exact string match)

Rules:
- Classify EVERY clause — do not skip any
- Each clause gets exactly one category
- Use the exact category names listed above
- For spans with >>>markers<<<, classify based on the highlighted text's role in context"""


def ensure_cuad_data() -> list:
    if not os.path.exists(CUAD_PATH):
        os.makedirs(CUAD_CACHE_DIR, exist_ok=True)
        os.system(f'cd {CUAD_CACHE_DIR} && curl -sL "{CUAD_URL}" -o data.zip && unzip -oq data.zip')
    with open(CUAD_PATH) as f:
        return json.load(f)["data"]


def build_span_pool(cuad_data: list) -> dict:
    def build_clause_text(span_text, context_before, context_after, is_short):
        if is_short and (context_before or context_after):
            parts = []
            if context_before:
                parts.append(f"...{context_before[-CONTEXT_WINDOW:]}")
            parts.append(f" >>>{span_text}<<< ")
            if context_after:
                parts.append(f"{context_after[:CONTEXT_WINDOW]}")
            return "".join(parts)
        text = span_text[:MAX_SPAN_LENGTH]
        if len(span_text) > MAX_SPAN_LENGTH:
            text += "..."
        return text

    span_pool = defaultdict(list)
    for doc in cuad_data:
        for para in doc["paragraphs"]:
            context = para["context"]
            for qa in para["qas"]:
                q = qa["question"]
                cat = q.split('"')[1] if '"' in q else q.strip()
                if cat not in EVAL_CONCEPTS:
                    continue
                for a in qa["answers"]:
                    if not a["text"].strip():
                        continue
                    span_text = a["text"].strip()
                    start = a["answer_start"]
                    end = start + len(a["text"])
                    ctx_before = context[max(0, start - CONTEXT_WINDOW):start].strip()
                    ctx_after = context[end:min(len(context), end + CONTEXT_WINDOW)].strip()
                    is_short = len(span_text) < SHORT_SPAN_THRESHOLD
                    span_pool[cat].append(build_clause_text(span_text, ctx_before, ctx_after, is_short))
    return span_pool


def build_bundles(span_pool: dict, n_bundles: int) -> list:
    random.seed(42)
    shuffled_pools = {}
    for concept in EVAL_CONCEPTS:
        pool = span_pool[concept].copy()
        random.shuffle(pool)
        shuffled_pools[concept] = pool

    pool_idx = {c: 0 for c in EVAL_CONCEPTS}
    bundles = []
    for bundle_i in range(n_bundles):
        clauses = []
        gt_mapping = {}
        for concept in EVAL_CONCEPTS:
            idx = pool_idx[concept] % len(shuffled_pools[concept])
            text = shuffled_pools[concept][idx]
            pool_idx[concept] += 1
            clauses.append((concept, text))
        random.shuffle(clauses)

        doc_parts = []
        for clause_num, (concept, text) in enumerate(clauses, 1):
            doc_parts.append(f"Clause {clause_num}:\n{text}")
            gt_mapping[clause_num] = concept

        bundles.append({
            "bundle_id": bundle_i,
            "text": "\n\n".join(doc_parts),
            "gt_mapping": gt_mapping,
        })
    return bundles


def parse_json_array(raw: str):
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
    return json.loads(raw)


def classify_bundle(client, model_id: str, bundle_text: str, max_retries=5, base_delay=2):
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_id,
                max_tokens=1024,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Classify each clause below:\n\n{bundle_text}"},
                ],
            )
            content = response.choices[0].message.content
            usage = response.usage
            cost = getattr(usage, "cost", None) if usage else None
            return parse_json_array(content), cost
        except Exception as e:
            is_rate_limit = "429" in str(e) or "rate_limit" in str(e).lower()
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt) if is_rate_limit else base_delay
                print(f"    retry {attempt + 1}/{max_retries - 1} in {delay}s: {type(e).__name__}: {e}")
                time.sleep(delay)
            else:
                raise


def array_to_pred_mapping(arr) -> dict:
    pred = {}
    for item in arr:
        if not isinstance(item, dict):
            continue
        clause = item.get("clause") or item.get("clause_number") or item.get("provision")
        concept = item.get("concept", "")
        if clause is not None and concept:
            pred[int(clause)] = concept
    return pred


def get_balance(client_kwargs) -> float | None:
    import httpx
    try:
        r = httpx.get(
            f"{GATEWAY_BASE_URL}/credits",
            headers={"Authorization": f"Bearer {client_kwargs['api_key']}"},
            timeout=10,
        )
        return float(r.json()["balance"])
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-bundles", type=int, default=3)
    parser.add_argument("--out", default="/tmp/cuad_pilot.json")
    parser.add_argument("--models", nargs="+", default=list(MODELS.keys()))
    args = parser.parse_args()

    api_key = os.environ.get("LLMGATEWAY_API_KEY")
    if not api_key:
        raise SystemExit("LLMGATEWAY_API_KEY not set in environment")

    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url=GATEWAY_BASE_URL)

    balance_before = get_balance({"api_key": api_key})
    print(f"Gateway balance before run: ${balance_before}")

    print("Loading CUAD data...")
    cuad_data = ensure_cuad_data()
    print(f"CUAD loaded: {len(cuad_data)} contracts")

    span_pool = build_span_pool(cuad_data)
    bundles = build_bundles(span_pool, args.n_bundles)
    print(f"Built {len(bundles)} bundles ({len(bundles) * 15} classifications per model)")

    results = {}
    total_cost = 0.0

    for model_name in args.models:
        model_id = MODELS[model_name]
        print(f"\n=== {model_name} ({model_id}) ===")
        bundle_results = []
        for b in bundles:
            arr, cost = classify_bundle(client, model_id, b["text"])
            pred_mapping = array_to_pred_mapping(arr)
            bundle_results.append({
                "bundle_id": b["bundle_id"],
                "gt_mapping": b["gt_mapping"],
                "pred_mapping": pred_mapping,
                "raw_array": arr,
                "cost": cost,
            })
            if cost:
                total_cost += cost
            print(f"  bundle {b['bundle_id']}: {len(pred_mapping)}/15 classified, cost=${cost}")
            time.sleep(1.0)

        # Score with the corrected Hungarian scorer, using clause_num as the "value"
        # (each clause is a fixed, uniquely-identified slot; the model classifies a
        # concept for a known clause rather than freely extracting a value).
        documents = []
        for br in bundle_results:
            gt_items = [(concept, str(cn)) for cn, concept in br["gt_mapping"].items()]
            pred_items = [(concept, str(cn)) for cn, concept in br["pred_mapping"].items()]
            documents.append((gt_items, pred_items))

        corpus = score_corpus(documents)
        results[model_name] = {
            "model_id": model_id,
            "scorer_result": corpus.to_dict(),
            "bundles": bundle_results,
        }
        print(f"  DONE: Rec_val={corpus.rec_val:.4f} CBA_cond={corpus.cba_cond:.4f} "
              f"ACC_joint={corpus.acc_joint:.4f} Delta={corpus.delta:.4f} "
              f"misbindings={corpus.n_misbindings}")

    balance_after = get_balance({"api_key": api_key})
    print(f"\nGateway balance after run: ${balance_after}")
    if balance_before is not None and balance_after is not None:
        print(f"Actual spend this run (from balance delta): ${balance_before - balance_after:.6f}")
    print(f"Summed per-call cost field: ${total_cost:.6f}")

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {args.out}")


if __name__ == "__main__":
    main()
